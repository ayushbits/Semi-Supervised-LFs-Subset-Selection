import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

import faulthandler

from cage import probability


class LearnMultiLambdaMeta(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
        ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
    """

    def __init__(self, trainloader, val_x, val_y, model, num_classes, N_trn, loss, device, fit, \
                 teacher_model, theta, pi_y, pi,k,continuous_mask,criterion_red, temp):
        """
        Constructer method
        """
        self.num_classes = num_classes
        self.device = device
        
        self.trainloader = trainloader  # assume its a sequential loader.
        self.val_x = torch.tensor(val_x).to(self.device)
        self.val_y = torch.tensor(val_y).to(self.device)
        self.model = model
        self.N_trn = N_trn

        self.fit = fit

        self.teacher_model = teacher_model
        self.theta = theta
        self.pi_y = pi_y
        self.pi = pi
        self.k = k 
        self.continuous_mask = continuous_mask
        self.criterion = loss
        self.criterion_red = criterion_red
        self.temp = temp
        print(N_trn)

    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)
        self.model.eval()

    
    def get_lambdas(self, eta,epoch,lam):

        self.model.eval()
        
        self.teacher_model.train()

        with torch.no_grad():

            outputs = self.model(self.val_x)
            loss = self.criterion_red(outputs, self.val_y)
            # print(loss.item(),end=",")
        # print()
        
        max_value = 2
        
        for batch_idx, (sample,indices) in enumerate(self.trainloader):

            unsupervised_indices = (1-sample[4]).nonzero().squeeze().view(-1)
           
            # Theta Model
            lr_outputs,l1 = self.model(sample[0][unsupervised_indices],last=True, freeze=True)

            #probs_theta = F.log_softmax(lr_outputs)
            #outputs = np.argmax(probs_theta.detach().numpy(), 1)
            
            # GM Model Test
            probs_graphical =(probability(self.theta, self.pi_y, self.pi,sample[2][unsupervised_indices],\
                sample[3][unsupervised_indices],self.k, self.num_classes, self.continuous_mask))
            targets = (probs_graphical.t() / probs_graphical.sum(1)).t()
            #outputs_tensor = torch.Tensor(outputs)
            targets_tensor = torch.Tensor(targets)

            loss_SL = self.criterion_red(lr_outputs, targets_tensor)
            
            l0_grads = (torch.autograd.grad(loss_SL, lr_outputs)[0]).detach().clone().to(self.device)
            l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
            l1_grads = l0_expand * l1.detach().repeat(1, self.num_classes).to(self.device)

            probs_theta_kl = F.log_softmax(lr_outputs / self.temp, dim =1)
            probs_lr_teacher = F.softmax(self.teacher_model(sample[0][unsupervised_indices])/self.temp, dim=1)
            loss_KD = self.temp * self.temp *torch.nn.KLDivLoss(reduction='batchmean')(probs_theta_kl, probs_lr_teacher)

            l0_grads_KD = (torch.autograd.grad(loss_KD, lr_outputs)[0]).detach().clone().to(self.device)
            l0_expand_KD = torch.repeat_interleave(l0_grads_KD, l1.shape[1], dim=1)
            l1_grads_KD = l0_expand_KD * l1.detach().repeat(1, self.num_classes).to(self.device)
            
            
            if batch_idx % self.fit == 0:
                SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                KD_grads = torch.cat((l0_grads_KD, l1_grads_KD), dim=1)
                batch_ind = list(indices[unsupervised_indices]) 
            else:
                SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                KD_grads = torch.cat((KD_grads, torch.cat((l0_grads_KD, l1_grads_KD), dim=1)), dim=0)
                batch_ind.extend(list(indices[unsupervised_indices]))
          
            if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):

                with torch.no_grad():

                    out, l1 = self.model(self.val_x.to(self.device), last=True, freeze=True)
                    self.init_out = out.detach().to(self.device)
                    self.init_l1 = l1.detach().to(self.device)
                
                for r in range(5):

                    lambdas = lam.clone().to(self.device)#, device=self.device)
                    lambdas.requires_grad = True
                    
                    lambdas1 = lambdas[batch_ind,0][:,None]
                    lambdas_2 = lambdas[batch_ind,1][:,None]

                    #print(lambdas1.shape,lambdas_2.shape,SL_grads.shape,KD_grads.shape)
                    comb_grad_all = lambdas1*SL_grads + lambdas_2*KD_grads

                    comb_grad = comb_grad_all.sum(0)

                    out_vec_val = self.init_out - (eta * comb_grad[:self.num_classes].view(1, -1).\
                        expand(self.init_out.shape[0], -1))

                    out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    #out_vec_val.requires_grad = True
                    loss_SL_val = self.criterion_red(out_vec_val, self.val_y.to(self.device))

                    alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas1,retain_graph=True)[0]).detach().clone().to(self.device)  
                    lam[batch_ind,0] = lam[batch_ind,0] - 100*alpha_grads.view(-1)

                    alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas_2)[0]).detach().clone().to(self.device) 
                    lam[batch_ind,1] = lam[batch_ind,1] - 100*alpha_grads.view(-1)

                    # if (batch_idx + 1) % (self.fit*10) ==0:
                        # if r ==0:
                        #     print(round(self.criterion_red(self.init_out, self.val_y.to(self.device)).item(),4))
                        # print(alpha_grads[0],round(loss_SL_val.item(),4),end=",")#"+",round(loss_KD_trn.item(),4), )
                        #print(self.init_out[r][self.y_val[r]],out_vec_val[r][self.y_val[r]])
                        #print(alpha_grads[:3])
                    del out_vec_val
                    
                    lam.clamp_(min=1e-7,max=max_value-1e-7)
                    
                if (batch_idx + 1) % (self.fit*40) ==0:
                    print()#"End for loop")

        #lambdas.clamp_(min=0.01,max=0.99)
        return lam