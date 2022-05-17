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
                 theta, pi_y, pi,k,continuous_mask,criterion_red, temp):
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

        #self.teacher_model = teacher_model
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

        with torch.no_grad():

            outputs = self.model(self.val_x)
            loss = self.criterion_red(outputs, self.val_y)
            # print(loss.item(),end=",")
        # print()
        
        max_value = 2
        
        for batch_idx, (sample,indices) in enumerate(self.trainloader):

            supervised_indices = sample[4].nonzero().view(-1)
            # unsupervised_indices = indices  ## Uncomment for entropy
            unsupervised_indices = (1-sample[4]).nonzero().squeeze().view(-1)

            # Theta Model
            lr_outputs,l1 = self.model(sample[0],last=True, freeze=True)
            
            # Case I - CE for theta model
            if len(supervised_indices) > 0:
                sup_lr = lr_outputs[supervised_indices]
                loss_1 = self.criterion_red(sup_lr, sample[1][supervised_indices])

                l0_grads = (torch.autograd.grad(loss_1, sup_lr)[0]).detach().clone().to(self.device)
                l0_expand = torch.repeat_interleave(l0_grads, l1[supervised_indices].shape[1], dim=1)
                l1_grads = l0_expand * l1[supervised_indices].detach().repeat(1, self.num_classes).to(self.device)

            un_lr = lr_outputs[unsupervised_indices]
            y_pred_unsupervised = np.argmax(probability(self.theta, self.pi_y, self.pi, sample[2][unsupervised_indices],\
                 sample[3][unsupervised_indices], self.k, self.num_classes, self.continuous_mask).detach().numpy(), 1)
            loss_3 = self.criterion_red(un_lr, torch.tensor(y_pred_unsupervised))

            l0_grads_3 = (torch.autograd.grad(loss_3, un_lr)[0]).detach().clone().to(self.device)
            l0_expand_3 = torch.repeat_interleave(l0_grads_3, l1[unsupervised_indices].shape[1], dim=1)
            l1_grads_3 = l0_expand_3 * l1[unsupervised_indices].detach().repeat(1, self.num_classes).to(self.device)

            if len(supervised_indices) >0:
                supervised_indices = supervised_indices.tolist()
                probs_graphical = probability(self.theta, self.pi_y, self.pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
                torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), self.k, self.num_classes, self.continuous_mask)
            else:
                probs_graphical = probability(self.theta, self.pi_y, self.pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices], self.k, self.num_classes, self.continuous_mask)

            probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
            probs_lr = torch.nn.Softmax()(lr_outputs)
            loss_6 = torch.nn.KLDivLoss(reduction='batchmean')(probs_lr, probs_graphical)

            l0_grads_KD = (torch.autograd.grad(loss_6, lr_outputs)[0]).detach().clone().to(self.device)
            l0_expand_KD = torch.repeat_interleave(l0_grads_KD, l1.shape[1], dim=1)
            l1_grads_KD = l0_expand_KD * l1.detach().repeat(1, self.num_classes).to(self.device)            
            
            if batch_idx % self.fit == 0:
                if len(supervised_indices) > 0:
                    SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                ls_3_grads = torch.cat((l0_grads_3, l1_grads_3), dim=1)
                KD_grads = torch.cat((l0_grads_KD, l1_grads_KD), dim=1)
                batch_un_ind = list(indices[unsupervised_indices]) 
                batch_sup_ind = list(indices[supervised_indices]) 
                batch_ind = list(indices) 
                curr_un_ind = list(np.array(unsupervised_indices)) 
                curr_sup_ind = list(supervised_indices) 
            else:
                if len(supervised_indices) > 0 and len(curr_sup_ind) >0:
                    SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                elif len(supervised_indices) > 0:
                    SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                ls_3_grads = torch.cat((ls_3_grads, torch.cat((l0_grads_3, l1_grads_3), dim=1)), dim=0)
                KD_grads = torch.cat((KD_grads, torch.cat((l0_grads_KD, l1_grads_KD), dim=1)), dim=0)
                batch_un_ind.extend(list(indices[unsupervised_indices]))
                batch_sup_ind.extend(list(indices[supervised_indices]))
                batch_ind.extend(list(indices))
                curr_un_ind.extend(list(len(curr_un_ind)+np.array(unsupervised_indices)))
                curr_sup_ind.extend(list(len(curr_sup_ind)+np.array(supervised_indices)))

            #print(curr_un_ind,len(curr_sup_ind),KD_grads.shape)
          
            if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):

                with torch.no_grad():

                    out, l1 = self.model(self.val_x.to(self.device), last=True, freeze=True)
                    self.init_out = out.detach().to(self.device)
                    self.init_l1 = l1.detach().to(self.device)
                
                #print(SL_grads.shape,ls_3_grads.shape, KD_grads.shape)
                for r in range(5):

                    lambdas = lam.clone().to(self.device)#, device=self.device)
                    lambdas.requires_grad = True
                        
                    lambdas_2 = lambdas[batch_un_ind,1][:,None] 
                    lambdas_3_un = lambdas[batch_un_ind,2][:,None]

                    #print(lambdas1.shape,lambdas_2.shape,lambdas_3_sup.shape)

                    if len(batch_sup_ind) > 0:
                        lambdas1 = lambdas[batch_sup_ind,0][:,None]
                        lambdas_3_sup = lambdas[batch_sup_ind,2][:,None]

                        comb_grad_all_1 = (lambdas1*SL_grads + lambdas_3_sup*KD_grads[curr_sup_ind]).sum(0)

                        out_vec_val = self.init_out - (eta * comb_grad_all_1[:self.num_classes].view(1, -1).\
                            expand(self.init_out.shape[0], -1))

                        out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad_all_1[self.num_classes:].\
                            view(self.num_classes, -1).transpose(0, 1)))

                        #out_vec_val.requires_grad = True
                        loss_SL_val = self.criterion_red(out_vec_val, self.val_y.to(self.device))

                        alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas1,retain_graph=True)[0]).detach().clone().to(self.device)  
                        lam[batch_sup_ind,0] = lam[batch_sup_ind,0] - 2000*alpha_grads.view(-1)

                        alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas_3_sup,retain_graph=True)[0]).detach().clone().to(self.device) 
                        lam[batch_sup_ind,2] = lam[batch_sup_ind,2] - 2000*alpha_grads.view(-1)

                    comb_grad_all_2 = (lambdas_2*ls_3_grads + lambdas_3_un*KD_grads[curr_un_ind]).sum(0)

                    out_vec_val = self.init_out - (eta * comb_grad_all_2[:self.num_classes].view(1, -1).\
                        expand(self.init_out.shape[0], -1))

                    out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad_all_2[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    #out_vec_val.requires_grad = True
                    loss_SL_val = self.criterion_red(out_vec_val, self.val_y.to(self.device))

                    alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas_2,retain_graph=True)[0]).detach().clone().to(self.device) 
                    lam[batch_un_ind,1] = lam[batch_un_ind,1] - 2000*alpha_grads.view(-1)
                    
                    alpha_grads =  (torch.autograd.grad(loss_SL_val, lambdas_3_un)[0]).detach().clone().to(self.device) 
                    lam[batch_un_ind,2] = lam[batch_un_ind,2] - 2000*alpha_grads.view(-1)

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