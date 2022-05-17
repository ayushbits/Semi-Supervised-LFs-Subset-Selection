# CUDA_VISIBLE_DEVICES=2 python3 alm_joint.py 1 dsets/MITR/  9 nn 512 0.0003 0.01 ''  macro  /tmp
# CUDA_VISIBLE_DEVICES=2 python3 alm_joint.py 1 dsets/YOUTUBE/  2 lr 32 0.0003 0.001 ''  accuracy  /tmp
import torch
import sys
import numpy as np
from logistic_regression import *
from deep_net import *
import warnings
warnings.filterwarnings("ignore")
from cage import *
from sklearn.feature_extraction.text import TfidfVectorizer
from losses import *
import pickle
from torch.utils.data import TensorDataset, DataLoader,Dataset 
import torch.nn.functional as F

from LearnMultiLambdaMeta import LearnMultiLambdaMeta
import copy

from sklearn.metrics import precision_score as prec_score
from sklearn.metrics import recall_score as recall_score
from sklearn.metrics import accuracy_score 
if sys.argv[9] == 'macro':
    from sklearn.metrics import f1_score as accuracy_score

Temp = 4 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.set_default_dtype(torch.float64)
seed = 7 #25, 42 , 7
torch.manual_seed(seed)
print('Seed is ', seed)

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]

        return sample, index

    def __len__(self):
        return len(self.dataset)


class TrainALM():

    def __init__(self):
        self.lam_learn = True
        self._lambda = 1.0
        self.best_epoch = 0
        self.best_score = 0
        self.num_runs = int(sys.argv[1])
        self.dset_directory = sys.argv[2]
        self.n_classes = int(sys.argv[3])
        self.feat_model = sys.argv[4]
        self.batch_size = int(sys.argv[5])
        self.lr_fnetwork = float(sys.argv[6])
        self.lr_gm = float(sys.argv[7])
        # self.name_dset = dset_directory.split("/")[-1].lower()
        # print('dset is ', name_dset)
        self.mode = sys.argv[8] #''
        self.metric = sys.argv[9]
        self.save_folder = sys.argv[10]
        self.continuous_epochs = 20 # val accuracy should be consecutively below these many number of epochs
        # if self.metric=='accuracy':
        #     from sklearn.metrics import accuracy_score as score
        #     print('inside accuracy')
        # else:
        #     from sklearn.metrics import f1_score as score
        #     metric_avg = 'macro'

    def processDataset(self):
        

        objs = []
        if self.mode != '':
            fname = self.dset_directory + "/" + self.mode + "_d_processed.p"
        else:
            fname = self.dset_directory + "/d_processed.p"
        with open(fname, 'rb') as f:
            while 1:
                try:
                    o = pickle.load(f)
                except EOFError:
                    break
                objs.append(o)

        x_supervised = torch.tensor(objs[0]).double()
        y_supervised = torch.tensor(objs[3]).long()
        l_supervised = torch.tensor(objs[2]).long()
        s_supervised = torch.tensor(objs[2]).double()
        print('supervised shape', objs[2].shape)

        objs = []
        if self.mode != '':
            fname = self.dset_directory + "/" + self.mode + "_U_processed.p"
        else:
            fname = self.dset_directory + "/U_processed.p"

        with open(fname, 'rb') as f:
            while 1:
                try:
                    o = pickle.load(f)
                except EOFError:
                    break
                objs.append(o)

        excl= []
        idx=0
        for x in objs[1]:
            if(all(x==int(self.n_classes))):
                excl.append(idx)
            idx+=1
        print('no of excludings are ', len(excl))

        x_unsupervised = torch.tensor(np.delete(objs[0],excl, axis=0)).double()
        y_unsupervised = torch.tensor(np.delete(objs[3],excl, axis=0)).long()
        l_unsupervised = torch.tensor(np.delete(objs[2],excl, axis=0)).long()
        s_unsupervised = torch.tensor(np.delete(objs[2],excl, axis=0)).double()
        print('UNsupervised shape', objs[2].shape)
        print('Length of U is', len(x_unsupervised))

        objs = []
        if self.mode != '':
            fname = self.dset_directory + "/" + self.mode + "_validation_processed.p"
        else:
            fname = self.dset_directory + "/validation_processed.p"

        with open(fname, 'rb') as f:
            while 1:
                try:
                    o = pickle.load(f)
                except EOFError:
                    break
                objs.append(o)

        self.x_valid = torch.tensor(objs[0]).double()
        self.y_valid = objs[3]
        self.l_valid = torch.tensor(objs[2]).long()
        self.s_valid = torch.tensor(objs[2]).double()
        print('Valid shape', objs[2].shape)
        objs1 = []
        if self.mode != '':
            fname = self.dset_directory + "/" + self.mode + "_test_processed.p"
        else:
            fname = self.dset_directory + "/test_processed.p"


        with open(fname, 'rb') as f:
            while 1:
                try:
                    o = pickle.load(f)
                except EOFError:
                    break
                objs1.append(o)
        self.x_test = torch.tensor(objs1[0]).double()
        self.y_test = objs1[3]
        self.l_test = torch.tensor(objs1[2]).long()
        self.s_test = torch.tensor(objs1[2]).double()
        print('Test shape', objs[2].shape)

        self.n_features = x_supervised.shape[1]

        # Labeling Function Classes
        # k = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])).long()
        #lf_classes_file = sys.argv[11]

        if self.mode != '':
            fname = self.dset_directory + '/' + self.mode + '_k.npy'
        else:
            fname = self.dset_directory + '/k.npy'
        self.k = torch.from_numpy(np.load(fname)).long()
        self.n_lfs = len(self.k)
        print('LFs are ',self.k)
        print('no of lfs are ', self.n_lfs)

        # if self.qg_available:
        #     self.a = torch.from_numpy(np.load(self.dset_directory+'/prec.npy')).double()
        # else:
        prec_lfs=[]
        for i in range(self.n_lfs):
            correct = 0
            for j in range(len(self.y_valid)):
                if self.y_valid[j] == self.l_valid[j][i]:
                    correct+=1
            prec_lfs.append(correct/len(self.y_valid))
        self.a = torch.tensor(prec_lfs).double()


        self.continuous_mask = torch.zeros(self.n_lfs).double()


        for i in range(s_supervised.shape[0]):
            for j in range(s_supervised.shape[1]):
                if s_supervised[i, j].item() > 0.999:
                    s_supervised[i, j] = 0.999
                if s_supervised[i, j].item() < 0.001:
                    s_supervised[i, j] = 0.001

        for i in range(s_unsupervised.shape[0]):
            for j in range(s_unsupervised.shape[1]):
                if s_unsupervised[i, j].item() > 0.999:
                    s_unsupervised[i, j] = 0.999
                if s_unsupervised[i, j].item() < 0.001:
                    s_unsupervised[i, j] = 0.001

        for i in range(self.s_valid.shape[0]):
            for j in range(self.s_valid.shape[1]):
                if self.s_valid[i, j].item() > 0.999:
                    self.s_valid[i, j] = 0.999
                if self.s_valid[i, j].item() < 0.001:
                    self.s_valid[i, j] = 0.001

        for i in range(self.s_test.shape[0]):
            for j in range(self.s_test.shape[1]):
                if self.s_test[i, j].item() > 0.999:
                    self.s_test[i, j] = 0.999
                if self.s_test[i, j].item() < 0.001:
                    self.s_test[i, j] = 0.001



        l = torch.cat([l_supervised, l_unsupervised])
        s = torch.cat([s_supervised, s_unsupervised])
        x_train = torch.cat([x_supervised, x_unsupervised])
        self.y_train = torch.cat([y_supervised, y_unsupervised])
        #selecting random instances here
        np.random.seed(seed)
        indices = np.random.choice(np.arange(x_train.shape[0]), len(x_supervised), replace=False)
        supervised_mask = torch.zeros(x_train.shape[0])
        supervised_mask[indices] = 1
        ######## 
        #handpicked here
        # supervised_mask = torch.cat([torch.ones(l_supervised.shape[0]), torch.zeros(l_unsupervised.shape[0])])
        print('X_train', x_train.shape, 'l',l.shape, 's', s.shape)

        

        self.lambdas = torch.full((len(self.y_train),2),self._lambda)
        self.dataset = TensorDataset(x_train, self.y_train, l, s, supervised_mask, self.lambdas)
        # return lambdas, dataset

    def check_stopping_cond(self, epoch, val_acc, continuous_epochs):
        save, cont = None, None
        if val_acc > self.best_score:
            self.best_score = val_acc
            self.best_epoch = epoch
            print('Best Score ', self.best_score , ' at epoch ', self.best_epoch)
            save = 1
            cont = 1
            

        if epoch > self.best_epoch + continuous_epochs-1:
            print('Leaving Early.. Bye !')
            print('Best Score ', self.best_score , ' at epoch ', self.best_epoch)
            save = 0
            cont = 0
        return save, cont

    def train(self):


        pi = torch.ones((self.n_classes, self.n_lfs)).double()
        pi.requires_grad = True

        theta = torch.ones((self.n_classes, self.n_lfs)).double() * 1
        theta.requires_grad = True

        pi_y = torch.ones(self.n_classes).double()
        pi_y.requires_grad = True

        if self.feat_model == 'lr':
            # teacher_lr_model = LogisticRegression(self.n_features, self.n_classes,  seed = seed)
            lr_model = LogisticRegression(self.n_features, self.n_classes,  seed = seed)
        elif self.feat_model =='nn':
            n_hidden = 512
            # teacher_lr_model = DeepNet(self.n_features, n_hidden, self.n_classes,  seed = seed)
            lr_model = DeepNet(self.n_features, n_hidden, self.n_classes,  seed = seed)
        else:
            print('Please provide feature based model : lr or nn')
            exit()


        # optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
        # teacher_optimizer_lr = torch.optim.Adam(teacher_lr_model.parameters(), lr=self.lr_fnetwork) #theta'
        optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=self.lr_fnetwork) # theta
        optimizer_gm = torch.optim.Adam([theta, pi, pi_y], lr=self.lr_gm, weight_decay=0)

        supervised_criterion = torch.nn.CrossEntropyLoss()
        supervised_criterion_nored = torch.nn.CrossEntropyLoss(reduction='none')

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        loader_ind = DataLoader(MyDataset(self.dataset), batch_size=self.batch_size, shuffle=True,pin_memory=True)
        # loading initial teacher_model 1 here


        
        self.best_epoch = 0
        self.best_score = 0
        val_acc = 0
        self.continuous_epochs = 100

        if self.lam_learn: 
            lelam = LearnMultiLambdaMeta(loader_ind, self.x_valid,self.y_valid, copy.deepcopy(lr_model), self.n_classes, len(self.y_train), \
                supervised_criterion_nored, "cpu", 2,teacher_lr_model,theta, pi_y, pi,self.k, self.continuous_mask,\
                    supervised_criterion, Temp)
                    
        ## Training student model here ------------
        
        
        for epoch in range(100):
            lr_model.train()

            for batch_ndx, sample in enumerate(loader):
                optimizer_lr.zero_grad()

                unsup = []
                sup = []
                supervised_indices = sample[4].nonzero().view(-1)
                # unsupervised_indices = indices  ## Uncomment for entropy
                unsupervised_indices = (1-sample[4]).nonzero().squeeze().view(-1)
                # Case I - CE for theta model
                if len(supervised_indices) > 0:
                    loss_1 = supervised_criterion(lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
                else:
                    loss_1 = 0

                y_pred_unsupervised = np.argmax(probability(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], self.k, self.n_classes,continuous_mask).detach().numpy(), 1)
                loss_3 = supervised_criterion(lr_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))

                if len(supervised_indices) > 0:
                    loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, sample[1][supervised_indices], sample[2][supervised_indices], sample[3][supervised_indices], self.k, self.n_classes, self.continuous_mask)
                else:
                    loss_4 = 0

                if len(supervised_indices) >0:
                    supervised_indices = supervised_indices.tolist()
                    probs_graphical = probability(theta, pi_y, pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
                    torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), self.k, self.n_classes, self.continuous_mask)
                else:
                    probs_graphical = probability(theta, pi_y, pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices], self.k, self.n_classes, self.continuous_mask)

                probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
                probs_lr = F.Softmax()(lr_model(sample[0]))
                loss_6 = torch.nn.KLDivLoss(probs_lr, probs_graphical)

                loss = loss_1 + loss_3 + loss_4 + loss_6
                
                loss.backward()
                optimizer_lr.step()
                optimizer_gm.step()
            # print('Epoch ', epoch, ' loss is ',  loss.item() )

            if self.lam_learn:

                cached_state_dictT = copy.deepcopy(lr_model.state_dict())
                clone_dict = copy.deepcopy(lr_model.state_dict())
                lelam.update_model(clone_dict)
                lr_model.load_state_dict(cached_state_dictT)

                self.lambdas = lelam.get_lambdas(optimizer_lr.param_groups[0]['lr'], epoch, self.lambdas)
                
                # for m in range(2):
                # if save:
                #     print(torch.median(self.lambdas[:,0]).item() , ' , ' , torch.median(self.lambdas[:,1]).item())
                    # print(self.lambdas[:,m].max(), self.lambdas[:,m].min(), torch.median(self.lambdas[:,m]),\
                    #         torch.quantile(self.lambdas[:,m], 0.75),torch.quantile(self.lambdas[:,m], 0.25))

                # self.dataset = TensorDataset(x_train, y_train, l, s, supervised_mask, lambdas)

                # loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True,pin_memory=True)

            probs = torch.nn.Softmax()(lr_model(self.x_valid))
            y_pred = np.argmax(probs.detach().numpy(), 1)
            val_acc = 0
            if self.metric =='macro':
                val_acc =accuracy_score(self.y_valid, y_pred, average='binary')
            else:
                val_acc =accuracy_score(self.y_valid, y_pred)
            # val_acc = accuracy_score(self.y_valid, y_pred)
            
            save, cont = self.check_stopping_cond( epoch, val_acc, self.continuous_epochs)
            if save==1:
                checkpoint = {'params': lr_model.state_dict()}
                torch.save(checkpoint, self.save_folder+"/student_lr_"+ str(epoch)+".pt")
                print(torch.median(self.lambdas[:,0]).item() , ' , ' , torch.median(self.lambdas[:,1]).item())
            if cont==0:
                break  

        student_best_epoch = self.best_epoch

        lr_model.load_state_dict(torch.load(self.save_folder+"/student_lr_"+ str(student_best_epoch)+".pt")['params'])            
        probs = torch.nn.Softmax()(lr_model(self.x_test))
        y_pred = np.argmax(probs.detach().numpy(), 1)
        lr_acc = 0
        if self.metric =='macro':
            print('inside macro')
            lr_acc = accuracy_score(self.y_test, y_pred, average='binary')
        else:
            print('inside else')
            lr_acc = accuracy_score(self.y_test, y_pred)
        lr_prec = prec_score(self.y_test, y_pred, average=None)
        lr_recall = recall_score(self.y_test, y_pred, average=None)
        print('Epoch ', self.best_epoch,  ' Test score is ', lr_acc)
        print('Seed is ', seed)

if __name__=='__main__':
    alm = TrainALM()
    alm.processDataset()
    alm.train()