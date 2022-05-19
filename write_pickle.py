
#python write_pickle.py dsets/YOUTUBE random/YOUTUBE_17 2 25 .. 25 is seed
import torch
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from cage import *
from sklearn.feature_extraction.text import TfidfVectorizer
from losses import *
import pickle, os
from torch.utils.data import TensorDataset, DataLoader,Dataset 
import torch.nn.functional as F

import torch.nn as nn



Temp = 4 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.set_default_dtype(torch.float64)
seed = int(sys.argv[4])   #25, 42 , 7, 17
torch.manual_seed(seed)
print('Seed is ', seed)



class TrainALM():

    def __init__(self):
        self.dset_directory = sys.argv[1].strip()
        self.mode = ''
        self.save_folder = sys.argv[2] + '_'+ str(seed)
        self.n_classes = sys.argv[3]
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        # self.name_dset = dset_directory.split("/")[-1].lower()
        # print('dset is ', name_dset)

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
        m_supervised = torch.tensor(objs[1]).long()
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
        m_unsupervised =  torch.tensor(np.delete(objs[1],excl, axis=0)).long()
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
        m = torch.cat([m_supervised, m_unsupervised])
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

        # def lsnork_to_l_m(lsnork, num_classes):
        #     print(lsnork)
        #     m = 1 - np.equal(lsnork, -1).astype(int)
        #     l = m*lsnork + (1-m)*num_classes
        #     return l,m

        def conv_l_to_lsnork(l,m):
            '''
            in snorkel convention is
            if a rule does not cover an instance assign it label -1
            we follow the convention where we assign the label num_classes instead of -1
            valid class labels range from {0,1,...num_classes-1}
            conv_l_to_lsnork:  converts l in our format to snorkel's format
            '''
            lsnork = l*m + -1*(1-m)
            return lsnork.astype(np.int)

        def write_pickle():
            
            os.system('cp ' + self.dset_directory + '/validation_processed.p ' + self.save_folder)
            os.system('cp ' + self.dset_directory + '/test_processed.p ' + self.save_folder)
            os.system('cp ' + self.dset_directory + '/k.npy ' + self.save_folder)
    
            supervised_mask = torch.zeros(x_train.shape[0])
            supervised_mask[indices] = 1
            supervised_indices = supervised_mask.nonzero().view(-1)
            unsupervised_indices = (1-supervised_mask).nonzero().squeeze().view(-1)


            file_name = self.save_folder + '/d_processed.p'
            with open(file_name,"wb") as f:
                # d_x = np.array([x_train[i] for i in supervised_indices])
                d_x = x_train[supervised_indices,:].numpy()
                d_l = l[supervised_indices,:].numpy()
                d_L = self.y_train[supervised_indices].numpy()
                d_m = m[supervised_indices,:].numpy()
                # d_m = conv_l_to_lsnork(d_m, d_l)
                print('len of d_m', d_m.shape)
                print('len of d_x', d_x.shape)
                print('len of d_L', d_L.shape)
                print('len of d_l', d_l.shape)
                d_d = np.array([1.0] * len(d_L))
                d_r = np.zeros(d_l.shape) #rule exemplar coupling unavailable
                
                
                pickle.dump(d_x,f)
                pickle.dump(d_m,f)
                pickle.dump(d_l,f)
                pickle.dump(d_L,f)
                pickle.dump(d_d,f)
                pickle.dump(d_r,f)

            print('Done writing d_processed')
            file_name = self.save_folder + '/U_processed.p'
            with open(file_name,"wb") as f:
                d_x = x_train[unsupervised_indices,:].numpy()
                d_l = l[unsupervised_indices,:].numpy()
                d_L = self.y_train[unsupervised_indices].numpy()
                d_m = m[unsupervised_indices,:].numpy()
                # d_m = conv_l_to_lsnork(d_m, d_l)
                d_d = np.array([1.0] * len(d_L))
                d_r = np.zeros(d_l.shape) #rule exemplar coupling unavailable
                print('Done writing U_processed')
                print('len of d_m', d_m.shape)
                print('len of d_x', d_x.shape)
                print('len of d_L', d_L.shape)
                print('len of d_l', d_l.shape)
                pickle.dump(d_x,f)
                pickle.dump(d_m,f)
                pickle.dump(d_l,f)
                pickle.dump(d_L,f)
                pickle.dump(d_d,f)
                pickle.dump(d_r,f)
        
        write_pickle()


if __name__=='__main__':
    alm = TrainALM()
    alm.processDataset()
    
