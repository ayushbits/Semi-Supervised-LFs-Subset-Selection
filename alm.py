#CUDA_VISIBLE_DEVICES=1 python3 alm.py /tmp l1 0 0 l4 0 l6 0 5 dsets/YOUTUBE 2 lr 0 32 0.0003 0.01 ''  accuracy

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

Temp = 4 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.set_default_dtype(torch.float64)
torch.set_printoptions(threshold=20)

objs = []
dset_directory = sys.argv[10]
n_classes = int(sys.argv[11])
feat_model = sys.argv[12]
qg_available = int(sys.argv[13])
batch_size = int(sys.argv[14])
lr_fnetwork = float(sys.argv[15])
lr_gm = float(sys.argv[16])
name_dset = dset_directory.split("/")[-1].lower()
print('dset is ', name_dset)
mode = sys.argv[17] #''
metric = sys.argv[18]

lam_learn = True

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]

        # Your transformations here (or set it in CIFAR10)

        return sample, index

    def __len__(self):
        return len(self.dataset)

from sklearn.metrics import precision_score as prec_score
from sklearn.metrics import recall_score as recall_score
if metric=='accuracy':
    from sklearn.metrics import accuracy_score as score
    print('inside accuracy')
else:
    from sklearn.metrics import f1_score as score
    metric_avg = 'macro'


if mode != '':
    fname = dset_directory + "/" + mode + "_d_processed.p"
else:
    fname = dset_directory + "/d_processed.p"
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
if mode != '':
    fname = dset_directory + "/" + mode + "_U_processed.p"
else:
    fname = dset_directory + "/U_processed.p"

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
    if(all(x==int(n_classes))):
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
if mode != '':
    fname = dset_directory + "/" + mode + "_validation_processed.p"
else:
    fname = dset_directory + "/validation_processed.p"

with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_valid = torch.tensor(objs[0]).double()
y_valid = objs[3]
l_valid = torch.tensor(objs[2]).long()
s_valid = torch.tensor(objs[2]).double()
print('Valid shape', objs[2].shape)
objs1 = []
if mode != '':
    fname = dset_directory + "/" + mode + "_test_processed.p"
else:
    fname = dset_directory + "/test_processed.p"


with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs1.append(o)
x_test = torch.tensor(objs1[0]).double()
y_test = objs1[3]
l_test = torch.tensor(objs1[2]).long()
s_test = torch.tensor(objs1[2]).double()
print('Test shape', objs[2].shape)

n_features = x_supervised.shape[1]

# Labeling Function Classes
# k = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])).long()
#lf_classes_file = sys.argv[11]

if mode != '':
    fname = dset_directory + '/' + mode + '_k.npy'
else:
    fname = dset_directory + '/k.npy'
k = torch.from_numpy(np.load(fname)).long()
n_lfs = len(k)
print('LFs are ',k)
print('no of lfs are ', n_lfs)

# a = torch.ones(n_lfs).double() * 0.9
# print('before ',a)
if qg_available:
    a = torch.from_numpy(np.load(dset_directory+'/prec.npy')).double()
else:
    # a = torch.ones(n_lfs).double() * 0.9

    prec_lfs=[]
    for i in range(n_lfs):
       correct = 0
       for j in range(len(y_valid)):
           if y_valid[j] == l_valid[j][i]:
               correct+=1
       prec_lfs.append(correct/len(y_valid))
    a = torch.tensor(prec_lfs).double()

# n_lfs = int(len(k))
# print('number of lfs ', n_lfs)
# a = torch.ones(n_lfs).double() * 0.9
continuous_mask = torch.zeros(n_lfs).double()


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

for i in range(s_valid.shape[0]):
    for j in range(s_valid.shape[1]):
        if s_valid[i, j].item() > 0.999:
            s_valid[i, j] = 0.999
        if s_valid[i, j].item() < 0.001:
            s_valid[i, j] = 0.001

for i in range(s_test.shape[0]):
    for j in range(s_test.shape[1]):
        if s_test[i, j].item() > 0.999:
            s_test[i, j] = 0.999
        if s_test[i, j].item() < 0.001:
            s_test[i, j] = 0.001



l = torch.cat([l_supervised, l_unsupervised])
s = torch.cat([s_supervised, s_unsupervised])
x_train = torch.cat([x_supervised, x_unsupervised])
y_train = torch.cat([y_supervised, y_unsupervised])
supervised_mask = torch.cat([torch.ones(l_supervised.shape[0]), torch.zeros(l_unsupervised.shape[0])])
print('X_train', x_train.shape, 'l',l.shape, 's', s.shape)

num_runs = int(sys.argv[9])

final_score_gm, final_score_lr, final_score_gm_val, final_score_lr_val = [],[],[],[]

final_score_lr_prec, final_score_lr_recall, final_score_gm_prec, final_score_gm_recall = [],[],[],[]



for lo in range(0,num_runs):
    pi = torch.ones((n_classes, n_lfs)).double()
    pi.requires_grad = True

    theta = torch.ones((n_classes, n_lfs)).double() * 1
    theta.requires_grad = True

    pi_y = torch.ones(n_classes).double()
    pi_y.requires_grad = True

    if feat_model == 'lr':
        teacher_lr_model = LogisticRegression(n_features, n_classes)
        lr_model = LogisticRegression(n_features, n_classes)
    elif feat_model =='nn':
        n_hidden = 512
        lr_model = DeepNet(n_features, n_hidden, n_classes)
    else:
        print('Please provide feature based model : lr or nn')
        exit()


    # optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
    teacher_optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=lr_fnetwork) #theta'
    optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=lr_fnetwork) # theta
    optimizer_gm = torch.optim.Adam([theta, pi, pi_y], lr=lr_gm, weight_decay=0)
    # optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
    supervised_criterion = torch.nn.CrossEntropyLoss()
    supervised_criterion_nored = torch.nn.CrossEntropyLoss(reduction='none')

    _lambda = 0.5 

    lambdas = torch.full((len(y_train),2),_lambda)
    dataset = TensorDataset(x_train, y_train, l, s, supervised_mask, lambdas)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    loader_ind = DataLoader(MyDataset(dataset), batch_size=batch_size, shuffle=True,pin_memory=True)
    
    # loading initial teacher_model 1 here
    for epoch in range(3):
        teacher_lr_model.train()
        for batch_ndx, sample in enumerate(loader):
            optimizer_lr.zero_grad()
            sup = []
            supervised_indices = sample[4].nonzero().view(-1)
            if len(supervised_indices) > 0:
                loss_1 = supervised_criterion(lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
                loss_1.backward( retain_graph=True)                
                teacher_optimizer_lr.step()
    # Finished teacher model 1 here -----------------

    # Training teacher Model 2 (GM) here ----
    for epoch in range (3):
        teacher_lr_model.train()
        for batch_ndx, sample in enumerate(loader):
            optimizer_lr.zero_grad()
            sup = []
            supervised_indices = sample[4].nonzero().view(-1)
            unsupervised_indices = (1-sample[4]).nonzero().squeeze().view(-1)
            loss_GM = log_likelihood_loss(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes, continuous_mask)
        
    # Finished training Teacher model 2 here ---------

    # Testing teacher model 1 here 
    probs = torch.nn.Softmax()(teacher_lr_model(x_test))
    y_pred = np.argmax(probs.detach().numpy(), 1)
    # if name_dset =='youtube' or name_dset=='census' or name_dset =='sms':
    if metric=='accuracy':
        # print('inside accuracy LR test')
        lr_acc =score(y_test, y_pred)
        lr_prec = prec_score(y_test, y_pred, average=None)
        lr_recall = recall_score(y_test, y_pred, average=None)

    print("Teacher Model 1 Test accuracy Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc))

    # Testing teacher model 2 here 
    y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
    gm_acc = score(y_test, y_pred)
    gm_prec = prec_score(y_test, y_pred, average=None)
    gm_recall = recall_score(y_test, y_pred, average=None)
    print("Teacher Model 2 (GM) Test accuracy Epoch: {}\tTest LR accuracy_score: {}".format(epoch, gm_acc))


    if lam_learn: 
        lelam = LearnMultiLambdaMeta(loader_ind, x_valid,y_valid, copy.deepcopy(lr_model), n_classes, len(y_train), \
                supervised_criterion_nored, "cuda", 2,teacher_lr_model,theta, pi_y, pi,k,continuous_mask,\
                     supervised_criterion, Temp)

    # Again initializing lr and gm parameters here
    best_score_lr,best_score_gm,best_epoch_lr,best_epoch_gm,best_score_lr_val, best_score_gm_val = 0,0,0,0,0,0
    best_score_lr_prec,best_score_lr_recall ,best_score_gm_prec,best_score_gm_recall= 0,0,0,0

    stop_pahle, stop_pahle_gm = [], []
    # wandb.watch(lr_model)
    for epoch in range(100):
        lr_model.train()

        for batch_ndx, sample in enumerate(loader):
            optimizer_lr.zero_grad()
            optimizer_gm.zero_grad()

            unsup = []
            sup = []
            supervised_indices = sample[4].nonzero().view(-1)
            # unsupervised_indices = indices  ## Uncomment for entropy
            unsupervised_indices = (1-sample[4]).nonzero().squeeze().view(-1)
            # print('sample[2][unsupervised_indices].shape', sample[2][unsupervised_indices].shape)
            # print('sample[3][unsupervised_indices].shape', sample[3][unsupervised_indices].shape)

            # Theta Model
            lr_outputs = (lr_model(sample[0][unsupervised_indices]))

            probs_theta = F.log_softmax(lr_outputs)
            outputs = np.argmax(probs_theta.detach().numpy(), 1)
            

            # GM Model Test
            probs_graphical =(probability(theta, pi_y, pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices],k, n_classes, continuous_mask))
            targets = (probs_graphical.t() / probs_graphical.sum(1)).t()
            outputs_tensor = torch.Tensor(outputs)
            targets_tensor = torch.Tensor(targets)
            # print('outputs ', outputs_tensor.shape)
            # print('targets ', targets_tensor.shape)

            # Case I - CE for theta model
            if len(supervised_indices) > 0:
                loss = supervised_criterion(lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
            else:
                loss = 0
            # print('Loss Case I ', loss)

            # Case II - Component 1
            loss_SL = supervised_criterion_nored(lr_outputs, targets_tensor)
            # print('loss_SL ', loss_SL.shape)
            # print('sample[5][unsupervised_indices][:,0]', sample[5][unsupervised_indices][:,0])
            loss_SL = sample[5][unsupervised_indices][:,0]*loss_SL # sample[5] are lambdas

            # loss_GM = log_likelihood_loss(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes, continuous_mask) # 2nd teacher model, bahar train hoga
    
            
            probs_theta_kl = F.log_softmax(lr_outputs / Temp, dim =1)
            probs_lr_teacher = F.softmax(teacher_lr_model(sample[0][unsupervised_indices])/Temp, dim=1)
            loss_KD = torch.nn.KLDivLoss(reduction='none')(probs_theta_kl, probs_lr_teacher)
            
            loss_KD =  Temp * Temp *sample[5][unsupervised_indices][:,1]*torch.sum(loss_KD, dim=1)
            loss += (loss_SL + loss_KD).mean()
            print('loss is ', loss, )
            loss.backward()
            # optimizer_gm.step()
            optimizer_lr.step()

        if lam_learn:

            cached_state_dictT = copy.deepcopy(lr_model.state_dict())
            clone_dict = copy.deepcopy(lr_model.state_dict())
            lelam.update_model(clone_dict)
            lr_model.load_state_dict(cached_state_dictT)

            lambdas = lelam.get_lambdas(optimizer_lr.param_groups[0]['lr'],i,lambdas)
            
            for m in range(2):
                print(lambdas[:,m].max(), lambdas[:,m].min(), torch.median(lambdas[:,m]),\
                        torch.quantile(lambdas[:,m], 0.75),torch.quantile(lambdas[:,m], 0.25))
            
#        wname = "Run_"+str(lo)+" Train Loss" #wandb
#        wandb.log({wname:loss, 'custom_step':epoch}) #wandb
#         y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
#         if metric=='accuracy':
#             gm_acc = score(y_test, y_pred)
#             lr_prec = prec_score(y_test, y_pred, average=None) 
#             lr_recall = recall_score(y_test, y_pred, average=None)
#             gm_prec, gm_recall = 0,0
#         else:
#             gm_acc = score(y_test, y_pred, average=metric_avg)
#             gm_prec = prec_score(y_test, y_pred, average=metric_avg)
#             gm_recall = recall_score(y_test, y_pred, average=metric_avg)
#         #Valid
#         y_pred = np.argmax(probability(theta, pi_y, pi, l_valid, s_valid, k, n_classes, continuous_mask).detach().numpy(), 1)
        
#         if metric=='accuracy':
#             gm_valid_acc = score(y_valid, y_pred)
#             gm_prec, gm_recall = 0,0
#         else:
#             gm_valid_acc = score(y_valid, y_pred, average=metric_avg)

#         #LR Test

#         probs = torch.nn.Softmax()(lr_model(x_test))
#         y_pred = np.argmax(probs.detach().numpy(), 1)
#         # if name_dset =='youtube' or name_dset=='census' or name_dset =='sms':
#         if metric=='accuracy':
#         	# print('inside accuracy LR test')
#         	lr_acc =score(y_test, y_pred)
#         	lr_prec = prec_score(y_test, y_pred, average=None)
#         	lr_recall = recall_score(y_test, y_pred, average=None)
#         	gm_prec, gm_recall = 0,0

    
#         else:
#             lr_acc =score(y_test, y_pred, average=metric_avg)
#             lr_prec = prec_score(y_test, y_pred, average=metric_avg)
#             lr_recall = recall_score(y_test, y_pred, average=metric_avg)
#         #LR Valid
#         probs = torch.nn.Softmax()(lr_model(x_valid))
#         y_pred = np.argmax(probs.detach().numpy(), 1)
        
#         if metric=='accuracy':
#             lr_valid_acc = score(y_valid, y_pred)
#             gm_prec, gm_recall = 0,0
#         else:
#             lr_valid_acc = score(y_valid, y_pred, average=metric_avg)
#         print("Epoch: {}\t Test GM accuracy_score: {}".format(epoch, gm_acc ))
#         print("Epoch: {}\tGM accuracy_score(Valid): {}".format(epoch, gm_valid_acc))
#         print("Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc))    
#         print("Epoch: {}\tLR accuracy_score(Valid): {}".format(epoch, lr_valid_acc))
#         # wname = "Run_"+str(lo)+" LR valid score"
#         # wnamegm = 'Run_' + str(lo) + ' GM valid score'
#         #wandb.log({wname:lr_valid_acc, 
#          #   wnamegm:gm_valid_acc,'custom_step':epoch})

#         if epoch > 1 and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_lr_val:
#             # print("Inside Best hu Epoch: {}\t Test GM accuracy_score: {}".format(epoch, gm_acc ))
#             # print("Inside Best hu Epoch: {}\tGM accuracy_score(Valid): {}".format(epoch, gm_valid_acc))
#             if gm_valid_acc == best_score_gm_val or gm_valid_acc == best_score_lr_val:
#                 if best_score_gm < gm_acc or best_score_lr < lr_acc:
#                     best_epoch_lr = epoch
#                     best_score_lr_val = lr_valid_acc
#                     best_score_lr = lr_acc

#                     best_epoch_gm = epoch
#                     best_score_gm_val = gm_valid_acc
#                     best_score_gm = gm_acc

#                     best_score_lr_prec = lr_prec
#                     best_score_lr_recall  = lr_recall
#                     best_score_gm_prec = gm_prec
#                     best_score_gm_recall  = gm_recall
#             else:
#                 best_epoch_lr = epoch
#                 best_score_lr_val = lr_valid_acc
#                 best_score_lr = lr_acc

#                 best_epoch_gm = epoch
#                 best_score_gm_val = gm_valid_acc
#                 best_score_gm = gm_acc

#                 best_score_lr_prec = lr_prec
#                 best_score_lr_recall  = lr_recall
#                 best_score_gm_prec = gm_prec
#                 best_score_gm_recall  = gm_recall
#                 stop_pahle = []
#                 stop_pahle_gm = []
#             checkpoint = {'theta': theta,'pi': pi}
#             # torch.save(checkpoint, save_folder+"/gm_"+str(epoch)    +".pt")
#             checkpoint = {'params': lr_model.state_dict()}
#             # torch.save(checkpoint, save_folder+"/lr_"+ str(epoch)+".pt")
            

#         if epoch > 1 and lr_valid_acc >= best_score_lr_val and lr_valid_acc >= best_score_gm_val:
#             # print("Inside Best hu Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc ))
#             # print("Inside Best hu Epoch: {}\tLR accuracy_score(Valid): {}".format(epoch, lr_valid_acc))
#             if lr_valid_acc == best_score_lr_val or lr_valid_acc == best_score_gm_val:
#                 if best_score_lr < lr_acc or best_score_gm < gm_acc:
                    
#                     best_epoch_lr = epoch
#                     best_score_lr_val = lr_valid_acc
#                     best_score_lr = lr_acc

#                     best_epoch_gm = epoch
#                     best_score_gm_val = gm_valid_acc
#                     best_score_gm = gm_acc

#                     best_score_lr_prec = lr_prec
#                     best_score_lr_recall  = lr_recall
#                     best_score_gm_prec = gm_prec
#                     best_score_gm_recall  = gm_recall
#             else:
#                 best_epoch_lr = epoch
#                 best_score_lr_val = lr_valid_acc
#                 best_score_lr = lr_acc
#                 best_epoch_gm = epoch
#                 best_score_gm_val = gm_valid_acc
#                 best_score_gm = gm_acc
#                 best_score_lr_prec = lr_prec
#                 best_score_lr_recall  = lr_recall
#                 best_score_gm_prec = gm_prec
#                 best_score_gm_recall  = gm_recall
#                 stop_pahle = []
#                 stop_pahle_gm = []
#             checkpoint = {'theta': theta,'pi': pi}
#             # torch.save(checkpoint, save_folder+"/gm_"+str(epoch)    +".pt")
#             checkpoint = {'params': lr_model.state_dict()}
#             # torch.save(checkpoint, save_folder+"/lr_"+ str(epoch)+".pt")
            


#         # if len(stop_pahle) > 10 and len(stop_pahle_gm) > 10 and (all(best_score_lr_val >= k for k in stop_pahle)
#         #  and all(best_score_gm_val >= k for k in stop_pahle_gm)):
#         if  len(stop_pahle_gm) > 10 and all(best_score_gm_val >= k for k in stop_pahle_gm):
        
#             print('Early Stopping at', best_epoch_gm, best_score_gm, best_score_lr)
#             print('Validation score Early Stopping at', best_epoch_gm, best_score_lr_val, best_score_gm_val)
#             break
#         else:
#             # print('inside else stop pahle epoch', epoch)
#             stop_pahle.append(lr_valid_acc)
#             stop_pahle_gm.append(gm_valid_acc)

#     # print("Run \t",lo, "Epoch Gm, Epoch LR, GM, LR \t", best_epoch_gm, best_epoch_lr,best_score_gm, best_score_lr)
#     # print("Run \t",lo, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)
#     print('Best Epoch LR', best_epoch_lr)
#     print('Best Precision LR', best_score_lr_prec)
#     print('Best Recall LR', best_score_lr_recall)
#     print('Best Epoch GM', best_epoch_gm)
#     print('Best Precision GM ', best_score_gm_prec)
#     print('Best Recall GM ', best_score_gm_recall)
#     print("Run \t",lo, "Epoch, GM, LR \t", best_score_gm, best_score_lr)
#     print("Run \t",lo, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)
#     final_score_gm.append(best_score_gm)
#     final_score_lr.append(best_score_lr)

#     final_score_gm_val.append(best_score_gm_val)
#     final_score_lr_val.append(best_score_lr_val)


# print("===================================================")
# print("TEST Averaged scores are for LR", np.mean(final_score_lr))
# print("TEST Precision average scores are for LR", np.mean(best_score_lr_prec))
# print("TEST Recall average scores are for LR", np.mean(best_score_lr_recall))
# print("===================================================")
# print("TEST Averaged scores are for GM",  np.mean(final_score_gm))
# print("TEST Precision average scores are for GM", np.mean(final_score_gm_prec))
# print("TEST Recall average scores are for GM", np.mean(final_score_gm_recall))
# print("===================================================")
# print("VALIDATION Averaged scores are for GM,LR", np.mean(final_score_gm_val), np.mean(final_score_lr_val))
# print("TEST STD  are for GM,LR", np.std(final_score_gm), np.std(final_score_lr))
# print("VALIDATION STD  are for GM,LR", np.std(final_score_gm_val), np.std(final_score_lr_val))

