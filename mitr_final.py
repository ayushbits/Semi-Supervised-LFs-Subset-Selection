import torch
import sys
import numpy as np
from logistic_regression import *
from sklearn.metrics import f1_score, accuracy_score
from cage import *
from sklearn.feature_extraction.text import TfidfVectorizer
from losses import *
import pickle


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        objs = []
        while 1:
            try:
                o = pickle.load(f)
            except EOFError:
                break
            objs.append(o)
    return objs


if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20)
test_objs = load_pickle("Data/MITR/test_processed.p")
train_objs = load_pickle("Data/MITR/d_processed.p")
u_objs = load_pickle("Data/MITR/U_processed.p")

x_train = torch.from_numpy(train_objs[0]).double()
y_train = torch.from_numpy(train_objs[3])
x_test = torch.from_numpy(test_objs[0]).double()
y_test = torch.from_numpy(test_objs[3])
x_u = torch.from_numpy(u_objs[0]).double()
y_u = torch.from_numpy(u_objs[3])

n_classes = 9
n_lfs = 15
n_features = x_train.shape[1]
train_instances = x_train.shape[0]
test_instances = x_test.shape[0]
u_instances = x_u.shape[0]
# Discrete lambda values
l = torch.from_numpy(train_objs[2]).long()
u_l = torch.from_numpy(u_objs[2]).long()
l_test = torch.from_numpy(test_objs[2]).long()

# Continuous score values
s = torch.ones(train_instances, n_lfs).double()
s_u = torch.ones(u_instances, n_lfs).double()
s_test = torch.ones(test_instances, n_lfs).double()

# Labeling Function Classes
k = torch.from_numpy(np.array([1, 5, 1, 4, 8, 2, 2, 3, 2, 7, 6, 4, 8, 3, 7])).long()

# True y
y_true = y_train
y_true_test = y_test

continuous_mask = torch.zeros(n_lfs).double()

a = torch.ones(n_lfs).double() * 0.9

for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        if s[i, j].item() > 0.999:
            s[i, j] = 0.999
        if s[i, j].item() < 0.001:
            s[i, j] = 0.001

for i in range(s_test.shape[0]):
    for j in range(s_test.shape[1]):
        if s_test[i, j].item() > 0.999:
            s_test[i, j] = 0.999
        if s_test[i, j].item() < 0.001:
            s_test[i, j] = 0.001

pi = torch.ones((n_classes, n_lfs)).double()
pi.requires_grad = True

theta = torch.ones((n_classes, n_lfs)).double() * 1
theta.requires_grad = True

pi_y = torch.ones(n_classes).double()
pi_y.requires_grad = True

lr_model = LogisticRegression(n_features, n_classes)

optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
# optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
supervised_criterion = torch.nn.CrossEntropyLoss()

n_supervised = int(len(y_true) * 1)
#n_supervised = int(1842)
l_unsupervised = u_l#[n_supervised:]
s_unsupervised = s_u#[n_supervised:]
features_unsupervised = x_u#x_train[n_supervised:]
y_unsupervised = torch.tensor(y_u).long()#torch.tensor(y_torch.tensor(y_true[n_supervised:]).long()#true[n_supervised:]).long()
y_supervised = torch.tensor(y_true[:n_supervised]).long()
features_supervised = x_train[:n_supervised]
l_supervised = l[:n_supervised]
s_supervised = s[:n_supervised]

X = torch.cat((features_supervised, features_unsupervised), 0)
y = torch.cat((y_supervised, y_unsupervised), 0)
l = torch.cat((l_supervised, l_unsupervised), 0)
s = torch.cat((s_supervised, s_unsupervised), 0)
r = torch.ones(X.shape[0])
r[features_supervised.shape[0]:] = 0
idx = np.arange(X.shape[0])
rand_idx = np.random.choice(idx, X.shape[0], replace=False)
X = X[rand_idx]
y = y[rand_idx]
l = l[rand_idx]
s = s[rand_idx]
r = r[rand_idx]

print('X', X.shape)
print('y', y.shape)
print('l', l.shape)
print('s', s.shape)

print('r', r.shape)
# print('s_supervised', s_supervised.shape)
# print('x_supervised', features_supervised.shape)
# print('y_supervised', y_supervised.shape)

gm_mac, gm_acc, lr_mac, lr_acc = 0,0,0,0
BATCH_SIZE = 10000
for epoch in range(100):
    for i in range(int(np.floor(len(features_unsupervised)/BATCH_SIZE))):
        if (i+1) * BATCH_SIZE < len(features_unsupervised):
            X_batch = X[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)]
            l_batch = l[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)]
            s_batch = s[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)]
            y_batch = y[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)]
            r_batch = r[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)]

            # print('X_batch', X_batch.shape)
            # print('y_batch', y_batch.shape)
            # print('l_batch', l_batch.shape)
            # print('s_batch', s_batch.shape)
            # print('r', r_batch)

            r_arr = r_batch.numpy()
            X_sup_batch = X_batch[np.where(r_arr == 1)]
            X_unsup_batch = X_batch[np.where(r_arr == 0)]
            y_sup_batch = y_batch[np.where(r_arr == 1)]
            y_unsup_batch = y_batch[np.where(r_arr == 0)]
            l_sup_batch = l_batch[np.where(r_arr == 1)]
            l_unsup_batch = l_batch[np.where(r_arr == 0)]
            s_sup_batch = s_batch[np.where(r_arr == 1)]
            s_unsup_batch = s_batch[np.where(r_arr == 0)]
            # print('X_sup_batch.shape, y_sup_batch',X_sup_batch.shape[0], y_sup_batch.shape)
        else:
            X_batch = X[i*BATCH_SIZE:]
            l_batch = y[i*BATCH_SIZE:]
            s_batch = s[i*BATCH_SIZE:]
            y_batch = y[i*BATCH_SIZE:]
            r_batch = r[i*BATCH_SIZE:]
            X_sup_batch = X_batch[np.where(r_arr == 1)]
            X_unsup_batch = X_batch[np.where(r_arr == 0)]
            y_sup_batch = y_batch[np.where(r_arr == 1)]
            y_unsup_batch = y_batch[np.where(r_arr == 0)]
            l_sup_batch = l_batch[np.where(r_arr == 1)]
            l_unsup_batch = l_batch[np.where(r_arr == 0)]
            s_sup_batch = s_batch[np.where(r_arr == 1)]
            s_unsup_batch = s_batch[np.where(r_arr == 0)]
            # print('X_sup_batch.shape, y_sup_batch',X_sup_batch.shape, y_sup_batch)

        lr_model.train()
        optimizer.zero_grad()

        if X_sup_batch.shape[0]!= 0:
            loss_1 = supervised_criterion(lr_model(X_sup_batch), y_sup_batch)
            loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, y_sup_batch, l_sup_batch, s_sup_batch, k, n_classes, continuous_mask)
            
        # unsupervised_lr_probability = torch.nn.Softmax()(lr_model(X_unsup_batch))
        # loss_2 = entropy(unsupervised_lr_probability)
        y_pred_unsupervised = np.argmax(probability(theta, pi_y, pi, l_unsup_batch, s_unsup_batch, k, n_classes, continuous_mask).detach().numpy(), 1)
        loss_3 = supervised_criterion(lr_model(X_unsup_batch), torch.tensor(y_pred_unsupervised))
        
        # loss_5 = log_likelihood_loss(theta, pi_y, pi, l_unsup_batch, s_unsup_batch, k, n_classes, continuous_mask)
        prec_loss = precision_loss(theta, k, n_classes, a)
        probs_graphical = probability(theta, pi_y, pi, l_batch, s_batch, k, n_classes, continuous_mask)
        # print('probs_graphical',probs_graphical.shape)
        probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
        probs_lr = torch.nn.Softmax()(lr_model(X_batch))
        loss_6 = kl_divergence(probs_graphical, probs_lr)
        if X_sup_batch.shape[0]!= 0:
            loss = loss_3 + loss_6 + loss_1 + loss_4
        else:
            loss = loss_3 + loss_6 #+ loss_3 + loss_4        
        # loss = loss + prec_loss # loss_4#+loss_2 #+  loss_3# + prec_loss
        # print(loss.item())
        loss.backward()
        optimizer.step()


        y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
#    print("Epoch: {}\tGM - f1_score: {}".format(epoch, f1_score(y_true_test, y_pred, average="macro")))
#    print("Epoch: {}\tGM - Accuracy: {}".format(epoch, accuracy_score(y_true_test, y_pred)))

    
    cur_gm_acc = accuracy_score(y_true_test, y_pred)
    cur_gm_f1 = f1_score(y_true_test, y_pred, average="macro")
    
    if (gm_acc <cur_gm_acc  or gm_mac < cur_gm_f1):
        gm_acc = cur_gm_acc
        gm_mac = cur_gm_f1
        highep = epoch
        print("GM" , lr_mac, lr_acc, epoch)
    
    probs = torch.nn.Softmax()(lr_model(x_test))#[:, 0]
    #probs = torch.stack([probs, 1 - probs]).T
    y_pred = np.argmax(probs.detach().numpy(), 1)
    cur_lr_acc = accuracy_score(y_true_test, y_pred)
    cur_lr_mac = f1_score(y_true_test, y_pred, average="macro")
    #print("Epoch: {}\tMacro f1_score: {}".format(epoch, lr_mac))   
    #print("Epoch: {}\t Accuracy: {}".format(epoch, lr_acc))

    if (lr_acc < cur_lr_acc or lr_mac < cur_lr_mac):
        lr_acc = accuracy_score(y_true_test, y_pred)
        lr_mac = f1_score(y_true_test, y_pred, average="macro")
        highep = epoch
        print("LR" , lr_mac, lr_acc, epoch)

print("LR Epoch: {}\tMacro f1_score: {}".format(highep, lr_mac))    
print("LR Epoch: {}\t Accuracy: {}".format(highep, lr_acc))
print("GM Epoch: {}\tMacro f1_score: {}".format(highep, gm_mac))    
print("GM Epoch: {}\t Accuracy: {}".format(highep, gm_acc))
