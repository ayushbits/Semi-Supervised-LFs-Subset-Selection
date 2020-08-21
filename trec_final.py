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
test_objs = load_pickle("Data/TREC/test_processed.p")
train_objs = load_pickle("Data/TREC/d_processed.p")
u_objs = load_pickle("Data/TREC/U_processed.p")

x_train = torch.from_numpy(train_objs[0]).double()
y_train = torch.from_numpy(train_objs[3])
x_test = torch.from_numpy(test_objs[0]).double()
y_test = torch.from_numpy(test_objs[3])
x_u = torch.from_numpy(u_objs[0]).double()
y_u = torch.from_numpy(u_objs[3])

n_classes = 6
n_lfs = 68
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
k = torch.from_numpy(np.array([1,0,0,4,2,1,0,2,5,4,2,1,4,0,1,5,4,0,1,4,2,0,2,1,0,0,4,4,5,0,2,5,1,0,5,0,5,5,3,5,5,0,4,1,2,1,0,2,5,0,0,2,1,0,0,5,1,2,1,5,2,2,4,5,5,4,1,3])).long()

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


for i in range(s_u.shape[0]):
    for j in range(s_u.shape[1]):
        if s_u[i, j].item() > 0.999:
            s_u[i, j] = 0.999
        if s_u[i, j].item() < 0.001:
            s_u[i, j] = 0.001

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

print('l', l.shape)
print('s', s.shape)
print('x', x_train.shape)
print('y', y_train.shape)

print('l_unsupervised', l_unsupervised.shape)
print('s_unsupervised', s_unsupervised.shape)
print('x_unsupervised', features_unsupervised.shape)
print('y_unsupervised', y_unsupervised.shape)

print('l_supervised', l_supervised.shape)
print('s_supervised', s_supervised.shape)
print('x_supervised', features_supervised.shape)
print('y_supervised', y_supervised.shape)

for epoch in range(1000):
    lr_model.train()
    optimizer.zero_grad()
    loss_1 = supervised_criterion(lr_model(features_supervised), y_supervised)
    unsupervised_lr_probability = torch.nn.Softmax()(lr_model(features_unsupervised))
    loss_2 = entropy(unsupervised_lr_probability)
    y_pred_unsupervised = np.argmax(probability(theta, pi_y, pi, l_unsupervised, s_unsupervised, k, n_classes, continuous_mask).detach().numpy(), 1)
    loss_3 = supervised_criterion(lr_model(features_unsupervised), torch.tensor(y_pred_unsupervised))
    loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, y_supervised, l_supervised, s_supervised, k, n_classes,
                                            continuous_mask)
    loss_5 = log_likelihood_loss(theta, pi_y, pi, l_unsupervised, s_unsupervised, k, n_classes, continuous_mask)
    prec_loss = precision_loss(theta, k, n_classes, a)
    probs_graphical = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)
# #    probs_graphical = (probs_graphical.T / probs_graphical.sum(1)).T
#   probs_lr = torch.nn.Softmax()(lr_model(x_train))
    probs_lr = torch.nn.Softmax()(lr_model(x_train))
    loss_6 = kl_divergence(probs_graphical, probs_lr)

    loss = loss_4 +  loss_5 + loss_1 + loss_2 + loss_3 + loss_6 + prec_loss
    #loss = loss_1+loss_2 #+  loss_3# + prec_loss 
    print(loss)
    loss.backward()
    optimizer.step()

    y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
    print("Epoch: {}\tGM - f1_score: {}".format(epoch, f1_score(y_true_test, y_pred, average="macro")))
    print("Epoch: {}\tGM - Accuracy: {}".format(epoch, accuracy_score(y_true_test, y_pred)))

    probs = torch.nn.Softmax()(lr_model(x_test))#[:, 0]
    #probs = torch.stack([probs, 1 - probs]).T
    y_pred = np.argmax(probs.detach().numpy(), 1)
    print("Epoch: {}\tMacro f1_score: {}".format(epoch, f1_score(y_true_test, y_pred, average="macro")))
    print("Epoch: {}\t Accuracy_score: {}".format(epoch, accuracy_score(y_true_test, y_pred)))
