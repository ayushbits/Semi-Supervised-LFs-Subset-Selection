import torch
import sys
import numpy as np
from logistic_regression import *
from sklearn.metrics import f1_score
from cage import *
from losses import *

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20)

# Load Dataset
x_train = torch.tensor(np.load("Data/Synthetic/Normal/train_min_x.npy", allow_pickle=True))
x_test = torch.tensor(np.load("Data/Synthetic/Normal/test_x.npy"))

# Discrete lambda values
l = torch.abs(torch.tensor(np.load("Data/Synthetic/Normal/train_min_l.npy")).long())
l_test = torch.abs(torch.tensor(np.load("Data/Synthetic/Normal/test_l.npy")).long())

# Continuous score values
s = l.clone().double()
s_test = l_test.clone().double()

# Labeling Function Classes
k = torch.tensor(np.load("Data/Synthetic/Normal/k.npy")).long()

# True y
y_train = np.load("Data/Synthetic/Normal/train_min_y.npy")
y_test = np.load("Data/Synthetic/Normal/test_y.npy")

n_classes = len(np.unique(y_train))
n_lfs = l.shape[1]
n_features = x_train.shape[1]

continuous_mask = torch.zeros(n_lfs).double()  # All labeling functions are continuous

a = torch.ones(n_lfs).double() * 0.2  # Quality  Guide all set to 0.9

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

# Initialize parameters

pi = torch.ones((n_classes, n_lfs)).double()
pi.requires_grad = True

theta = torch.ones((n_classes, n_lfs)).double() * 1
theta.requires_grad = True

pi_y = torch.ones(n_classes).double()
pi_y.requires_grad = True

lr_model = LogisticRegression(n_features, n_classes)
# parameters = [p for p in lr_model.parameters()]
# parameters.append(pi)
# parameters.append(theta)
# parameters.append(pi_y)
# print(parameters)

optimizer = torch.optim.SGD([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.1)
# optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
supervised_criterion = torch.nn.CrossEntropyLoss()
crit = torch.nn.NLLLoss()

n_supervised = int(len(y_train) * 0.1)


l_unsupervised = l[n_supervised:len(y_train)]
s_unsupervised = s[n_supervised:len(y_train)]
x_unsupervised = x_train[n_supervised:len(y_train)]
y_unsupervised = torch.tensor(y_train[n_supervised:]).long()
y_supervised = torch.tensor(y_train[:n_supervised]).long()
x_supervised = x_train[:n_supervised]
l_supervised = l[:n_supervised]
s_supervised = s[:n_supervised]

print('l', l.shape)
print('s', s.shape)
print('x', x_train.shape)
print('y', y_train.shape)

print('l_unsupervised', l_unsupervised.shape)
print('s_unsupervised', s_unsupervised.shape)
print('x_unsupervised', x_unsupervised.shape)
print('y_unsupervised', y_unsupervised.shape)

print('l_supervised', l_supervised.shape)
print('s_supervised', s_supervised.shape)
print('x_supervised', x_supervised.shape)
print('y_supervised', y_supervised.shape)



for epoch in range(1000):
    lr_model.train()
    optimizer.zero_grad()
    loss_1 = supervised_criterion(lr_model(x_supervised), y_supervised)

    unsupervised_lr_probability = torch.nn.Softmax()(lr_model(x_unsupervised))
    loss_2 = entropy(unsupervised_lr_probability)

    y_pred_unsupervised = np.argmax(
        probability(theta, pi_y, pi, l_unsupervised, s_unsupervised, k, n_classes, continuous_mask).detach().numpy(), 1)
    loss_3 = supervised_criterion(lr_model(x_unsupervised), torch.tensor(y_pred_unsupervised))

    loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, y_supervised, l_supervised, s_supervised, k, n_classes,
                                            continuous_mask)
    # p = probability(theta, pi_y, pi, l_supervised, s_supervised, k, n_classes, continuous_mask)
    # print(crit(torch.log(p), y_supervised))

    loss_5 = log_likelihood_loss(theta, pi_y, pi, l_unsupervised, s_unsupervised, k, n_classes, continuous_mask)
    prec_loss = precision_loss(theta, k, n_classes, a)

    probs_graphical = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)
    probs_graphical = (probs_graphical.T / probs_graphical.sum(1)).T

    probs_lr = torch.nn.Softmax()(lr_model(x_train))
    loss_6 = kl_divergence(probs_graphical, probs_lr)

    #loss = loss_1 + loss_2 #+ loss_3 + loss_4 + loss_5 + loss_6 #+ prec_loss
    loss = loss_4 + loss_5 + prec_loss

    loss.backward()
    optimizer.step()

    y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
    print("Epoch: {}\tf1_score: {}".format(epoch, f1_score(y_test, y_pred, average="micro")))

    probs = torch.nn.Softmax()(lr_model(x_test))
    y_pred = np.argmax(probs.detach().numpy(), 1)
    print("Epoch: {}\tf1_score: {}".format(epoch, f1_score(y_test, y_pred, average="micro")))
