import torch
import sys
import numpy as np
from logistic_regression import *
from sklearn.metrics import f1_score, accuracy_score
from cage import *
from sklearn.feature_extraction.text import TfidfVectorizer
from losses import *
import pickle
from torch.utils.data import TensorDataset, DataLoader
import apricot
import statistics


def find_indices(data, data_sub):
    indices = []
    for ele in data_sub:
        x = np.where(data == ele)[0]
        indices.append(statistics.mode(x))
    return indices

torch.set_default_dtype(torch.float64)
torch.set_printoptions(threshold=20)
run = 5
objs = []

with open('Data/Youtube/d_processed.p', 'rb') as f:
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

objs = []
with open('Data/Youtube/U_processed.p', 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_unsupervised = torch.tensor(objs[0]).double()
y_unsupervised = torch.tensor(objs[3]).long()
l_unsupervised = torch.tensor(objs[2]).long()
s_unsupervised = torch.tensor(objs[2]).double()

objs = []
with open('Data/Youtube/validation_processed.p', 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_valid = torch.tensor(objs[0]).double()[:100]
y_valid = objs[3][:100]
l_valid = torch.tensor(objs[2]).long()[:100]
s_valid = torch.tensor(objs[2])[:100]

objs1 = []
with open('Data/Youtube/test_processed.p', 'rb') as f:
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

n_classes = 2
n_lfs = 10
n_features = x_supervised.shape[1]

# Labeling Function Classes
k = torch.from_numpy(np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])).long()

continuous_mask = torch.zeros(n_lfs).double()

# a = torch.ones(n_lfs).double() * 0.9

a = torch.tensor(np.load("Data/Youtube/prec.npy")).double()

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
fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, n_samples=len(x_supervised))
X_sub = fl.fit_transform(x_train)
indices = find_indices(x_train, X_sub)
supervised_mask = torch.zeros(x_train.shape[0])
supervised_mask[indices] = 1

pi = torch.ones((n_classes, n_lfs)).double()
pi.requires_grad = True

theta = torch.ones((n_classes, n_lfs)).double() * 1
theta.requires_grad = True

pi_y = torch.ones(n_classes).double()
pi_y.requires_grad = True

lr_model = LogisticRegression(n_features, n_classes)

optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=0.0003)
optimizer_gm = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
# optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
supervised_criterion = torch.nn.CrossEntropyLoss()


dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
best_score = 0
best_epoch = 0

for epoch in range(100):
    lr_model.train()
    for batch_ndx, sample in enumerate(loader):
        optimizer_lr.zero_grad()
        optimizer_gm.zero_grad()
        supervised_indices = sample[4].nonzero().view(-1)
        unsupervised_indices = (1-sample[4]).nonzero().squeeze()

        if len(supervised_indices) > 0:
            loss_1 = supervised_criterion(lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
        else:
            loss_1 = 0

        unsupervised_lr_probability = torch.nn.Softmax()(lr_model(sample[0][unsupervised_indices]))
        loss_2 = entropy(unsupervised_lr_probability)

        y_pred_unsupervised = np.argmax(
            probability(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes,
                        continuous_mask).detach().numpy(), 1)
        loss_3 = supervised_criterion(lr_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))

        if len(supervised_indices) > 0:
            loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, sample[1][supervised_indices], sample[2][supervised_indices], sample[3][supervised_indices], k, n_classes,
                                                    continuous_mask)
        else:
            loss_4 = 0

        loss_5 = log_likelihood_loss(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes, continuous_mask)

        prec_loss = precision_loss(theta, k, n_classes, a)

        # loss = loss_1 #+ loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + prec_loss
        # loss = loss_4 + loss_5 + prec_loss

        probs_graphical = probability(theta, pi_y, pi, sample[2], sample[3], k, n_classes, continuous_mask)
        probs_graphical = (probs_graphical.T / probs_graphical.sum(1)).T
        probs_lr = torch.nn.Softmax()(lr_model(sample[0]))
        loss_6 = kl_divergence(probs_graphical, probs_lr)
        # loss_6 = - torch.log(1 - probs_graphical * (1 - probs_lr)).sum(1).mean()

        loss = loss_1 + loss_2  + loss_4 + loss_3 + loss_6 + loss_5# + prec_loss

        loss.backward()
        optimizer_gm.step()
        optimizer_lr.step()
    y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
    print("Epoch: {}\tGM accuracy_score: {}".format(epoch, accuracy_score(y_test, y_pred)))

    probs = torch.nn.Softmax()(lr_model(x_test))
    y_pred = np.argmax(probs.detach().numpy(), 1)
    print("Epoch: {}\tLR accuracy_score: {}".format(epoch, accuracy_score(y_test, y_pred)))

    y_pred = np.argmax(probability(theta, pi_y, pi, l_valid, s_valid, k, n_classes, continuous_mask).detach().numpy(), 1)
    print("Epoch: {}\tGM accuracy_score(Valid): {}".format(epoch, accuracy_score(y_valid, y_pred)))

    probs = torch.nn.Softmax()(lr_model(x_valid))
    y_pred = np.argmax(probs.detach().numpy(), 1)

    score = accuracy_score(y_valid, y_pred)
    if score > best_score:
        best_epoch = epoch
        best_score = score
        loss_type = "123456"
        checkpoint = {
                    'theta': theta,
                    'pi': pi,
                }

        torch.save(checkpoint, "models/youtube/run {}/youtube_gm_{}.pt".format(run, loss_type))
        torch.save(lr_model.state_dict(), "models/youtube/run {}/youtube_lr_{}.pt".format(run, loss_type))
    print("Epoch: {}\tLR accuracy_score(Valid): {}".format(epoch, accuracy_score(y_valid, y_pred)))

print(best_epoch, best_score)
