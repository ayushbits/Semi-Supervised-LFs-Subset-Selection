import torch
import sys
import numpy as np
from deep_net import *
from logistic_regression import *
from sklearn.metrics import f1_score
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

objs = []

with open('Data/SMS/d_processed.p', 'rb') as f:
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
with open('Data/SMS/U_processed.p', 'rb') as f:
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
covered_indices = l_unsupervised.sum(1).nonzero().squeeze()
x_unsupervised = x_unsupervised[covered_indices]
y_unsupervised = y_unsupervised[covered_indices]
l_unsupervised = l_unsupervised[covered_indices]
s_unsupervised = s_unsupervised[covered_indices]

objs = []
with open('Data/SMS/validation_processed.p', 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_valid = torch.tensor(objs[0]).double()[-69:]
y_valid = torch.tensor(objs[3]).long()[-69:]
l_valid = torch.tensor(objs[2]).long()[-69:]
s_valid = torch.tensor(objs[2]).double()[-69:]

objs1 = []
with open('Data/SMS/test_processed.p', 'rb') as f:
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
n_lfs = 73
n_features = x_supervised.shape[1]
n_hidden = 512

k = torch.from_numpy(np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])).long()
k = 1 - k

continuous_mask = torch.zeros(n_lfs).double()

a = torch.ones(n_lfs).double() * 0.9

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
pi_y = torch.ones(n_classes).double()
x_train = torch.cat([x_supervised, x_unsupervised])
y_train = torch.cat([y_supervised, y_unsupervised])
indices = np.random.choice(np.arange(x_train.shape[0]), len(x_supervised), replace=False)
supervised_mask = torch.zeros(x_train.shape[0])
supervised_mask[indices] = 1
supervised_criterion = torch.nn.CrossEntropyLoss()
best_score = 0
best_epoch = 0
save_folder = sys.argv[1]
num_runs = int(sys.argv[9])
print('num runs are ', sys.argv[1], num_runs)
best_score_lr, best_score_gm, best_epoch_lr, best_epoch_gm, best_score_lr_val, best_score_gm_val = 0, 0, 0, 0, 0, 0
stop_pahle = []
stop_pahle_gm = []
final_score_gm_val, final_score_lr_val, final_score_gm, final_score_lr = [],[], [],[]
for lo in range(0, num_runs):
    pi = torch.ones((n_classes, n_lfs)).double()
    pi.requires_grad = True

    theta = torch.ones((n_classes, n_lfs)).double() * 1
    theta.requires_grad = True

    pi_y = torch.ones(n_classes).double()
    pi_y.requires_grad = True

    # lr_model = LogisticRegression(n_features, n_classes)
    lr_model = DeepNet(n_features, n_hidden, n_classes)

    optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
    optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=0.0003)
    optimizer_gm = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
    # optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
    supervised_criterion = torch.nn.CrossEntropyLoss()

    dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
    save_folder = sys.argv[1]
    print('num runs are ', sys.argv[1], num_runs)
    best_score_lr, best_score_gm, best_epoch_lr, best_epoch_gm, best_score_lr_val, best_score_gm_val = 0, 0, 0, 0, 0, 0
    stop_pahle = []
    stop_pahle_gm = []
    for epoch in range(100):

        lr_model.train()

        for batch_ndx, sample in enumerate(loader):
            optimizer_lr.zero_grad()
            optimizer_gm.zero_grad()
            unsup = []
            sup = []
            supervised_indices = sample[4].nonzero().view(-1)
            unsupervised_indices = (1 - sample[4]).nonzero().squeeze()
            if (sys.argv[2] == 'l1'):
                if len(supervised_indices) > 0:
                    loss_1 = supervised_criterion(lr_model(sample[0][supervised_indices]),
                                                  sample[1][supervised_indices])
                else:
                    loss_1 = 0
            if (sys.argv[3] == 'l2'):
                unsupervised_lr_probability = torch.nn.Softmax()(lr_model(sample[0][unsupervised_indices]))
                loss_2 = entropy(unsupervised_lr_probability)
            else:
                loss_2 = 0
            if (sys.argv[4] == 'l3'):
                y_pred_unsupervised = np.argmax(
                    probability(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k,
                                n_classes, continuous_mask).detach().numpy(), 1)
                loss_3 = supervised_criterion(lr_model(sample[0][unsupervised_indices]),
                                              torch.tensor(y_pred_unsupervised))
            else:
                loss_3 = 0

            if (sys.argv[5] == 'l4' and len(supervised_indices) > 0):
                loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, sample[1][supervised_indices],
                                                        sample[2][supervised_indices], sample[3][supervised_indices], k,
                                                        n_classes,
                                                        continuous_mask)
            else:
                loss_4 = 0

            if (sys.argv[6] == 'l5'):
                loss_5 = log_likelihood_loss(theta, pi_y, pi, sample[2][unsupervised_indices],
                                             sample[3][unsupervised_indices], k, n_classes, continuous_mask)
            else:
                loss_5 = 0

            if (sys.argv[7] == 'l6'):
                if (len(supervised_indices) > 0):
                    supervised_indices = supervised_indices.tolist()
                    probs_graphical = probability(theta, pi_y, pi, torch.cat(
                        [sample[2][unsupervised_indices], sample[2][supervised_indices]]), \
                                                  torch.cat(
                                                      [sample[3][unsupervised_indices], sample[3][supervised_indices]]),
                                                  k, n_classes, continuous_mask)
                else:
                    probs_graphical = probability(theta, pi_y, pi, sample[2][unsupervised_indices],
                                                  sample[3][unsupervised_indices], \
                                                  k, n_classes, continuous_mask)
                probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
                probs_lr = torch.nn.Softmax()(lr_model(sample[0]))
                loss_6 = kl_divergence(probs_graphical, probs_lr)
            else:
                loss_6 = 0
            # loss_6 = - torch.log(1 - probs_graphical * (1 - probs_lr)).sum(1).mean()
            if (sys.argv[8] == 'qg'):
                prec_loss = precision_loss(theta, k, n_classes, a)
            else:
                prec_loss = 0
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_6 + loss_5 + prec_loss
            # print('loss is',loss)
            if loss != 0:
                loss.backward()
                optimizer_gm.step()
                optimizer_lr.step()
        # Test
        y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(),
                           1)
        gm_acc = f1_score(y_test, y_pred, average="binary")
        # Valid
        y_pred = np.argmax(
            probability(theta, pi_y, pi, l_valid, s_valid, k, n_classes, continuous_mask).detach().numpy(), 1)
        gm_valid_acc = f1_score(y_valid, y_pred, average="binary")
        # LR Test
        probs = torch.nn.Softmax()(lr_model(x_test))
        y_pred = np.argmax(probs.detach().numpy(), 1)
        lr_acc = f1_score(y_test, y_pred, average="binary")
        # LR Valid
        probs = torch.nn.Softmax()(lr_model(x_valid))
        y_pred = np.argmax(probs.detach().numpy(), 1)
        lr_valid_acc = f1_score(y_valid, y_pred, average="binary")
        if gm_valid_acc > best_score_gm_val and gm_valid_acc > best_score_lr_val:
            best_epoch_gm = epoch
            best_score_gm_val = gm_valid_acc
            best_score_gm = gm_acc
            best_epoch_lr = epoch
            best_score_lr_val = lr_valid_acc
            best_score_lr = lr_acc
            checkpoint = {'theta': theta, 'pi': pi}
            torch.save(checkpoint, save_folder + "/gm_" + str(epoch) + ".pt")
            checkpoint = {'params': lr_model.state_dict()}
            torch.save(checkpoint, save_folder + "/lr_" + str(epoch) + ".pt")
            stop_pahle = []
            stop_pahle_gm = []

        if lr_valid_acc > best_score_lr_val and lr_valid_acc > best_score_gm_val:
            best_epoch_lr = epoch
            best_score_lr_val = lr_valid_acc
            best_score_lr = lr_acc
            best_epoch_gm = epoch
            best_score_gm_val = gm_valid_acc
            best_score_gm = gm_acc
            checkpoint = {'theta': theta, 'pi': pi}
            torch.save(checkpoint, save_folder + "/gm_" + str(epoch) + ".pt")
            checkpoint = {'params': lr_model.state_dict()}
            torch.save(checkpoint, save_folder + "/lr_" + str(epoch) + ".pt")
            stop_pahle = []
            stop_pahle_gm = []

        if len(stop_pahle) > 7 and len(stop_pahle_gm) > 7 and (all(best_score_lr_val >= k for k in stop_pahle) or \
                                                               all(best_score_gm_val >= k for k in stop_pahle_gm)):
            print('Early Stopping at', best_epoch_gm, best_score_gm, best_score_lr)
            print('Validation score Early Stopping at', best_epoch_gm, best_score_lr_val, best_score_gm_val)
            break
        else:
            stop_pahle.append(lr_valid_acc)
            stop_pahle_gm.append(gm_valid_acc)

    print("Run \t", lo, "Epoch, GM, LR \t", best_epoch_lr, best_score_gm, best_score_lr)
    print("Run \t", lo, "GM Val, LR Val \t", epoch, best_score_gm_val, best_score_lr_val)
    final_score_gm.append(best_score_gm)
    final_score_lr.append(best_score_lr)
    final_score_gm_val.append(best_score_gm_val)
    final_score_lr_val.append(best_score_lr_val)

print("Averaged scores are for GM,LR", np.sum(final_score_gm) / num_runs, np.sum(final_score_lr) / num_runs)
print("VALIDATION Averaged scores are for GM,LR", np.sum(final_score_gm_val) / num_runs,
      np.sum(final_score_lr_val) / num_runs)
