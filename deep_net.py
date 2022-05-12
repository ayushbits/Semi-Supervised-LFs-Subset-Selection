import torch.nn as nn
import torch.nn.functional as F
import torch

class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNet, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x,last=False, freeze=False):

        if freeze:
            with torch.no_grad():
                out = F.relu(self.linear_1(x))
                out = F.relu(self.linear_2(out))
        else:
            out = F.relu(self.linear_1(x))
            out = F.relu(self.linear_2(out))

        if last:
            return self.out(out), out
        else:
            return self.out(out)