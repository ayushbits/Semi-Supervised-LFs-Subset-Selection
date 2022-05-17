import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size, seed=7):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        torch.manual_seed(seed)

    def forward(self, x,last=False, freeze=False):

        if last:
            return self.linear(x), x
        else:
            return self.linear(x)