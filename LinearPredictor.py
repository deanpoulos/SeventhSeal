from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.l1 = nn.Linear(784,10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.log_softmax(self.l1(x), dim=1)
        return x

            nn.ReLU(),
        )
        self.l5 = nn.Sequential(
            nn.Linear(192, 10),
            nn.BatchNorm1d(10)
        )