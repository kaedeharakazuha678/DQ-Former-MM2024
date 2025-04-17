import torch
import torch.nn as nn

class ConNet(nn.Module):
    def __init__(self, input_dim):
        super(ConNet, self).__init__()
        self.conf_net = nn.Linear(input_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conf_net(x))