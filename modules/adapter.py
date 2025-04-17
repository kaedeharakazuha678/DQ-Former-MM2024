import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
    ):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, in_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return residual + x


class BottleneckAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(BottleneckAdapter, self).__init__()
        self.down_project = nn.Linear(input_dim, output_dim)
        self.up_project = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        x = self.down_project(x)
        x = F.relu(x)
        x = self.up_project(x)
        return x
    

class FeedForwardNetwork(nn.Module):
    "Implement the FFN function"
    def __init__(self, dim, FFNdim,dropout = 0.3) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.FFN1 = nn.Linear(dim, FFNdim)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.FFN2 = nn.Linear(FFNdim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.FFN1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.FFN2(x1)
        x1 = self.dropout2(x1)
        return x1