import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Permute
from math import sqrt


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.linear(x)
        x = self.permute2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, features, ):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.norm(x)
        x = self.permute2(x)
        return x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.gelu(x)
        x = self.permute2(x)
        return x


class SoftGatedSkipConnection(nn.Module):
    def __init__(self, block, dim):
        super().__init__()
        self.block = block
        self.alphas = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return (x.clone() * self.sigmoid(self.alphas)) + self.block(x)
