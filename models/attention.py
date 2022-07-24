import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Permute
from math import sqrt
from utils import Residual


class ScaledDotProduct(nn.Module):
    def __init__(self, dim, scale_dim):
        super().__init__()
        self.q = nn.Linear(dim, dim // scale_dim)
        self.k = nn.Linear(dim, dim // scale_dim)
        self.v = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.dim = sqrt(dim // scale_dim)

        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.normal_(self.q.bias, std=1e-6)
        nn.init.normal_(self.k.bias, std=1e-6)
        nn.init.normal_(self.v.bias, std=1e-6)

    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        z = torch.bmm(q, k.transpose(-1, -2))
        z = self.sigmoid(z / self.dim)
        z = self.bmm(z, v)
        return z


class PEG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.permute1 = Permute([0, 3, 1, 2])
        self.permute2 = Permute([0, 2, 3, 1])

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.normal_(self.conv.bias, std=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        H = int(sqrt(N))
        W = N // H
        x = x.view(B, H, W, C)
        x = self.permute1(x)
        x = self.conv(x)
        x = self.permute2(x)
        return x.view(B, B, C)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, scale_dim):
        super().__init__()
        self.sdps = []
        self.pegs = []
        for i in range(num_heads):
            self.sdps.append(ScaledDotProduct(dim=dim, scale_dim=scale_dim))
        self.sdps = nn.ModuleList(self.sdps)
        self.projector = nn.Linear(dim * num_heads, dim)
        self.num_heads = num_heads

        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.normal_(self.projector.bias, std=1e-6)

    def forward(self, q, k, v):
        _outs = []
        for m in self.sdps:
            _outs.append(m(q=q, k=k, v=v))
        return self.projector(torch.cat(_outs, dim=-1))


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * mlp_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * mlp_dim, dim)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear2.bias, std=1e-6)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, scale_dim, mlp_dim):
        super().__init__()
        self.self_atten_1 = MultiHeadAttention(dim=dim, num_heads=num_heads, scale_dim=scale_dim)
        self.self_atten_2 = MultiHeadAttention(dim=dim, num_heads=num_heads, scale_dim=scale_dim)
