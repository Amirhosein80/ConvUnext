import torch
import torch.nn as nn
from torchvision.ops.misc import Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from math import sqrt
from utils import LayerNormalization


class ScaledDotProduct(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.q = nn.Linear(dim, dim // num_heads)
        self.k = nn.Linear(dim, dim // num_heads)
        self.v = nn.Linear(dim, dim // num_heads)
        self.sigmoid = nn.Sigmoid()
        self.scale = sqrt(dim // num_heads)

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
        z = q @ k.transpose(-1, -2)
        z = self.sigmoid(z / self.scale)
        z = z @ v
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
        return x.view(B, N, C)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.sdps = []
        self.pegs = []
        for i in range(num_heads):
            self.sdps.append(ScaledDotProduct(dim=dim, num_heads=num_heads))
        self.sdps = nn.ModuleList(self.sdps)
        self.projector = nn.Linear(dim, dim)
        self.num_heads = num_heads

        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.normal_(self.projector.bias, std=1e-6)

    def forward(self, q, k, v):
        _outs = []
        for m in self.sdps:
            _outs.append(m(q=q, k=k, v=v))
        return self.projector(torch.cat(_outs, dim=-1))


class CNBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3, dilation=1, layer_scale=1e-6, stochastic_depth_prob=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.alphas = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input * self.alphas
        return result


class OverlapPatch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormalization(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.normal_(self.conv.bias, std=1e-6)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class UnPatch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.normal_(self.conv.bias, std=1e-6)

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, scale_dim, mlp_dim_scale):
        super().__init__()
        self.head = CNBlock(dim=dim)
        self.self_atten_1 = MultiHeadAttention(dim=dim, num_heads=num_heads)
        self.self_atten_2 = MultiHeadAttention(dim=dim, num_heads=num_heads)
        self.alphas_1 = nn.Parameter(torch.ones(1, 1, dim))
        self.alphas_2 = nn.Parameter(torch.ones(1, 1, dim))
        self.alphas_3 = nn.Parameter(torch.ones(1, 1, dim))
        self.peg = PEG(dim=dim)
        self.norm_1 = nn.LayerNorm(normalized_shape=dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=dim)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def _preprocess_input(self, x):
        B, C, H, W = x.shape
        x = self.permute1(x)
        x = x.view(B, H * W, C)
        return x

    def _postprocess_output(self, x):
        B, N, C = x.shape
        H = int(sqrt(N))
        W = N // H
        x = x.view(B, H, W, C)
        x = self.permute2(x)
        return x

    def forward(self, decoder, encoder=None):
        decoder = self._preprocess_input(decoder)
        if encoder is not None:
            encoder = self._preprocess_input(encoder)
        decoder = (self.alphas_1 * decoder.clone()) + self.self_atten_1(q=decoder, k=decoder, v=decoder)
        decoder = self.peg(decoder)
        decoder = self.norm_1(decoder)
        if encoder is not None:
            decoder = (self.alphas_2 * decoder.clone()) + self.self_atten_2(q=decoder, k=encoder, v=encoder)
        else:
            decoder = (self.alphas_2 * decoder.clone()) + self.self_atten_2(q=decoder, k=decoder, v=decoder)
        decoder = self.norm_2(decoder)
        decoder = self._postprocess_output(decoder)
        decoder = self.head(decoder)
        return self._postprocess_output(decoder)
