import torch
import math
import torch.nn as nn
from .. import functional as F
import torch.nn.functional as Ft


class EquivariantSpectralConv3d(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.C = C
        self.L = L
        q = torch.fft.fftfreq(L, d=1/L)
        k = torch.fft.fftfreq(L, d=1/L)
        q1, q2, q3 = torch.meshgrid(q, q, k, indexing='ij')
        q_square = (q1**2 + q2**2 + q3**2).int()
        self.register_buffer('key', torch.searchsorted(q_square.unique(), q_square))
        self.opt_weight = nn.Parameter(torch.randn(q_square.unique().numel(), C, 2) / C**0.5, requires_grad=True)

    def forward(self, input):
        weight = torch.view_as_complex(self.opt_weight[self.key]).permute(3, 0, 1, 2)
        return F.spectral_conv3d(input, weight)

class EFNO3d(nn.Module):
    def __init__(self, C_in, C_out, L):
        super().__init__()
        q = torch.fft.fftfreq(L, d=1/L)
        k1, k2, k3 = torch.meshgrid(q, q, q, indexing='ij')
        self.register_buffer('k_square', k1**2 + k2**2 + k3**2)
        k_square_set = self.k_square.int().unique()
        self.register_buffer('key', torch.searchsorted(k_square_set, self.k_square.int()))
        self.opt_weight = nn.Parameter(torch.randn(k_square_set.numel(), C_in, C_out, 2) / C_in**0.5, requires_grad=True)

    def forward(self, input):
        weight = torch.view_as_complex(self.opt_weight[self.key]).permute(3, 4, 0, 1, 2) / (self.k_square + 1)
        return self.spectral_conv3d(input, weight)
    
    def spectral_conv3d(self, input, weight):
        output = torch.fft.fftn(input,dim=(-3,-2,-1))
        output_fre = torch.einsum('B C x y z, C O x y z -> B O x y z', output, weight)
        return output_fre.mean(dim=[-3,-2,-1]).real