import torch
import torch.nn as nn
from .. import functional as F


class EquivariantSpectralConv3d(nn.Module):
    def __init__(self, in_channels, opt_size):
        super().__init__()
        self.in_channels = in_channels
        self.opt_size = opt_size
        q1 = torch.fft.fftfreq(opt_size[0], d=1/opt_size[0])
        q2 = torch.fft.fftfreq(opt_size[1], d=1/opt_size[1])
        q3 = torch.fft.rfftfreq(opt_size[2], d=1/opt_size[2])
        q1, q2, q3 = torch.meshgrid(q1, q2, q3, indexing='ij')
        q_norm = (q1**2 + q2**2 + q3**2).int()
        self.register_buffer('key', torch.searchsorted(q_norm.unique(), q_norm))
        self.opt_weight = nn.Parameter(torch.zeros(q_norm.unique().numel(), in_channels, 2), requires_grad=True)

    def forward(self, input):
        weight = torch.view_as_complex(self.opt_weight[self.key]).permute(3, 0, 1, 2)
        return F.spectral_conv3d(input, weight)