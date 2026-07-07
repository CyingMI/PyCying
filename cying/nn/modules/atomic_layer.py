import torch
import torch.nn as nn
from .equivariant_spectral_conv import EquivariantSpectralConv3d


class AtomicLayer(nn.Module):
    def __init__(self, size, in_channels, out_channels, hidden_width, opt_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_width = hidden_width
        self.opt_size = opt_size
        self.spectral_opts = EquivariantSpectralConv3d(
            in_channels=self.in_channels,
            opt_size=opt_size
        )
        self.nonlinear_opts = nn.Sequential(
            nn.Conv3d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv3d(self.hidden_width,self.out_channels,1)
        )
        self.linear_skip = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1
        )
    
    def forward(self, input):
        lin_opts = self.spectral_opts(input) + input
        return self.nonlinear_opts(lin_opts) + self.linear_skip(lin_opts)