import torch
import torch.nn as nn
from .equivariant_conv import EquivariantConv3d
from .equivariant_spectral_conv import EquivariantSpectralConv3d


class AtomicLayer3d(nn.Module):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_width = hidden_width
        self.spe_opt_size = spe_opt_size
        self.spa_opt_size = spa_opt_size
        self.spectral_opts = EquivariantSpectralConv3d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
        self.spatial_opts = EquivariantConv3d(
            size=self.size,
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
        self.nonlinear_opts = nn.Sequential(
            nn.Conv3d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv3d(self.hidden_width,self.out_channels,1)
        )
        self.linear_skip = nn.Conv3d(self.in_channels,self.out_channels,1)

    def forward(self, input):
        lin_opts = self.spectral_opts(input) + self.spatial_opts(input) + input
        return self.nonlinear_opts(lin_opts) + self.linear_skip(lin_opts)

    def _create_spectral_opts(self):
        raise NotImplementedError

    def _create_spatial_opts(self):
        raise NotImplementedError

    def _create_nonlinear_opts(self):
        raise NotImplementedError
    
    def _create_linear_skip(self):
        raise NotImplementedError