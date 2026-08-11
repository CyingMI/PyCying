import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv1d, Conv2d, Conv3d
from .spectral_conv import SpectralConv1d, SpectralConv2d, SpectralConv3d


class BaseOperatorLayer(nn.Module):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_width = hidden_width
        self.spe_opt_size = spe_opt_size
        self.spa_opt_size = spa_opt_size
        self.spectral_opts = self._create_spectral_opts()
        self.spatial_opts = self._create_spatial_opts()
        self.nonlinear_opts = self._create_nonlinear_opts()
        self.linear_skip = self._create_linear_skip()

    def forward(self, input):
        input = self._interpolate(input)
        lin_opts = self.spectral_opts(input) + self.spatial_opts(input) + input
        return self.nonlinear_opts(lin_opts) + self.linear_skip(lin_opts)

    def _interpolate(self, input):
        raise NotImplementedError

    def _create_spectral_opts(self):
        raise NotImplementedError

    def _create_spatial_opts(self):
        raise NotImplementedError

    def _create_nonlinear_opts(self):
        raise NotImplementedError
    
    def _create_linear_skip(self):
        raise NotImplementedError


class OperatorLayer1d(BaseOperatorLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _interpolate(self, input):
        return F.interpolate(input, size=self.size, mode='linear')

    def _create_spectral_opts(self):
        return SpectralConv1d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return Conv1d(
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
    
    def _create_nonlinear_opts(self):
        return nn.Sequential(
            nn.Conv1d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv1d(self.hidden_width,self.out_channels,1)
        )
    
    def _create_linear_skip(self):
        return nn.Conv1d(self.in_channels,self.out_channels,1)


class OperatorLayer2d(BaseOperatorLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _interpolate(self, input):
        return F.interpolate(input, size=self.size, mode='bilinear')

    def _create_spectral_opts(self):
        return SpectralConv2d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return Conv2d(
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
    
    def _create_nonlinear_opts(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv2d(self.hidden_width,self.out_channels,1)
        )
    
    def _create_linear_skip(self):
        return nn.Conv2d(self.in_channels,self.out_channels,1)


class OperatorLayer3d(BaseOperatorLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _interpolate(self, input):
        return F.interpolate(input, size=self.size, mode='trilinear')

    def _create_spectral_opts(self):
        return SpectralConv3d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return Conv3d(
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
    
    def _create_nonlinear_opts(self):
        return nn.Sequential(
            nn.Conv3d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv3d(self.hidden_width,self.out_channels,1)
        )
    
    def _create_linear_skip(self):
        return nn.Conv3d(self.in_channels,self.out_channels,1)