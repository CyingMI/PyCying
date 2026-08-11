import torch
import torch.nn as nn
from .. import functional as F


class BaseSpectralConv(nn.Module):
    def __init__(self, in_channels, opt_size):
        super().__init__()
        self.in_channels = in_channels
        self.opt_size = opt_size
        self.opt_weight = nn.Parameter(torch.zeros(in_channels, *self.opt_size, 2),requires_grad=True)

    def _get_conv_function(self):
        raise NotImplementedError

    def forward(self, input):
        weight = torch.view_as_complex(self.opt_weight)
        return self._get_conv_function()(input, weight)


class SpectralConv1d(BaseSpectralConv):
    def __init__(self, in_channels, opt_size):
        super().__init__(in_channels, opt_size)

    def _get_conv_function(self):
        return F.spectral_conv1d


class SpectralConv2d(BaseSpectralConv):
    def __init__(self, in_channels, opt_size):
        super().__init__(in_channels, opt_size)

    def _get_conv_function(self):
        return F.spectral_conv2d


class SpectralConv3d(BaseSpectralConv):
    def __init__(self, in_channels, opt_size):
        super().__init__(in_channels, opt_size)

    def _get_conv_function(self):
        return F.spectral_conv3d