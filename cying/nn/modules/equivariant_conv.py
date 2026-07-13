'''
Operator Convolutional Layers
'''
import torch
import torch.nn as nn
from .. import functional as F


class EquivariantConv3d(nn.Module):
    def __init__(self, size, in_channels, kernel_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.kernel_size = self._modify_kernel_size(kernel_size)
        x1 = torch.arange(-(self.kernel_size[0]//2), self.kernel_size[0]//2+1)
        x2 = torch.arange(-(self.kernel_size[1]//2), self.kernel_size[1]//2+1)
        x3 = torch.arange(-(self.kernel_size[2]//2), self.kernel_size[2]//2+1)
        x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')
        x_norm = (x1**2 + x2**2 + x3**2).int()
        self.register_buffer('key', torch.searchsorted(x_norm.unique(), x_norm))
        self.opt_weight = nn.Parameter(torch.zeros([x_norm.unique().numel(),in_channels]),requires_grad=True)

    def _modify_kernel_size(self, size):
        if isinstance(size, int):
            return (size,)*3
        elif isinstance(size, tuple) and len(size==3):
            return size
        else:
            raise ValueError('The parameter "kernel_size" does not match the current dimension.')
    
    def forward(self, input):
        weight = self.opt_weight[self.key].permute(3, 0, 1, 2)
        return F.conv3d(input,weight,self.size)