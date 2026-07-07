'''
Operator Convolutional Layers
'''
import torch
import torch.nn as nn
from .. import functional as F


class BaseConv(nn.Module):
    def __init__(self, size, in_channels, kernel_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.kernel_size = self._modify_kernel_size(kernel_size)
        self.padding = tuple((s-1)//2 for s in self.kernel_size for _ in range(2))
        self.opt_weight = nn.Parameter(torch.zeros([in_channels,*self.kernel_size]),requires_grad=True)

    def _modify_kernel_size(self, size):
        raise NotImplementedError
    
    def _get_conv_function(self):
        raise NotImplementedError
    
    def forward(self, input):
        return self._get_conv_function()(input,self.opt_weight,self.size)


class Conv1d(BaseConv):
    def __init__(self, size, in_channels, kernel_size):
        super().__init__(size, in_channels, kernel_size)

    def _modify_kernel_size(self, size):
        if isinstance(size, int):
            return (size,)
        elif isinstance(size, tuple) and len(size==1):
            return size
        else:
            raise ValueError('The parameter "kernel_size" does not match the current dimension.')
    
    def _get_conv_function(self):
        return F.conv1d


class Conv2d(BaseConv):
    def __init__(self, size, in_channels, kernel_size):
        super().__init__(size, in_channels, kernel_size)

    def _modify_kernel_size(self, size):
        if isinstance(size, int):
            return (size,)*2
        elif isinstance(size, tuple) and len(size==2):
            return size
        else:
            raise ValueError('The parameter "kernel_size" does not match the current dimension.')
    
    def _get_conv_function(self):
        return F.conv2d


class Conv3d(BaseConv):
    def __init__(self, size, in_channels, kernel_size):
        super().__init__(size, in_channels, kernel_size)

    def _modify_kernel_size(self, size):
        if isinstance(size, int):
            return (size,)*3
        elif isinstance(size, tuple) and len(size==3):
            return size
        else:
            raise ValueError('The parameter "kernel_size" does not match the current dimension.')
    
    def _get_conv_function(self):
        return F.conv3d