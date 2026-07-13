import torch
import math
import torch.nn as nn
from .. import functional as F
import torch.nn.functional as Ft


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
        self.opt_weight = nn.Parameter(torch.randn(q_norm.unique().numel(), in_channels, 2), requires_grad=True)

    def forward(self, input):
        weight = torch.view_as_complex(self.opt_weight[self.key]).permute(3, 0, 1, 2)
        return F.spectral_conv3d(input, weight)
    
class EFNO3d(nn.Module):
    def __init__(self, in_channels, out_channels, opt_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.opt_size = opt_size
        q1 = torch.fft.fftfreq(opt_size[0], d=1/opt_size[0])
        q2 = torch.fft.fftfreq(opt_size[1], d=1/opt_size[1])
        q3 = torch.fft.rfftfreq(opt_size[2], d=1/opt_size[2])
        q1, q2, q3 = torch.meshgrid(q1, q2, q3, indexing='ij')
        q_norm = (q1**2 + q2**2 + q3**2).int()
        self.register_buffer('key', torch.searchsorted(q_norm.unique(), q_norm))
        self.opt_weight = nn.Parameter(torch.randn(q_norm.unique().numel(), in_channels, out_channels, 2) / math.sqrt(in_channels), requires_grad=True)

    def forward(self, input):
        weight = torch.view_as_complex(self.opt_weight[self.key]).permute(3, 4, 0, 1, 2)
        return self.spectral_conv3d(input, weight)
    
    def spectral_conv3d(self, input, weight):
        if math.prod(weight.shape) == 0:
            return 0
        _,_,Sw1,Sw2,Sw3 = weight.shape
        output = torch.fft.rfftn(input,dim=(-3,-2,-1))
        _,_,S1,S2,S3 = output.shape
        weight = torch.fft.ifftshift(Ft.pad(torch.fft.fftshift(weight,dim=[-2,-3]),(0,S3-Sw3,(S2-Sw2-1)//2+1,S2-Sw2-((S2-Sw2-1)//2+1),(S1-Sw1-1)//2+1,S1-Sw1-((S1-Sw1-1)//2+1)),value=0),dim=[-2,-3])
        output_fre = torch.einsum('B C x y z, C O x y z -> B O x y z', output, weight)
        return torch.fft.irfftn(output_fre, dim=(-3,-2,-1))