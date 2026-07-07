import math
import torch
import torch.nn.functional as F


def spectral_conv1d(input, weight):
    if math.prod(weight.shape) == 0:
        return 0
    _,Sw = weight.shape
    output = torch.fft.rfft(input)
    _,_,S = output.shape
    weight = F.pad(weight,(0,S-Sw),value=0)
    return torch.fft.irfft(weight * output)


def spectral_conv2d(input, weight):
    if math.prod(weight.shape) == 0:
        return 0
    _,Sw1,Sw2 = weight.shape
    output = torch.fft.rfft2(input)
    _,_,S1,S2 = output.shape
    weight = torch.fft.ifftshift(F.pad(torch.fft.fftshift(weight,dim=-2),(0,S2-Sw2,(S1-Sw1-1)//2+1,S1-Sw1-((S1-Sw1-1)//2+1)),value=0),dim=-2)
    return torch.fft.irfft2(weight * output)


def spectral_conv3d(input, weight):
    if math.prod(weight.shape) == 0:
        return 0
    _,Sw1,Sw2,Sw3 = weight.shape
    output = torch.fft.rfftn(input,dim=(-3,-2,-1))
    _,_,S1,S2,S3 = output.shape
    weight = torch.fft.ifftshift(F.pad(torch.fft.fftshift(weight,dim=[-2,-3]),(0,S3-Sw3,(S2-Sw2-1)//2+1,S2-Sw2-((S2-Sw2-1)//2+1),(S1-Sw1-1)//2+1,S1-Sw1-((S1-Sw1-1)//2+1)),value=0),dim=[-2,-3])
    return torch.fft.irfftn(weight * output,dim=(-3,-2,-1))