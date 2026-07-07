import math
import torch.nn.functional as F

def conv1d(input, weight, size):
    if math.prod(weight.shape) == 0:
        return 0
    in_size = input.shape[-1:]
    input = F.interpolate(input, size=size, mode='linear')
    _,C,_ = input.shape
    _,S = weight.shape
    padding = ((S-1)//2,S-1-(S-1)//2)
    input = F.pad(input,padding,'circular')
    return F.interpolate(F.conv1d(input,weight.unsqueeze(1),groups=C), size=in_size, mode='linear')

def conv2d(input, weight, size):
    if math.prod(weight.shape) == 0:
        return 0
    in_size = input.shape[-2:]
    input = F.interpolate(input, size=size, mode='bilinear')
    _,C,_,_ = input.shape
    _,S1,S2 = weight.shape
    padding = ((S2-1)//2,S2-1-(S2-1)//2,(S1-1)//2,S1-1-(S1-1)//2)
    input = F.pad(input,padding,'circular')
    return F.interpolate(F.conv2d(input,weight.unsqueeze(1),groups=C), size=in_size, mode='bilinear')

def conv3d(input, weight, size):
    if math.prod(weight.shape) == 0:
        return 0
    in_size = input.shape[-3:]
    input = F.interpolate(input, size=size, mode='trilinear')
    _,C,_,_,_ = input.shape
    _,S1,S2,S3 = weight.shape
    padding = ((S3-1)//2,S3-1-(S3-1)//2,(S2-1)//2,S2-1-(S2-1)//2,(S1-1)//2,S1-1-(S1-1)//2)
    input = F.pad(input,padding,'circular')
    return F.interpolate(F.conv3d(input,weight.unsqueeze(1),groups=C), size=in_size, mode='trilinear')