import math
import torch.nn.functional as F

def conv1d(input, weight):
    if math.prod(weight.shape) == 0:
        return 0
    _,C,_ = input.shape
    _,S = weight.shape
    padding = ((S-1)//2, S-1-(S-1)//2)
    input = F.pad(input, padding, 'circular')
    return F.conv1d(input, weight.unsqueeze(1), groups=C)

def conv2d(input, weight):
    if math.prod(weight.shape) == 0:
        return 0
    _,C,_,_ = input.shape
    _,S1,S2 = weight.shape
    padding = ((S2-1)//2, S2-1-(S2-1)//2, (S1-1)//2, S1-1-(S1-1)//2)
    input = F.pad(input, padding, 'circular')
    return F.conv2d(input, weight.unsqueeze(1), groups=C)

def conv3d(input, weight):
    if math.prod(weight.shape) == 0:
        return 0
    _,C,_,_,_ = input.shape
    _,S1,S2,S3 = weight.shape
    padding = ((S3-1)//2, S3-1-(S3-1)//2, (S2-1)//2, S2-1-(S2-1)//2, (S1-1)//2, S1-1-(S1-1)//2)
    input = F.pad(input, padding, 'circular')
    return F.conv3d(input, weight.unsqueeze(1), groups=C)