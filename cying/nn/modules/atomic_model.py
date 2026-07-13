import numpy as np
import math
import torch
import torch.nn as nn
from .atomic_layer import AtomicLayer3d

class AtomicModel(nn.Module):
    def __init__(self, size, params, cut_off_r=6):
        super().__init__()
        self.size = size
        self.params = params
        self.cut_off_r = cut_off_r
        self.atomic_model = nn.Sequential(*[
            AtomicLayer3d(self.size, **param) for param in self.params
        ])

        q1 = torch.fft.fftfreq(size[0], d=1/size[0])
        q2 = torch.fft.fftfreq(size[1], d=1/size[1])
        q3 = torch.fft.rfftfreq(size[2], d=1/size[2])
        q1, q2, q3 = torch.meshgrid(q1, q2, q3, indexing='ij')
        self.register_buffer('q', torch.stack((q1, q2, q3), dim=-1))
        q_norm = (self.q**2).sum(dim=-1).int()
        self.register_buffer('key', torch.searchsorted(q_norm.unique(), q_norm))

        self.emb = nn.Embedding(119, q_norm.unique().numel())

        self.out_linear = nn.Linear(self.params[-1]['out_channels'], 1)

        self.norm = nn.RMSNorm(self.size)


    def forward(self, cell, ato_num, ato_pos):
        ato_emb = self.emb(ato_num)[:,self.key] + 1j*0
        # 局部坐标
        scaled_pos = ato_pos @ torch.inverse(cell)
        rel_scaled_pos = scaled_pos[None,...] - scaled_pos[:,None,:]
        rel_scaled_pos = rel_scaled_pos - torch.round(rel_scaled_pos)
        rel_pos = torch.einsum('N M i, i j -> N M j', rel_scaled_pos, cell)
        rel_dis = rel_pos.norm(dim=-1)
        ato_mask = (rel_dis < self.cut_off_r) * self.sp(rel_dis/self.cut_off_r) + 1j * 0

        V_fre = torch.exp(-1j * torch.einsum('N M d, x y z d -> N M x y z', rel_pos, self.q))
        V_fre = torch.einsum('N M x y z, N M, M x y z -> N x y z', V_fre, ato_mask, ato_emb) * math.prod(self.size)
        V_rel = self.norm(torch.fft.irfftn(V_fre, dim=(-3,-2,-1)))

        V_rel = self.atomic_model(V_rel[:,None]).mean(dim=[2,3,4])

        energy = self.out_linear(V_rel).sum()

        return energy
    
    def sp(self, r, p=5):
        ap = - (p+1) * (p+2) * (p+3) / 6
        bp = p * (p + 2) * (p + 3) / 2
        cp = - p * (p + 1) * (p + 3) / 2
        dp = p * (p + 1) * (p + 2) / 6
        return (1 + r**p * (ap + bp * r + cp * r**2 + dp * r**3)) * (r < 1)