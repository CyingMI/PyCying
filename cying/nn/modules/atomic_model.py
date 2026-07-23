import torch
import torch.nn as nn
from .atomic_layer import AtomicLayer
from .equivariant_spectral_conv import EFNO3d


class AtomicModel(nn.Module):
    def __init__(self, d_model, hid_width, num_layers, cutoff_r = 6):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.num_layers = num_layers
        self.cutoff_r = cutoff_r

        self.ato_emb = nn.Embedding(119, d_model, 0)

        self.layers = nn.ModuleList([
            AtomicLayer(d_model, hid_width) for _ in range(num_layers)
        ])

        self.ene_lin = nn.Linear(d_model, 1)

    def forward(self, ato_num, scaled_pos, cell):
        rel_scaled_pos = scaled_pos[:,None,...] - scaled_pos[:,:,None,:]
        rel_scaled_pos = rel_scaled_pos - torch.round(rel_scaled_pos) * (rel_scaled_pos.abs() < 2)
        rel_pos = torch.einsum('B N M i, B i j -> B N M j', rel_scaled_pos, cell)
        rel_dis = rel_pos.norm(dim=-1)

        rel_pos = rel_pos / (rel_dis + (rel_dis < 1e-2))[...,None]
        rel_pos = torch.cat([rel_pos, torch.ones_like(rel_pos[...,:1])], dim=-1)
        srij = self.s5(rel_dis) * (rel_dis > 1e-2)

        rel_pos = rel_pos * (srij / (rel_dis + (rel_dis < 1e-2)))[...,None]

        mul_pos = torch.einsum('B N M d, B N O d -> B N M O', rel_pos, rel_pos)

        ato_emb = self.ato_emb(ato_num)

        for layer in self.layers:
            ato_emb = layer(ato_emb, mul_pos)

        return self.ene_lin(ato_emb)
    
    def s5(self, x):
        x = x / self.cutoff_r
        return (1 - x**5*(56 - 140*x + 120*x**2 - 35*x**3)) * (x < 1)

# class AtomicModel(nn.Module):
#     def __init__(self, C, L, num_layers, cutoff_r = 6):
#         super().__init__()
#         self.C = C
#         self.L = L
#         self.num_layers = num_layers
#         self.cutoff_r = cutoff_r

#         q = torch.fft.fftfreq(L, 1/L)
#         k1, k2, k3 = torch.meshgrid(q, q, q, indexing='ij')
#         self.register_buffer('k', torch.stack([k1, k2, k3], dim=-1) * torch.pi / cutoff_r)
#         k_square = (k1**2 + k2**2 + k3**2).int()
#         opt_len = k_square.unique().numel()
#         self.key = torch.searchsorted(k_square.unique(), k_square)

#         self.atomic_emb = nn.Embedding(119, C, 0)

#         self.layers = nn.ModuleList([
#             AtomicLayer(C, L, opt_len, self.key) for _ in range(num_layers)
#         ])
#         self.ene_weight = nn.Parameter(torch.zeros(opt_len, C), requires_grad=True)

#     def forward(self, ato_num, scaled_pos, cell):
#         ato_emb = self.atomic_emb(ato_num)
#         rel_scaled_pos = scaled_pos[:,None,...] - scaled_pos[:,:,None,:]
#         rel_scaled_pos = rel_scaled_pos - torch.round(rel_scaled_pos) * (rel_scaled_pos.abs() < 2)
#         rel_pos = torch.einsum('B N M i, B i j -> B N M j', rel_scaled_pos, cell)
#         rel_dis = rel_pos.norm(dim=-1)

#         global_mask = (rel_dis < self.cutoff_r).float()
#         radial_mask = self.s5(rel_dis) * (rel_dis > 1e-2)
#         expijk = torch.cos(torch.einsum('i j k d, B N M d -> B N M i j k', self.k, rel_pos))
#         weight = self.ene_weight[self.key].permute(3, 0, 1, 2)
#         energy = torch.einsum(
#             'B N M i j k, B N C, B M C, C i j k, B N M -> B N i j k',
#             expijk,
#             ato_emb,
#             ato_emb,
#             weight,
#             global_mask
#         ).mean(dim=[-3,-2,-1])
#         for layer in self.layers:
#             energy, ato_emb = layer(energy, ato_emb, expijk, radial_mask, global_mask)

#         return energy
    
#     def s5(self, x):
#         x = x / self.cutoff_r
#         return (1 - x**5*(56 - 140*x + 120*x**2 - 35*x**3)) * (x < 1)