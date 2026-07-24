import torch
import torch.nn as nn
from .atomic_layer import AtomicLayer
from .equivariant_spectral_conv import EFNO3d

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

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

    def forward(self, ato_num, ato_pos, neb_matrix, cell_shifts, cell):
        env_num = torch.take_along_dim(
            ato_num[:,None,:],
            neb_matrix,
            dim=2
        )
        neb_pos = torch.take_along_dim(
            ato_pos[:,None,:],
            neb_matrix[...,None],
            dim=2
        ) + torch.einsum('B N M i, B i j -> B N M j', cell_shifts, cell)
        rel_pos = neb_pos - ato_pos[...,None,:]
        rel_dis = rel_pos.norm(dim=-1)

        rel_pos = rel_pos / (rel_dis + (rel_dis < 1e-2))[...,None]
        rel_pos = torch.cat([rel_pos, torch.ones_like(rel_pos[...,:1])], dim=-1)
        srij = self.s5(rel_dis) * (rel_dis > 1e-2) * (env_num > 0)

        ene_emb = self.ato_emb(ato_num)
        env_emb = self.ato_emb(env_num)

        for layer in self.layers:
            ene_emb, env_emb = layer(ene_emb,env_emb,rel_pos,rel_dis,srij)
        
        return self.ene_lin(ene_emb)
    
    def s5(self, x):
        x = x / self.cutoff_r
        x = x * (x < 1)
        return (1 - x**5*(56 - 140*x + 120*x**2 - 35*x**3)) * (x < 1)