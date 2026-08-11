import torch
import numpy as np
import torch.nn as nn
from .atomic_layer import AtomicLayer

class AtomicModel(nn.Module):
    def __init__(self, d_model, hid_width, num_layers, cutoff_r = 6, L=32, N=128):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.num_layers = num_layers
        self.cutoff_r = cutoff_r
        self.L = L
        self.N = N
        k, w = self.gauss_legendre_ab(N, 0, L)
        self.register_buffer('k', k)
        self.register_buffer('w', w)

        self.ato_emb = nn.Embedding(119, d_model, 0)

        self.layers = nn.ModuleList([
            AtomicLayer(d_model, hid_width, L, N, cutoff_r) for _ in range(num_layers)
        ])

        self.lin_out = nn.Linear(d_model, 1)

    def forward(self, ato_num, ato_pos, neb_matrix, cell_shifts, cell):
        ato_emb = self.ato_emb(ato_num)
        env_num = torch.take_along_dim(
            ato_num[:,None,:],
            neb_matrix,
            dim=2
        )
        ato_env = torch.take_along_dim(
            ato_emb[:,None,:],
            neb_matrix[...,None],
            dim=2
        )

        neb_pos = torch.take_along_dim(
            ato_pos[:,None,:],
            neb_matrix[...,None],
            dim=2
        ) + torch.einsum('B N M i, B i j -> B N M j', cell_shifts, cell)
        rel_posij = neb_pos - ato_pos[...,None,:]
        rel_disij = rel_posij.norm(dim=-1)

        rel_posijk = rel_posij[...,None,:] - rel_posij[...,None,:,:]
        rel_disijk = rel_posijk.norm(dim=-1)

        padding = env_num[...,None] > 0
        
        J0ij = torch.special.bessel_j0(torch.einsum('B N M, L -> B N M L', rel_disij, self.k)) * (rel_disij[...,None] > 1e-2)

        J0ijk = torch.special.bessel_j0(torch.einsum('B N M O, L -> B N M O L', rel_disijk, self.k)) * (rel_disijk[...,None] > 1e-2)
        
        for layer in self.layers:
            ato_emb, ato_env = layer(ato_emb, ato_env, rel_disij, J0ij, J0ijk, padding, self.w)
        
        return self.lin_out(ato_emb)

    def gauss_legendre_ab(self, n, a, b):
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        
        x = torch.from_numpy(x_np).float()
        w = torch.from_numpy(w_np).float()
        
        t = 0.5 * (b - a) * x + 0.5 * (a + b)
        w = 0.5 * (b - a) * w
        
        return t, w