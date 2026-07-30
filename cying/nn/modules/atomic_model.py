import torch
import torch.nn as nn
from .atomic_layer import AtomicLayer

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class AtomicModel(nn.Module):
    def __init__(self, d_model, hid_width, num_layers, cutoff_r = 6, L=32):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.num_layers = num_layers
        self.cutoff_r = cutoff_r
        self.L = L

        k = torch.fft.fftfreq(L, 1/L)
        k1, k2, k3 = torch.meshgrid(k, k, k, indexing='ij')
        k = torch.stack([k1, k2, k3],dim=-1)
        k2_flat = (k**2).sum(dim=-1).flatten().int()
        k2_unique, key = torch.unique(
            k2_flat,
            return_inverse=True,
            return_counts=False
        )
        self.register_buffer('key', key)
        self.register_buffer('k', k * torch.pi / cutoff_r)
        self.opt_size = len(k2_unique)

        self.ato_emb = nn.Embedding(119, d_model, 0)

        self.fnn_cutoff = nn.Sequential(
            nn.Linear(1, hid_width),
            SIREN(),
            nn.Linear(hid_width, hid_width),
            SIREN(),
            nn.Linear(hid_width, d_model)
        )

        self.layers = nn.ModuleList([
            AtomicLayer(d_model, hid_width, self.opt_size, L) for _ in range(num_layers)
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
        rel_pos = neb_pos - ato_pos[...,None,:]
        rel_dis = rel_pos.norm(dim=-1, keepdim=True)

        srij = self.s5(rel_dis) * self.fnn_cutoff(rel_dis/self.cutoff_r) * (rel_dis > 1e-2) * (env_num[...,None] > 0)

        expijk_ = torch.cos(torch.einsum('B N M d, i j k d -> B N M i j k', rel_pos, self.k))
        B, N, M = expijk_.shape[:3]
        
        key = self.key[None,None,None,...]
        
        key = key.expand(B, N, M, -1)
        
        expijk = torch.zeros(B, N, M, self.opt_size, dtype=expijk_.dtype, device=expijk_.device)

        expijk.scatter_add_(dim=-1, index=key, src=expijk_.flatten(-3,-1))

        expijk_count = torch.zeros(self.opt_size, dtype=torch.float32, device=expijk.device)

        ones = torch.ones_like(self.key, dtype=torch.float32)
        
        expijk_count.scatter_add_(dim=-1, index=self.key, src=ones)
        
        for layer in self.layers:
            ato_emb, ato_env = layer(ato_emb, ato_env, expijk, expijk_count, srij)
        
        return self.lin_out(ato_emb)

    def s5(self, x):
        x = x / self.cutoff_r
        x = x * (x < 1)
        return (1 - x**5*(56 - 140*x + 120*x**2 - 35*x**3)) * (x < 1)