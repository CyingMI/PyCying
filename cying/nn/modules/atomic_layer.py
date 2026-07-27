import torch
import torch.nn as nn

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class AtomicLayer(nn.Module):
    def __init__(self, d_model, hid_width, L=32):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.L = L

        k = torch.fft.fftfreq(L, 1/L)
        k1, k2, k3 = torch.meshgrid(k, k, k, indexing='ij')
        k = torch.stack([k1, k2, k3],dim=-1)
        k_square = (k**2).sum(dim=-1).int()
        k_square_unique, key = torch.unique(
            k_square, 
            return_inverse=True,
            return_counts=False
        )
        length = k_square_unique.size(0)

        self.register_buffer('key', key)

        self.opt_weight = nn.Parameter(torch.zeros(length, d_model), requires_grad=True)

        # self.opt_linear = nn.Linear(d_model, d_model, bias=False)

        self.fnn = nn.Sequential(
            nn.Linear(d_model, hid_width),
            SIREN(),
            nn.Linear(hid_width, d_model)
        )

        self.norm_in = nn.RMSNorm(d_model)
        self.norm_out = nn.RMSNorm(d_model)

    def forward(self, ato_emb, srij, expijk, neb_matrix):
        env_emb = torch.take_along_dim(
            ato_emb[:,None,:],
            neb_matrix[...,None],
            dim=2
        )

        w = self.opt_weight[self.key]

        fij = torch.einsum('i j k n, B N M i j k -> B N M n', w, expijk) / self.L**3

        env = torch.einsum('B N M n, B N M n, B N n, B N M -> B N n', fij, env_emb, ato_emb, srij)

        ato_emb = self.norm_in(env + ato_emb)

        return self.norm_out(self.fnn(ato_emb) + ato_emb)

# class SIREN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.sin(x)

# class AtomicLayer(nn.Module):
#     def __init__(self, d_model, hid_width):
#         super().__init__()
#         self.d_model = d_model
#         self.hid_width = hid_width

#         self.opt_weight = nn.Parameter(torch.zeros(d_model // 2 + 1, 2), requires_grad=True)
#         self.qk_linear = nn.Linear(2*d_model, d_model, bias=False)
#         self.v_linear = nn.Linear(d_model, d_model, bias=False)

#         self.ene_fnn = nn.Sequential(
#             nn.Linear(d_model, hid_width),
#             nn.GELU(),
#             nn.Linear(hid_width, d_model)
#         )

#         self.env_fnn = nn.Sequential(
#             nn.Linear(d_model, hid_width),
#             nn.GELU(),
#             nn.Linear(hid_width, d_model)
#         )

#         self.gr = nn.Sequential(
#             nn.Linear(1, 32),
#             SIREN(),
#             nn.Linear(32, 32),
#             SIREN(),
#             nn.Linear(32, 32),
#             SIREN(),
#             nn.Linear(32, 1)
#         )

#         self.env_norm = nn.RMSNorm(d_model)

#         self.envout_norm = nn.RMSNorm(d_model)

#         self.ene_norm = nn.RMSNorm(d_model)

#         self.eneout_norm = nn.RMSNorm(d_model)

#     def forward(self, ene_emb, env_emb, rel_pos, rel_dis, srij):
#         rel_pos = rel_pos * self.gr(rel_dis[...,None])
#         mul_pos = torch.einsum('B N M d, B N O d -> B N M O', rel_pos, rel_pos)

#         qk = self.qk_linear(torch.cat([ene_emb[...,None,:].expand(*env_emb.shape),env_emb], dim=-1))
#         opt_weight = torch.view_as_complex(self.opt_weight)
#         env_fre = torch.fft.rfft(qk)
#         v = self.v_linear(env_emb)

#         s = torch.einsum(
#             'B N M d, B N O d, d -> B N M O',
#             env_fre,
#             env_fre.conj(),
#             opt_weight
#         ).real / self.d_model * mul_pos

#         env_emb = self.env_norm(torch.einsum('B N M O, B N O d, B N O -> B N M d', s, v, srij) + env_emb)

#         env_emb = self.envout_norm(self.env_fnn(env_emb) + env_emb)

#         ene_emb = self.ene_norm(torch.einsum('B N M d, B N M -> B N d', env_emb, srij) + ene_emb)

#         ene_emb = self.eneout_norm(self.ene_fnn(ene_emb) + ene_emb)

#         return ene_emb, env_emb