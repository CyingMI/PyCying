import torch
import torch.nn as nn

class AtomicLayer(nn.Module):
    def __init__(self, d_model, hid_width):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        # self.q_linear = nn.Linear(d_model, d_model, bias=False)
        # self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.opt_weight = nn.Parameter(torch.zeros(d_model // 2 + 1, 2), requires_grad=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid_width),
            nn.GELU(),
            nn.Linear(hid_width, d_model)
        )

        self.ato_norm = nn.RMSNorm(d_model)

        self.out_norm = nn.RMSNorm(d_model)

    def forward(self, ato_emb, mul_pos):
        # ato_emb: B*N*d, mul_pos: B*N*M*O
        ato_mul = torch.einsum('B N d, B M d -> B N M d', ato_emb, ato_emb)
        opt_weight = torch.view_as_complex(self.opt_weight)
        ato_mul_fre = torch.fft.rfft(ato_mul)
        v = self.v_linear(ato_emb)

        s = torch.einsum(
            'B N M d, B N O d, d, B N M O -> B N O', 
            ato_mul_fre,
            ato_mul_fre.conj(),
            opt_weight, 
            mul_pos + 1j*0
        ).real / self.d_model

        env_emb = torch.einsum('B N M, B M d -> B N d', s, v)

        return self.out_norm(self.ffn(self.ato_norm(ato_emb + env_emb)) + ato_emb)

# class AtomicLayer(nn.Module):
#     def __init__(self, C, L, opt_len, key):
#         super().__init__()
#         self.C = C
#         self.L = L
#         self.opt_len = opt_len
#         self.key = key
#         self.ene_weight = nn.Parameter(torch.zeros(opt_len, C), requires_grad=True)
#         self.opt_weight = nn.Parameter(torch.zeros(opt_len, C, C), requires_grad=True)

#     def forward(self, energy, ato_emb, expijk, radial_mask, global_mask):
#         weight = self.opt_weight[self.key].permute(3, 4, 0, 1, 2)
#         env_emb = torch.einsum(
#             'B N M i j k, B N C, B M C, C O i j k, B N M -> B N O i j k',
#             expijk,
#             ato_emb,
#             ato_emb,
#             weight,
#             radial_mask
#         ).mean(dim=[-3,-2,-1])
#         ato_emb = ato_emb + env_emb

#         weight = self.ene_weight[self.key].permute(3, 0, 1, 2)
#         energy = torch.einsum(
#             'B N M i j k, B N C, B M C, C i j k, B N M -> B N i j k',
#             expijk,
#             ato_emb,
#             ato_emb,
#             weight,
#             global_mask
#         ).mean(dim=[-3,-2,-1]) + energy
#         return energy, ato_emb