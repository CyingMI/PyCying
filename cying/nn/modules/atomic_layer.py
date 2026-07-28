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

        self.ato_weight = nn.Parameter(torch.zeros(length, d_model, d_model), requires_grad=True)

        self.env_weight = nn.Parameter(torch.zeros(length, d_model, d_model), requires_grad=True)

        self.env_linear = nn.Linear(2*d_model, d_model, bias=False)

        self.fnn_ato = nn.Sequential(
            nn.Linear(d_model, hid_width),
            SIREN(),
            nn.Linear(hid_width, d_model)
        )

        self.fnn_env = nn.Sequential(
            nn.Linear(d_model, hid_width),
            SIREN(),
            nn.Linear(hid_width, d_model)
        )

        self.env_norm_in = nn.RMSNorm(d_model)
        self.env_norm_out = nn.RMSNorm(d_model)
        
        self.ato_norm_in = nn.RMSNorm(d_model)
        self.ato_norm_out = nn.RMSNorm(d_model)

    def forward(self, ato_emb, ato_env, env_field, expijk):
        ato_env = self.env_linear(torch.cat([ato_env,ato_emb[...,None,:].expand(*ato_env.shape)],dim=-1))

        env_induc = torch.einsum('B N n s, B N M n, s n m, B N M s -> B N M m', env_field, ato_env + 1j*0, self.env_weight + 1j*0, expijk.conj()).real / self.L**3

        ato_env = self.env_norm_in(env_induc + ato_env)

        ato_env = self.env_norm_out(self.fnn_env(ato_env) + ato_env)

        env_field = torch.einsum('B N M d, B N M s -> B N d s', ato_env + 1j*0, expijk)

        ato_induc = torch.einsum('B N n s, B N n, s n m -> B N m', env_field.real, ato_emb, self.ato_weight) / self.L**3

        ato_emb = self.ato_norm_in(ato_induc + ato_emb)

        ato_emb = self.ato_norm_out(self.fnn_ato(ato_emb) + ato_emb)

        return ato_emb, ato_env, env_field