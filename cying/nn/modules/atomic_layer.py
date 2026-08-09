import torch
import torch.nn as nn

class AtomicLayer(nn.Module):
    def __init__(self, d_model, hid_width, N, cutoff_r=6):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.N = N
        self.cutoff_r = cutoff_r

        self.ato_weight = nn.Parameter(torch.zeros(N, d_model, d_model), requires_grad=True)

        self.env_weight = nn.Parameter(torch.zeros(N, d_model, d_model), requires_grad=True)

        self.fnn_ato = nn.Sequential(
            nn.Linear(d_model, hid_width),
            nn.SiLU(),
            nn.Linear(hid_width, d_model)
        )

        self.fnn_env = nn.Sequential(
            nn.Linear(d_model, hid_width),
            nn.SiLU(),
            nn.Linear(hid_width, d_model)
        )

        self.fnn_cutoff = nn.Sequential(
            nn.Linear(1, hid_width),
            nn.SiLU(),
            nn.Linear(hid_width, hid_width),
            nn.SiLU(),
            nn.Linear(hid_width, 1)
        )

        self.ato_norm_in = nn.RMSNorm(d_model)
        self.ato_norm_out = nn.RMSNorm(d_model)
        self.env_norm_in = nn.RMSNorm(d_model)
        self.env_norm_out = nn.RMSNorm(d_model)

    def forward(self, ato_emb, ato_env, rel_disij, J0ij, J0ijk, padding):
        srij = self.s5(rel_disij) * (rel_disij > 1e-2) * padding

        env_field = torch.einsum(
            'B N M d, B N M s -> B N d s',
            ato_env * srij,
            J0ij
        ) + ato_emb[...,None]

        env_probe = torch.einsum(
            'B N M d, B N M O s -> B N O d s',
            ato_env * srij,
            J0ijk
        ) + torch.einsum(
            'B N d, B N M s -> B N M d s',
            ato_emb,
            J0ij
        )

        env_probe = torch.einsum(
            'B N M n s, B N M n, s n m -> B N M m',
            env_probe,
            ato_env,
            self.env_weight
        ) / self.N

        ato_env = self.env_norm_in(env_probe + ato_env)
        ato_env = self.env_norm_out(self.fnn_env(ato_env) + ato_env)

        ato_probe = torch.einsum(
            'B N n s, B N n, s n m -> B N m',
            env_field,
            ato_emb,
            self.ato_weight
        ) / self.N

        ato_emb = self.ato_norm_in(ato_probe + ato_emb)
        ato_emb = self.ato_norm_out(self.fnn_ato(ato_emb) + ato_emb)

        return ato_emb, ato_env

    def s5(self, x):
        x = x / self.cutoff_r
        x = x * (x < 1)
        return (1 - x**5*(56 - 140*x + 120*x**2 - 35*x**3)) * self.fnn_cutoff(x) * (x < 1)