import torch
import torch.nn as nn

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class AtomicLayer(nn.Module):
    def __init__(self, d_model, hid_width, L, N):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.L = L
        self.N = N

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

        self.env_tensor_env = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )
        self.env_tensor_ato = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )
        self.env_cutoff = nn.Sequential(
            nn.Linear(1, d_model),
            SIREN()
        )
        self.env_tensor_linear = nn.Linear(d_model, d_model, bias=False)
        self.env_merge_norm = nn.RMSNorm(d_model)

        self.env_norm_in = nn.RMSNorm(d_model)
        self.env_norm_out = nn.RMSNorm(d_model)

        self.ato_tensor_env = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )
        self.ato_cutoff = nn.Sequential(
            nn.Linear(1, d_model),
            SIREN()
        )
        self.ato_tensor_linear = nn.Linear(d_model, d_model, bias=False)

        self.ato_norm_in = nn.RMSNorm(d_model)
        self.ato_norm_out = nn.RMSNorm(d_model)

        self.merge_norm = nn.RMSNorm(d_model)

    def forward(self, ato_emb, neb_matrix, rel_disij, J0ij, J0ijk, srij):        
        ato_env = torch.take_along_dim(
            ato_emb[:,None,:],
            neb_matrix[...,None],
            dim=2
        )

        ato_env = self.merge_norm(self.env_tensor_linear(self.env_tensor_ato(ato_emb)[...,None,:] * self.env_tensor_env(ato_env) * self.env_cutoff(rel_disij[...,None]) * srij) + ato_env)

        env_probe = torch.einsum(
            'B N M n, B N M O s, s n m -> B N O m',
            ato_env,
            J0ijk,
            self.env_weight
        )

        ato_env = self.env_norm_in(env_probe + ato_env)
        ato_env = self.env_norm_out(self.fnn_env(ato_env) + ato_env)

        ato_env = self.ato_tensor_linear(self.ato_tensor_env(ato_env) * self.ato_cutoff(rel_disij[...,None]) * srij)

        ato_probe = torch.einsum(
            'B N M n, B N M s, s n m -> B N m',
            ato_env,
            J0ij,
            self.ato_weight
        )

        ato_emb = self.ato_norm_in(ato_probe + ato_emb)
        ato_emb = self.ato_norm_out(self.fnn_ato(ato_emb) + ato_emb)

        return ato_emb