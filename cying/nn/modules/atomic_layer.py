import torch
import torch.nn as nn

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class AtomicLayer(nn.Module):
    def __init__(self, d_model, hid_width, opt_size, L=32):
        super().__init__()
        self.d_model = d_model
        self.hid_width = hid_width
        self.opt_size = opt_size
        self.L = L

        self.ato_weight = nn.Parameter(torch.zeros(opt_size, d_model, d_model), requires_grad=True)

        self.env_weight = nn.Parameter(torch.zeros(opt_size, d_model, d_model), requires_grad=True)

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


    def forward(self, ato_emb, ato_env, expijk, expijk_count, srij):
        ato_env = self.env_linear(torch.cat([ato_env,ato_emb[...,None,:].expand(*ato_env.shape)],dim=-1))

        env_field = torch.einsum(
            'B N M d, B N M s -> B N d s',
            ato_env * srij,
            expijk
        )
        
        ato_emb = self.atom_probe(ato_emb, env_field)

        env_field = torch.einsum(
            'B N d s, B N M s -> B N M d s',
            env_field / expijk_count,
            expijk
        )

        ato_env = self.env_probe(ato_env, env_field)

        return ato_emb, ato_env

    def atom_probe(self, ato_emb, env_field):
        atom_probe = torch.einsum(
            'B N n s, B N n, s n m -> B N m',
            env_field,
            ato_emb,
            self.ato_weight
        ) / self.L**3
        
        ato_emb = self.ato_norm_in(atom_probe + ato_emb)
        
        ato_emb = self.ato_norm_out(self.fnn_ato(ato_emb) + ato_emb)

        return ato_emb

    def env_probe(self, ato_env, env_field):
        env_probe = torch.einsum(
            'B N M n s, B N M n, s n m -> B N M m',
            env_field,
            ato_env,
            self.env_weight
        ) / self.L**3
        
        ato_env = self.env_norm_in(env_probe + ato_env)

        ato_env = self.env_norm_out(self.fnn_env(ato_env) + ato_env)

        return ato_env