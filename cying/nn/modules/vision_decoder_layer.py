import torch
import torch.nn as nn

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class TensorAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_basis = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )
        self.t_basis = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )
        self.merge_weight = nn.Parameter(torch.randn(d_model, num_heads) / self.d_model**0.5, requires_grad=True)

        self.v_linear = nn.Linear(d_model, d_model, bias=False)

        self.att_linear = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.RMSNorm(d_model)

    def forward(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        t_position_embeddings,
        padding_mask=None
    ):
        scores, values = self._get_score_value(quary_seq, target_seq, q_position_embeddings,t_position_embeddings)

        scores = scores.masked_fill(
            mask=padding_mask[:,None,:,None],
            value=-float("inf"),
        ) if padding_mask is not None else scores
        out = torch.einsum(
            "b t s n, b s n d -> b t n d",
            scores.softmax(dim=-2),
            values
        ).flatten(-2, -1)

        return self.norm(self.att_linear(out) + out)

    def _get_score_value(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        t_position_embeddings
    ):
        q_basis = self.q_basis(quary_seq) * q_position_embeddings
        t_basis = self.t_basis(target_seq) * t_position_embeddings.conj()
        scores = torch.einsum(
            'b t d, b s d, d h -> b t s h',
            q_basis,
            t_basis,
            self.merge_weight + 1j*0
        ).real

        values = self.v_linear(target_seq).view(
            *target_seq.shape[:2],
            self.num_heads,
            self.d_head
        )
        return scores, values

class VisionDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_width
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_width = hidden_width
        self.d_head = d_model // num_heads

        self.self_att = TensorAttention(d_model, num_heads)
        self.cross_att = TensorAttention(d_model, num_heads)

        self.nonlinear = nn.Sequential(
            nn.Linear(d_model, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, d_model),
        )

        self.out_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        t_position_embeddings,
        padding_mask
    ):
        query_seq = self.self_att(
            quary_seq,
            quary_seq,
            q_position_embeddings,
            q_position_embeddings
        )
    
        quary_att = self.cross_att(
            query_seq,
            target_seq,
            q_position_embeddings,
            t_position_embeddings,
            padding_mask
        )

        return self.out_norm(self.nonlinear(quary_att) + quary_att)