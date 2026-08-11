import torch
import torch.nn as nn


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

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.t_linear = nn.Linear(d_model, d_model, bias=False)

        self.qk_weight = nn.Parameter(torch.zeros(num_heads, self.d_head // 2 + 1, 2), requires_grad=True)

        self.att_linear = nn.Linear(d_model, d_model, bias=False)

        self.nonlinear = nn.Sequential(
            nn.Linear(d_model, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, d_model),
        )

        self.self_att_norm = nn.RMSNorm(d_model)
        self.cross_att_norm = nn.RMSNorm(d_model)
        self.out_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        t_position_embeddings,
        padding_mask
    ):

        query_seq = self.self_att_norm(self._multi_head_attention(
            quary_seq,
            quary_seq,
            q_position_embeddings,
            q_position_embeddings
        ) + quary_seq)
    
        quary_att = self.cross_att_norm(self._multi_head_attention(
            query_seq,
            target_seq,
            q_position_embeddings,
            t_position_embeddings,
            padding_mask
        ) + query_seq)

        return self.out_norm(self.nonlinear(quary_att) + quary_att)

    def _multi_head_attention(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        t_position_embeddings,
        padding_mask=None
    ):
        scores, values = self._get_score_value(quary_seq, target_seq, q_position_embeddings,t_position_embeddings)

        scores = scores.masked_fill(
            mask=padding_mask,
            value=-float("inf"),
        ) if padding_mask is not None else scores

        out = torch.einsum(
            "n b t s, b s n d -> b t n d",
            scores.softmax(dim=-1),
            values
        )

        return self.att_linear(out.flatten(-2, -1))

    def _get_score_value(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        t_position_embeddings
    ):
        quary_seq = self.q_linear(quary_seq).view(
            *quary_seq.shape[:2],
            self.num_heads,
            self.d_head
        )

        target_seq = self.t_linear(target_seq).view(
            *target_seq.shape[:2],
            self.num_heads,
            self.d_head
        )

        q_freq = torch.fft.rfft(quary_seq) * q_position_embeddings / self.d_head

        k_freq = torch.fft.rfft(target_seq) * t_position_embeddings / self.d_head

        scores = torch.einsum(
            "b t n d, n d, b s n d -> n b t s",
            q_freq,
            torch.view_as_complex(self.qk_weight),
            k_freq.conj()
        ).real

        return scores, target_seq