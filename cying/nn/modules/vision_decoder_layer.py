import torch
import torch.nn as nn


class VisionDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_width,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_width = hidden_width
        self.d_head = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.t_linear = nn.Linear(d_model, d_model, bias=False)

        self.qk_weight = nn.Parameter(torch.stack((torch.zeros(num_heads, self.d_head // 2 + 1),torch.zeros(num_heads, self.d_head // 2 +1)), dim=-1))

        self.att_linear = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer(
            "theta",
            10000 ** (2 * torch.arange(0, self.d_head // 2 + 1) / self.d_head),
        )

        self.nonlinear = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.d_model),
        )

        self.norm = nn.RMSNorm(d_model, elementwise_affine=False)

    def forward(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        padding_mask
    ):

        query_seq2 = self.norm(self._multi_head_attention(
            quary_seq,
            quary_seq,
            q_position_embeddings
        ) + quary_seq)
        
        quary_att = self.norm(self._multi_head_attention(
            query_seq2,
            target_seq,
            q_position_embeddings,
            padding_mask
        ) + query_seq2)

        return self.norm(self.nonlinear(quary_att) + quary_att)

    def _multi_head_attention(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings,
        padding_mask=None
    ):
        scores, values = self._get_score_value(quary_seq, target_seq, q_position_embeddings)

        if padding_mask is not None:
            scores = scores.masked_fill(
                mask=padding_mask,
                value=-float("inf"),
            )

        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum(
            "n b t s, b s n d -> b t n d",
            attention,
            values
        )

        return self.att_linear(out.flatten(-2, -1))

    def _get_score_value(
        self,
        quary_seq,
        target_seq,
        q_position_embeddings
    ):
        quary_len = quary_seq.size(1)
        target_len = target_seq.size(1)

        position_embeddings = torch.exp(
            1j * torch.arange(
                0,
                target_len,
                device=quary_seq.device
            )[..., None, None] / self.theta[None,None,:]
        )

        quary_seq = self.q_linear(quary_seq).view(
            -1,
            quary_len,
            self.num_heads,
            self.d_head
        )


        target_seq = self.t_linear(target_seq).view(
            -1,
            target_len,
            self.num_heads,
            self.d_head
        )

        q_freq = torch.fft.rfft(quary_seq) * q_position_embeddings / self.d_head

        k_freq = torch.fft.rfft(target_seq) * position_embeddings / self.d_head

        scores = torch.einsum(
            "b t n d, n d, b s n d -> n b t s",
            q_freq,
            torch.view_as_complex(self.qk_weight),
            k_freq.conj()
        ).real

        return scores, target_seq