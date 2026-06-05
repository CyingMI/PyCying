"""Operator language layer module.

This module defines a single optimized language layer that includes
multi-head frequency-domain attention and a nonlinear transformation.
"""
import torch
import torch.nn as nn


class OptLangLayer(nn.Module):
    """Operator language layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        hidden_width (int): Hidden dimension for nonlinear layer.
    """

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

        self.h_linear = nn.Linear(d_model, d_model, bias=False)
        qk_weight = torch.stack(
            [
                torch.ones(num_heads, self.d_head // 2 + 1),
                torch.zeros(num_heads, self.d_head // 2 + 1),
            ],
            dim=-1,
        )
        self.qk_weight = nn.Parameter(qk_weight)

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
        token_seq,
        padding_mask,
        causal_mask
    ):
        """Forward function.

        Args:
            token_seq (Tensor): Input features with shape (B, L, D).
            padding_mask (Tensor): Padding mask.
            causal_mask (Tensor): Causal mask.

        Returns:
            Tensor: Output features with shape (B, L, D).
        """
        token_att = self.norm(self._multi_head_attention(
            token_seq,
            padding_mask,
            causal_mask,
        ) + token_seq)

        token_out = self.norm(self.nonlinear(token_att) + token_att)
        
        return token_out

    def _multi_head_attention(
        self,
        token_seq,
        padding_mask,
        causal_mask
    ):
        """Compute multi-head attention result."""
        batch_size, seq_len, _ = token_seq.shape

        scores, values = self._get_score_value(token_seq)

        if padding_mask is not None:
            scores = scores.masked_fill(
                mask=padding_mask.view((1, batch_size, 1, seq_len)),
                value=-float("inf"),
            )

        if causal_mask is not None:
            scores = scores.masked_fill(
                mask=causal_mask,
                value=-float("inf"),
            )

        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum(
            "n b t s, b s n d -> b t n d",
            attention,
            values,
        )

        return self.att_linear(out.flatten(-2, -1))

    def _get_score_value(
        self,
        token_seq
    ):
        """Generate attention scores and value features."""
        batch_size, seq_len, _ = token_seq.shape

        position_embeddings = torch.exp(
            1j * torch.arange(
                0,
                seq_len,
                device=token_seq.device
            )[..., None, None] / self.theta[None, None, :]
        )

        token_seq = self.h_linear(token_seq).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_head
        )

        qk_freq = torch.fft.rfft(token_seq) * position_embeddings / self.d_head

        scores = torch.einsum(
            "b t n d, n d, b s n d -> n b t s",
            qk_freq,
            torch.view_as_complex(self.qk_weight),
            qk_freq.conj()
        ).real

        return scores, token_seq