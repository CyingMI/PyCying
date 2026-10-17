"""Operator language layer module.

This module defines a single optimized language layer that includes
multi-head frequency-domain attention and a nonlinear transformation.
"""
import torch
import torch.nn as nn

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class LangLayer(nn.Module):
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

        self.q_tensor = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )
        self.k_tensor = nn.Sequential(
            nn.Linear(d_model, d_model),
            SIREN()
        )

        self.v_linear = nn.Linear(d_model, d_model, bias=False)

        self.att_linear = nn.Linear(d_model, d_model, bias=False)

        self.s_weight = nn.Parameter(torch.randn(d_model, num_heads), requires_grad=True)

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
        position_embeddings,
        padding_mask,
        causal_mask
    ):
        """Compute multi-head attention result."""
        batch_size, seq_len, _ = token_seq.shape

        scores, values = self._get_score_value(token_seq, position_embeddings)

        if padding_mask is not None:
            scores = scores.masked_fill(
                mask=padding_mask.view((batch_size, 1, seq_len, 1)),
                value=-float("inf"),
            )
        
        if causal_mask is not None:
            scores = scores.masked_fill(
                mask=causal_mask[...,None],
                value=-float("inf"),
            )

        out = torch.einsum(
            "b t s n, b s d n -> b t d n",
            scores.softmax(dim=-2),
            values
        ).flatten(-2,-1)
        
        return self.att_linear(out)

    def _get_score_value(
        self,
        token_seq,
        position_embeddings
    ):
        """Generate attention scores and value features."""

        q_tensor = self.q_tensor(token_seq)
        k_tensor = self.k_tensor(token_seq)

        scores = torch.einsum(
            "b t d, b s d, t s d, d n -> b t s n",
            q_tensor,
            k_tensor,
            position_embeddings,
            self.s_weight
        )

        values = self.v_linear(token_seq).view(*token_seq.shape[:-1], self.d_head, self.num_heads)

        return scores, values