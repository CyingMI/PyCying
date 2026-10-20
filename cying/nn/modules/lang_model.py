"""Operator language model module.

This module defines a language model architecture based on
`OptLangLayer`. It uses an embedding layer, normalization and
attention residuals to produce final logits over the vocabulary.
"""
import torch
import torch.nn as nn
from .lang_layer import LangLayer


class LangModel(nn.Module):
    """Operator language model.

    Args:
        vocab_size (int): Vocabulary size.
        d_model (int): Model embedding dimension.
        num_heads (int): Number of attention heads.
        hidden_width (int): Hidden dimension for nonlinear layers.
        num_layers (int): Number of stacked layers.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        hidden_width,
        num_layers,
        padding_idx=0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_width = hidden_width
        self.num_layers = num_layers

        # Input embedding layer: map token ids to model-dimension vectors
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Stack of OptLangLayer instances
        self.layers = nn.ModuleList(
            [LangLayer(d_model, num_heads, hidden_width) for _ in range(num_layers)]
        )

        self.register_buffer(
            "theta",
            10000 ** (2 * torch.arange(0, d_model) / d_model),
        )

        # Query weights for attention residual connections
        self.q_weights = nn.Parameter(torch.zeros((num_layers * 2, d_model)))

        # Output layer
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_seq,
        padding_mask=None,
        causal_mask=None,
    ):
        """Forward function.

        Args:
            token_seq (Tensor): Input token id sequence of shape (batch, seq_len).
            padding_mask (Tensor, optional): Attention mask for padding positions.
            causal_mask (Tensor, optional): Causal mask for autoregressive modeling.

        Returns:
            Tensor: Output logits of shape (batch, seq_len, vocab_size).
        """

        # Embed tokens
        token_seq = [self.embedding(token_seq)]

        position_embeddings = torch.exp(
            1j * torch.arange(
                0,
                token_seq[0].size(1),
                device=token_seq[0].device
            )[..., None] / self.theta[None, :]
        )

        for i in range(self.num_layers):
            # Compute normalized attention input and attention output
            att_output = self.layers[i]._multi_head_attention(
                token_seq[-1], position_embeddings, padding_mask, causal_mask
            )
            token_seq.append(att_output)

            # Mix attention output with previous states
            weights = []
            for j in range(i * 2 + 2):
                weights.append(token_seq[j] @ self.q_weights[i * 2])
            scores = torch.stack(weights, dim=-1).softmax(dim=-1)
            token_seq[-1] = self.layers[i].att_norm(
                sum(
                    token_seq[j] * scores[..., j:j + 1] for j in range(i * 2 + 2)
                ) + token_seq[-1]
            )

            # Apply nonlinear transformation
            nonlin_output = self.layers[i].nonlinear(token_seq[-1])
            token_seq.append(nonlin_output)

            # Mix nonlinear output with historical states
            weights = []
            for j in range(i * 2 + 3):
                weights.append(token_seq[j] @ self.q_weights[i * 2 + 1])
            scores = torch.stack(weights, dim=-1).softmax(dim=-1)
            token_seq[-1] = self.layers[i].out_norm(
                sum(
                    token_seq[j] * scores[..., j:j + 1] for j in range(i * 2 + 3)
                ) + token_seq[-1]
            )

        return self.linear_out(token_seq[-1])