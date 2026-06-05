"""Operator language model module.

This module defines a language model architecture based on
`OptLangLayer`. It uses an embedding layer, normalization and
attention residuals to produce final logits over the vocabulary.
"""
import torch
import torch.nn as nn
from .opt_lang_layer import OptLangLayer


class OptLangModel(nn.Module):
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_width = hidden_width
        self.num_layers = num_layers

        # Input embedding layer: map token ids to model-dimension vectors
        self.embedding = nn.Embedding(
            vocab_size, d_model, max_norm=d_model ** 0.5
        )

        # Stack of OptLangLayer instances
        self.layers = nn.ModuleList(
            [OptLangLayer(d_model, num_heads, hidden_width) for _ in range(num_layers)]
        )

        # Query weights for attention residual connections
        self.q_weights = nn.Parameter(torch.zeros((num_layers * 2, d_model)))
        self.norm = nn.RMSNorm(d_model, elementwise_affine=False)

        # 输出线性层，将模型维度映射回词表大小
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

        for i in range(self.num_layers):
            # Compute normalized attention input and attention output
            att_input = self.norm(token_seq[-1])
            att_output = self.layers[i]._multi_head_attention(
                att_input, padding_mask, causal_mask
            )
            token_seq.append(att_output)

            # Mix attention output with previous states
            weights = []
            for j in range(i * 2 + 2):
                weights.append(self.norm(token_seq[j]) @ self.q_weights[i * 2])
            scores = torch.stack(weights, dim=-1).softmax(dim=-1)
            token_seq[-1] = sum(
                token_seq[j] * scores[..., j:j + 1] for j in range(i * 2 + 2)
            )

            # Apply nonlinear transformation
            nonlin_input = self.norm(token_seq[-1])
            nonlin_output = self.layers[i].nonlinear(nonlin_input)
            token_seq.append(nonlin_output)

            # Mix nonlinear output with historical states
            weights = []
            for j in range(i * 2 + 3):
                weights.append(self.norm(token_seq[j]) @ self.q_weights[i * 2 + 1])
            scores = torch.stack(weights, dim=-1).softmax(dim=-1)
            token_seq[-1] = sum(
                token_seq[j] * scores[..., j:j + 1] for j in range(i * 2 + 3)
            )

        return self.linear_out(self.norm(token_seq[-1]))