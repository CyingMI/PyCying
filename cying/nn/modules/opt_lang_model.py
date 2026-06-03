"""算子语言模型模块。

本模块定义了一个基于 `OptLangLayer` 的语言模型架构，使用嵌入层、
归一化、注意力残差来生成最终的词表概率输出。
"""

import torch
import torch.nn as nn
from .opt_lang_layer import OptLangLayer


class OptLangModel(nn.Module):
    """算子语言模型类。

    参数:
        vocab_size (int): 词表大小。
        d_model (int): 模型嵌入维度。
        num_heads (int): 多头注意力头数。
        hidden_width (int): 非线性层隐藏维度。
        num_layers (int): 堆叠层数。
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

        # 输入嵌入层，将 token id 映射到模型维度向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 逐层堆叠的 OptLangLayer
        self.layers = nn.ModuleList(
            [OptLangLayer(d_model, num_heads, hidden_width) for _ in range(num_layers)]
        )

        # 注意力残差连接的查询权重
        self.q_weights = nn.Parameter(
            torch.zeros((num_layers * 2, d_model))
        )
        self.norm = nn.RMSNorm(d_model, elementwise_affine=False)

        # 输出线性层，将模型维度映射回词表大小
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_seq,
        padding_mask=None,
        causal_mask=None,
    ):
        """模型前向计算。

        Args:
            token_seq (Tensor): 输入 token id 序列，形状为 (batch, seq_len)。
            padding_mask (Tensor, optional): 用于遮蔽填充位置的注意力掩码。
            causal_mask (Tensor, optional): 因果注意力掩码，用于自回归建模。

        Returns:
            Tensor: 输出 logits，形状为 (batch, seq_len, vocab_size)。
        """
        # 先进行词嵌入
        token_seq = [self.embedding(token_seq)]

        for i in range(self.num_layers):
            # 计算当前序列的注意力输入和输出
            att_input = self.norm(token_seq[-1])
            att_output = self.layers[i]._multi_head_attention(
                att_input, padding_mask, causal_mask
            )
            token_seq.append(att_output)

            # 用当前层的注意力输出与历史状态进行混合
            weights = []
            for j in range(i * 2 + 2):
                weights.append(self.norm(token_seq[j]) @ self.q_weights[i * 2])
            scores = torch.stack(weights, dim=-1).softmax(dim=-1)
            token_seq[-1] = sum(
                token_seq[j] * scores[..., j:j + 1] for j in range(i * 2 + 2)
            )

            # 经过非线性层变换
            nonlin_input = self.norm(token_seq[-1])
            nonlin_output = self.layers[i].nonlinear(nonlin_input)
            token_seq.append(nonlin_output)

            # 再次混合非线性输出与当前历史状态
            weights = []
            for j in range(i * 2 + 3):
                weights.append(self.norm(token_seq[j]) @ self.q_weights[i * 2 + 1])
            scores = torch.stack(weights, dim=-1).softmax(dim=-1)
            token_seq[-1] = sum(
                token_seq[j] * scores[..., j:j + 1] for j in range(i * 2 + 3)
            )

        # 输出最终 logits
        return self.linear_out(self.norm(token_seq[-1]))