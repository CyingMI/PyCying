import torch
import torch.nn as nn
from .vision_decoder_layer import VisionDecoderLayer

class VisionDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_width,
        num_layers,
        target_len = 64,
        query_len = 100
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_width = hidden_width
        self.num_layers = num_layers
        self.target_len = target_len
        self.query_len = query_len
        self.d_head = d_model // num_heads

        self.layers = nn.ModuleList([
            VisionDecoderLayer(d_model,
                num_heads,
                hidden_width
            ) for _ in range(num_layers)
        ])

        self.query_seq = nn.Parameter(torch.randn(query_len, d_model), requires_grad=True)

        self.q_position_embeddings = nn.Parameter(torch.randn(query_len, 1, self.d_head // 2 + 1), requires_grad=True)

        self.register_buffer(
            't_position_embeddings',
            torch.arange(0, target_len)[:,None,None] / (10000 ** (2 * torch.arange(0, self.d_head // 2 + 1) / self.d_head)[None,None,:])
        )

    def forward(self, target_seq, padding_mask):
        q_position_embeddings = torch.exp(1j * self.q_position_embeddings)
        t_position_embeddings = torch.exp(1j * self.t_position_embeddings)

        query_seq = self.query_seq.expand(*target_seq.shape)

        for layer in self.layers:
            query_seq = layer(
                query_seq,
                target_seq,
                q_position_embeddings,
                t_position_embeddings,
                padding_mask
            )
        return query_seq
