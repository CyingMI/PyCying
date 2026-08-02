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
            num_query = 100
        ):
        super().__init__()
        self.layers = nn.ModuleList([VisionDecoderLayer(d_model,
                                                        num_heads,
                                                        hidden_width
                                                        ) 
                                     for _ in range(num_layers)]
                                     )

        self.query_seq = nn.Parameter(torch.randn(num_query, d_model))
        
        self.d_head = d_model // num_heads

        self.q_position_embeddings = torch.exp(1j * nn.Parameter(
            torch.randn(num_query, self.d_head // 2 + 1) * 0.02
        ).unsqueeze(1)
        )

    def t_position_embedings(self, target_seq):

        target_len = target_seq.size(1)
        
        position = torch.exp(
                    1j * torch.arange(
                        0,
                        target_len,
                        device=target_seq.device
                    )[..., None, None] / (10000 ** (2 * torch.arange(0, self.d_head // 2 + 1) / self.d_head)[None,None,:])
                )
        
        return position


    def forward(self, target_seq, padding_mask):

        query_seq = self.query_seq.unsqueeze(0)

        t_position_embedings = self.t_position_embedings(target_seq)

        for layer in self.layers:

            query_seq = layer(query_seq,
                              target_seq,
                              self.q_position_embeddings,
                              t_position_embedings,
                              padding_mask
                              )
            
        return query_seq
