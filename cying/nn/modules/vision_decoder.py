import torch
import torch.nn as nn
from .vision_decoder_layer import VisionDecoderLayer

class VisionDecoder(nn.Module):
    theta : torch.Tensor
    def __init__(
            self,
            d_model,
            num_heads,
            hidden_width,
            num_layers,
            num_query = 100
        ):
        super().__init__()
        self.layers = nn.ModuleList([VisionDecoderLayer(d_model=d_model,num_heads=num_heads,hidden_width=hidden_width) 
                                     for _ in range(num_layers)]
                                     )

        self.query_seq = nn.Parameter(torch.randn(num_query, d_model))
        
        self.d_head = d_model // num_heads

        self.register_buffer(
                    "theta",
                    10000 ** (2 * torch.arange(0, self.d_head // 2 + 1) / self.d_head),
                )
        
    def q_position_embedings(self):

        query_len = self.query_seq.size(0)
        
        position = torch.exp(
                    1j * torch.arange(
                        0,
                        query_len,
                        device=self.query_seq.device
                    )[..., None, None] / self.theta[None,None,:]
                )
        
        return position


    def forward(self,target_seq,padding_mask):

        q_position_embedings = self.q_position_embedings()

        query_seq = self.query_seq.unsqueeze(0)

        for layer in self.layers:

            query_seq = layer(query_seq,target_seq,q_position_embedings,padding_mask)
            
        return query_seq


def build_VisonDecoder(target_seq,
                       padding_mask,
                       d_model = 512,
                       num_heads = 8,
                       hidden_width = 2048,
                       num_layers = 6,
                       ):

    
    visiondecoder = VisionDecoder(d_model,num_heads,hidden_width,num_layers)

    out = visiondecoder(target_seq,padding_mask)

    return out


##test
target_seq = torch.randn(2,32*32,512)

print(target_seq.size(1))

out = build_VisonDecoder(target_seq = target_seq,padding_mask=None)

print(out.shape)

