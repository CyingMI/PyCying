import torch
import torch.nn as nn


class OptLangLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        hid_width
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.hid_width = hid_width
        self.d_head = d_model // n_head

        self.h_linear = nn.Linear(d_model,d_model,bias=False)
        self.qk_weight = nn.Parameter(torch.stack([torch.ones(n_head,self.d_head//2+1),torch.zeros(n_head,self.d_head//2+1)],dim=-1))
        self.att_linear = nn.Linear(d_model,d_model,bias=False)

        self.nonlinear = nn.Sequential(
            nn.Linear(self.d_model,hid_width),
            nn.GELU(),
            nn.Linear(hid_width,self.d_model)
        )
        
        self.memory = None

    def forward(
        self,
        token_seq,
        padding_mask,
        causal_mask
    ):
        token_att = self._multi_head_attention(
            token_seq,
            padding_mask,
            causal_mask
        )

        return self.nonlinear(token_att)

    def _multi_head_attention(
        self,
        token_seq,
        padding_mask,
        causal_mask
    ):
        B,L,_= token_seq.shape
        
        s, v = self._get_score_value(token_seq)

        s = s.masked_fill(
            mask=padding_mask.view((1,B,1,L)),
            value=-float('inf')
        ) if padding_mask is not None else s

        s = s.masked_fill(
            mask=causal_mask,
            value=-float('inf')
        ) if causal_mask is not None else s

        out = torch.einsum('n b t s, b s n d -> b t n d', torch.softmax(s, dim=-1), v)

        return self.att_linear(out.flatten(-2,-1))

    def _get_score_value(
        self,
        token_seq
    ):
        qkv = self.h_linear(token_seq).view(*token_seq.shape[:2],self.n_head,self.d_head)
        
        qkv_fre = torch.fft.rfft(qkv) / self.d_head

        s = torch.einsum('b t n d, n d, b s n d -> n b t s', qkv_fre, torch.view_as_complex(self.qk_weight), qkv_fre.conj()).real

        return s, qkv