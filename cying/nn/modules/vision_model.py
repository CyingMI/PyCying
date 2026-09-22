import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operator_model import OperatorModel2d
from .vision_decoder import VisionDecoder

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class VisionPredictionHeads(nn.Module):
    def __init__(
        self,
        d_model,
        d_mask,
        num_classes,
        hid_width
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mask = d_mask
        self.num_classes = num_classes
        self.hid_width = hid_width

        self.class_head = nn.Linear(d_model, num_classes+1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, hid_width),
            nn.SiLU(),
            nn.Linear(hid_width, 4),
            nn.Sigmoid()
        )
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, hid_width),
            nn.SiLU(),
            nn.Linear(hid_width, d_mask)
        )

    def forward(
        self,
        queries,
        pixel_features,
    ):
        box = self.box_head(queries)
        box = torch.cat([box[...,:2],box[...,2:] * (1 - box[...,:2])], dim=-1)
        mask = torch.sigmoid(
            torch.einsum(
                "b n d, b d h w -> b n h w",
                self.mask_head(queries),
                pixel_features
            )
        )
        return self.class_head(queries), box, mask

class VisionModel(nn.Module):
    def __init__(
        self,
        backbone_params,
        encoder_params,
        num_classes,
        num_heads,
        decoder_hidden_width,
        decoder_layers,
        num_query
    ):
        super().__init__()
        self.backbone_params = backbone_params
        self.encoder_params = encoder_params
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.decoder_hidden_width = decoder_hidden_width
        self.decoder_layers = decoder_layers
        self.num_query = num_query

        self.d_model = encoder_params[-1]['out_channels']
        self.d_mask = backbone_params[-1]['out_channels']
        self.target_len = math.prod(encoder_params[-1]['size'])
        self.token_grid_size = encoder_params[-1]['size']

        self.backbone = OperatorModel2d(backbone_params)

        self.encoder = OperatorModel2d(encoder_params)

        self.decoder = VisionDecoder(
            d_model=self.d_model,
            num_heads=num_heads,
            hid_width=decoder_hidden_width,
            num_layers=decoder_layers,
            target_len=self.target_len,
            query_len=num_query
        )

        self.prediction_heads = VisionPredictionHeads(
            d_model=self.d_model,
            d_mask=self.d_mask,
            num_classes=num_classes,
            hid_width=decoder_hidden_width
        )

    def forward(
        self,
        images,
        image_padding_mask = None,
        mode = 'train'
    ):
        pixel_features = self.backbone(images, mode)

        target_seq = self.encoder(pixel_features, mode).flatten(-2,-1).permute(0,2,1)

        token_padding_mask = F.adaptive_max_pool2d(
            image_padding_mask.float()[:,None,...],
            self.token_grid_size,
        )[:,0].bool().flatten(-2,-1)

        queries = self.decoder(target_seq, token_padding_mask)

        return self.prediction_heads(queries, pixel_features)

    def get_mask_weight(self):
        return self.backbone.get_mask_weight() + self.encoder.get_mask_weight()