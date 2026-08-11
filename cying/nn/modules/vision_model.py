import torch
import torch.nn as nn
import torch.nn.functional as F
from .operator_model import OperatorModel2d
from .vision_decoder import VisionDecoder


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

        self.class_head = nn.Linear(d_model, num_classes +1)
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
        return {
            "pred_logits": self.class_head(queries),
            "pred_boxes": self.box_head(queries),
            "pred_masks": torch.sigmoid(
                torch.einsum(
                    "b n d, b d h w -> b n h w", 
                    self.mask_head(queries), 
                    pixel_features,
                )
            )
        }

class VisionModel(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model,
        num_heads,
        decoder_hidden_width,
        decoder_layers,
        num_query,
        mask_dim,
        token_grid_size,
        backbone_stride,
        mask_stride,
        backbone_params,
        encoder_params
    ):
        super().__init__()

        self.d_model = d_model
        self.num_query = num_query
        self.token_grid_size = tuple(token_grid_size)
        self.backbone_stride = backbone_stride
        self.mask_stride = mask_stride
        self.config = {
            "num_classes": num_classes,
            "d_model": d_model,
            "num_heads": num_heads,
            "decoder_hidden_width": decoder_hidden_width,
            "decoder_layers": decoder_layers,
            "num_query": num_query,
            "mask_dim": mask_dim,
            "token_grid_size": tuple(token_grid_size),
            "backbone_stride": backbone_stride,
            "mask_stride": mask_stride,
            "backbone_params": backbone_params,
            "encoder_params": encoder_params,
        }
        self.backbone = OperatorModel2d(backbone_params)
        self.backbone_norm = nn.GroupNorm(1, d_model)

        self.encoder = OperatorModel2d(encoder_params)
        self.encoder_norm = nn.GroupNorm(1, d_model)

        self.decoder = VisionDecoder(
            d_model=d_model,
            num_heads=num_heads,
            hid_width=decoder_hidden_width,
            num_layers=decoder_layers,
            num_query=num_query,
        )

        self.prediction_heads = VisionPredictionHeads(
            d_model=d_model,
            d_mask=mask_dim,
            num_classes=num_classes,
            hid_width=decoder_hidden_width
        )

    def forward(
        self,
        images,
        image_padding_mask = None,
    ):
        pixel_features = self.backbone(images)

        target_seq = self.encoder_norm(self.encoder(pixel_features)).flatten(-2,-1).permute(0,2,1)

        token_padding_mask = F.adaptive_max_pool2d(
            image_padding_mask.float()[:,None,...],
            self.token_grid_size,
        )[:,0].bool().flatten(-2,-1)

        queries = self.decoder(target_seq, token_padding_mask)

        return self.prediction_heads(queries, self.backbone_norm(pixel_features))

