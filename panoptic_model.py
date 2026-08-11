
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operator_model import OperatorModel2d
from .vision_decoder import VisionDecder, VisionDecoder

class MLP(nn.Module):
    def __init__(
            self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()

        dimensions = (
            [input_dim]
            + [hidden_dim] * (num_layers - 1)
            + [output_dim]
        )
        layers = []
        for index in range(num_layers):
            layers.append(
                nn.Linear(dimensions[index], dimensions[index+1])
            )
            if index < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class PanopticPredictionHeads(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        mask_dim: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.mask_dim = mask_dim

        self.query_norm = nn.RMSNorm(d_model)
        self.class_head = nn.Linear(d_model, num_classes +1)
        self.box_head = MLP(d_model, d_model, 4, num_layers=3)
        self.mask_embed_head = MLP(d_model, d_model, mask_dim, num_layers=3)

    def forward(
        self,
        queries: torch.Tensor,
        pixel_features: torch.Tensor,

    ):
        queries = self.query_norm(queries)
        mask_embeddings = self.mask_embed_head(queries)

        return {
            "pred_logits": self.class_head(queries),
            "pred_boxes": torch.sigmoid(self.box_head(queries)),
            "pred_masks": torch.sigmoid(torch.einsum(
                "bnd,bdhw->bnhw", 
                mask_embeddings, 
                pixel_features,
            )),
        }

class OperatorPanopticModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int ,
        num_heads: int ,
        decoder_hidden_width: int ,
        decoder_layers: int ,
        num_query: int ,
        mask_dim: int ,
        token_grid_size: Tuple[int, int] ,
        backbone_stride: int ,
        mask_stride: int ,
        backbone_params: Optional[Sequence[Dict]] ,
        encoder_params: Optional[Sequence[Dict]] ,
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
            hidden_width=decoder_hidden_width,
            num_layers=decoder_layers,
            num_query=num_query,
        )

        self.prediction_heads = PanopticPredictionHeads(
            d_model=d_model,
            num_classes=num_classes,
            mask_dim=mask_dim,
        )


    def forward(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
    ):
        pixel_features = self.backbone(images)

        target_seq = self.encoder_norm(self.encoder(pixel_features)).flatten(-2,-1).permute(0,2,1)

        token_padding_mask = F.adaptive_max_pool2d(
            image_padding_mask.float()[:,None,...],
            self.token_grid_size,
        )[:,0].bool().flatten(-2,-1)

        queries = self.decoder(target_seq, token_padding_mask)

        return self.prediction_heads(queries, self.backbone_norm(pixel_features))

