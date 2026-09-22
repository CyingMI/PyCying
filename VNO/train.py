import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nn.modules.vision_model import VisionModel
from cityscapes import CityscapesDataset
from loss import HungarianMatcher,SetCriterion

cityscapes_root = "/root/autodl-tmp/cityscapes"
split = "train"
target_size = (256, 512)
batch_size = 8
num_query = 291
num_classes = 19
backbone_stride = 16
d_model = 128
num_heads = 8
decoder_layers = 3
decoder_hidden_width = 256
mask_dim = d_model
feature_map_size = (target_size[0] // backbone_stride,
                        target_size[1] // backbone_stride)

num_epochs   = 50
lr           = 1e-4
weight_decay = 1e-4
grad_clip    = 0.1
num_workers  = 4
log_every    = 20

device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_targets(masks, labels, padding_mask, bboxes):
    return {
        "masks":        masks.float(),   # [B, Q, H, W]
        "labels":       labels,          # [B, Q]  无效 = -1
        "boxes":        bboxes,          # [B, Q, 4] xywh 归一化
        "padding_mask": padding_mask,    # [B, H, W] bool
    }


dataset = CityscapesDataset(
    root=cityscapes_root,
    split=split,
    target_size=target_size,
    num_query=num_query,
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)

backbone_params = [
    {
        'size':          target_size,
        'in_channels':   3,
        'out_channels':  d_model,
        'hidden_width':  decoder_hidden_width,
        'spe_opt_size':  feature_map_size,
        'spa_opt_size':  11,
    },
    {
        'size':          target_size,
        'in_channels':   d_model,
        'out_channels':  d_model,
        'hidden_width':  decoder_hidden_width,
        'spe_opt_size':  feature_map_size,
        'spa_opt_size':  11,
    },
]
encoder_params = [
    {
        'size':          feature_map_size,
        'in_channels':   d_model,
        'out_channels':  d_model,
        'hidden_width':  decoder_hidden_width,
        'spe_opt_size':  feature_map_size,
        'spa_opt_size':  11,
    } for _ in range(3)
]

model = VisionModel(
    num_classes=num_classes,
    d_model=d_model,
    num_heads=num_heads,
    decoder_hidden_width=decoder_hidden_width,
    decoder_layers=decoder_layers,
    num_query=num_query,
    mask_dim=mask_dim,
    token_grid_size=feature_map_size,
    backbone_stride=backbone_stride,
    mask_stride=4,
    backbone_params=backbone_params,
    encoder_params=encoder_params,
).to(device)

matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0,
    num_classes=num_classes,
)
weight_dict = {
    "loss_ce":    1.0,
    "loss_bbox":  5.0,
    "loss_giou":  2.0,
    "loss_dice":  5.0,
    "loss_bce":   2.0,
}
criterion = SetCriterion(
    num_classes=num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

print(f"[INFO] device = {device}")
print(f"[INFO] dataset = {len(dataset)}  steps/epoch = {len(dataloader)}")

model.train()

for epoch in range(num_epochs):
    for step, (imgs, masks, labels, padding_mask, bboxes) in enumerate(dataloader):
        imgs         = imgs.to(device, non_blocking=True)
        masks        = masks.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        bboxes       = bboxes.to(device, non_blocking=True)

        targets = collate_targets(masks, labels, padding_mask, bboxes)

        outputs = model(imgs, image_padding_mask=padding_mask)
        losses  = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()          
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if (step + 1) % log_every == 0:
            msg = "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"[E{epoch:3d}] [S{step+1:4d}/{len(dataloader)}] {msg}")