import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment
#bbox输入同一为xyxy

def generalized_box_iou(boxes1, boxes2):
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * \
            (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * \
            (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)

    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, :, None] + area2[:, None, :] - inter

    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb_c = torch.max(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class, cost_bbox, cost_giou, num_classes):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]  # [B,N,C+1]
        pred_boxes = outputs["pred_boxes"]    # [B,N,4] xyxy
        tgt_labels = targets["labels"]        # [B,M]，-1 为 padding
        tgt_boxes = targets["boxes"]          # [B,M,4] xyxy

        B, N = pred_logits.shape[:2]
        valid_tgt = tgt_labels >= 0           # [B,M]
        pred_prob = pred_logits.softmax(-1)   # [B,N,C+1]

        cost_class = -pred_prob.gather(
            2, tgt_labels.clamp(min=0).unsqueeze(1).expand(-1, N, -1)
        )                                     # [B,N,M]
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)              # [B,N,M]
        cost_giou = -generalized_box_iou(pred_boxes, tgt_boxes)          # [B,N,M]

        C = (self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou)    # [B,N,M]

        C = C.masked_fill(~valid_tgt.unsqueeze(1), 1e6)

        assignment = batch_linear_assignment(C)  # [B,N]，target 索引或 -1     事实上因为我预处理已经实例数填充为N，所以必定时索引没有-1（即对应无）
        assignment = assignment.masked_fill(
            ~valid_tgt.gather(1, assignment.clamp(min=0)), -1              #填充无效实例为-1
        )

        return assignment

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        tgt_idx = self.matcher(outputs, targets)          # [B,N] 
        valid = tgt_idx >= 0                              # [B,N]

        pred_logits = outputs["pred_logits"]              # [B,N,C+1]
        pred_boxes  = outputs["pred_boxes"]               # [B,N,4] xyxy
        pred_masks  = outputs["pred_masks"]               # [B,N,h,w] 

        B, N = pred_logits.shape[:2]
        C = self.num_classes

        tgt_labels_pad = targets["labels"]                # [B,M]
        tgt_boxes_pad  = targets["boxes"]                 # [B,M,4] xyxy
        tgt_masks_pad  = targets["masks"].float()         # [B,M,H_t,W_t]
        padding_mask   = targets["padding_mask"]          # [B,H_t,W_t] True=padding
        H_t, W_t = tgt_masks_pad.shape[-2:]

        safe_idx = tgt_idx.clamp(min=0)
        matched_labels = tgt_labels_pad.gather(1, safe_idx)                     # [B,N]
        matched_boxes  = tgt_boxes_pad.gather(
            1, safe_idx[..., None].expand(-1, -1, 4))                           # [B,N,4]
        matched_masks  = tgt_masks_pad.gather(
            1, safe_idx[:, :, None, None].expand(-1, -1, H_t, W_t))             # [B,N,H_t,W_t]

        target_classes = matched_labels.masked_fill(~valid, C)                  # [B,N]
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),                                        # [B,C+1,N]
            target_classes,
            weight=self.empty_weight,
        )

        num_matched = valid.sum().clamp(min=1.0)

        l1_per = F.l1_loss(pred_boxes, matched_boxes, reduction="none").sum(-1)  # [B,N]
        giou = generalized_box_iou(pred_boxes, matched_boxes).diagonal(dim1=-2, dim2=-1)
        loss_bbox = (l1_per * valid).sum() / num_matched
        loss_giou = ((1.0 - giou) * valid).sum() / num_matched

        valid_pixel = (~padding_mask).unsqueeze(1)                              # [B,1,H_t,W_t] 布尔
        pred_m = pred_masks * valid_pixel                                       # [B,N,H_t,W_t]
        tgt_m  = matched_masks * valid_pixel                                    # [B,N,H_t,W_t]

        inter = (pred_m * tgt_m).sum((-1, -2))                                  # [B,N]
        union = pred_m.sum((-1, -2)) + tgt_m.sum((-1, -2))
        dice  = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        loss_dice = (dice * valid).sum() / num_matched

        n_pix = valid_pixel.sum((-1, -2)).clamp(min=1.0)                        # [B]
        bce = F.binary_cross_entropy(
            pred_m.clamp(1e-6, 1.0 - 1e-6), tgt_m, reduction="none"
        ) * valid_pixel                                                          # [B,N,H_t,W_t]
        loss_bce = (bce.sum((-1, -2)) / n_pix[:, None] * valid).sum() / num_matched

        loss_dict = {
            "loss_ce":   loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_dice": loss_dice,
            "loss_bce":  loss_bce,
        }
        loss_dict["total_loss"] = sum(
            self.weight_dict.get(k, 1.0) * v for k, v in loss_dict.items()
        )
        return loss_dict