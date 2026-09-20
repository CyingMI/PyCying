import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from .hungarian_gpu_batch import hungarian_gpu

def HungarianMatching(input, target):
    lambda_cls = 1.0
    lambda_l1 = 5.0
    lambda_giou = 2.0

    B, M, _ = input["pred_logits"].shape
    N = target["labels"].shape[1]

    # classification cost
    valid_mask = target["valid_mask"].bool()
    valid_labels = target["labels"].clone()
    valid_labels = valid_labels.masked_fill(~valid_mask, 0)
    target_labels = valid_labels.unsqueeze(1).expand(-1, M, -1)
    class_probs = torch.softmax(input["pred_logits"], -1)
    class_cost = -torch.gather(
        class_probs,
        dim=2,
        index=target_labels.long()
    )

    # L1 cost
    pred_boxes = input["pred_boxes"][...,None,:]
    target_boxes = target["boxes"][:,None,...]
    l1_cost = (pred_boxes - target_boxes).abs().sum(-1)

    # GIoU cost
    eps = 1e-6
    
    p0, p1, p2, p3 = torch.unbind(pred_boxes, dim=-1)
    t0, t1, t2, t3 = torch.unbind(target_boxes, dim=-1)
    inter_x1 = torch.maximum(p0, t0)
    inter_y1 = torch.maximum(p1, t1)
    inter_x2 = torch.minimum(p2, t2)
    inter_y2 = torch.minimum(p3, t3)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    I = inter_w * inter_h

    A = (p2 - p0) * (p3 - p1)
    B_area = (t2 - t0) * (t3 - t1)
    U = A + B_area - I + eps

    enc_x1 = torch.minimum(p0, t0)
    enc_y1 = torch.minimum(p1, t1)
    enc_x2 = torch.maximum(p2, t2)
    enc_y2 = torch.maximum(p3, t3)
    C = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + eps

    IoU = I / U
    giou = IoU - (C - U) / C
    giou_cost = 1 - giou

    cost = lambda_cls * class_cost \
        + lambda_l1 * l1_cost \
        + lambda_giou * giou_cost

    Ns = target["valid_mask"].sum(1).to(torch.int32)
    # hungarian matching
    
    cost_gpu = torch.full(
        (B, 300, 300),
        1e10,
        device=cost.device
    )
    cost_gpu[:, :M, :N] = cost.float()

    assignment = hungarian_gpu(
        cost_gpu.contiguous(),
        Ns
    )

    assignment = assignment[:, :M, :]
 
    return assignment

def Loss(input, match, target):
    lambda_cls = 1.0
    lambda_l1 = 5.0
    lambda_giou = 2.0
    
    B, M, C = input["pred_logits"].shape
    no_object_idx = C - 1
    pred_target = torch.full(
        (B, M),
        no_object_idx,
        dtype = torch.long,
        device=input["pred_logits"].device
    )

    query_idx = match[:, :, 0]
    target_idx = match[:, :, 1]

    matched_masked = target_idx != -1

    target_idx = target_idx.masked_fill(
        target_idx == -1,
        0
    )

    matched_labels = torch.gather(
            target["labels"],
            dim = 1,
            index = target_idx
    )
    matched_labels = torch.where(
        matched_masked,
        matched_labels,
        no_object_idx
    )

    pred_target.scatter_(
        1,
        query_idx,
        matched_labels
    )

    matched_pred_boxes = torch.gather(
        input["pred_boxes"],
        dim = 1,
        index=query_idx.unsqueeze(-1).expand(-1, -1, 4)
    )
    matched_target_boxes = torch.gather(
        target["boxes"],
        dim = 1,
        index = target_idx.unsqueeze(-1).expand(-1, -1, 4)
    )
    matched_pred_boxes = matched_pred_boxes[matched_masked]
    matched_target_boxes = matched_target_boxes[matched_masked]

    logits = input["pred_logits"].flatten(0,1)
    pred_target = pred_target.flatten(0,1)
    class_weight = torch.ones(C, device=input["pred_logits"].device)
    class_weight[-1] = 0.1
    classes_cost = F.cross_entropy(
        logits, 
        pred_target, 
        weight=class_weight
    )

    num_boxes = matched_pred_boxes.shape[0]
    l1_cost = (matched_pred_boxes - matched_target_boxes).abs().sum(-1)
    l1_cost = l1_cost.sum() / num_boxes

    eps = 1e-6
        
    p0, p1, p2, p3 = torch.unbind(matched_pred_boxes, dim=-1)
    t0, t1, t2, t3 = torch.unbind(matched_target_boxes, dim=-1)
    inter_x1 = torch.maximum(p0, t0)
    inter_y1 = torch.maximum(p1, t1)
    inter_x2 = torch.minimum(p2, t2)
    inter_y2 = torch.minimum(p3, t3)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    I = inter_w * inter_h

    A = (p2 - p0) * (p3 - p1)
    B_area = (t2 - t0) * (t3 - t1)
    U = A + B_area - I + eps
    
    enc_x1 = torch.minimum(p0, t0)
    enc_y1 = torch.minimum(p1, t1)
    enc_x2 = torch.maximum(p2, t2)
    enc_y2 = torch.maximum(p3, t3)
    C = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + eps

    IoU = I / U
    giou = IoU - (C - U) / C
    giou_cost = (1 - giou).sum() / num_boxes

    loss = lambda_cls * classes_cost \
        + lambda_l1 * l1_cost \
        + lambda_giou * giou_cost

    return loss