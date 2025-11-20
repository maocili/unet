import torch
from torch import Tensor

def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # Intersection: TP
    inter = (input * target).sum(dim=sum_dim)

    # Union: (Input + Target) - Intersection
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    union = sets_sum - inter

    # IoU calculation
    iou = (inter + epsilon) / (union + epsilon)
    return iou.mean()

def iou_loss(input: Tensor, target: Tensor):
    # IoU loss (objective to minimize)
    return 1 - iou_coeff(input, target, reduce_batch_first=True)