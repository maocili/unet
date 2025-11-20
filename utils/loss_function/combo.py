import torch
import torch.nn as nn
from utils.loss_function.dice import dice_loss
import segmentation_models_pytorch as smp
from torchvision.ops import sigmoid_focal_loss


def combo_ce_dice(inpt: torch.Tensor, target: torch.Tensor):
    ce_loss = nn.CrossEntropyLoss()
    dice_loos = dice_loss

    loss = ce_loss(inpt, target)
    mask = torch.softmax(inpt, dim=1)[:, 1, :, :]
    loss += dice_loss(mask, target, multiclass=False)
    return loss


def combo_focal_dice(inpt: torch.Tensor, target: torch.Tensor):
    focal_loss = smp.losses.FocalLoss(mode="multiclass", alpha=0.25)
    dice_loos = dice_loss

    loss = focal_loss(inpt, target)
    mask = torch.softmax(inpt, dim=1)[:, 1, :, :]
    loss += dice_loss(mask, target, multiclass=False)
    return loss


def combo_tversky_loss(inpt: torch.Tensor, target: torch.Tensor):
    focal_loss = smp.losses.FocalLoss(mode="multiclass", alpha=0.25)
    tversky_loss = smp.losses.TverskyLoss(mode="multiclass", alpha=0.6, beta=0.4)

    loss = focal_loss(inpt, target)
    loss += tversky_loss(inpt, target)
    return loss


def combo_loss_for_micro(inpt: torch.Tensor, target: torch.Tensor):
    return combo_focal_dice(inpt=inpt, target=target)
