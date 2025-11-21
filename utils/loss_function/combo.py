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


# 0.25 2.0
# .5 .5
def combo_tversky_loss(inpt: torch.Tensor, target: torch.Tensor):
    focal_loss = smp.losses.FocalLoss(mode="multiclass", alpha=0.25,gamma=2.0)
    tversky_loss = smp.losses.TverskyLoss(mode="multiclass", alpha=0.5, beta=0.5)

    loss = focal_loss(inpt, target)

    masks_pred = torch.softmax(inpt, dim=1)[:,1,:,:]
    loss += 1-tversky_loss.compute_score(masks_pred, target)
    return loss

def combo_focal_logdice(inpt: torch.Tensor, target: torch.Tensor):
    focal_loss = smp.losses.FocalLoss(mode="multiclass", alpha=0.25)
    dice_val = dice_loss(torch.softmax(inpt, dim=1)[:, 1, :, :], target, multiclass=False)
    loss =  torch.log(torch.cosh(dice_val))
    loss += focal_loss(inpt,target)
    return loss

def combo_loss_for_micro(inpt: torch.Tensor, target: torch.Tensor):
    return combo_tversky_loss(inpt=inpt, target=target)
