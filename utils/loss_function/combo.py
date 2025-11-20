import torch
import torch.nn as nn
from utils.loss_function.dice import dice_loss

def combo_loss_for_micro(inpt: torch.Tensor, target: torch.Tensor):
    ce_loss = nn.CrossEntropyLoss()
    dice_loos = dice_loss

    loss = ce_loss(inpt, target)
    mask = torch.argmax(inpt, dim=1)
    loss += dice_loss(mask, target, multiclass=False)
    return loss
