import sys
import os
from model import UNet
from utils.dataset import TiffDataset
from utils.transformers import MicroImageTransformers, MicroLabelTransformers
from utils.loss_function import dice_loss

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt


if sys.platform.startswith('win'):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
torch.device(device)
print("Using device:", device)


# Load Data
dataset = TiffDataset(
    'data/image', 'data/label', MicroImageTransformers, MicroLabelTransformers)
indices = len(dataset)
train_size = len(dataset) - int(0.2*len(dataset))
test_size = len(dataset) - train_size


# Create random splits for train and test sets
train_set, test_set = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)

print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")


BATCH_SIZE = 4
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=False)


# Test
MODEL_PATH = "best_unet_model.pth"
model = UNet(in_channels=1, out_channels=3).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
except FileNotFoundError:
    exit()

model.eval()

ce_weight = torch.Tensor([0.5, 1, 0.5])
ce_criterion = nn.CrossEntropyLoss(weight=ce_weight).to(device=device)
dice_criterion = dice_loss

total_test_loss = 0.0
image_list = []
masks_list = []
preds_list = []
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device).long()
        masks_pred = model(images)

        batch_loss = ce_criterion(masks_pred, masks)
        masks_pred = torch.argmax(masks_pred, dim=1)
        batch_loss += dice_criterion(masks_pred, masks, multiclass=False)
        total_test_loss += batch_loss

        image_list.append(images.cpu().numpy())
        masks_list.append(masks.cpu().numpy())
        preds_list.append(masks_pred.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
print(f" (Test Loss): {avg_test_loss:.4f}")

page = 4
for i in range(0, page):
    images_np = image_list[i]
    masks_np = masks_list[i]
    preds_np = preds_list[i]

    fig, axes = plt.subplots(BATCH_SIZE, 3, figsize=(15, BATCH_SIZE * 5))
    if BATCH_SIZE == 1:
        axes = [axes]

    for i in range(BATCH_SIZE):
        axes[i, 0].imshow(np.squeeze(images_np[i]), cmap='gray')
        axes[i, 0].set_title("(Original Image)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.squeeze(masks_np[i]), cmap='gray')
        axes[i, 1].set_title("(True Mask)")
        axes[i, 1].axis('off')

        # final_prediction = np.argmax(preds_np[i], axis=0)
        print(np.unique(preds_np[i]),preds_np[i].shape)

        axes[i, 2].imshow(preds_np[i], cmap='gray')
        axes[i, 2].set_title("(Predicted Mask)")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()
