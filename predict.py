import sys
import os
from model import UNet
from utils.dataset import TiffDataset
from utils.transformers import MicroImageTransformers, MicroLabelTransformers
from utils.loss_function.combo import combo_loss_for_micro
from utils.loss_function.iou import iou_coeff

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
    'data/test/image', 'data/test/label', MicroImageTransformers, MicroLabelTransformers)
indices = len(dataset)

print(f"Test set size: {len(dataset)}")


BATCH_SIZE = 4
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=False, drop_last=False)


# Test
MODEL_PATH = "best_unet_model.pth"
model = UNet(in_channels=1, out_channels=2).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
except FileNotFoundError:
    exit()

model.eval()

criterion = combo_loss_for_micro

total_test_loss = 0.0
image_list = []
masks_list = []
preds_list = []
with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()
        masks_pred = model(images)

        batch_loss = criterion(masks_pred, masks)
        total_test_loss += batch_loss

        masks_pred = torch.argmax(masks_pred, dim=1)
        image_list.append(images.cpu())
        masks_list.append(masks.cpu())
        preds_list.append(masks_pred.cpu())

avg_test_loss = total_test_loss / len(loader)
print(f" (Test Loss): {avg_test_loss:.4f}")

page = 1
for i in range(0, page):
    images_np = image_list[i]
    masks_np = masks_list[i]
    preds_np = preds_list[i]

    size = len(images_np)

    fig, axes = plt.subplots(size, 3, figsize=(15, size * 5))
    if size == 1:
        axes = [axes]

    for i in range(size):
        axes[i, 0].imshow(np.squeeze(images_np[i]), cmap='gray')
        axes[i, 0].set_title("(Original Image)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.squeeze(masks_np[i]), cmap='gray')
        axes[i, 1].set_title("(True Mask)")
        axes[i, 1].axis('off')

        iou = iou_coeff(masks_np[i], preds_np[i])
        axes[i, 2].imshow(preds_np[i], cmap='gray')
        axes[i, 2].set_title(f"(Predicted Mask) iou = {iou :.4f}")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()
