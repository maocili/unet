import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from model import UNet
from utils.dataset import TiffDataset
from utils.weights import kaiming_init_weights
from utils.loss_function.combo import combo_loss_for_micro
from utils.loss_function.iou import iou_coeff
from utils.transformers import MicroTransformers



device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
torch.device(device)
print("Using device:", device)

# Load Data
tif_train_data = TiffDataset("data/tif/train/image/","data/tif/train/label/", transforms=MicroTransformers(geo_augment=True))
png_train_data = TiffDataset("data/png/train/image/","data/png/train/label", transforms=MicroTransformers(geo_augment=True))
tif_test_data = TiffDataset("data/tif/test/image/","data/tif/test/label/", transforms=MicroTransformers(geo_augment=False))
png_test_data = TiffDataset("data/png/test/image/","data/png/test/label", transforms=MicroTransformers(geo_augment=False))

train_set = png_train_data
test_set = png_test_data

print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")

batch_size = 1
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, drop_last=False)


num_epochs = 20
LEARNING_RATE = 1e-4

model = UNet(in_channels=1, out_channels=2).to(device=device)
model.apply(kaiming_init_weights)

criterion = combo_loss_for_micro

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

best_val_loss = float('inf')
best_iou = 0.0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for images, masks in loop:
        images = images.to(device=device)
        masks = masks.to(device=device).long()

        masks_pred = model(images)

        batch_loss = criterion(masks_pred, masks)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_train_loss += batch_loss.item()

    avg_train_loss = total_train_loss/float(len(train_set))

    model.eval()
    val_loss = 0.0
    iou = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).long()

            masks_pred = model(images)

            loss = criterion(masks_pred, masks)

            masks_pred = torch.softmax(masks_pred, dim=1)[:, 1, :, :]
            iou += iou_coeff(masks_pred, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / float(len(test_set))
    avg_iou = iou / float(len(test_set))

    if epoch % (num_epochs/num_epochs) == 0:
        print(
            f"Epoch [{epoch + 1}] | training loss : {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f} | IoU={avg_iou :.4f} ")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_unet_model.pth')
        print("Saved best_unet_model")

    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), 'best_iou_unet_model.pth')
        print("Saved best_IoU_unet_model")

    if epoch+1 == num_epochs:
        torch.save(model.state_dict(), 'least_unet_model.pth')
        print("Saved least_unet_model.pth")
