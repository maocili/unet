import torch
import numpy as np
import pandas as pd
from model import UNet
from data import TiffSegmentationDataset

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
torch.device(device)
print("Using device:", device)

# Load Data
dataset = TiffSegmentationDataset(
    'data_isbi/train/images', 'data_isbi/train/labels')

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

batch_size = 1
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, drop_last=False)


num_epochs = 10
LEARNING_RATE = 1e-3

model = UNet(in_channels=1, out_channels=2).to(device=device)
criterion = nn.CrossEntropyLoss() if model.out_channels > 1 else nn.BCEWithLogitsLoss()
# criterion = model.loss_fn
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=0.99)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device=device)
        masks = masks.to(device=device)

        masks_pred = model(images)

        batch_loss = criterion(masks_pred, masks)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_train_loss += batch_loss.item()

    avg_train_loss = total_train_loss/float(len(train_set))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(masks, outputs)

            val_loss += loss.item()

    avg_val_loss = val_loss / float(len(test_set))

    if epoch % (num_epochs/10) == 0:
        print(
            f"Epoch [{epoch}] | trainning loss : {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_unet_model.pth')

# torch.save(model.state_dict(),"unet_model.pth")
