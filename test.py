import torch
import numpy as np
import pandas as pd 

from model import UNet
from data import TiffSegmentationDataset

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
torch.device(device)
print("Using device:", device)

# Load Data
DATA_DIR = "./data/"
dataset = TiffSegmentationDataset(data_dir=DATA_DIR)

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

batch_size = 4
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, drop_last=False)


# Test
BATCH_SIZE=4
MODEL_PATH="best_unet_model.pth"
model = UNet(in_channels=1, out_channels=1).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    exit()

model.eval()
# criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss() if model.out_channels > 1 else nn.BCEWithLogitsLoss()

total_test_loss = 0.0

with torch.no_grad(): 
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f" (Test Loss): {avg_test_loss:.4f}")

# --- 6. 可视化预测结果 ---
try:
    images, masks = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    # 将输 (logits) 转换为 0-1 之间的概率
    preds = torch.sigmoid(outputs)
    preds = (preds > 0.5).float() # 使用 0.5 作为阈值

    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    num_to_show = min(BATCH_SIZE, 4)
    fig, axes = plt.subplots(num_to_show, 3, figsize=(15, num_to_show * 5))
    
    if num_to_show == 1:
        axes = [axes]

    for i in range(num_to_show):
        axes[i, 0].imshow(np.squeeze(images_np[i]), cmap='gray')
        axes[i, 0].set_title("(Original Image)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.squeeze(masks_np[i]), cmap='gray')
        axes[i, 1].set_title("(True Mask)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(np.squeeze(preds_np[i]), cmap='gray')
        axes[i, 2].set_title("(Predicted Mask)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show() # 如果你想在脚本运行时立即看到图像，取消这行注释

except StopIteration:
    print(f"Error: StopIteration")

