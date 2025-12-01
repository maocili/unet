import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd 
from datetime import datetime

from models.unet import UNet
from utils.dataset import TiffDataset
from utils.weights import kaiming_init_weights
from utils.loss_function.combo import combo_loss_for_micro
from utils.loss_function.iou import iou_coeff
from utils.transformers import MicroTransformers
from utils.plt import save_loss_data

# --- Configuration ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
NAME = "U-Net"
log_csv_path = f'training_{NAME}_log_{timestamp}.csv'

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 2 

RESUME = False 
RESUME_CHECKPOINT_PATH = ""

def train(loader, model, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss

def validate(loader, model, criterion):
    model.eval()
    val_loss = 0.0
    iou = 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            # Calculate IoU
            # assuming outputs shape [B, C, H, W] and class 1 is the target
            preds = torch.softmax(outputs, dim=1)[:, 1, :, :]
            iou += iou_coeff(preds, masks)
            val_loss += loss.item()

    avg_loss = val_loss / len(loader)
    avg_iou = iou / len(loader)
    return avg_loss, avg_iou

def main():
    print(f"Using device: {DEVICE}")

    # --- Load Data ---
    train_dataset = TiffDataset(
        "data/tif/train/image/", 
        "data/tif/train/label",
        transforms=MicroTransformers(geo_augment=True)
    )
    test_dataset = TiffDataset(
        "data/png/test/image/", 
        "data/png/test/label",
        transforms=MicroTransformers(geo_augment=False)
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # --- Model Initialization ---
    model = UNet(in_channels=1, out_channels=2).to(DEVICE)
    
    # 默认使用 Kaiming 初始化
    model.apply(kaiming_init_weights)

    # --- Resume Logic (新增逻辑) ---
    if RESUME:
        if os.path.isfile(RESUME_CHECKPOINT_PATH):
            print(f"Loading checkpoint from: {RESUME_CHECKPOINT_PATH}")
            try:
                model.load_state_dict(torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
            except TypeError:
                model.load_state_dict(torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE))
            print("Checkpoint loaded successfully.")
        else:
            print(f"Warning: Checkpoint file not found at {RESUME_CHECKPOINT_PATH}. Training from scratch.")

    # --- Optimizer & Criterion ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = combo_loss_for_micro

    best_iou = 0.0

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train(train_loader, model, optimizer, criterion, epoch)
        
        # Validate
        val_loss, val_iou = validate(test_loader, model, criterion)

        # Print metrics
        print(f"Epoch [{epoch + 1}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Log Data
        log_data = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Val_IoU': val_iou
        }
        save_loss_data(pd.DataFrame([log_data]),log_csv_path)
        
        # Save Best Model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f'best_iou_{NAME.lower()}_model.pth')
            print(f"Saved best model with IoU: {best_iou:.4f}")

    # Save Last Model
    torch.save(model.state_dict(), f'last_{NAME.lower()}_model.pth')
    print("Saved last model.")

if __name__ == "__main__":
    main()