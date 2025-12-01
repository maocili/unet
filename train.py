import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd 
from datetime import datetime
import argparse 

from models.unet import UNet
from utils.dataset import TiffDataset
from utils.weights import kaiming_init_weights
from utils.loss_function.combo import combo_loss_for_micro
from utils.loss_function.iou import iou_coeff
from utils.transformers import MicroTransformers
from utils.plt import save_loss_data

def parse_args():
    parser = argparse.ArgumentParser(description="U-Net Training Script")
    
    parser.add_argument("--name", type=str, default="U-Net", help="Experiment name (default: U-Net)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu). Default is auto-detect.")
    
    parser.add_argument("--epochs", type=int, default=20, help="Number of total epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    
    # Resume 
    parser.add_argument("--resume", action="store_true", help="Flag to resume training from a checkpoint")
    parser.add_argument("--resume-path", type=str, default="last_u-net_model.pth", help="Path to the checkpoint file to resume from")

    return parser.parse_args()

def train(loader, model, optimizer, criterion, epoch, device, num_epochs):
    model.train()
    total_loss = 0.0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).long()

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

def validate(loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    iou = 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()

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
    args = parse_args()
    
    if args.device:
        device = args.device
    else:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv_path = f'training_{args.name}_log_{timestamp}.csv'
    print(f"Log file will be saved to: {log_csv_path}")

    train_dataset = TiffDataset(
        "data/png/train/image/", 
        "data/png/train/label",
        transforms=MicroTransformers(geo_augment=True)
    )
    test_dataset = TiffDataset(
        "data/png/test/image/", 
        "data/png/test/label",
        transforms=MicroTransformers(geo_augment=False)
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    model = UNet(in_channels=1, out_channels=2).to(device)
    model.apply(kaiming_init_weights)

    # Resume Logic
    if args.resume:
        if os.path.isfile(args.resume_path):
            print(f"Loading checkpoint from: {args.resume_path}")
            try:
                model.load_state_dict(torch.load(args.resume_path, map_location=device, weights_only=True))
            except TypeError:
                model.load_state_dict(torch.load(args.resume_path, map_location=device))
            print("Checkpoint loaded successfully.")
        else:
            print(f"Warning: Checkpoint file not found at {args.resume_path}. Training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = combo_loss_for_micro

    best_iou = 0.0

    for epoch in range(args.epochs):
        # Train
        train_loss = train(train_loader, model, optimizer, criterion, epoch, device, args.epochs)
        
        # Validate
        val_loss, val_iou = validate(test_loader, model, criterion, device)

        print(f"Epoch [{epoch + 1}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        log_data = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Val_IoU': val_iou
        }
        save_loss_data(pd.DataFrame([log_data]), log_csv_path)
        
        if val_iou > best_iou:
            best_iou = val_iou
            save_name = f'best_iou_{args.name.lower()}_model.pth'
            torch.save(model.state_dict(), save_name)
            print(f"Saved best model ({save_name}) with IoU: {best_iou:.4f}")

    last_name = f'last_{args.name.lower()}_model.pth'
    torch.save(model.state_dict(), last_name)
    print(f"Saved last model to {last_name}")

if __name__ == "__main__":
    main()