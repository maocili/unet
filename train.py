import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import argparse

# Models & Utils
from models.unet import UNet
from utils.dataset import TiffDataset
from utils.weights import kaiming_init_weights
from utils.loss_function.combo import combo_loss_for_micro
from utils.loss_function.iou import iou_coeff
from utils.transformers import MicroTransformers
from utils.plt import save_loss_data

# Mean Teacher dependencies
from mean_teacher import ramps, losses


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net Training Script (Standard or Mean Teacher)")

    # Basic Config
    parser.add_argument("--name", type=str, default="U-Net", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, mps, cpu)")

    # Paths
    parser.add_argument("--train-img", type=str, default="data/mix/train/image/", help="Path to training images")
    parser.add_argument("--train-lbl", type=str, default="data/mix/train/label", help="Path to training labels")
    parser.add_argument("--test-img", type=str, default="data/mix/test/image/", help="Path to test images")
    parser.add_argument("--test-lbl", type=str, default="data/mix/test/label", help="Path to test labels")

    # Data Augmentation
    parser.add_argument("--disable-denoise", action="store_true", help="Disable data preprocess to denoise")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-path", type=str, default="last_u-net_model.pth", help="Checkpoint path")

    # --- Mean Teacher Arguments ---
    parser.add_argument("--mean-teacher", action="store_true", help="Enable Mean Teacher training")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--consistency", type=float, default=30.0, help="Consistency weight")
    parser.add_argument("--consistency-rampup", type=float, default=15.0, help="Consistency rampup length")
    parser.add_argument("--noise", action="store_true", default=True, help="Add noise to inputs (default: True)")

    return parser.parse_args()

# Mean Teacher Helper Functions


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch, weight, rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(epoch, rampup)


def train_standard(loader, model, optimizer, criterion, epoch, device, num_epochs):
    """Standard Supervised Training"""
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Std]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader), 0.0, 0.0  # Return 0 for consistency/class parts


def train_mean_teacher(loader, model, ema_model, optimizer, criterion, epoch, device, num_epochs, args, global_step):
    """Mean Teacher Training"""
    model.train()
    ema_model.train()

    total_loss = 0.0
    total_class_loss = 0.0
    total_cons_loss = 0.0

    enable_unlabeled = epoch >= args.consistency_rampup

    consistency_weight = get_current_consistency_weight(epoch, args.consistency, args.consistency_rampup)


    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [MT]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).long()

        is_unlabeled = torch.max(masks) == torch.min(masks)
        print(is_unlabeled)

        # Skip unlabel data before rampup
        if not enable_unlabeled and is_unlabeled:
            continue

        model_input, ema_input = images, images

        # Noise Injection
        if args.noise:
            noise_shape = images.shape
            model_noise = torch.randn(noise_shape).to(device) * 0.1
            ema_model_noise = torch.randn(noise_shape).to(device) * 0.1
            model_input = images + model_noise
            ema_input = images + ema_model_noise

        # Forward Passes
        model_output = model(model_input)
        with torch.no_grad():
            ema_model_output = ema_model(ema_input)

        # Losses
        class_loss = 0.0
        if not is_unlabeled:
            class_loss = criterion(model_output, masks)
        consistency_loss = losses.softmax_mse_loss(model_output, ema_model_output)
       
        loss = class_loss + consistency_weight * consistency_loss

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # Logging
        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_cons_loss += consistency_loss.item()

        loop.set_postfix(
            loss=loss.item(),
            cls=class_loss.item(),
            cons=consistency_loss.item(),
            w=consistency_weight
        )

    avg_total = total_loss / len(loader)
    avg_class = total_class_loss / len(loader)
    avg_cons = total_cons_loss / len(loader)

    return avg_total, avg_class, avg_cons, global_step


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

            # Calculate IoU (class 1)
            preds = torch.softmax(outputs, dim=1)[:, 1, :, :]
            iou += iou_coeff(preds, masks).item()
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

    # Logging Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv_path = f'training_{args.name}_log_{timestamp}.csv'
    print(f"Log file: {log_csv_path}")

    denoise = (args.disable_denoise == False)
    train_dataset = TiffDataset(
        args.train_img, args.train_lbl,
        transforms=MicroTransformers(geo_augment=True, denoise=denoise)
    )
    test_dataset = TiffDataset(
        args.test_img, args.test_lbl,
        transforms=MicroTransformers(geo_augment=False)
    )

    print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # Model Setup
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.apply(kaiming_init_weights)

    ema_model = None
    if args.mean_teacher:
        print("--- Mean Teacher Mode Enabled ---")
        ema_model = UNet(in_channels=1, out_channels=2).to(device)
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.detach_()  # Teacher does not update via backprop
    else:
        print("--- Standard Training Mode ---")

    # Resume Logic
    if args.resume and os.path.isfile(args.resume_path):
        print(f"Loading checkpoint from: {args.resume_path}")
        try:
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint)
            if args.mean_teacher and ema_model:
                # If resuming MT, ideally we load EMA state.
                # If not available, re-cloning model state is a fallback.
                print("Warning: Re-initializing EMA model from loaded Student model.")
                ema_model.load_state_dict(model.state_dict())
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = combo_loss_for_micro

    best_iou = 0.0
    global_step = 0  # For MT rampup

    for epoch in range(args.epochs):

        # --- TRAIN ---
        if args.mean_teacher:
            train_loss, class_loss, cons_loss, global_step = train_mean_teacher(
                train_loader, model, ema_model, optimizer, criterion, epoch, device, args.epochs, args, global_step
            )
        else:
            train_loss, class_loss, cons_loss = train_standard(
                train_loader, model, optimizer, criterion, epoch, device, args.epochs
            )

        # --- VALIDATE ---
        # If MT is on, we validate the Teacher (ema_model), otherwise the Student (model)
        eval_model = ema_model if args.mean_teacher else model
        val_loss, val_iou = validate(test_loader, eval_model, criterion, device)

        # Console Output
        if args.mean_teacher:
            print(
                f"Epoch [{epoch+1}] | Train: {train_loss:.4f} (Cls:{class_loss:.3f} Cons:{cons_loss:.3f}) | Val IoU: {val_iou:.4f}")
        else:
            print(f"Epoch [{epoch+1}] | Train: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Logging
        log_data = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss,
            'Class_Loss': class_loss if args.mean_teacher else 0,
            'Cons_Loss': cons_loss if args.mean_teacher else 0,
            'Val_Loss': val_loss,
            'Val_IoU': val_iou,
            'Mode': 'MeanTeacher' if args.mean_teacher else 'Standard'
        }

        save_loss_data(pd.DataFrame([log_data]), log_csv_path)

        # Save Best Model
        if val_iou > best_iou:
            best_iou = val_iou
            prefix = "MT_Teacher" if args.mean_teacher else "Standard"
            save_name = f'best_iou_{prefix}_{args.name}_{timestamp}.pth'
            torch.save(eval_model.state_dict(), save_name)
            print(f"Saved best model: {save_name}")

    # Save Last Model
    last_name = f'last_{args.name}_model.pth'
    torch.save(model.state_dict(), last_name)
    if args.mean_teacher:
        torch.save(ema_model.state_dict(), f'last_{args.name}_ema_model.pth')

    print("Training Finished.")


if __name__ == "__main__":
    main()
