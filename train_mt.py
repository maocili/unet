import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.unet import UNet
from utils.dataset import TiffDataset
from utils.weights import kaiming_init_weights
from utils.loss_function.combo import combo_loss_for_micro
from utils.loss_function.iou import iou_coeff
from utils.transformers import MicroTransformers
from mean_teacher import ramps, losses


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"


LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 2

NOISE = True
EMA_DECAY = 0.999
CONSISTENCY = 10.0
CONSISTENCY_RAMPUP = 15.0

global_step = 0


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return CONSISTENCY * ramps.sigmoid_rampup(epoch, CONSISTENCY_RAMPUP)


def train(train_loader, model, ema_model, optimizer, epoch):
    global global_step
    model.train()
    ema_model.train()

    total_loss = 0.0
    total_class_loss = 0.0
    total_cons_loss = 0.0

    consistency_weight = get_current_consistency_weight(epoch=epoch)

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    classification_criterion = combo_loss_for_micro
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()

        model_input, ema_input = images, images
        if NOISE:
            noise_shape = images.shape
            model_noise = torch.randn(noise_shape).to(DEVICE) * 0.1
            ema_model_noise = torch.randn(noise_shape).to(DEVICE) * 0.1

            model_input = images + model_noise
            ema_input = images + ema_model_noise

        model_output = model(model_input)
        with torch.no_grad():
            ema_model_output = ema_model(ema_input)

        class_loss = classification_criterion(model_output, masks)

        consistency_loss = losses.softmax_mse_loss(model_output, ema_model_output)

        loss = class_loss + consistency_weight * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA weights
        global_step += 1
        update_ema_variables(model, ema_model, EMA_DECAY, global_step)

        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_cons_loss += consistency_loss.item()
        loop.set_postfix(
            loss=loss.item(),
            cls=class_loss.item(),
            cons=consistency_loss.item(),
            w=consistency_weight
        )
    return total_loss, total_class_loss, total_cons_loss


def validate(model, loader, device):
    model.eval()
    iou_sum = 0.0
    loss_sum = 0.0
    classification_criterion = combo_loss_for_micro
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)

            loss_sum += classification_criterion(outputs, masks)

            preds = torch.softmax(outputs, dim=1)[:, 1, :, :]
            iou_sum += iou_coeff(preds, masks)

    return iou_sum / len(loader), loss_sum / len(loader)


def main():
    global global_step
    print(f"Using device: {DEVICE}")

    # Load Data
    tif_train_data = TiffDataset("data/tif/train/image/", "data/tif/train/label/",
                                 transforms=MicroTransformers(geo_augment=True))
    png_train_data = TiffDataset("data/png/train/image/", "data/png/train/label",
                                 transforms=MicroTransformers(geo_augment=True))
    tif_test_data = TiffDataset("data/tif/test/image/", "data/tif/test/label/",
                                transforms=MicroTransformers(geo_augment=False))
    png_test_data = TiffDataset("data/png/test/image/", "data/png/test/label",
                                transforms=MicroTransformers(geo_augment=False))

    train_set = png_train_data
    test_set = png_test_data

    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    assert len(train_set) != 0
    assert len(test_set) != 0

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    model = UNet(in_channels=1, out_channels=2).to(DEVICE)
    model.apply(kaiming_init_weights)

    ema_model = UNet(in_channels=1, out_channels=2).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.detach_()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_iou = 0

    for epoch in range(NUM_EPOCHS):
        total_loss, total_class_loss, total_cons_loss = train(train_loader, model, ema_model, optimizer, epoch)

        avg_iou, avg_val_loss = validate(ema_model, test_loader, DEVICE)
        avg_train_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_cons_loss = total_cons_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}] | Train Loss: {avg_train_loss:.4f} "
              f"(Class: {avg_class_loss:.4f} + {consistency_weight:.2f}*Cons: {avg_cons_loss:.4f})")
        print(f"Validation IoU (Teacher): {avg_iou:.4f} Validate Loss: {avg_val_loss:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(ema_model.state_dict(), 'best_mt_iou_model.pth')
            print("Saved best Teacher model")

    torch.save(ema_model.state_dict(), 'last_mt_model.pth')
    torch.save(model.state_dict(), 'last_mt_ema_model.pth')


if __name__ == "__main__":
    main()
