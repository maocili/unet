import sys
import os
from models import UNet
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio.v2 as iio
from monai.inferers import sliding_window_inference

from utils.dataset import TiffDataset
from utils.transformers import MicroTransformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.plt import show_predictions

MODEL_PATH = "best_iou_unet_model.pth"

DEVICE = "cuda"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
torch.device(DEVICE)
print("Using device:", DEVICE)


def main():
    dataset = TiffDataset(single_dir=True, image_path="data/datasets/10min_HT/",
                          masks_path="", transforms=MicroTransformers(geo_augment=False))

    batch_size = 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = UNet(in_channels=1, out_channels=2).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters:", {total_params})
    try:
        print(f"Loading model: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        exit()
    model.eval()

    p_list = []
    count = 0
    page = 16
    with torch.no_grad():

        loop = tqdm(loader, desc=f'Predict')
        for img in loop:
            img = img.to(DEVICE)
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0)

            roi_size = (768, 768)
            sw_batch_size = 2
            overlap = 0.5

            with torch.no_grad():
                output_mask = sliding_window_inference(
                    inputs=img,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode='gaussian'
                )

            masks_pred = torch.argmax(output_mask, dim=1)
            img = img.cpu().numpy()
            masks_pred = masks_pred.cpu().numpy()

            p_list.append((img, masks_pred))

            if count >= page:
                break
            count += 1

    for p in p_list:
        show_predictions(p)


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
