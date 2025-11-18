import sys
import os
if sys.platform.startswith('win'):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataset import TiffDataset
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt


def show_img(img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title("Contrast-Stretched Micro-CT Image")
    plt.axis('off')
    plt.show()


def stretch_img(img):
    img_stretched = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img_stretched = img_stretched.astype(np.uint8)
    return img_stretched


def load_img(ipath, lpath):
    img = tiff.imread(ipath)
    lable = tiff.imread(lpath)

    if img.ndim == 3:
        print(f"3D stack detected with shape: {img.shape}")
        img = img[img.shape[0] // 2]
    if lable.ndim == 3:
        print(f"3D stack detected with shape: {lable.shape}")
        lable = lable[lable.shape[0] // 2]

    return stretch_img(img), stretch_img(lable)


transform_pipeline = T.Compose([
    T.ToImage(),  # Converts PIL Image (H, W, C) to Tensor (C, H, W)
    T.ToDtype(torch.float32, scale=True),  # Normalize to [0.0, 1.0]
    # --- Transforms from your list ---
    T.RandomHorizontalFlip(p=0.8),
    T.RandomVerticalFlip(p=0.8),
    T.ToDtype(torch.uint8, scale=True)
])


def augment(img, lable):
    img, lable = transform_pipeline(img, lable)
    return img.squeeze().numpy(), lable.squeeze().numpy()


dataset = TiffDataset('data/Original Images', 'data/Original Masks')
dataset = dataset.get_list()


ilist = []
llist = []
# for img,lable in dataset:
for ipath, lpath in dataset:
    img, lable = load_img(ipath, lpath)
    ilist.append(img)
    llist.append(lable)
    for i in range(0,4):
        new_img, new_lable = augment(img, lable)
        ilist.append(new_img)
        llist.append(new_lable)

import imageio.v2 as iio
def make_path(idx):
    ipath = "data/image/"+ str(idx) + ".png"
    lpath = "data/lable/"+ str(idx) + ".png"

    os.makedirs("data/lable/",exist_ok=True)
    os.makedirs("data/image/",exist_ok=True)
    return ipath,lpath

for idx,(i,l) in enumerate(zip(ilist,llist)):
    ipath,lpath = make_path(idx)
    iio.imwrite(ipath, i.squeeze())
    iio.imwrite(lpath, l.squeeze())


