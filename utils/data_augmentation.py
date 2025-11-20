import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

from dataset import TiffDataset


if sys.platform.startswith('win'):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_path(idx, target):
    ipath = "data/" + target + "/image/" + str(idx) + ".png"
    lpath = "data/" + target + "/label/" + str(idx) + ".png"

    os.makedirs("data/" + target + "/label/", exist_ok=True)
    os.makedirs("data/" + target + "/image/" , exist_ok=True)
    return ipath, lpath


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

conv2img_pipline = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.uint8, scale=True)
])


def augment(img, lable):
    img, lable = transform_pipeline(img, lable)
    return img.squeeze().numpy(), lable.squeeze().numpy()


def conv2img(img, label):
    img, label = conv2img_pipline(img, label)
    return img.squeeze().numpy(), label.squeeze().numpy()


dataset = TiffDataset('data/Original Images', 'data/Original Masks')
dataset = dataset.get_list()

indices = len(dataset)
train_size = indices - int(0.2*indices)
test_size = indices - train_size

# Create random splits for train and test sets
train_set, test_set = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)

# Train Dataset
ilist = []
llist = []
for pairs in train_set:
    ipath, lpath = pairs[0],pairs[1]
    img, lable = load_img(ipath, lpath)
    ilist.append(img)
    llist.append(lable)
    for i in range(0, 4):
        new_img, new_lable = augment(img, lable)
        ilist.append(new_img)
        llist.append(new_lable)


for idx, (i, l) in enumerate(zip(ilist, llist)):
    ipath, lpath = make_path(idx, "train")
    iio.imwrite(ipath, i.squeeze())
    iio.imwrite(lpath, l.squeeze())

# Test Dataset
ilist = []
llist = []
for pairs in test_set:
    ipath, lpath = pairs[0],pairs[1]

    img, lable = load_img(ipath, lpath)
    new_img, new_lable = conv2img(img, lable)
    ilist.append(new_img)
    llist.append(new_lable)


for idx, (i, l) in enumerate(zip(ilist, llist)):
    ipath, lpath = make_path(idx, "test")
    iio.imwrite(ipath, i.squeeze())
    iio.imwrite(lpath, l.squeeze())
