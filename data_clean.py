from dataset import TiffDataset

import os
import shutil
import kagglehub
import numpy as np
import pandas as pd
import imageio.v2 as iio

tran
dataset = TiffDataset(
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

batch_size = 4

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, drop_last=False)

images, mask = next(iter(train_loader))
print(images.shape, mask.shape)

print(len(images))

TiffDataset.show_img(images[0], streth=True, title="a")
TiffDataset.show_img(mask[0], streth=True, title="a")

#TODO:
# 1. smooth
# 2. non-rigid
# 3. de-noise
# 4. ....
