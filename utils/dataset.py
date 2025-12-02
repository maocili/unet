import os
import re
import sys
import torch
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


class TiffDataset(Dataset):
    def __init__(self, image_path: str, masks_path: str, img_transforms=None, label_transforms=None, transforms=None, image_only=False):
        super().__init__()

        self.file_supports = ("tif", "tiff", "png")
        self.image_path_dir = image_path
        self.mask_path_dir = masks_path
        self.image_only = image_only  # True, read image path only

        self.img_transforms = img_transforms
        self.label_transforms = label_transforms
        self.transforms = transforms

        self.file_pairs = self.__get_file_pairs()

    def __get_numeric_key(self, filename):
        for t in self.file_supports:
            pattern = f"(\d+)(?=\.{t})"
            matches = re.findall(pattern, filename)
            if matches:
                return int(matches[-1])
        return None

    def get_list(self):
        return self.__get_file_pairs()

    def __get_file_pairs(self):
        pairs_map = {}
        try:
            ilist = os.listdir(self.image_path_dir)
        except FileNotFoundError:
            print(f"Error: file not found {self.image_path_dir}")
            return []

        for f in ilist:
            if not f.endswith(self.file_supports):
                continue
            idx = self.__get_numeric_key(f)
            if idx is not None:
                pairs_map[idx] = os.path.join(self.image_path_dir, f)

        if not self.image_only:
            try:
                mlist = os.listdir(self.mask_path_dir)
            except FileNotFoundError:
                print(f"Error: file not found {self.mask_path_dir}")
                return []

            for f in mlist:
                if not f.endswith(self.file_supports):
                    continue

                idx = self.__get_numeric_key(f)
                if idx in pairs_map:
                    img_path = pairs_map[idx]
                    pairs_map[idx] = (img_path, os.path.join(self.mask_path_dir, f))

        print(f"Found {len(pairs_map)}  (image, mask) pairs")
        return [v for k, v in sorted(pairs_map.items())]

    def __len__(self):
        return len(self.file_pairs)

    def __image_strech_uint8(self, image):
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        return image.astype(np.uint8)

    def __getitem__(self, idx):
        if self.image_only:
            return self._get_single_items(idx)
        return self._get_pair_items(idx)

    def _get_single_items(self, idx):
        image_path = self.file_pairs[idx]
        image = iio.imread(image_path)  # (H, W)

        image = self.__image_strech_uint8(image)

        if self.transforms:
            image = self.transforms(image)
        elif self.img_transforms:
            image = self.img_transforms(image)
        return (image)

    def _get_pair_items(self, idx):
        image_path, label_path = self.file_pairs[idx]
        image = iio.imread(image_path)  # (H, W)
        label = iio.imread(label_path)  # (H, W)

        # streched
        image = self.__image_strech_uint8(image)

        if self.transforms:
            image, label = self.transforms(image, label)
        else:
            if self.img_transforms:
                image = self.img_transforms(image)
            if self.label_transforms:
                label = self.label_transforms(label)
        return (image, label)

    @staticmethod
    def show_img(
        img_data: np.ndarray, streth: bool = True, title: str = "Micro-CT Image"
    ):
        if isinstance(img_data, torch.Tensor):
            if img_data.requires_grad:
                img_data = img_data.detach().cpu().numpy()
            else:
                img_data = img_data.cpu().numpy()

        if img_data.shape[0] == 1:
            img_data = img_data[0]

        if img_data.ndim == 3:
            print(f"3D stack detected with shape: {img_data.shape}")
            img_data = img_data[img_data.shape[0] // 2]

        print("Image dtype:", img_data.dtype)
        print("Min intensity:", np.min(img_data))
        print("Max intensity:", np.max(img_data))

        # Show image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_data, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
