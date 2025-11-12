import os
import re
import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.transforms import v2

from transformers import ToBinaryMask

"""
# dataset = TiffSegmentationDataset('data_isbi/train/images','data_isbi/train/labels')

# indices = len(dataset)

# train_size = len(dataset) - int(0.2*len(dataset))
# test_size = len(dataset) - train_size

# # Create random splits for train and test sets
# train_set, test_set = torch.utils.data.random_split(
#     dataset, 
#     [train_size, test_size]
# )
# print(f"Training set size: {len(train_set)}")
# print(f"Test set size: {len(test_set)}")

# batch_size = 4

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

# images,mask = next(iter(train_loader))
# print(images.shape,mask.shape)

# print(len(images))
# TiffSegmentationDataset.show_img(images[0],streth=True,title="a")
# TiffSegmentationDataset.show_img(mask[0],streth=True,title="a")

"""

ISBIImageTransformers = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.15], std=[0.35]),
    v2.ElasticTransform(alpha=80.0, sigma=8.0),
])

ISBILableTransformers = v2.Compose([
    ToBinaryMask(),
    v2.ToDtype(torch.long, scale=True)
])


class TiffDataset(Dataset):
    def __init__(self, image_path: str, masks_path: str, img_transforms=None, label_transforms=None):
        super().__init__()

        self.file_supports = ("tif", "tiff")
        self.image_path_dir = image_path
        self.mask_path_dir = masks_path
        self.file_pairs = self.__get_file_pairs()

        self.img_transforms = img_transforms
        self.lable_transforms = label_transforms

    def __get_numeric_key(self, filename):
        for t in self.file_supports:
            pattern = f"(\d+)(?=\.{t})"
            matches = re.findall(pattern, filename)
            if matches:
                return int(matches[-1])
        return None

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
                pairs_map[idx] = [os.path.join(self.image_path_dir, f)]

        try:
            mlist = os.listdir(self.mask_path_dir)
        except FileNotFoundError:
            print(f"Error: file not found {self.mask_path_dir}")
            return []

        final_pairs_list = []
        for f in mlist:
            if not f.endswith(self.file_supports):
                continue
            idx = self.__get_numeric_key(f)
            if idx in pairs_map:
                pairs_map[idx].append(os.path.join(self.mask_path_dir, f))
                final_pairs_list.append(pairs_map[idx])

        print(f"Found {len(final_pairs_list)}  (image, mask) pairs")
        return final_pairs_list

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_path, lable_path = self.file_pairs[idx]
        image = tiff.imread(image_path)  # (H, W)
        lable = tiff.imread(lable_path)

        if self.img_transforms:
            image = self.img_transforms(image)

        if self.lable_transforms:
            lable = self.lable_transforms(lable)

        return (image, lable)

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

        if streth:
            p_min = np.min(img_data)
            p_max = np.max(img_data)
            if p_max - p_min > 1e-5:
                img_stretched = (img_data - p_min) / (p_max - p_min) * 255
            else:
                img_stretched = img_data * 0

            img_stretched = img_stretched.astype(np.uint8)
            img_data = img_stretched

        # Show image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_data, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    dataset = TiffDataset('data_isbi/train/images', 'data_isbi/train/labels',
                          img_transforms=ISBIImageTransformers, label_transforms=ISBILableTransformers)

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    images, lables = next(iter(train_loader))
    print(images.shape, lables.shape)

    for i in images:
        TiffDataset.show_img(i, streth=True, title="Orginal")
    # TiffDataset.show_img(lables[0], streth=True, title="Orginal")
    # TiffDataset.show_img(blurred_images[0], streth=True, title="Lable")
    # TiffDataset.show_img(blurred_images[1], streth=True, title="Lable")
    # TiffDataset.show_img(blurred_images[2], streth=True, title="Lable")
