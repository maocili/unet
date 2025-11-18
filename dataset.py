import os
import re
import sys
import torch
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from transformers import ISBIImageTransformers, ISBILabelTransformers
from transformers import MicroImageTransformers, MicroLabelTransformers


class TiffDataset(Dataset):
    def __init__(self, image_path: str, masks_path: str, img_transforms=None, label_transforms=None):
        super().__init__()

        self.file_supports = ("tif", "tiff", "png")
        self.image_path_dir = image_path
        self.mask_path_dir = masks_path
        self.file_pairs = self.__get_file_pairs()

        self.img_transforms = img_transforms
        self.label_transforms = label_transforms

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
        image_path, label_path = self.file_pairs[idx]
        image = iio.imread(image_path)  # (H, W)
        label = iio.imread(label_path)  # (H, W)

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


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # dataset = TiffDataset('data_isbi/train/images', 'data_isbi/train/labels',

    #                         img_transforms=ISBIImageTransformers, label_transforms=ISBIlabelTransformers)

    dataset = TiffDataset('data/image', 'data/label', img_transforms=MicroImageTransformers,
                          label_transforms=MicroLabelTransformers)

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
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    images, labels = next(iter(train_loader))

    # print(images.shape, labels.shape, old)

    for i, j in zip(images, labels):
        TiffDataset.show_img(i, streth=True, title="Orginal")
        TiffDataset.show_img(j, streth=True, title="Masks")
        break
