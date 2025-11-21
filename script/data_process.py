import sys
import os
import argparse
import numpy as np
import imageio.v2 as iio
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T


def parse_args():
    parser = argparse.ArgumentParser(description="Data Augmentation Pipeline for Micro-CT Images")

    parser.add_argument('--src_img', type=str, default='data/Original Images', help='Path to original images folder')
    parser.add_argument('--src_mask', type=str, default='data/Original Masks', help='Path to original masks folder')
    parser.add_argument('--output_dir', type=str, default='data_processed',
                        help='Root directory for saving processed data')

    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test set size (default: 0.1)')

    parser.add_argument('--aug_count', type=int, default=0,
                        help='Number of augmented copies per training image (default: 0)')
    parser.add_argument('--aug_prob', type=float, default=0.6,
                        help='Probability for flip transformations (default: 0.6)')

    return parser.parse_args()


def make_path(base_dir, idx, subset_name):
    img_dir = os.path.join(base_dir, subset_name, "image")
    lbl_dir = os.path.join(base_dir, subset_name, "label")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    ipath = os.path.join(img_dir, f"{idx}.png")
    lpath = os.path.join(lbl_dir, f"{idx}.png")
    return ipath, lpath


def stretch_img(img):
    min_val, max_val = np.min(img), np.max(img)
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)

    img_stretched = (img - min_val) / (max_val - min_val) * 255
    img_stretched = img_stretched.astype(np.uint8)
    return img_stretched


def load_img(ipath, lpath):
    img = tiff.imread(ipath)
    label = tiff.imread(lpath)

    if img.ndim == 3:
        print(f"3D stack detected with shape: {img.shape}")
        img = img[img.shape[0] // 2]
    if label.ndim == 3:
        print(f"3D stack detected with shape: {label.shape}")
        label = label[label.shape[0] // 2]

    return stretch_img(img), stretch_img(label)


def get_transforms(prob=0.8):
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=prob),
        T.RandomVerticalFlip(p=prob),
        T.ToDtype(torch.uint8, scale=True)
    ])


conv2img_pipeline = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.uint8, scale=True)
])


def augment(img, label, transform_pipeline):
    img, label = transform_pipeline(img, label)
    return img.squeeze().numpy(), label.squeeze().numpy()


def conv2img(img, label):
    img, label = conv2img_pipeline(img, label)
    return img.squeeze().numpy(), label.squeeze().numpy()


def process_and_save(dataset_split, output_dir, subset_name, augment_pipeline=None, aug_count=0):
    processed_imgs = []
    processed_lbls = []

    print(f"Processing {subset_name} set...")

    for pairs in dataset_split:
        ipath, lpath = pairs[0], pairs[1]
        img, label = load_img(ipath, lpath)

        if subset_name == 'train':
            i_base, l_base = conv2img(img, label)
            processed_imgs.append(i_base)
            processed_lbls.append(l_base)

            if augment_pipeline and aug_count > 0:
                for _ in range(aug_count):
                    new_img, new_label = augment(img, label, augment_pipeline)
                    processed_imgs.append(new_img)
                    processed_lbls.append(new_label)
        else:
            new_img, new_label = conv2img(img, label)
            processed_imgs.append(new_img)
            processed_lbls.append(new_label)

    print(f"Saving {len(processed_imgs)} images to {os.path.join(output_dir, subset_name)}...")
    for idx, (i, l) in enumerate(zip(processed_imgs, processed_lbls)):
        ipath, lpath = make_path(output_dir, idx, subset_name)
        iio.imwrite(ipath, i)
        iio.imwrite(lpath, l)


def main():
    if sys.platform.startswith('win'):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    from utils.dataset import TiffDataset

    args = parse_args()

    if not os.path.exists(args.src_img) or not os.path.exists(args.src_mask):
        print(f"Error: Input directories not found: {args.src_img} or {args.src_mask}")
        sys.exit(1)

    print(f"Loading data from: {args.src_img}")
    dataset = TiffDataset(args.src_img, args.src_mask)
    dataset_list = dataset.get_list()

    indices = len(dataset_list)
    test_size = int(indices * args.test_ratio)
    train_size = indices - test_size

    print(f"Total images: {indices} | Train: {train_size} | Test: {test_size}")

    train_set, test_set = torch.utils.data.random_split(
        dataset_list,
        [train_size, test_size]
    )
    train_transform = get_transforms(prob=args.aug_prob)

    process_and_save(
        train_set,
        args.output_dir,
        "train",
        augment_pipeline=train_transform,
        aug_count=args.aug_count
    )

    process_and_save(
        test_set,
        args.output_dir,
        "test",
        augment_pipeline=None,
        aug_count=0
    )

    print("Done!")


if __name__ == "__main__":
    main()
