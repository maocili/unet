
import os
import argparse
import sys
import numpy as np
import imageio.v2 as iio
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Unlabeled Data with Random Crop & Padding")

    parser.add_argument('--src_img', type=str, default='data/datasets/10min_HT/',
                        help='Path to source images folder (unlabeled)')
    parser.add_argument('--output_dir', type=str, default='data/unlabeled_processed',
                        help='Directory to save processed images')

    parser.add_argument('--crop_size', type=int, default=512,
                        help='Size of the random crop')
    parser.add_argument('--padding', type=int, default=0,
                        help='Optional extra padding on borders before cropping')
    parser.add_argument('--aug_count', type=int, default=5,
                        help='Number of random crops to generate per original image')

    return parser.parse_args()


def get_transforms(crop_size, padding):
    return T.Compose([
        T.ToImage(),  # 转换为 Tensor 格式
        # T.ToDtype(torch.uint8, scale=True), # 确保数据类型正确

        # 实现随机裁剪 + 填充
        # pad_if_needed=True: 如果原图小于 crop_size，会自动填充
        # padding=padding: 额外的边缘填充
        # fill=0: 填充值为 0 (黑色)
        T.RandomCrop(size=(crop_size, crop_size), padding=padding, pad_if_needed=True, fill=0, padding_mode='constant'),
    ])


def make_output_dirs(output_root):
    img_dir = os.path.join(output_root, "image")
    lbl_dir = os.path.join(output_root, "label")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return img_dir, lbl_dir


def main():
    args = parse_args()
    if not os.path.exists(args.src_img):
        print(f"Error: Source directory not found: {args.src_img}")
        return

    img_out_dir, lbl_out_dir = make_output_dirs(args.output_dir)

    print(f"Loading data from: {args.src_img}")
    dataset = TiffDataset(image_path=args.src_img, masks_path="", image_only=True)

    if len(dataset) == 0:
        print("No images found in the dataset.")
        return

    transform = get_transforms(crop_size=args.crop_size, padding=args.padding)
    print(f"Start processing... Total images: {len(dataset)}")
    print(f"Config: Crop Size={args.crop_size}, Padding={args.padding}, Crops per Image={args.aug_count}")
    print(f"Saving to: {img_out_dir} AND {lbl_out_dir}")

    total_saved = 0

    black_mask = np.zeros((args.crop_size, args.crop_size), dtype=np.uint8)

    for idx, img in tqdm(enumerate(dataset), total=len(dataset), desc="Generating"):
        for i in range(args.aug_count):
            augmented_tensor = transform(img)

            augmented_np = augmented_tensor.permute(1, 2, 0).numpy(
            ) if augmented_tensor.ndim == 3 else augmented_tensor.numpy()
            if augmented_np.shape[-1] == 1:
                augmented_np = augmented_np.squeeze(-1)

            filename = f"{10000 + total_saved}.png"

            iio.imwrite(os.path.join(img_out_dir, filename), augmented_np)
            iio.imwrite(os.path.join(lbl_out_dir, filename), black_mask)

            total_saved += 1

    print(f"Done! Saved {total_saved} pairs (image + black mask) to '{args.output_dir}'")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from utils import TiffDataset

    # Windows 下可能需要此设置避免库冲突
    if sys.platform.startswith('win'):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
