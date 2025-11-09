import os
import re
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader,Subset

'''

DATA_DIR = "./data/"
dataset = TiffSegmentationDataset(data_dir=DATA_DIR)

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

images,mask = next(iter(train_loader))
print(images.shape,mask.shape)

print(len(images))
TiffSegmentationDataset.show_img(images[0],streth=True,title="a")
TiffSegmentationDataset.show_img(images[1],streth=True,title="a")

'''

class DatasetSubsetWithTransform(Subset):
    
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def _transform_and_translate(self, x, y):
        # Convert PIL image to RGB (some of them are greyscale)
        x = x.convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = translate_label(y, keep_classes)
        return x, y

    def __getitem__(self, idx):
        return self._transform_and_translate(*super().__getitem__(idx))
    
    def __getitems__(self, indices):
        items = super().__getitems__(indices)
        for i in range(len(items)):
            items[i] = self._transform_and_translate(*items[i])
        return items


class TiffSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        
        IMAGE_PATH = "Original Images"
        MASKS_PATH = "Original Masks"
        
        self.image_path_dir = os.path.join(data_dir, IMAGE_PATH)
        self.mask_path_dir = os.path.join(data_dir, MASKS_PATH)
        self.transform = transform
        
        # 这个列表将存储 (image_path, mask_path) 对
        self.file_pairs = self.__get_file_pairs()

    def __get_numeric_key(self, filename):
        pattern = r"(\d+)(?=\.tif)"
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
            if not f.endswith(('.tif', '.tiff')): continue
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
            if not f.endswith(('.tif', '.tiff')): continue
            idx = self.__get_numeric_key(f)
            if idx in pairs_map:
                pairs_map[idx].append(os.path.join(self.mask_path_dir, f))
                final_pairs_list.append(pairs_map[idx])
        
        print(f"Found {len(final_pairs_list)}  (image, mask) pairs")
        return final_pairs_list

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.file_pairs[idx]
        image = tiff.imread(image_path) # (H, W) or (D, H, W)
        mask = tiff.imread(mask_path)
        
        
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0) 
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float() 
        if self.transform:
            # TODO:
            pass 
            
        return image_tensor, mask_tensor

    @staticmethod
    def show_img(img_data: np.ndarray, streth: bool = True, title: str = "Micro-CT Image"):
        if isinstance(img_data, torch.Tensor):
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
            # 避免除以零
            if p_max - p_min > 1e-5:
                img_stretched = (img_data - p_min) / \
                    (p_max - p_min) * 255
            else:
                img_stretched = img_data * 0 # 如果所有值都一样，则为黑色
                
            img_stretched = img_stretched.astype(np.uint8)
            img_data = img_stretched

        # Show image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_data, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

