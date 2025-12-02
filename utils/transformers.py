import numpy as np
import torch
import kornia
from torchvision.transforms import v2
from torchvision import tv_tensors
from typing import Union, Dict, Any, Tuple

from torchvision.utils import _log_api_usage_once


class ToMasks(v2.Transform):
    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    _transformed_types = (torch.Tensor, np.ndarray)

    def _transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        return self.transform(inpt, params)

    def transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        if isinstance(inpt, torch.Tensor):
            inpt = inpt.cpu().numpy()

        # process .tif(0,1) and .png(0,255) both
        img_stretched = (inpt - np.min(inpt)) / (np.max(inpt) - np.min(inpt)) * 255
        img_stretched = img_stretched.astype(np.uint8)

        img_binary = (img_stretched >= 255 / 2).astype(np.uint8)
        mask_tensor = torch.from_numpy(img_binary).long()
        return tv_tensors.Mask(mask_tensor)


ISBIImageTransformers = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.15], std=[0.35]),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(2.0)),
])

ISBILabelTransformers = v2.Compose([
    ToMasks(),
])


class Clahe(v2.Transform):
    def __init__(self,  clip_limit: float = 40.0, grid_size: Tuple[int, int] = (8, 8)) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.clip_limit = clip_limit
        self.grid_size = grid_size

    _transformed_types = (torch.Tensor, np.ndarray)

    def _transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        return self.transform(inpt, params)

    def transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        inpt = torch.clamp(inpt, 0.0, 1.0)
        inpt = kornia.enhance.equalize_clahe(inpt, clip_limit=self.clip_limit, grid_size=self.grid_size)
        return inpt


class SobelFilter(v2.Transform):
    def __init__(self):
        super(SobelFilter, self).__init__()

    def _transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        return self.transform(inpt, params)

    def transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        is_unbatched = inpt.ndim == 3
        if is_unbatched:
            inpt = inpt.unsqueeze(0)  # [B, C, H, W]

        clean_input = kornia.filters.gaussian_blur2d(inpt, (5, 5), (2.0, 2.0))
        grads = kornia.filters.spatial_gradient(clean_input, mode='sobel')

        grad_x = grads[:, :, 0, :, :]
        grad_y = grads[:, :, 1, :, :]

        output = 0.5 * grad_x + 0.5 * grad_y

        if is_unbatched:
            output = output.squeeze(0)  # [C, H, W]
        return output


class MicroTransformers:
    def __init__(self, geo_augment=True, denoise=True):
        self.to_img = v2.ToImage()
        self.to_mask = ToMasks()
        self.geo_augment = geo_augment
        self.denoise = denoise

        self.geom_aug_func = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ElasticTransform(alpha=30.0, sigma=5.0),
        ])

        # Only for image augment
        self.pixel_aug_func = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),  # Scale [0,1]
            Clahe(clip_limit=40.0, grid_size=(8, 8)),
            v2.Normalize(mean=[0.15], std=[0.35]),
        ])

        self.denoise_func = v2.Compose([
            SobelFilter(),
        ])

    def _joint_call(self, img, label):
        img = self.to_img(img)
        label = self.to_mask(label)

        if self.geo_augment:
            img, label = self.geom_aug_func(img, label)
        img = self.pixel_aug_func(img)

        return img, label

    def __call__(self, img: Any = None, label: Any = None):
        if img is not None and label is not None:
            return self._joint_call(img=img, label=label)

        if img is not None:
            img = self.to_img(img)

            if self.geo_augment:
                img = self.geom_aug_func(img)
            img = self.pixel_aug_func(img)
            if self.denoise:
                img = self.denoise_func(img)
            return img

        label = self.to_mask(label)
        return label
