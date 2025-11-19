import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from typing import Union, Dict, Any

from torchvision.utils import _log_api_usage_once


class ToBinaryMask(v2.Transform):
    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    _transformed_types = (torch.Tensor, np.ndarray)

    def transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        if isinstance(inpt, torch.Tensor):
            inpt = inpt.cpu().numpy()

        inpt = (inpt >= 255 / 2).astype(np.uint8)

        mask_tensor = torch.from_numpy(inpt).long()
        return tv_tensors.Mask(mask_tensor)


class ToMicroMasks(v2.Transform):
    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    _transformed_types = (torch.Tensor, np.ndarray)

    def transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        if isinstance(inpt, torch.Tensor):
            inpt = inpt.cpu().numpy()

        img_stretched = (inpt - np.min(inpt)) / (np.max(inpt) - np.min(inpt)) * 255
        img_stretched = img_stretched.astype(np.uint8)

        final_masks = np.zeros_like(img_stretched, dtype=np.uint8)

        threhold = [5, 250]
        final_masks[(img_stretched > threhold[0]) & (img_stretched < threhold[1])] = 0 # 127 background
        final_masks[img_stretched < threhold[0]] = 1 # 0 white area
        final_masks[(img_stretched > threhold[1])] = 1 # 255 black area

        return final_masks


ISBIImageTransformers = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.15], std=[0.35]),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(2.0)),
])

ISBILabelTransformers = v2.Compose([
    ToBinaryMask(),
])


MicroImageTransformers = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.15], std=[0.35]),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(2.0)),
])

MicroLabelTransformers = v2.Compose([
    ToMicroMasks(),
])
