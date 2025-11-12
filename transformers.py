
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
