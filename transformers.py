
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from typing import Union, Dict, Any


class ToBinaryMask(v2.Transform):
    _transformed_types = (tv_tensors.Image, np.ndarray)

    def _transform(
        self, inpt: Union[torch.Tensor, np.ndarray], params: Dict[str, Any]
    ) -> tv_tensors.Image:
        inpt = (inpt >= 255 / 2).astype(np.uint8)
        return torch.from_numpy(inpt).long()
