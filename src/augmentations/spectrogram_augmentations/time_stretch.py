import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)
        
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        x = self._aug(data).squeeze(1)
        return x


class RandomTimeStretch(RandomApply):
    def __init__(self, p: float, *args, **kwargs):
        super().__init__(TimeStretch(*args, **kwargs), p)
