import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.spectrogram_augmentations import FrequencyMasking, TimeMasking
from src.augmentations.random_apply import RandomApply


class SpecAugment(AugmentationBase):
    def __init__(self, num_freq_mask: int, num_time_mask: int, freq_mask_param: int, 
                 time_mask_param: int, *args, **kwargs):
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
        self._freq_masking = FrequencyMasking(freq_mask_param)
        self._time_masking = TimeMasking(time_mask_param)
        
    def __call__(self, data: Tensor):
        for _ in range(self.num_mask_f):
            x = self._freq_masking(x)
        
        for _ in range(self.num_mask_t):
            x = self._time_masking(x)
            
        return x


class RandomSpecAugment(RandomApply):
    def __init__(self, p: float, *args, **kwargs):
        super().__init__(SpecAugment(*args, **kwargs), p)
