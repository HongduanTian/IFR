from locale import normalize
import random
import torch

import numpy as np
import torchvision.transforms as transforms

from PIL import ImageFilter


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]) -> None:
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DataAugmentation(object):

    def __init__(self, num_aug=1) -> None:
        self.num_aug = num_aug
        self.transforms = [
            transforms.RandomSizedCrop(84, scale=(0.75, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            self._normalize,
        ]
    
    def _normalize(self, x):
        return 2 * (x / 255.0 - 0.5)

    def generate_augmentations(self, x):
        # 1. restore the image
        original_imgs = (x * 0.5 + 0.5) * 255.0
        
        augmented_data = []
        for _ in range(self.num_aug):
            augmented_data.append(transforms.Compose(self.transforms)(original_imgs))
        augmented_data = torch.cat(augmented_data, dim=0)
        return augmented_data.contiguous()


class OnlyCropAugmentation(object):
    
    def __init__(self, num_aug=3) -> None:
        self.num_aug = num_aug
        self.crop1 = transforms.RandomSizedCrop(84, scale=(0.9, 1.))
        self.crop2 = transforms.RandomSizedCrop(84, scale=(0.2, 1.))
    
    def _normalize(self, x):
        return 2 * (x / 255.0 - 0.5)

    def generate_augmentations(self, x):
        # 1. restore the image
        original_imgs = (x * 0.5 + 0.5) * 255.0
        
        augmented_data = []
        augmented_data.append(self._normalize(self.crop1(original_imgs)))
        for _ in range(self.num_aug-1):
            augmented_data.append(self._normalize(self.crop2(original_imgs)))
        augmented_data = torch.cat(augmented_data, dim=0)
        return augmented_data.contiguous()