import math
import numpy as np
import torch.nn as nn
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import transforms
from torchvision import transforms, datasets


class TrainAugment:
    def __init__(self, size, s=1):
        #self.root_folder = root_folder
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        blur = transforms.GaussianBlur((9, 9), (0.1, 2.0))
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomApply([blur], p=0.5),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                              ])
    def __call__(self, x):
        return self.data_transforms(x)
    

class TestAugment:
    def __init__(self, size, s=1):
        #self.root_folder = root_folder
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        blur = transforms.GaussianBlur((9, 9), (0.1, 2.0))
        self.data_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def __call__(self, x):
        return self.data_transforms(x)

'''
class UniformTemporalSubsample(nn.Module):
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # ...THW
        t_dim = -3
        t = x.shape[t_dim]
        indices = torch.linspace(0, t - 1, self.num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, dim=t_dim, index=indices)


class RandomUniformTemporalSubsample(UniformTemporalSubsample):
    def __init__(self, num_samples: int):
        super().__init__(num_samples)
        self.num_samples = num_samples

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # ...THW
        t_dim = -3
        t = x.shape[t_dim]
        if t < self.num_samples:
            return super().forward(x)
        else:
            indices = torch.randperm(t)[:self.num_samples].sort()[0]
            return torch.index_select(x, dim=t_dim, index=indices)


class Repeat(nn.Module):
    def __init__(self, num_repeats: int, transform: nn.Module):
        super().__init__()
        self.num_repeats = num_repeats
        self.transform = transform

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        samples = [self.transform(x) for _ in range(self.num_repeats)]
        if len(x.shape) == 4:
            samples = torch.stack(samples, dim=0)
        else:
            samples = torch.cat(samples, dim=0)
        return samples


class ShortSideScale(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def scale(self, x, size):
        # CTWH
        h, w = x.shape[-2], x.shape[-1]
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        return resize(x, size=(new_h, new_w))

    @torch.no_grad()
    def forward(self, x):
        # CTWH
        return self.scale(x, self.size)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    @torch.no_grad()
    def forward(self, x):
        return (x - self.mean) / self.std
'''
