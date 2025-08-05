# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
import torch


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)




class TileDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tile_paths = []

        for slide_dir in os.listdir(root_dir):
            slide_path = os.path.join(root_dir, slide_dir)
            if os.path.isdir(slide_path):
                for tile_file in os.listdir(slide_path):
                    if tile_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.tile_paths.append(os.path.join(slide_path, tile_file))

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        image = Image.open(tile_path).convert("RGB")
        return image


class MyCollateFunction:

    def __init__(self, base_transform1, base_transform2, stain_augmentation, arg_gpu=None):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.stain_augmentation = stain_augmentation
        self.arg_gpu = arg_gpu
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    """
    ImageFolder that was used in default MoCo v3 automatically assign a labels to images 
    corresponding to the folder of the images. 
    This label isn't use so we just return None in addition of data to have the correct 
    output form but it is never used. 
    """
    def __call__(self, img_list): 
        if self.arg_gpu is not None:
            imgs = torch.stack(self.to_tensor(img_list)).cuda(self.arg_gpu)
        else:
            imgs = torch.stack(self.to_tensor(img_list)).cuda()
        stained_batch1 = self.stain_augmentation(imgs)
        stained_batch2 = self.stain_augmentation(imgs)
 
        batch1 = []
        batch2 = []

        for i in range(len(img_list)):
            first_crop = self.base_transform1(stained_batch1[i].cpu())
            batch1.append(first_crop)
            second_crop = self.base_transform2(stained_batch2[i].cpu())
            batch2.append(second_crop)

        batch1 = torch.stack(batch1)
        batch2 = torch.stack(batch2)

        return (batch1, batch2), None
