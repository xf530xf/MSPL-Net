from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiCDDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        
        imgA = Image.open(os.path.join(self.root, 'images', id)).convert('RGB')

        
        mask = np.array(Image.open(os.path.join(self.root, 'masks', id)).convert('L'))

        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))
        

        if self.mode == 'val':
            imgA, mask = normalize(imgA, mask)

            return imgA, mask, id
        
        imgA, mask = resize(imgA, mask, (0.8, 1.2))
        imgA,  mask = crop(imgA,  mask, self.size)
        imgA, mask = hflip(imgA, mask, p=0.5)

        if self.mode == 'train_l':
            imgA, mask = normalize(imgA, mask)
            return imgA,  mask

        imgA_w,imgA_s1, imgA_s2,imgA_s3 = deepcopy(imgA), deepcopy(imgA),deepcopy(imgA),deepcopy(imgA)


        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(imgA_s1.size[0], p=0.5)


        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=0.5)
        imgA_s2=cutout(imgA_s2)
        cutmix_box2 = obtain_cutmix_box(imgA_s2.size[0], p=0.5)
        

        if random.random() < 0.8:
            imgA_s3 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s3)
        imgA_s3 = blur(imgA_s3, p=0.5)
        imgA_s3=cutout(imgA_s3)
        cutout_box3 = obtain_cutmix_box(imgA_s3.size[0], p=0.5)

        
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        mask = torch.from_numpy(np.array(mask)).long()
        # print(mask.shape,"##################################################")
        ignore_mask[mask == 255] = 255

        return normalize(imgA_w), normalize(imgA_s1),normalize(imgA_s2),normalize(imgA_s3), ignore_mask, cutmix_box1, cutmix_box2, cutout_box3

    def __len__(self):
        return len(self.ids)
