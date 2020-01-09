import os
from os.path import join as pjoin
import collections
import torch
import numpy as np
from glob import glob
from random import shuffle
import re

from PIL import Image
from torch.utils import data
from torchvision import transforms

class ImageFolderLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=True,
        img_size=256,
        split="train",
        test_mode=False,
        img_norm=True,
        n_classes=2,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([125.08347, 124.99436, 124.99769])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.files = collections.defaultdict(list)
        self.n_classes = n_classes

        if not self.test_mode:
            for split in ["train", "test"]:
                file_list = []
                for sub_classes in os.listdir(self.root + split):
                    path = pjoin(root, split, sub_classes + "/*png")
                    file_list += glob(path)
                self.files[split] = file_list
            # self.setup_annotations()

        # normMean = [0.498, 0.497, 0.497]
        # normStd = [0.206, 0.206, 0.206]
        normMean = [0.498]
        normStd = [0.206]

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normMean, normStd),
            ]
        )

    def __len__(self):
        k = len(self.files[self.split])
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]

        im = Image.open(im_name)
        lbl = int(re.split(r'[/\\]', im_name)[-2])

        if self.is_transform:
            im = self.transform(im)
        return im, lbl

    def transform(self, img):
        if img.size == self.img_size:
            pass
        else:
            img = img.resize(self.img_size, Image.ANTIALIAS)  # uint8 with RGB mode
        img_rgb = self.tf(img)
        return img_rgb

