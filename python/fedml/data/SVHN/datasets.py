import logging

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import SVHN


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def default_loader(path):
    return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class SVHN_truncated(data.Dataset):
    def __init__(
        self, root, dataidxs=None, split="train", transform=None, target_transform=None, download=False,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        svhn_dataobj = SVHN(self.root, self.split, self.transform, self.target_transform, self.download)

        data = svhn_dataobj.data
        target = np.array(svhn_dataobj.labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        # 对svhn数据做通道转换
        data=np.transpose(data, (0, 2, 3, 1))
        # logging.info("img_dimmision" + str(data.shape))
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # 对svhn数据做通道转换
        # img=np.transpose(img, (2, 0, 1, 3))
        # logging.info("img_dimmision" + str(img.shape))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
