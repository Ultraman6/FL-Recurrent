import numpy as np
import torch.utils.data as data
from torchvision.datasets import CIFAR10, SVHN

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


class SVHN_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, split='train', transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.labels = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        svhn_dataobj = SVHN(self.root, self.split, self.transform, self.target_transform, self.download)

        data = svhn_dataobj.data
        targets = np.array(svhn_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        # 对svhn数据做通道转换
        data=np.transpose(data, (0, 2, 3, 1))
        return data, targets

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
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class SVHN_truncated_WO_reload(data.Dataset):
    def __init__(self, root, dataidxs=None, transform=None, target_transform=None, full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        data = self.full_dataset.data
        targets = np.array(self.full_dataset.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

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
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)
