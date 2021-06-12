import os
import torch
from torchvision import transforms, datasets
from typing import List
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

# list of all datasets
DATASETS = ["imagenet", "cifar10"]

# CIFAR10


class CIFAR10_Trainset(Dataset):
    def __init__(self):
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.cifar10 = CIFAR10(
            root='datasets',
            download=True,
            train=True,
            transform=t_train
        )

    def __getitem__(self, index: slice):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIFAR10_Testset(Dataset):
    def __init__(self):
        self.cifar10 = CIFAR10(
            root='datasets',
            download=True,
            train=False,
            transform=transforms.ToTensor()
        )

    def __getitem__(self, index: slice):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


def cifar10(batch_sz: int) -> (DataLoader, DataLoader, int, int, int):
    """Return CIFAR10 dataloaders and metadata

    Args:
        batch_sz (int): desired training and testing batch size

    Returns:
        DataLoader, DataLoader, int, int, int: train and test loaders,
        input size, length of train and test sets
    """
    img_sz = [3, 32, 32]
    trainset, testset = CIFAR10_Trainset(), CIFAR10_Testset()
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=4, drop_last=False)
    return train_loader, test_loader, img_sz, len(trainset), len(testset)


# ImageNet
def ImageNet(
        batch_sz: int, directory: str
) -> (DataLoader, DataLoader, int, int, int):
    """Return ImageNet dataloaders and metadata

    Args:
        batch_sz (int): desired training and testing batch size
        directory (str): directory on disk must be provided

    Returns:
        DataLoader, DataLoader, int, int, int: train and test loaders,
        input size, length of train and test sets
    """
    img_sz = [3, 224, 224]
    trainset, testset = ImageNet_Trainset(
        directory), ImageNet_Testset(directory)
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=4)
    return train_loader, test_loader, img_sz, len(trainset), len(testset)


class ImageNet_Trainset(Dataset):
    def __init__(self, path: str):
        subdir = os.path.join(path, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.imgnet = datasets.ImageFolder(subdir, transform)

    def __getitem__(self, index: slice) -> (torch.Tensor, torch.Tensor, slice):
        data, target = self.imgnet[index]
        return data, target, index

    def __len__(self) -> int:
        return len(self.imgnet)


class ImageNet_Testset(Dataset):
    def __init__(self, path: str):
        subdir = os.path.join(path, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        indices = list(range(0, 50000, 100))
        testset = datasets.ImageFolder(subdir, transform)
        self.imgnet = torch.utils.data.Subset(testset, indices)

    def __getitem__(self, index: slice) -> (torch.Tensor, torch.Tensor, slice):
        data, target = self.imgnet[index]
        return data, target, index

    def __len__(self) -> int:
        return len(self.imgnet)


def get_dataset(dataset: str, split: str, folder: str = None) -> Dataset:
    """Return the dataset as a PyTorch Dataset object

    Args:
        dataset (str): dataset identifier, imagenet or cifar10
        split (str): train or test dataset
        folder (str, optional): if imagenet, its directory on disk must be
            provided

    Returns:
        Dataset: desired dataset object
    """
    if dataset == "imagenet":
        return _imagenet(split, directory=folder)
    elif dataset == "cifar10":
        return _cifar10(split)
    else:
        raise ValueError(
            "datasets available are 'cifar10' or 'imagenet'"
        )


def get_num_classes(dataset: str) -> int:
    """Return the number of classes in the dataset. 

    Args:
        dataset (str): string corresponding to imagenet or cifar10

    Returns:
        int: number of classes in dataset
    """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    else:
        raise ValueError(
            "datasets available are 'cifar10' or 'imagenet'"
        )


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer

    Args:
        dataset (str): string corresponding to imagenet or cifar10

    Returns:
        torch.nn.Module: dataset normalization layer
    """
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    else:
        raise ValueError(
            "datasets available are 'cifar10' or 'imagenet'"
        )


def _cifar10(split: str) -> Dataset:
    """Retrive the CIFAR10 training or testing dataset

    Args:
        split (str): train or test dataset

    Returns:
        Dataset: dataset object
    """
    if split == "train":
        return datasets.CIFAR10(
            "./dataset_cache",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        )
    elif split == "test":
        return datasets.CIFAR10(
            "./dataset_cache",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )


def _imagenet(split: str, directory: str) -> Dataset:
    """Retrive the ImageNet training or testing dataset

    Args:
        split (str): train or test dataset
        directory (str): ImageNet dataset folder on disk

    Returns:
        Dataset: dataset object
    """
    if directory is None:
        raise ValueError(
            "to use ImageNet, please provide its correct directory on disk."
        )

    if split == "train":
        subdir = os.path.join(directory, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(directory, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(
            "split parameter should be 'train' or 'test'"
        )

    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset
    mean and dividing by the dataset standard deviation.

    In order to certify radii in original coordinates rather than standardized
    coordinates, we add the Gaussian noise _before_ standardizing, which is
    why we have standardization be the first layer of the classifier rather
    than as a part of preprocessing as is typical.
    """

    def __init__(self, means: List[float], sds: List[float]):
        """
        Args:
            means (List[float]): the channel means
            sds (List[float]): the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass on this layer
        """
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat(
            (batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat(
            (batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
