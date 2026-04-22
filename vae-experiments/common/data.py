"""Unified dataset loading. All data downloaded to project data/ directory."""

import os

# Set data directories before any torch imports
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")

os.environ["TORCH_HOME"] = _DATA_DIR
os.environ["HF_HOME"] = _DATA_DIR

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(batch_size: int = 128, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=_DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=_DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_cifar10(batch_size: int = 128, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root=_DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=_DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
