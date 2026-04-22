"""Unified dataset loading. All data downloaded to project data/ directory."""

import os

# Set data directories before any torch imports
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")

os.environ["TORCH_HOME"] = _DATA_DIR
os.environ["HF_HOME"] = _DATA_DIR

import torch
from torch.utils.data import DataLoader, TensorDataset
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


def get_dsprites(batch_size: int = 128, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """Load dSprites dataset. Downloads from DeepMind if not cached."""
    import numpy as np
    import urllib.request

    data_path = os.path.join(_DATA_DIR, "dsprites_ndarray_co1sh3sc6or40x32x32_64x64.npz")
    url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32x32_64x64.npz"

    if not os.path.exists(data_path):
        print("Downloading dSprites dataset (~2.7GB)...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    dataset_zip = np.load(data_path, allow_pickle=True)
    imgs = dataset_zip["imgs"]  # (737280, 64, 64) bool
    imgs = torch.from_numpy(imgs).float().unsqueeze(1)  # (N, 1, 64, 64)

    n = len(imgs)
    n_train = int(0.9 * n)
    train_data = TensorDataset(imgs[:n_train])
    test_data = TensorDataset(imgs[n_train:])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
