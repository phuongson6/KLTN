"""
dataset.py - Tải và chuẩn bị dữ liệu CIFAR-10 (Sprint 1 / Gate 1).

CIFAR-10: 60.000 ảnh màu 32x32, 10 lớp (50.000 train / 10.000 test).
Output của DataLoader: images (B, 3, 32, 32), labels (B,)
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Đường dẫn gốc: PROJECT/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tự nhận biết môi trường: Kaggle (có GPU, chạy huấn luyện) hay máy cá nhân
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    # /kaggle/working là thư mục ghi được; file ở đây hiện trong tab Output để tải về
    DATA_DIR = "/kaggle/working/data"
    RESULTS_DIR = "/kaggle/working/results"
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Giá trị trung bình / độ lệch chuẩn của CIFAR-10 trên 3 kênh RGB
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả chạy lại giống nhau (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool = True):
    """Train: có tăng cường dữ liệu (lật ngang, cắt ngẫu nhiên). Test: chỉ chuẩn hóa."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_raw_datasets():
    """Tập dữ liệu thô (chưa transform) - dùng cho thống kê và vẽ hình."""
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True)
    return train_set, test_set


def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    """Trả về train_loader, test_loader. images: (B, 3, 32, 32), labels: (B,)."""
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=get_transforms(train=True))
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                transform=get_transforms(train=False))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
