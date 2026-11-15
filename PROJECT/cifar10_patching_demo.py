"""
cifar10_patching_demo.py - SPRINT 1 / GATE 1 (Dữ liệu - 1.0 điểm)

Việc script làm:
  1. Tải CIFAR-10, in thống kê (Check 1.1, 1.2)
  2. Vẽ biểu đồ phân bố 10 lớp       -> results/figures/class_distribution.png (Check 1.3)
  3. Vẽ lưới ảnh mẫu mỗi lớp           -> results/figures/sample_images.png
  4. Minh họa chia ảnh thành 64 patches -> results/figures/patch_grid_demo.png (Check 1.4)

Chạy:  python cifar10_patching_demo.py   (đứng trong thư mục PROJECT/)
"""
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # lưu file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import CLASSES, FIG_DIR, get_raw_datasets, set_seed

PATCH_SIZE = 4  # p = 4 -> lưới 8x8 = 64 patches (theo Tensor Shape Contract)


def save_fig(name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Đã lưu] {path}")


# ---------------------------------------------------------------- 1. Thống kê
def print_statistics(train_set, test_set) -> Counter:
    train_counts = Counter(train_set.targets)
    test_counts = Counter(test_set.targets)
    img = np.array(train_set[0][0])

    print("=" * 55)
    print(" THỐNG KÊ TẬP DỮ LIỆU CIFAR-10")
    print("=" * 55)
    print(f" Số ảnh train : {len(train_set):,}")
    print(f" Số ảnh test  : {len(test_set):,}")
    print(f" Kích thước   : {img.shape}  (H, W, C)")
    print(f" Số lớp       : {len(CLASSES)}")
    print("-" * 55)
    print(f" {'Lớp':<12}{'Train':>10}{'Test':>10}")
    for i, name in enumerate(CLASSES):
        print(f" {name:<12}{train_counts[i]:>10,}{test_counts[i]:>10,}")
    print("=" * 55)

    # Check 1.2 - kiểm tra tự động
    assert len(train_set) == 50_000 and len(test_set) == 10_000
    assert all(train_counts[i] == 5_000 for i in range(10)), "Train không cân bằng!"
    print(" [OK] Check 1.2: 50.000 train / 10.000 test, 5.000 ảnh/lớp\n")
    return train_counts


# ------------------------------------------------------ 2. Biểu đồ phân bố lớp
def plot_class_distribution(train_counts: Counter) -> None:
    counts = [train_counts[i] for i in range(10)]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(CLASSES, counts, color=plt.cm.tab10(np.arange(10)))
    for bar, c in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, c + 50, f"{c:,}",
                 ha="center", fontsize=9)
    plt.title("Phân bố số lượng ảnh theo lớp - CIFAR-10 (tập Train)")
    plt.xlabel("Lớp")
    plt.ylabel("Số lượng ảnh")
    plt.ylim(0, max(counts) * 1.15)
    plt.xticks(rotation=30)
    save_fig("class_distribution.png")


# ------------------------------------------------------- 3. Ảnh mẫu mỗi lớp
def plot_samples(train_set, per_class: int = 6) -> None:
    targets = np.array(train_set.targets)
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 1.2, 12))
    for c in range(10):
        idxs = np.random.choice(np.where(targets == c)[0], per_class, replace=False)
        for j, idx in enumerate(idxs):
            ax = axes[c, j]
            ax.imshow(train_set[idx][0])
            ax.axis("off")
        axes[c, 0].text(-5, 16, CLASSES[c], ha="right", va="center", fontsize=10)
    plt.suptitle("Ảnh mẫu CIFAR-10 (32x32)", y=0.92)
    save_fig("sample_images.png")


# --------------------------------------------- 4. Minh họa chia ảnh thành patch
def image_to_patches(img: torch.Tensor, p: int = PATCH_SIZE) -> torch.Tensor:
    """
    Chia ảnh thành các patch không chồng lấn (đúng như bước Patch Embedding sẽ làm).
    Input : img (3, 32, 32)
    Output: patches (64, 3, p, p)   với 64 = (32/p) * (32/p)
    """
    C, H, W = img.shape
    patches = img.unfold(1, p, p).unfold(2, p, p)          # (3, 8, 8, p, p)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, p, p)  # (64, 3, p, p)
    assert patches.shape == ((H // p) * (W // p), C, p, p)
    return patches


def plot_patch_grid(train_set, idx: int = 7) -> None:
    pil_img, label = train_set[idx]
    img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0  # (3,32,32)
    patches = image_to_patches(img)                     # (64, 3, 4, 4)
    n = 32 // PATCH_SIZE                                # 8

    fig = plt.figure(figsize=(15, 5))

    # (a) Ảnh gốc
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img.permute(1, 2, 0))
    ax1.set_title(f"(a) Ảnh gốc 32x32 - lớp '{CLASSES[label]}'")
    ax1.axis("off")

    # (b) Ảnh kẻ lưới + đánh số node
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img.permute(1, 2, 0))
    for k in range(1, n):
        ax2.axhline(k * PATCH_SIZE - 0.5, color="yellow", lw=0.8)
        ax2.axvline(k * PATCH_SIZE - 0.5, color="yellow", lw=0.8)
    for r in range(n):
        for c in range(n):
            ax2.text(c * PATCH_SIZE + 1.5, r * PATCH_SIZE + 1.5, str(r * n + c),
                     color="white", fontsize=6, ha="center", va="center",
                     bbox=dict(facecolor="black", alpha=0.4, pad=0.5, lw=0))
    ax2.set_title(f"(b) Lưới {n}x{n} = {n*n} patches ({PATCH_SIZE}x{PATCH_SIZE} px)")
    ax2.axis("off")

    # (c) Các patch tách rời = các Node của đồ thị
    sub = fig.add_gridspec(n, n, left=0.68, right=0.95, wspace=0.15, hspace=0.15)
    for i in range(n * n):
        ax = fig.add_subplot(sub[i // n, i % n])
        ax.imshow(patches[i].permute(1, 2, 0))
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.815, 0.93, "(c) 64 patches tách rời = 64 Nodes", ha="center", fontsize=12)

    save_fig("patch_grid_demo.png")
    print(f"  Mỗi patch: {PATCH_SIZE}x{PATCH_SIZE}x3 = {PATCH_SIZE*PATCH_SIZE*3} giá trị "
          f"-> sẽ được chiếu thành vector D=128 ở Sprint 2")


if __name__ == "__main__":
    set_seed(42)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("[1/4] Tải CIFAR-10 (lần đầu ~170MB)...")
    train_set, test_set = get_raw_datasets()
    print(" [OK] Check 1.1: tải dữ liệu thành công\n")

    counts = print_statistics(train_set, test_set)
    print("[2/4] Vẽ biểu đồ phân bố lớp...")
    plot_class_distribution(counts)
    print("[3/4] Vẽ ảnh mẫu...")
    plot_samples(train_set)
    print("[4/4] Vẽ minh họa chia patch...")
    plot_patch_grid(train_set)
    print("\n>>> GATE 1 HOÀN THÀNH: kiểm tra 3 ảnh trong results/figures/")
