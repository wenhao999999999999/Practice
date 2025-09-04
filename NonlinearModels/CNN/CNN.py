# cnn_classic.py
# 说明：
# - 公开数据集：CIFAR-10（默认）或 MNIST（--dataset mnist）
# - 自动下载数据、训练、验证、测试，并保存最佳权重到 best_cnn.pt
# - 适配 CPU/GPU，支持自动混合精度（AMP）
# 运行示例：
#   python cnn_classic.py --dataset cifar10 --epochs 20 --batch-size 128 --lr 1e-3

import os
import math
import argparse
from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# 工具函数
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 使得 cudnn 可复现（会牺牲少许速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
# 经典小型 CNN（适配 CIFAR-10 / MNIST）
# - Block: Conv -> BN -> ReLU -> MaxPool
# - 最后做 AdaptiveAvgPool + 全连接
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 尺寸减半

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 不再池化，保留更多空间信息
        )
        # 自适应池化到 1x1，便于不同输入尺寸复用
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)          # [B, 128, H', W']
        x = self.gap(x)               # [B, 128, 1, 1]
        x = torch.flatten(x, 1)       # [B, 128]
        x = self.classifier(x)        # [B, num_classes]
        return x

# ----------------------------
# 数据加载
# ----------------------------
def get_dataloaders(dataset: str, data_dir: str, batch_size: int, num_workers: int):
    dataset = dataset.lower()
    if dataset not in {"cifar10", "mnist"}:
        raise ValueError("dataset 仅支持 cifar10 或 mnist")

    if dataset == "cifar10":
        # 标准均值方差
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
        in_channels, num_classes = 3, 10

    else:  # MNIST
        mean, std = (0.1307,), (0.3081,)
        train_tf = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_tf)
        test_set  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_tf)
        in_channels, num_classes = 1, 10

    # 划一小部分验证集（从训练集中切分 10%）
    val_size = max(1, int(0.1 * len(train_set)))
    train_size = len(train_set) - val_size
    g = torch.Generator().manual_seed(123)
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, in_channels, num_classes

# ----------------------------
# 训练与评估
# ----------------------------
@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    lr: float

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().item()
    return correct / targets.size(0)

def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        bs = labels.size(0)
        epoch_loss += loss.item() * bs
        epoch_acc  += accuracy_top1(logits, labels) * bs
        total      += bs

    return epoch_loss / total, epoch_acc / total

# ----------------------------
# 主训练入口
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Classic CNN on CIFAR-10 / MNIST (PyTorch)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"], help="选择数据集")
    parser.add_argument("--data-dir", type=str, default="./data", help="数据下载/缓存目录")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="优化器")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--step-lr", type=int, default=0, help="StepLR 的 step_size（0 表示不用 StepLR）")
    parser.add_argument("--cosine", action="store_true", help="使用 CosineAnnealingLR（与 StepLR 互斥）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 工作线程（Windows 可设 0/2）")
    parser.add_argument("--amp", action="store_true", help="启用自动混合精度（AMP）")
    parser.add_argument("--save-path", type=str, default="best_cnn.pt", help="最佳模型保存路径")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    train_loader, val_loader, test_loader, in_channels, num_classes = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )

    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    print(f"[Info] Model params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    # 学习率调度器（可选）
    scheduler = None
    if args.cosine:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.step_lr and args.step_lr > 0:
        scheduler = StepLR(optimizer, step_size=args.step_lr, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_acc = 0.0
    best_epoch = -1

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, scaler=None, train=False)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "args": vars(args),
            }, args.save_path)

        lr_now = optimizer.param_groups[0]["lr"]
        stat = TrainStats(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc, lr=lr_now)
        history.append(stat)

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  ValAcc={val_acc*100:.2f}%  LR={lr_now:.6f}")

    print(f"[Info] Best Val Acc = {best_val_acc*100:.2f}% @ epoch {best_epoch}, saved to: {args.save_path}")

    # 测试集评估（加载最佳权重）
    if os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Info] Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

    test_loss, test_acc = run_one_epoch(model, test_loader, criterion, optimizer, device, scaler=None, train=False)
    print(f"[Test] Loss={test_loss:.4f}  Acc={test_acc*100:.2f}%")

    # 可选：打印每类准确率（以 CIFAR-10/MNIST 的 10 类为例）
    try:
        model.eval()
        class_correct = torch.zeros(num_classes, dtype=torch.long)
        class_total = torch.zeros(num_classes, dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    class_total[t] += 1
                    class_correct[t] += int(t == p)
        # 类别名（仅对常见数据集提供）
        if args.dataset == "cifar10":
            classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        else:
            classes = [str(i) for i in range(10)]
        print("[Per-Class Acc]")
        for i, name in enumerate(classes):
            if class_total[i] > 0:
                acc_i = 100.0 * class_correct[i].item() / class_total[i].item()
                print(f"  {name:>10s}: {acc_i:5.2f}%  (n={class_total[i].item()})")
    except Exception as e:
        print(f"[Warn] Per-class accuracy skipped: {e}")

if __name__ == "__main__":
    main()
