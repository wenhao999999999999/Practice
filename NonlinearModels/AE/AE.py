# -*- coding: utf-8 -*-
"""
自编码器(AutoEncoder) - Fashion-MNIST / MNIST
修复：输出与输入尺寸不一致的问题（28x28）。改为两次下采样(28->14->7)与两次上采样(7->14->28)。
功能:
1) 训练卷积自编码器并保存权重
2) 输出训练/验证损失曲线
3) 保存原图 vs 重建图对比
4) 评估测试集 MSE
默认输出目录: D:/WenHao/code/practice/NonlinearModels/AE
可传参: --epochs --batch_size --lr --latent --dataset
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

def get_default_outdir():
    win_path = r"D:\WenHao\code\practice\NonlinearModels\AE"
    try:
        os.makedirs(win_path, exist_ok=True)
        test_file = os.path.join(win_path, "__writable__.tmp")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return win_path
    except Exception:
        local = os.path.abspath("./AE_outputs")
        os.makedirs(local, exist_ok=True)
        return local

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class ConvAutoEncoder(nn.Module):
    """
    两层编码器/解码器，确保输入输出均为 [B,1,28,28]
    编码: 28 -> 14 -> 7
    解码: 7 -> 14 -> 28
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # [B,1,28,28] -> [B,16,14,14]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # [B,16,14,14] -> [B,32,7,7]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc_out_shape = (32, 7, 7)
        enc_feat_dim = 32 * 7 * 7
        self.to_latent = nn.Linear(enc_feat_dim, latent_dim)

        # Decoder
        self.from_latent = nn.Linear(latent_dim, enc_feat_dim)
        self.decoder = nn.Sequential(
            # [B,32,7,7] -> [B,16,14,14]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),
            # [B,16,14,14] -> [B,1,28,28]
            nn.ConvTranspose2d(16, 1,  kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)                 # [B,32,7,7]
        h = h.flatten(1)                    # [B, 1568]
        z = self.to_latent(h)               # [B, latent]
        return z

    def decode(self, z):
        h = self.from_latent(z)             # [B, 1568]
        h = h.view(-1, *self.enc_out_shape) # [B,32,7,7]
        x_hat = self.decoder(h)             # [B,1,28,28]
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        # 断言尺寸一致，便于早发现问题
        if x_hat.shape != x.shape:
            raise RuntimeError(f"Shape mismatch: pred={tuple(x_hat.shape)} vs target={tuple(x.shape)}")
        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        total_loss += criterion(x_hat, x).item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def visualize_reconstruction(model, loader, device, save_path, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    x_hat, _ = model(x)
    grid = utils.make_grid(torch.cat([x, x_hat], dim=0), nrow=n, padding=2)
    plt.figure(figsize=(n*1.5, 3))
    plt.axis("off")
    plt.title("Top: Original | Bottom: Reconstructed")
    plt.imshow(grid.cpu().permute(1, 2, 0).numpy(), interpolation='nearest')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("AE Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def get_data(dataset_name, out_dir, batch_size=256, val_ratio=0.1, num_workers=2):
    # 输入保持 28x28
    tfm = transforms.ToTensor()
    root = os.path.join(out_dir, "data")
    os.makedirs(root, exist_ok=True)

    name = dataset_name.lower()
    if name in ["fashion", "fashion-mnist", "fashion_mnist"]:
        full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)
    elif name in ["mnist"]:
        full = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset. Use 'fashion' or 'mnist'.")

    val_len = int(len(full) * val_ratio)
    train_len = len(full) - val_len
    train_set, val_set = random_split(full, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,      batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="fashion", choices=["fashion", "mnist"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = get_default_outdir()
    run_dir = os.path.join(out_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    train_loader, val_loader, test_loader = get_data(args.dataset, out_dir, batch_size=args.batch_size)

    model = ConvAutoEncoder(latent_dim=args.latent).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch:02d}/{args.epochs}] train={tr_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(run_dir, "ckpt", "best.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)

        vis_path = os.path.join(run_dir, "figs", f"recon_epoch_{epoch:02d}.png")
        visualize_reconstruction(model, val_loader, device, vis_path, n=8)

    curve_path = os.path.join(run_dir, "figs", "training_curve.png")
    plot_curves(train_losses, val_losses, curve_path)

    test_mse = evaluate(model, test_loader, criterion, device)
    print(f"[Test] MSE = {test_mse:.6f}")

    final_path = os.path.join(run_dir, "ckpt", "final.pth")
    torch.save({"model": model.state_dict(), "epochs": args.epochs, "test_mse": test_mse}, final_path)

    test_vis = os.path.join(run_dir, "figs", f"recon_test.png")
    visualize_reconstruction(model, test_loader, device, test_vis, n=8)

    with torch.no_grad():
        model.eval()
        zs, ys = [], []
        for x, y in test_loader:
            x = x.to(device)
            _, z = model(x)
            zs.append(z.cpu())
            ys.append(y)
        Z = torch.cat(zs, dim=0).numpy()
        Y = torch.cat(ys, dim=0).numpy()
        np.save(os.path.join(run_dir, "latent_test.npy"), Z)
        np.save(os.path.join(run_dir, "labels_test.npy"), Y)

    print("\n======= 运行完成 =======")
    print(f"输出目录: {run_dir}")
    print(f"- 最优权重: {os.path.join(run_dir, 'ckpt', 'best.pth')}")
    print(f"- 最终权重: {final_path}")
    print(f"- 训练曲线: {curve_path}")
    print(f"- 可视化: {test_vis}")

if __name__ == "__main__":
    main()
