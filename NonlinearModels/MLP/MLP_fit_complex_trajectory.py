#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch GPU MLP for fitting complex 2D trajectories: t -> (x, y)
Trajectories: lissajous / lemniscate / spiral / butterfly
Saves metrics, plots, and model checkpoint.
"""

import argparse, os, math, time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Trajectories ----------------
def traj_lissajous(t, ax=3, ay=2, dx=np.pi/3):
    x = np.sin(ax * t + dx)
    y = np.sin(ay * t)
    return x, y

def traj_lemniscate_bern(t, a=1.0):
    u = (t % (2*np.pi))
    mask = np.cos(2*u) >= 0
    r = np.zeros_like(u)
    r[mask] = a * np.sqrt(np.cos(2*u[mask]) + 1e-12)
    x = r * np.cos(u)
    y = r * np.sin(u)
    x[~mask] = 0.0
    y[~mask] = 0.0
    return x, y

def traj_log_spiral(t, a=0.1, b=0.20):
    r = a * np.exp(b * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

def traj_butterfly(t):
    A = np.exp(np.cos(t)) - 2*np.cos(4*t) - (np.sin(t/12))**5
    x = np.sin(t) * A
    y = np.cos(t) * A
    return x, y

TRAJ_FUNCS = {
    "lissajous": traj_lissajous,
    "lemniscate": traj_lemniscate_bern,
    "spiral": traj_log_spiral,
    "butterfly": traj_butterfly,
}

def add_noise(x, y, sigma=0.02, seed=42):
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, sigma, size=x.shape), y + rng.normal(0, sigma, size=y.shape)

def path_rmse(y_true_xy: np.ndarray, y_pred_xy: np.ndarray) -> float:
    d = np.linalg.norm(y_true_xy - y_pred_xy, axis=1)
    return float(np.sqrt(np.mean(d**2)))

# ---------------- Model ----------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=(256,128,64), out_dim=2, act="relu"):
        super().__init__()
        acts = dict(relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU)
        Act = acts.get(act.lower(), nn.ReLU)

        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), Act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------- Train / Eval ----------------
def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_model(model, X_np, Y_np, device) -> Tuple[float, float, float, float]:
    model.eval()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    pred = model(X).cpu().numpy()
    mse_x = mean_squared_error(Y_np[:,0], pred[:,0])
    mse_y = mean_squared_error(Y_np[:,1], pred[:,1])
    r2_x = r2_score(Y_np[:,0], pred[:,0])
    r2_y = r2_score(Y_np[:,1], pred[:,1])
    return mse_x, mse_y, r2_x, r2_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, default="butterfly",
                        choices=list(TRAJ_FUNCS.keys()))
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--tmax", type=float, default=24.0)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,128,64")
    parser.add_argument("--act", type=str, default="relu", choices=["relu","tanh","gelu"])
    parser.add_argument("--out_dir", type=str, default="torch_traj_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    t = np.linspace(0, args.tmax, args.n)
    x_true, y_true = TRAJ_FUNCS[args.traj](t)
    x_noisy, y_noisy = add_noise(x_true, y_true, sigma=args.noise, seed=args.seed)

    # Inputs/targets
    X = t.reshape(-1,1).astype(np.float32)
    Y = np.vstack([x_noisy, y_noisy]).T.astype(np.float32)
    Y_true = np.vstack([x_true, y_true]).T.astype(np.float32)  # for evaluation/plots

    # Standardize time input (常见小技巧，提升收敛稳定性）
    t_mean, t_std = X.mean(), X.std() + 1e-8
    Xn = (X - t_mean) / t_std

    # Split (shuffle)
    idx = np.arange(args.n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    split = int(args.n * (1 - args.val_ratio))
    tr_idx, va_idx = idx[:split], idx[split:]

    Xtr, Ytr = Xn[tr_idx], Y[tr_idx]
    Xva, Yva = Xn[va_idx], Y_true[va_idx]  # 验证用真值评估（不含噪），更客观

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    hidden = tuple(int(h.strip()) for h in args.hidden.split(",") if h.strip())
    model = MLP(in_dim=1, hidden=hidden, out_dim=2, act=args.act).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    train_curve, val_curve = [], []

    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        mse_x, mse_y, r2_x, r2_y = eval_model(model, Xva, Yva, device)
        val_metric = (mse_x + mse_y) / 2.0  # 简单汇总指标做早停

        train_curve.append(tr_loss)
        val_curve.append(val_metric)

        improved = val_metric < best_val - 1e-8
        if improved:
            best_val = val_metric
            bad = 0
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1

        if epoch % 20 == 0 or improved:
            print(f"[{epoch:4d}] train_loss={tr_loss:.6f}  "
                  f"val_mse_x={mse_x:.6f} val_mse_y={mse_y:.6f}  "
                  f"R2x={r2_x:.4f} R2y={r2_y:.4f}")

        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval on full test/val split
    with torch.no_grad():
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
        Ypred = model(Xva_t).cpu().numpy()

    mse_x = mean_squared_error(Yva[:,0], Ypred[:,0])
    mse_y = mean_squared_error(Yva[:,1], Ypred[:,1])
    r2_x  = r2_score(Yva[:,0], Ypred[:,0])
    r2_y  = r2_score(Yva[:,1], Ypred[:,1])
    prmse = path_rmse(Yva, Ypred)

    elapsed = time.time() - t0
    print("\n=== PyTorch MLP Complex Trajectory ===")
    print(f"Trajectory: {args.traj}")
    print(f"Hidden: {hidden}  Act: {args.act}  Epochs used: {len(train_curve)}  Time: {elapsed:.2f}s")
    print(f"MSE_x: {mse_x:.6f} | MSE_y: {mse_y:.6f}")
    print(f"R2_x : {r2_x:.4f} | R2_y : {r2_y:.4f}")
    print(f"Path RMSE: {prmse:.6f}")

    # ---------- Plots ----------
    # 1) XY trajectory on val
    plt.figure(figsize=(6,6))
    plt.scatter(Yva[:,0], Yva[:,1], s=6, alpha=0.6, label="True (val)")
    plt.scatter(Ypred[:,0], Ypred[:,1], s=6, alpha=0.6, label="Pred (MLP)")
    plt.title(f"XY Trajectory (val) - {args.traj}")
    plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.traj}_xy_val.png"), dpi=150)

    # 2) x(t), y(t) curves on val (sorted by t)
    # 还原 t_val
    t_val = X[va_idx, 0]
    idx_sort = np.argsort(t_val)
    t_val_sorted = t_val[idx_sort]
    x_true_sorted = Yva[idx_sort, 0]
    y_true_sorted = Yva[idx_sort, 1]
    x_pred_sorted = Ypred[idx_sort, 0]
    y_pred_sorted = Ypred[idx_sort, 1]

    plt.figure(figsize=(8,4))
    plt.plot(t_val_sorted, x_true_sorted, label="x_true", linewidth=1)
    plt.plot(t_val_sorted, x_pred_sorted, label="x_pred", linestyle="--", linewidth=1)
    plt.title("x(t): true vs pred (val)"); plt.xlabel("t"); plt.ylabel("x"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.traj}_xt_val.png"), dpi=150)

    plt.figure(figsize=(8,4))
    plt.plot(t_val_sorted, y_true_sorted, label="y_true", linewidth=1)
    plt.plot(t_val_sorted, y_pred_sorted, label="y_pred", linestyle="--", linewidth=1)
    plt.title("y(t): true vs pred (val)"); plt.xlabel("t"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.traj}_yt_val.png"), dpi=150)

    # 3) Training / validation curves
    plt.figure(figsize=(6,4))
    plt.plot(train_curve, label="train_loss")
    plt.plot(val_curve, label="val_metric(mse_x+y)/2")
    plt.title("Training Curve"); plt.xlabel("epoch"); plt.ylabel("loss / metric"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.traj}_training_curve.png"), dpi=150)

    # Save checkpoint
    ckpt_path = os.path.join(args.out_dir, f"{args.traj}_mlp.pt")
    torch.save({
        "model_state": model.state_dict(),
        "t_mean": float(t_mean),
        "t_std": float(t_std),
        "hidden": hidden,
        "act": args.act
    }, ckpt_path)
    print(f"Saved plots to {args.out_dir} and checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
