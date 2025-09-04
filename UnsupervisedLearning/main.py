# -*- coding: utf-8 -*-
"""
Unsupervised Learning Benchmark (Clustering / Dim-Reduction / Density Estimation)
Author: ChatGPT
Usage:
  python run_unsupervised_benchmark.py

Outputs:
  outputs/
    <dataset>_embeddings.png          # PCA / t-SNE / (UMAP)
    <dataset>_clustering.png          # KMeans / Agglo / DBSCAN / GMM on PCA-2D
    <dataset>_density.png             # (2D) KDE heatmap + samples (if feasible)
  outputs/metrics_summary.csv         # ARI, NMI, Silhouette, n_clusters, noise%, gmm_test_ll
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

# ============ Optional imports ============
HAS_UMAP = False
try:
    import umap
    HAS_UMAP = True
except Exception:
    pass

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    pass

# --------- Utils ----------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_silhouette(X, labels):
    """Compute silhouette only when there are >=2 clusters and labels not all same."""
    labs = np.array(labels)
    unique = np.unique(labs)
    # DBSCAN may have -1 as noise; require at least 2 non-noise clusters or total >=2 clusters
    if len(unique) < 2:
        return np.nan
    # If all points are considered the same label (edge case)
    if np.all(labs == labs[0]):
        return np.nan
    # Silhouette cannot handle single-sample clusters robustly; try-catch
    try:
        return float(silhouette_score(X, labs, metric='euclidean'))
    except Exception:
        return np.nan

def num_clusters_and_noise(labels):
    labs = np.array(labels)
    uniq = np.unique(labs)
    n_noise = int(np.sum(labs == -1))
    # count clusters excluding noise (-1)
    n_clusters = int(np.sum(uniq[uniq != -1].shape[0]))
    noise_ratio = n_noise / len(labs)
    return n_clusters, noise_ratio

# --------- Datasets ----------
def prepare_datasets():
    """Return dict: name -> (X, y, n_classes)"""
    data = {}

    iris = load_iris()
    data['iris'] = (iris.data, iris.target, len(np.unique(iris.target)))

    wine = load_wine()
    data['wine'] = (wine.data, wine.target, len(np.unique(wine.target)))

    # digits: 8x8=64 features, 10 classes
    digits = load_digits()
    data['digits'] = (digits.data, digits.target, len(np.unique(digits.target)))

    return data

# --------- Dimensionality Reduction ----------
def embed_all(X_std, random_state=RANDOM_STATE):
    """Return dict of 2D embeddings: pca, tsne, (umap)"""
    out = {}
    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(X_std)
    out['pca'] = pca2

    tsne2 = TSNE(n_components=2, random_state=random_state, init='pca', perplexity=30).fit_transform(X_std)
    out['tsne'] = tsne2

    if HAS_UMAP:
        umap2 = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1).fit_transform(X_std)
        out['umap'] = umap2

    return out

def plot_embeddings(name, embeds, y_true, outdir):
    cols = list(embeds.keys())
    n = len(cols)
    plt.figure(figsize=(5*n, 4))
    for i, key in enumerate(cols, 1):
        E = embeds[key]
        plt.subplot(1, n, i)
        plt.scatter(E[:,0], E[:,1], c=y_true, s=12, alpha=0.8)
        plt.title(f'{name.upper()} - {key.upper()} (colored by true label)')
        plt.xlabel('Dim-1'); plt.ylabel('Dim-2')
        plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{name}_embeddings.png'), dpi=160)
    plt.close()

# --------- Clustering ----------
def auto_dbscan(X, pca_for_eps=None, random_state=RANDOM_STATE):
    """
    Auto-select eps by scanning a range on PCA(10) space if provided,
    target clusters in [2, 20], maximize silhouette.
    """
    if pca_for_eps is None:
        pca_for_eps = PCA(n_components=min(10, X.shape[1]), random_state=random_state).fit_transform(X)

    best = None
    best_score = -np.inf
    best_model = None

    for eps in np.linspace(0.2, 3.0, 15):
        model = DBSCAN(eps=float(eps), min_samples=5, n_jobs=-1)
        labels = model.fit_predict(pca_for_eps)
        n_clusters, _ = num_clusters_and_noise(labels)
        if n_clusters >= 2 and n_clusters <= 20:
            sc = safe_silhouette(pca_for_eps, labels)
            if not np.isnan(sc) and sc > best_score:
                best_score = sc
                best = (eps, labels)
                best_model = model

    if best_model is None:
        # fallback with default eps
        best_model = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1).fit(X)
    else:
        # refit on original space with chosen eps for consistency
        best_model = DBSCAN(eps=best[0], min_samples=5, n_jobs=-1).fit(X)

    return best_model

def run_clustering_suite(X_std, y_true, n_classes, name, embeds_for_plot, outdir):
    """
    Fit KMeans / Agglo / DBSCAN / GMM on standardized features.
    Plot predictions on PCA-2D space.
    Return metrics list of dicts.
    """
    metrics = []
    pca2 = embeds_for_plot['pca']  # 2D for visualization

    # 1) KMeans
    kmeans = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init='auto')
    pred_km = kmeans.fit_predict(X_std)
    metrics.append(dict(
        dataset=name, algo='KMeans',
        ari=adjusted_rand_score(y_true, pred_km),
        nmi=normalized_mutual_info_score(y_true, pred_km),
        silhouette=safe_silhouette(X_std, pred_km),
        n_clusters=len(np.unique(pred_km)),
        noise_ratio=0.0,
        gmm_test_ll=np.nan
    ))

    # 2) Agglomerative (Ward)
    agg = AgglomerativeClustering(n_clusters=n_classes, linkage='ward')
    pred_ag = agg.fit_predict(X_std)
    metrics.append(dict(
        dataset=name, algo='Agglomerative',
        ari=adjusted_rand_score(y_true, pred_ag),
        nmi=normalized_mutual_info_score(y_true, pred_ag),
        silhouette=safe_silhouette(X_std, pred_ag),
        n_clusters=len(np.unique(pred_ag)),
        noise_ratio=0.0,
        gmm_test_ll=np.nan
    ))

    # 3) DBSCAN (auto eps)
    dbs = auto_dbscan(X_std)
    pred_db = dbs.labels_
    ncls, noise_ratio = num_clusters_and_noise(pred_db)
    metrics.append(dict(
        dataset=name, algo='DBSCAN',
        ari=adjusted_rand_score(y_true, pred_db),
        nmi=normalized_mutual_info_score(y_true, pred_db),
        silhouette=safe_silhouette(X_std, pred_db),
        n_clusters=ncls,
        noise_ratio=noise_ratio,
        gmm_test_ll=np.nan
    ))

    # 4) GMM (clustering用)
    gmm_c = GaussianMixture(n_components=n_classes, covariance_type='full', random_state=RANDOM_STATE)
    pred_gmm = gmm_c.fit_predict(X_std)
    metrics.append(dict(
        dataset=name, algo='GMM(Cluster)',
        ari=adjusted_rand_score(y_true, pred_gmm),
        nmi=normalized_mutual_info_score(y_true, pred_gmm),
        silhouette=safe_silhouette(X_std, pred_gmm),
        n_clusters=len(np.unique(pred_gmm)),
        noise_ratio=0.0,
        gmm_test_ll=np.nan
    ))

    # ---- Plot predictions on PCA-2D ----
    preds = [('KMeans', pred_km), ('Agglo', pred_ag), ('DBSCAN', pred_db), ('GMM', pred_gmm)]
    plt.figure(figsize=(16, 10))
    rows = 2; cols = 2
    for i, (title, labels) in enumerate(preds, 1):
        plt.subplot(rows, cols, i)
        plt.scatter(pca2[:,0], pca2[:,1], c=labels, s=12, alpha=0.85)
        plt.title(f'{name.upper()} - {title} on PCA(2D)')
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{name}_clustering.png'), dpi=180)
    plt.close()

    return metrics

# --------- Density Estimation ----------
def run_density_estimation(X_std, name, outdir, y_true=None):
    """
    Fit GMM for likelihood on held-out test set.
    If X has >=2 dims, also try 2D KDE on PCA2 for visualization.
    """
    Xtr, Xte = train_test_split(X_std, test_size=0.3, random_state=RANDOM_STATE, shuffle=True)
    # Choose n_components by BIC scan small range (1..10)
    best_bic = np.inf; best = None
    for k in range(1, min(11, max(2, X_std.shape[1] + 1))):
        gm = GaussianMixture(n_components=k, covariance_type='full', random_state=RANDOM_STATE)
        gm.fit(Xtr)
        bic = gm.bic(Xtr)
        if bic < best_bic:
            best_bic = bic
            best = gm
    gmm = best
    test_ll = float(np.mean(gmm.score_samples(Xte)))  # average log-likelihood on test
    # Visualization via KDE on PCA-2D (if dim>=2)
    if X_std.shape[1] >= 2:
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_std)
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(pca2)
        # grid
        x_min, x_max = pca2[:,0].min()-1, pca2[:,0].max()+1
        y_min, y_max = pca2[:,1].min()-1, pca2[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = kde.score_samples(grid).reshape(xx.shape)

        plt.figure(figsize=(6,5))
        plt.contourf(xx, yy, zz, levels=30, alpha=0.8)
        plt.scatter(pca2[:,0], pca2[:,1], s=6, c='k', alpha=0.5)
        plt.title(f'{name.upper()} - KDE on PCA(2D)')
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{name}_density.png'), dpi=160)
        plt.close()

    return test_ll

# --------- (Optional) Simple Autoencoder on Digits ----------
class SimpleAE(nn.Module):
    def __init__(self, in_dim=64, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out, z

def run_autoencoder_digits(outdir):
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # normalize to [0,1]
    y = digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleAE(in_dim=64, latent=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, device=device)
    Xte_t = torch.tensor(Xte, device=device)

    model.train()
    for epoch in range(30):
        opt.zero_grad()
        out, _ = model(Xtr_t)
        loss = crit(out, Xtr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out_te, z_te = model(Xte_t)
        recon_mse = float(crit(out_te, Xte_t).item())

    # t-SNE on latent
    z_np = z_te.cpu().numpy()
    tsne2 = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(z_np)

    plt.figure(figsize=(6,5))
    plt.scatter(tsne2[:,0], tsne2[:,1], c=yte, s=10, alpha=0.85)
    plt.title(f'Digits AE latent (MSE={recon_mse:.4f})')
    plt.xlabel('t-SNE-1'); plt.ylabel('t-SNE-2'); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'digits_autoencoder_latent.png'), dpi=170)
    plt.close()

    return recon_mse

# --------- (Optional) Association Rules (synthetic) ----------
def run_association_rules_optional(outdir):
    """
    If mlxtend is installed, generate a small synthetic market-basket dataset,
    mine frequent itemsets and association rules; save CSVs.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        # Synthetic baskets: 1000 transactions, items A..H with correlated occurrences
        rng = np.random.default_rng(RANDOM_STATE)
        n_tx = 1000
        items = list('ABCDEFGH')
        P = np.array([0.35,0.30,0.25,0.2,0.15,0.12,0.1,0.08])
        # inject a relation: if A then B with prob 0.6
        baskets = []
        for _ in range(n_tx):
            pick = rng.random(len(items)) < P
            if pick[0] and rng.random() < 0.6:  # A implies B
                pick[1] = True
            baskets.append(pick.astype(int))
        df = pd.DataFrame(baskets, columns=items).astype(bool)
        freq = apriori(df, min_support=0.1, use_colnames=True)
        rules = association_rules(freq, metric='lift', min_threshold=1.0)
        freq.to_csv(os.path.join(outdir, 'assoc_frequent_itemsets.csv'), index=False)
        rules.to_csv(os.path.join(outdir, 'assoc_rules.csv'), index=False)
        return True
    except Exception:
        return False

# --------- Main Pipeline ----------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base_dir, 'outputs')
    ensure_dir(outdir)
    print(f'[INFO] Outputs will be saved to: {outdir}')

    datasets = prepare_datasets()
    all_metrics = []

    for name, (X, y, n_classes) in datasets.items():
        print(f'\n===== Dataset: {name} | n={X.shape[0]}, d={X.shape[1]}, classes={n_classes} =====')
        X_std = StandardScaler().fit_transform(X)

        # Embeddings
        embeds = embed_all(X_std)
        plot_embeddings(name, embeds, y_true=y, outdir=outdir)

        # Clustering suite
        metrics = run_clustering_suite(X_std, y_true=y, n_classes=n_classes, name=name,
                                       embeds_for_plot=embeds, outdir=outdir)
        all_metrics.extend(metrics)

        # Density estimation via GMM LL, KDE viz
        test_ll = run_density_estimation(X_std, name=name, outdir=outdir, y_true=y)
        # attach the test_ll to each algo row for this dataset as a reference
        for m in metrics:
            if m['dataset'] == name:
                m['gmm_test_ll'] = test_ll

        print(f'[OK] Finished {name}: saved embeddings/clustering/density plots.')

    # Optional: Autoencoder on digits
    if HAS_TORCH:
        print('[INFO] torch detected. Running a small autoencoder on digits...')
        mse = run_autoencoder_digits(outdir)
        print(f'[OK] AE reconstruction MSE (digits test): {mse:.6f}')
    else:
        print('[INFO] torch not found. Skipping autoencoder experiment.')

    # Optional: Association rules
    ar_ok = run_association_rules_optional(outdir)
    if ar_ok:
        print('[OK] Association rules CSVs saved.')
    else:
        print('[INFO] mlxtend not found. Skipping association rules (optional).')

    # Save metrics summary
    df = pd.DataFrame(all_metrics)
    # order columns nicely
    cols = ['dataset','algo','ari','nmi','silhouette','n_clusters','noise_ratio','gmm_test_ll']
    df = df[cols]
    df.sort_values(['dataset','algo'], inplace=True)
    csv_path = os.path.join(outdir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n[RESULT] Metrics saved to: {csv_path}')
    print(df)

if __name__ == '__main__':
    main()
