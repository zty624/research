"""Visualization utilities for VAE experiments."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def save_reconstructions(originals: torch.Tensor, reconstructions: torch.Tensor,
                         save_path: str, n: int = 8, title: str = "Reconstructions"):
    originals = originals[:n].cpu()
    reconstructions = reconstructions[:n].cpu()
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    fig.suptitle(title)
    for i in range(n):
        img_o = originals[i].squeeze()
        img_r = reconstructions[i].squeeze()
        if img_o.ndim == 3 and img_o.shape[0] in (1, 3):
            img_o = img_o.permute(1, 2, 0)
            img_r = img_r.permute(1, 2, 0)
        axes[0, i].imshow(img_o, cmap="gray" if img_o.ndim == 2 else None)
        axes[0, i].axis("off")
        axes[1, i].imshow(img_r, cmap="gray" if img_r.ndim == 2 else None)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_latent_space(latents: np.ndarray, labels: np.ndarray,
                      save_path: str, title: str = "Latent Space", method: str = "pca"):
    from sklearn.decomposition import PCA
    if latents.shape[1] > 2:
        if method == "tsne":
            from sklearn.manifold import TSNE
            proj = TSNE(n_components=2, random_state=42).fit_transform(latents)
        else:
            proj = PCA(n_components=2).fit_transform(latents)
    else:
        proj = latents
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", alpha=0.5, s=1)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_samples(samples: torch.Tensor, save_path: str, n: int = 8, title: str = "Samples"):
    samples = samples[:n * n].cpu()
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
    fig.suptitle(title)
    for i in range(n):
        for j in range(n):
            img = samples[i * n + j].squeeze()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            axes[i, j].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[i, j].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_loss_curve(train_losses: list, val_losses: list | None,
                    save_path: str, title: str = "Loss Curve"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train")
    if val_losses is not None:
        ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_latent_traversal(decoder, latent_dim: int, save_path: str,
                          n_values: int = 11, range_val: float = 3.0,
                          title: str = "Latent Traversal", device: str | None = None,
                          image_shape: tuple | None = None):
    if device is None:
        try:
            device = next(decoder.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    fig, axes = plt.subplots(latent_dim, n_values, figsize=(n_values, latent_dim))
    if latent_dim == 1:
        axes = axes[np.newaxis, :]
    for dim in range(latent_dim):
        for i, val in enumerate(np.linspace(-range_val, range_val, n_values)):
            z = torch.zeros(1, latent_dim, device=device)
            z[0, dim] = val
            with torch.no_grad():
                sample = decoder(z).cpu().squeeze()
            if sample.ndim == 1 and image_shape is not None:
                sample = sample.view(*image_shape)
            if sample.ndim == 3 and sample.shape[0] in (1, 3):
                sample = sample.permute(1, 2, 0)
            axes[dim, i].imshow(sample, cmap="gray" if sample.ndim == 2 else None)
            axes[dim, i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
