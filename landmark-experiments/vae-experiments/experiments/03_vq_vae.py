"""VQ-VAE on MNIST and CIFAR-10 (van den Oord et al., 2017).

Usage: uv run experiments/03_vq_vae.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

from common.data import get_mnist, get_cifar10
from common.viz import save_reconstructions, save_samples, save_loss_curve


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)
        self.register_buffer("_ema_cluster_size", torch.zeros(n_embeddings))
        self._ema_w = nn.Parameter(self.embedding.weight.data.clone(), requires_grad=False)
        self._decay = 0.99
        self._eps = 1e-5

    def forward(self, z):
        flat_z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        distances = (flat_z.unsqueeze(1) - self.embedding.weight.unsqueeze(0)).pow(2).sum(-1)
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding(encoding_indices).view(z.permute(0, 2, 3, 1).shape)
        z_q = z_q.permute(0, 3, 1, 2)

        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z.detach())

        if self.training:
            one_hot = F.one_hot(encoding_indices, self.n_embeddings).float()
            cluster_sizes = one_hot.sum(0)
            self._ema_cluster_size.data = self._decay * self._ema_cluster_size + (1 - self._decay) * cluster_sizes
            n = self._ema_cluster_size.sum()
            self._ema_cluster_size = (self._ema_cluster_size + self._eps) / (n + self.n_embeddings * self._eps) * n
            dw = one_hot.T @ flat_z
            self._ema_w.data = self._decay * self._ema_w + (1 - self._decay) * dw
            self.embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        z_q_st = z + (z_q - z).detach()
        avg_probs = F.one_hot(encoding_indices, self.n_embeddings).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, commitment_loss, perplexity


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, n_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 3, 1, 1),
        )
        self.vq = VectorQuantizer(n_embeddings, embedding_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, commitment_loss, perplexity = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, commitment_loss, perplexity

    def loss_function(self, x_recon, x, commitment_loss):
        recon_loss = F.mse_loss(x_recon, x)
        return {"loss": recon_loss + commitment_loss, "recon": recon_loss, "commitment": commitment_loss}


def train_vqvae(dataset_name, model, train_loader, test_loader, device, result_dir, epochs):
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    sample_dir = os.path.join(result_dir, "samples")
    metric_dir = os.path.join(result_dir, "metrics")
    for d in [ckpt_dir, sample_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    train_losses, perplexities = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_perp = 0, 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, commitment_loss, perplexity = model(data)
            x_recon = torch.sigmoid(x_recon)
            losses = model.loss_function(x_recon, data, commitment_loss)
            losses["loss"].backward()
            optimizer.step()
            total_loss += losses["loss"].item()
            total_perp += perplexity.item()
        avg_loss = total_loss / len(train_loader)
        avg_perp = total_perp / len(train_loader)
        train_losses.append(avg_loss)
        perplexities.append(avg_perp)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{dataset_name}] Epoch {epoch}: Loss={avg_loss:.4f}, Perplexity={avg_perp:.1f}")
            data, _ = next(iter(test_loader))
            data = data[:8].to(device)
            with torch.no_grad():
                x_recon, _, _ = model(data)
                x_recon = torch.sigmoid(x_recon)
            save_reconstructions(data, x_recon,
                                os.path.join(sample_dir, f"recon_epoch{epoch}.png"))

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"vqvae_{dataset_name}.pt"))
    save_loss_curve(train_losses, None, os.path.join(metric_dir, "loss_curve.png"),
                    title=f"VQ-VAE ({dataset_name})")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(perplexities)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Codebook Perplexity")
    ax.set_title(f"VQ-VAE ({dataset_name}) Codebook Utilization")
    ax.axhline(y=model.vq.n_embeddings, color="r", linestyle="--", label="Max")
    ax.legend()
    plt.savefig(os.path.join(metric_dir, "codebook_perplexity.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "03-vq-vae")
    os.makedirs(result_dir, exist_ok=True)

    print("\n=== VQ-VAE on MNIST ===")
    train_loader, test_loader = get_mnist(batch_size=128)
    model = VQVAE(in_channels=1).to(device)
    train_vqvae("mnist", model, train_loader, test_loader, device,
                os.path.join(result_dir, "mnist"), epochs=30)

    print("\n=== VQ-VAE on CIFAR-10 ===")
    train_loader, test_loader = get_cifar10(batch_size=128)
    model = VQVAE(in_channels=3).to(device)
    train_vqvae("cifar10", model, train_loader, test_loader, device,
                os.path.join(result_dir, "cifar10"), epochs=50)

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
