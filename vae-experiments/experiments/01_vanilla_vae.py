"""Vanilla VAE on MNIST (Kingma & Welling, 2013).

Usage: uv run experiments/01_vanilla_vae.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from common.data import get_mnist
from common.viz import save_reconstructions, save_latent_space, save_samples, save_loss_curve


class VanillaVAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        BCE = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return {"loss": BCE + KLD, "BCE": BCE, "KLD": KLD}

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z).view(-1, 1, 28, 28)


def train(epoch, model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)
        losses = model.loss_function(x_recon, data, mu, logvar)
        losses["loss"].backward()
        optimizer.step()
        total_loss += losses["loss"].item()
    avg = total_loss / len(loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {avg:.2f}")
    return avg


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            x_recon, mu, logvar = model(data)
            losses = model.loss_function(x_recon, data, mu, logvar)
            total_loss += losses["loss"].item()
    avg = total_loss / len(loader.dataset)
    print(f"  Val Loss = {avg:.2f}")
    return avg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "01-vanilla-vae")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    sample_dir = os.path.join(result_dir, "samples")
    metric_dir = os.path.join(result_dir, "metrics")
    for d in [ckpt_dir, sample_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    train_loader, test_loader = get_mnist(batch_size=128)
    model = VanillaVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, device)
        val_loss = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            data, _ = next(iter(test_loader))
            data = data.to(device)
            with torch.no_grad():
                x_recon, _, _ = model(data)
            save_reconstructions(data, x_recon.view(-1, 1, 28, 28),
                                os.path.join(sample_dir, f"recon_epoch{epoch}.png"))

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "vanilla_vae_final.pt"))

    with torch.no_grad():
        samples = model.sample(64, device)
    save_samples(samples, os.path.join(sample_dir, "prior_samples.png"), n=8)

    # Latent space visualization
    latents, labels = [], []
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            mu, _ = model.encode(data.to(device).view(-1, 784))
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    save_latent_space(latents, labels, os.path.join(metric_dir, "latent_pca.png"), method="pca")
    save_latent_space(latents, labels, os.path.join(metric_dir, "latent_tsne.png"), method="tsne")

    save_loss_curve(train_losses, val_losses, os.path.join(metric_dir, "loss_curve.png"))
    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
