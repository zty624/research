"""Simplified NVAE on CIFAR-10 (Vahdat & Kautz, 2020).

Usage: uv run experiments/05_nvae.py
Simplified hierarchical VAE with 2 groups of latent variables,
adapted for RTX 4060 (8GB VRAM).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from common.data import get_cifar10
from common.viz import save_reconstructions, save_samples, save_loss_curve


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels), Swish(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels), Swish(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class SimplifiedNVAE(nn.Module):
    """2-level hierarchical VAE: z1 (8x8 spatial) + z2 (global scalar)."""

    def __init__(self, in_channels=3, base_ch=64, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 32x32 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, 1, 1),
            ResidualBlock(base_ch),
            nn.Conv2d(base_ch, base_ch, 3, 2, 1),  # /2
            ResidualBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1),  # /4 -> 8x8
            ResidualBlock(base_ch * 2),
        )

        # Level 1: spatial latent at 8x8
        self.enc_mu1 = nn.Conv2d(base_ch * 2, latent_dim, 1)
        self.enc_logvar1 = nn.Conv2d(base_ch * 2, latent_dim, 1)

        # Level 2: global latent from z1 pooled
        self.enc_mu2 = nn.Linear(latent_dim * 8 * 8, latent_dim)
        self.enc_logvar2 = nn.Linear(latent_dim * 8 * 8, latent_dim)

        # Decoder: z2 expanded + z1 -> reconstruct
        self.dec_expand2 = nn.Linear(latent_dim, latent_dim * 8 * 8)

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * 2, base_ch * 2, 3, 1, 1),
            ResidualBlock(base_ch * 2),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),  # 8->16
            ResidualBlock(base_ch),
            nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1),  # 16->32
            ResidualBlock(base_ch),
            nn.Conv2d(base_ch, in_channels, 3, 1, 1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x)

        mu1 = self.enc_mu1(h)
        logvar1 = self.enc_logvar1(h)
        z1 = self.reparameterize(mu1, logvar1)

        z1_flat = z1.view(z1.shape[0], -1)
        mu2 = self.enc_mu2(z1_flat)
        logvar2 = self.enc_logvar2(z1_flat)
        z2 = self.reparameterize(mu2, logvar2)

        z2_expanded = self.dec_expand2(z2).view(-1, self.latent_dim, 8, 8)
        dec_input = torch.cat([z1, z2_expanded], dim=1)
        x_recon = self.decoder(dec_input)

        return x_recon, mu1, logvar1, mu2, logvar2

    def loss_function(self, x_recon, x, mu1, logvar1, mu2, logvar2):
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kld1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        kld2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        return {"loss": recon_loss + kld1 + kld2, "recon": recon_loss, "kld1": kld1, "kld2": kld2}

    def sample(self, n, device):
        z2 = torch.randn(n, self.latent_dim, device=device)
        z2_expanded = self.dec_expand2(z2).view(-1, self.latent_dim, 8, 8)
        z1 = torch.randn(n, self.latent_dim, 8, 8, device=device)
        dec_input = torch.cat([z1, z2_expanded], dim=1)
        return self.decoder(dec_input)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "05-nvae")
    for d in ["checkpoints", "samples", "metrics"]:
        os.makedirs(os.path.join(result_dir, d), exist_ok=True)

    train_loader, test_loader = get_cifar10(batch_size=16)

    model = SimplifiedNVAE(in_channels=3, base_ch=64, latent_dim=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs = 100
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, total_kld1, total_kld2 = 0, 0, 0, 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, mu1, logvar1, mu2, logvar2 = model(data)
            losses = model.loss_function(x_recon, data, mu1, logvar1, mu2, logvar2)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses["loss"].item()
            total_recon += losses["recon"].item()
            total_kld1 += losses["kld1"].item()
            total_kld2 += losses["kld2"].item()
        scheduler.step()

        n_train = len(train_loader.dataset)
        avg_loss = total_loss / n_train
        train_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: Loss={avg_loss:.2f} "
                  f"Recon={total_recon/n_train:.2f} "
                  f"KLD1={total_kld1/n_train:.2f} KLD2={total_kld2/n_train:.2f} "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")
            data, _ = next(iter(test_loader))
            data = data[:8].to(device)
            with torch.no_grad():
                x_recon, _, _, _, _ = model(data)
            save_reconstructions(data, x_recon,
                                os.path.join(result_dir, "samples", f"recon_epoch{epoch}.png"),
                                title=f"NVAE Epoch {epoch}")

    torch.save(model.state_dict(), os.path.join(result_dir, "checkpoints", "nvae_final.pt"))

    with torch.no_grad():
        samples = model.sample(64, device)
    save_samples(samples, os.path.join(result_dir, "samples", "prior_samples.png"), n=8,
                 title="NVAE Samples")

    save_loss_curve(train_losses, None, os.path.join(result_dir, "metrics", "loss_curve.png"),
                    title="NVAE Training Loss")

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
