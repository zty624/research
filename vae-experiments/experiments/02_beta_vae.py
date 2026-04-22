"""β-VAE on dSprites (Higgins et al., 2017).

Usage: uv run experiments/02_beta_vae.py
Sweeps β ∈ {1, 2, 4, 10} and compares disentanglement via latent traversal.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from common.data import get_dsprites
from common.viz import save_reconstructions, save_latent_traversal, save_loss_curve, save_samples


class BetaVAE(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 10, beta: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        BCE = F.binary_cross_entropy(x_recon, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return {"loss": BCE + self.beta * KLD, "BCE": BCE, "KLD": KLD}

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def train_one_beta(beta, train_loader, test_loader, device, result_dir, epochs=30):
    beta_dir = os.path.join(result_dir, f"beta={beta}")
    ckpt_dir = os.path.join(beta_dir, "checkpoints")
    sample_dir = os.path.join(beta_dir, "samples")
    metric_dir = os.path.join(beta_dir, "metrics")
    for d in [ckpt_dir, sample_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    model = BetaVAE(latent_dim=10, beta=beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (data,) in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            losses = model.loss_function(x_recon, data, mu, logvar)
            losses["loss"].backward()
            optimizer.step()
            total_loss += losses["loss"].item()
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (data,) in test_loader:
                data = data.to(device)
                x_recon, mu, logvar = model(data)
                val_loss += model.loss_function(x_recon, data, mu, logvar)["loss"].item()
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  β={beta} Epoch {epoch}: Train={train_loss:.1f} Val={val_loss:.1f}")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"beta_vae_beta{beta}.pt"))

    (data,) = next(iter(test_loader))
    data = data[:8].to(device)
    with torch.no_grad():
        x_recon, _, _ = model(data)
    save_reconstructions(data, x_recon, os.path.join(sample_dir, "recon.png"))

    save_latent_traversal(model.decode, model.latent_dim,
                          os.path.join(sample_dir, "traversal.png"), n_values=11)

    save_loss_curve(train_losses, val_losses, os.path.join(metric_dir, "loss_curve.png"),
                    title=f"β-VAE (β={beta}) Loss")

    return train_losses[-1], val_losses[-1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "02-beta-vae")
    os.makedirs(result_dir, exist_ok=True)

    print("Loading dSprites dataset...")
    train_loader, test_loader = get_dsprites(batch_size=256)

    betas = [1, 2, 4, 10]
    results = {}
    for beta in betas:
        print(f"\n{'='*40}")
        print(f"Training β-VAE with β={beta}")
        print(f"{'='*40}")
        t, v = train_one_beta(beta, train_loader, test_loader, device, result_dir, epochs=30)
        results[beta] = (t, v)

    print("\n" + "="*40)
    print("Summary:")
    for beta, (t, v) in results.items():
        print(f"  β={beta:2d}: Train={t:.1f}, Val={v:.1f}")
    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
