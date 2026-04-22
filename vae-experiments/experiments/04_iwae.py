"""IWAE on MNIST (Burda et al., 2015).

Usage: uv run experiments/04_iwae.py
Compares IWAE bound with K ∈ {1, 5, 10, 50} samples. K=1 reduces to vanilla VAE.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

from common.data import get_mnist
from common.viz import save_reconstructions, save_samples, save_loss_curve


class IWAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, n_samples=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        mu, logvar = self.encode(x_flat)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn(batch_size, self.n_samples, self.latent_dim, device=x.device)
        z = mu.unsqueeze(1) + eps * std.unsqueeze(1)

        z_flat = z.view(-1, self.latent_dim)
        x_recon_flat = self.decoder(z_flat)
        x_recon = x_recon_flat.view(batch_size, self.n_samples, -1)

        return x_recon, mu, logvar, z, x_flat

    def loss_function(self, x_recon, mu, logvar, z, x_flat):
        x_expanded = x_flat.unsqueeze(1).expand_as(x_recon)
        log_p_x_z = -F.binary_cross_entropy_with_logits(x_recon, x_expanded, reduction="none").sum(-1)

        log_p_z = -0.5 * (self.latent_dim * torch.log(torch.tensor(2 * 3.14159265, device=z.device)) + (z ** 2).sum(-1))

        log_q_z_x = -0.5 * (
            self.latent_dim * torch.log(torch.tensor(2 * 3.14159265, device=z.device))
            + logvar.sum(-1).unsqueeze(1)
            + ((z - mu.unsqueeze(1)) ** 2 / torch.exp(logvar.unsqueeze(1)).clamp(min=1e-10)).sum(-1)
        )

        log_w = log_p_x_z + log_p_z - log_q_z_x
        log_w_max = log_w.max(dim=1, keepdim=True)[0]
        iwae_bound = (log_w_max.squeeze(1)
                      + torch.log(torch.exp(log_w - log_w_max).sum(dim=1))
                      - torch.log(torch.tensor(self.n_samples, dtype=torch.float32, device=z.device)))

        return {"loss": -iwae_bound.sum(), "iwae_bound": iwae_bound.mean().item()}

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decoder(z)).view(-1, 1, 28, 28)


def compute_test_ll(model, test_loader, device, n_samples=50):
    model.eval()
    total_ll, n = 0, 0
    orig_k = model.n_samples
    model.n_samples = n_samples
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, mu, logvar, z, x_flat = model(data)
            ll = model.loss_function(x_recon, mu, logvar, z, x_flat)["iwae_bound"]
            total_ll += ll * data.shape[0]
            n += data.shape[0]
    model.n_samples = orig_k
    return total_ll / n


def train_one_k(k, train_loader, test_loader, device, result_dir, epochs=20):
    k_dir = os.path.join(result_dir, f"K={k}")
    for d in ["checkpoints", "samples", "metrics"]:
        os.makedirs(os.path.join(k_dir, d), exist_ok=True)

    model = IWAE(n_samples=k, latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, z, x_flat = model(data)
            losses = model.loss_function(x_recon, mu, logvar, z, x_flat)
            losses["loss"].backward()
            optimizer.step()
            total_loss += losses["loss"].item()
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  K={k} Epoch {epoch}: Loss={avg_loss:.2f}")

    test_ll = compute_test_ll(model, test_loader, device, n_samples=50)
    print(f"  K={k}: Test log-likelihood (IWAE-50) = {test_ll:.2f}")

    torch.save(model.state_dict(), os.path.join(k_dir, "checkpoints", f"iwae_K{k}.pt"))

    data, _ = next(iter(test_loader))
    data = data[:8].to(device)
    with torch.no_grad():
        x_recon, mu, logvar, z, x_flat = model(data)
    save_reconstructions(data, torch.sigmoid(x_recon[:, 0, :]).view(-1, 1, 28, 28),
                        os.path.join(k_dir, "samples", "recon.png"))

    with torch.no_grad():
        samples = model.sample(64, device)
    save_samples(samples, os.path.join(k_dir, "samples", "prior_samples.png"), n=8)

    save_loss_curve(train_losses, None, os.path.join(k_dir, "metrics", "loss_curve.png"),
                    title=f"IWAE (K={k}) Loss")

    return test_ll


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "04-iwae")
    os.makedirs(result_dir, exist_ok=True)

    train_loader, test_loader = get_mnist(batch_size=128)

    ks = [1, 5, 10, 50]
    ll_results = {}
    for k in ks:
        print(f"\n{'='*40}")
        print(f"Training IWAE with K={k}")
        print(f"{'='*40}")
        ll_results[k] = train_one_k(k, train_loader, test_loader, device, result_dir, epochs=20)

    fig, ax = plt.subplots(figsize=(8, 5))
    ks_sorted = sorted(ll_results.keys())
    ax.plot(ks_sorted, [ll_results[k] for k in ks_sorted], "o-")
    ax.set_xlabel("K (number of importance samples)")
    ax.set_ylabel("Test Log-Likelihood (IWAE-50)")
    ax.set_title("IWAE: Log-Likelihood vs K")
    ax.set_xscale("log")
    plt.savefig(os.path.join(result_dir, "ll_vs_k.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "="*40)
    print("Summary (Test LL with IWAE-50):")
    for k in ks_sorted:
        print(f"  K={k:2d}: {ll_results[k]:.2f}")
    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
