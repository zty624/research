"""
Minimal GAN Reproduction
=========================
Reproduces core ideas from GAN literature:
1. Original GAN (1406.2661): adversarial min-max training
2. WGAN (1701.07875): Wasserstein distance for stable training
3. Spectral Normalization (1802.05983): discriminator regularization
4. Compare: vanilla GAN vs WGAN vs SN-GAN on 2D data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Generator ──

class Generator(nn.Module):
    def __init__(self, latent_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, z):
        return self.net(z)


# ── Discriminators ──

class Discriminator(nn.Module):
    """Standard discriminator for vanilla GAN."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class SNDiscriminator(nn.Module):
    """Discriminator with Spectral Normalization."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2, hidden)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, x):
        return self.net(x)


# ── Data ──

def sample_2d_data(n, device='cpu'):
    n1 = n // 4
    n2 = n // 4
    n3 = n // 4
    n4 = n - n1 - n2 - n3
    c1 = torch.randn(n1, 2, device=device) * 0.2 + torch.tensor([2.0, 2.0], device=device)
    c2 = torch.randn(n2, 2, device=device) * 0.2 + torch.tensor([-2.0, 2.0], device=device)
    c3 = torch.randn(n3, 2, device=device) * 0.2 + torch.tensor([-2.0, -2.0], device=device)
    c4 = torch.randn(n4, 2, device=device) * 0.2 + torch.tensor([2.0, -2.0], device=device)
    return torch.cat([c1, c2, c3, c4])


# ── Training Functions ──

def train_vanilla_gan(G, D, n_steps=5000, batch_size=256, lr=2e-4, latent_dim=2, device='cpu'):
    """Vanilla GAN: min-max game with BCE loss."""
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_losses, d_losses = [], []

    for step in range(n_steps):
        # Train D
        real = sample_2d_data(batch_size, device)
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z).detach()

        d_real = D(real)
        d_fake = D(fake)

        d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                 F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train G
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)
        d_fake = D(fake)

        g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    return g_losses, d_losses


def train_wgan(G, D, n_steps=5000, batch_size=256, lr=2e-4, latent_dim=2,
               n_critic=5, clip_value=0.01, device='cpu'):
    """WGAN: Wasserstein distance with weight clipping."""
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_losses, d_losses, w_distances = [], [], []

    for step in range(n_steps):
        # Train critic multiple times
        for _ in range(n_critic):
            real = sample_2d_data(batch_size, device)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z).detach()

            d_loss = D(fake).mean() - D(real).mean()  # Wasserstein distance (maximize)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train G
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)
        g_loss = -D(fake).mean()

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        w_distances.append(-d_loss.item())

    return g_losses, d_losses, w_distances


def train_sngan(G, D, n_steps=5000, batch_size=256, lr=2e-4, latent_dim=2, device='cpu'):
    """SN-GAN: Spectral Normalization for stable training."""
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_losses, d_losses = [], []

    for step in range(n_steps):
        # Train D
        real = sample_2d_data(batch_size, device)
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z).detach()

        d_real = D(real)
        d_fake = D(fake)

        # Hinge loss
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train G
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)
        d_fake = D(fake)

        g_loss = -d_fake.mean()

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    return g_losses, d_losses


# ── Evaluation ──

def compute_mode_coverage(samples, centers, threshold=1.0):
    """How many modes of the mixture does the generator cover?"""
    covered = 0
    for center in centers:
        dist = (samples - center.unsqueeze(0)).norm(dim=1)
        if (dist < threshold).any():
            covered += 1
    return covered / len(centers)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "20-gan"
    results_dir.mkdir(parents=True, exist_ok=True)

    latent_dim = 2
    n_steps = 5000

    # Train Vanilla GAN
    print("=== Training Vanilla GAN ===")
    G_vanilla = Generator(latent_dim).to(device)
    D_vanilla = Discriminator().to(device)
    vg_gl, vg_dl = train_vanilla_gan(G_vanilla, D_vanilla, n_steps, device=device)

    # Train WGAN
    print("\n=== Training WGAN ===")
    G_wgan = Generator(latent_dim).to(device)
    D_wgan = Discriminator().to(device)
    wg_gl, wg_dl, wg_wd = train_wgan(G_wgan, D_wgan, n_steps, device=device)

    # Train SN-GAN
    print("\n=== Training SN-GAN ===")
    G_sn = Generator(latent_dim).to(device)
    D_sn = SNDiscriminator().to(device)
    sn_gl, sn_dl = train_sngan(G_sn, D_sn, n_steps, device=device)

    # ── Evaluate ──
    centers = torch.tensor([[2., 2.], [-2., 2.], [-2., -2.], [2., -2.]], device=device)
    target = sample_2d_data(1000, device).cpu().numpy()

    with torch.no_grad():
        z = torch.randn(1000, latent_dim, device=device)
        s_vanilla = G_vanilla(z).cpu().numpy()
        s_wgan = G_wgan(z).cpu().numpy()
        s_sn = G_sn(z).cpu().numpy()

    mc_vanilla = compute_mode_coverage(G_vanilla(torch.randn(1000, latent_dim, device=device)), centers)
    mc_wgan = compute_mode_coverage(G_wgan(torch.randn(1000, latent_dim, device=device)), centers)
    mc_sn = compute_mode_coverage(G_sn(torch.randn(1000, latent_dim, device=device)), centers)

    print(f"\nMode Coverage: Vanilla={mc_vanilla:.2f}, WGAN={mc_wgan:.2f}, SN-GAN={mc_sn:.2f}")

    # ── Visualization ──
    window = 30

    # 1. Generated samples comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].scatter(target[:, 0], target[:, 1], alpha=0.2, s=3)
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(s_vanilla[:, 0], s_vanilla[:, 1], alpha=0.2, s=3, color='red')
    axes[1].set_title(f"Vanilla GAN (coverage={mc_vanilla:.0%})")
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(s_wgan[:, 0], s_wgan[:, 1], alpha=0.2, s=3, color='blue')
    axes[2].set_title(f"WGAN (coverage={mc_wgan:.0%})")
    axes[2].set_xlim(-5, 5); axes[2].set_ylim(-5, 5)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    axes[3].scatter(s_sn[:, 0], s_sn[:, 1], alpha=0.2, s=3, color='green')
    axes[3].set_title(f"SN-GAN (coverage={mc_sn:.0%})")
    axes[3].set_xlim(-5, 5); axes[3].set_ylim(-5, 5)
    axes[3].set_aspect('equal'); axes[3].grid(True, alpha=0.3)

    plt.suptitle("GAN Comparison: Generated Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "sample_comparison.png", dpi=150)
    plt.close()

    # 2. Training dynamics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vg_gl_s = np.convolve(vg_gl, np.ones(window)/window, mode='valid')
    wg_gl_s = np.convolve(wg_gl, np.ones(window)/window, mode='valid')
    sn_gl_s = np.convolve(sn_gl, np.ones(window)/window, mode='valid')

    axes[0].plot(vg_gl_s, label='Vanilla', color='red')
    axes[0].plot(wg_gl_s, label='WGAN', color='blue')
    axes[0].plot(sn_gl_s, label='SN-GAN', color='green')
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    vg_dl_s = np.convolve(vg_dl, np.ones(window)/window, mode='valid')
    wg_dl_s = np.convolve(wg_dl, np.ones(window)/window, mode='valid')
    sn_dl_s = np.convolve(sn_dl, np.ones(window)/window, mode='valid')

    axes[1].plot(vg_dl_s, label='Vanilla', color='red')
    axes[1].plot(wg_dl_s, label='WGAN', color='blue')
    axes[1].plot(sn_dl_s, label='SN-GAN', color='green')
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # WGAN Wasserstein distance
    wg_wd_s = np.convolve(wg_wd, np.ones(window)/window, mode='valid')
    axes[2].plot(wg_wd_s, color='blue')
    axes[2].set_title("WGAN: Wasserstein Distance Estimate")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("-D_loss (smoothed)")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("GAN Training Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_dynamics.png", dpi=150)
    plt.close()

    # 3. Mode coverage comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Vanilla GAN', 'WGAN', 'SN-GAN']
    coverages = [mc_vanilla, mc_wgan, mc_sn]
    colors = ['red', 'blue', 'green']
    ax.bar(methods, coverages, color=colors, alpha=0.7)
    ax.set_ylabel("Mode Coverage")
    ax.set_title("GAN: Mode Coverage (4-mode mixture)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(coverages):
        ax.text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "mode_coverage.png", dpi=150)
    plt.close()

    # 4. GAN objective visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    texts = [
        ("Vanilla GAN", "min_max E[log D(x)] + E[log(1-D(G(z)))]\n→ Mode collapse, unstable", 0.17, 'red'),
        ("WGAN", "min max E[D(G(z))] - E[D(x)]\n→ Weight clipping, stable\nWasserstein distance", 0.5, 'blue'),
        ("SN-GAN", "min max with ||D|| ≤ 1\n→ Spectral normalization\n→ Best stability", 0.83, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.7, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("GAN Evolution: From Instability to Stability", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "gan_evolution.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
