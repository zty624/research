"""
Minimal VAE vs GAN Comparison on Image Generation
==================================================
Compares VAE (1312.6114) and GAN (1406.2661) on the same synthetic image data:
1. Train both VAE and GAN on synthetic pattern images (colored shapes)
2. Compare: sample quality, diversity, mode collapse, training stability
3. Latent space analysis: VAE is smooth/continuous, GAN has sharper
   but potentially less structured latent space
4. Show: generated samples side by side, latent interpolation comparison,
   training curves, diversity metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Data ──

def generate_pattern_dataset(n_samples=5000, img_size=28, device='cpu'):
    """Generate simple pattern images: colored shapes on dark background.
    4 modes: horizontal stripe, vertical stripe, circle, diagonal."""
    images = torch.zeros(n_samples, 1, img_size, img_size, device=device)
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    n_per_mode = n_samples // 4
    x = torch.linspace(-1, 1, img_size, device=device)
    y = torch.linspace(-1, 1, img_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    idx = 0
    # Mode 0: horizontal stripe
    for _ in range(n_per_mode):
        offset = torch.rand(1, device=device) * 0.4 - 0.2
        img = ((Y - offset).abs() < 0.15).float()
        images[idx, 0] = img
        labels[idx] = 0
        idx += 1

    # Mode 1: vertical stripe
    for _ in range(n_per_mode):
        offset = torch.rand(1, device=device) * 0.4 - 0.2
        img = ((X - offset).abs() < 0.15).float()
        images[idx, 0] = img
        labels[idx] = 1
        idx += 1

    # Mode 2: circle
    for _ in range(n_per_mode):
        r = torch.rand(1, device=device) * 0.3 + 0.2
        img = ((X**2 + Y**2 - r**2).abs() < 0.08).float()
        images[idx, 0] = img
        labels[idx] = 2
        idx += 1

    # Mode 3: diagonal
    for _ in range(n_per_mode):
        offset = torch.rand(1, device=device) * 0.4 - 0.2
        img = ((X - Y - offset).abs() < 0.12).float()
        images[idx, 0] = img
        labels[idx] = 3
        idx += 1

    # Fill remaining
    while idx < n_samples:
        images[idx, 0] = images[idx % n_per_mode, 0]
        labels[idx] = labels[idx % n_per_mode]
        idx += 1

    # Add small noise
    images = images + torch.randn_like(images) * 0.02
    images = images.clamp(0, 1)
    return images, labels


# ── VAE ──

class VAE(nn.Module):
    """Variational Autoencoder with convolutional encoder/decoder."""

    def __init__(self, latent_dim=16, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        ch = 32

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.ReLU(),
        )
        enc_out = (img_size // 8) ** 2 * ch * 4
        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, enc_out)
        self.ch4 = ch * 4
        self.dec_spatial = img_size // 8
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, self.ch4, self.dec_spatial, self.dec_spatial)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z)


# ── GAN ──

class GAN_Generator(nn.Module):
    """GAN generator with conv transpose layers."""

    def __init__(self, latent_dim=16, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        ch = 32
        self.fc = nn.Linear(latent_dim, ch * 4 * (img_size // 8) ** 2)
        self.ch4 = ch * 4
        self.dec_spatial = img_size // 8

        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.ch4, self.dec_spatial, self.dec_spatial)
        return self.net(h)


class GAN_Discriminator(nn.Module):
    """GAN discriminator with conv layers."""

    def __init__(self, img_size=28):
        super().__init__()
        ch = 32
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.LeakyReLU(0.2),
        )
        enc_out = (img_size // 8) ** 2 * ch * 4
        self.fc = nn.Linear(enc_out, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


# ── Training ──

def train_vae(model, data, n_epochs=100, batch_size=128, lr=2e-3, device='cpu'):
    """Train VAE with reconstruction + KL divergence loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses, recon_losses, kl_losses = [], [], []

    for epoch in range(n_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar, _ = model(batch)

            recon_loss = F.mse_loss(recon, batch, reduction='sum') / batch.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

        if epoch % 20 == 0:
            print(f"  [VAE] Epoch {epoch}: loss={loss.item():.2f} "
                  f"(recon={recon_loss.item():.2f}, kl={kl_loss.item():.2f})")

    return losses, recon_losses, kl_losses


def train_gan(G, D, data, n_epochs=100, batch_size=128, lr=2e-4,
              latent_dim=16, device='cpu'):
    """Train GAN with BCE loss."""
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    g_losses, d_losses = [], []

    for epoch in range(n_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            bs = batch.size(0)

            # Train D
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z).detach()
            d_real = D(batch)
            d_fake = D(fake)
            d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                     F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train G
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z)
            d_fake = D(fake)
            g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if epoch % 20 == 0:
            print(f"  [GAN] Epoch {epoch}: G_loss={g_loss.item():.4f}, "
                  f"D_loss={d_loss.item():.4f}")

    return g_losses, d_losses


# ── Evaluation ──

def compute_diversity(samples, n_bins=10):
    """Compute a simple diversity metric: coverage of mode grid."""
    # Flatten and discretize
    flat = samples.view(samples.size(0), -1)
    # Compute pairwise distances, measure average minimum distance
    n = min(500, flat.size(0))
    subset = flat[:n]
    dists = torch.cdist(subset, subset)
    # Exclude self-distances
    mask = ~torch.eye(n, dtype=torch.bool, device=samples.device)
    avg_min_dist = dists[mask].view(n, n-1).min(dim=1).values.mean().item()
    return avg_min_dist


def compute_mode_coverage(samples, labels, n_modes=4):
    """Check how many of the 4 modes are covered by generated samples."""
    # Use a simple classifier approach: for each generated sample,
    # find nearest real mode center
    if isinstance(samples, torch.Tensor):
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples
    # Approximate: check variety in image statistics
    # Use mean and std as 2D features
    if samples_np.ndim == 4:
        features = np.array([
            [s.mean(), s.std()] for s in samples_np
        ])
    else:
        return 1.0

    # Cluster and count distinct clusters (simple approach)
    from collections import Counter
    # Assign to grid cells
    x_bins = np.linspace(features[:, 0].min() - 0.01, features[:, 0].max() + 0.01, 3)
    y_bins = np.linspace(features[:, 1].min() - 0.01, features[:, 1].max() + 0.01, 3)
    xi = np.digitize(features[:, 0], x_bins)
    yi = np.digitize(features[:, 1], y_bins)
    cells = Counter(zip(xi, yi))
    # Count cells with more than 5% of samples
    threshold = len(features) * 0.05
    covered = sum(1 for c in cells.values() if c > threshold)
    return covered / n_modes


def compute_fid_simple(real, fake):
    """Simplified FID-like metric using feature statistics."""
    def feat_stats(x):
        flat = x.view(x.size(0), -1)
        return flat.mean(dim=0), flat.var(dim=0)

    mu_r, var_r = feat_stats(real)
    mu_f, var_f = feat_stats(fake)

    # Simplified Frechet distance
    diff = mu_r - mu_f
    fid = diff.dot(diff) + var_r.sum() + var_f.sum() - 2 * torch.sqrt(var_r * var_f + 1e-8).sum()
    return fid.item()


def latent_interpolation_vae(model, n_steps=10, device='cpu'):
    """Linear interpolation in VAE latent space."""
    z1 = torch.randn(1, model.latent_dim, device=device)
    z2 = torch.randn(1, model.latent_dim, device=device)
    alphas = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
    z_interp = z1 * (1 - alphas) + z2 * alphas

    with torch.no_grad():
        images = model.decode(z_interp)
    return images.cpu()


def latent_interpolation_gan(G, latent_dim, n_steps=10, device='cpu'):
    """Linear interpolation in GAN latent space."""
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
    z_interp = z1 * (1 - alphas) + z2 * alphas

    with torch.no_grad():
        images = G(z_interp)
    return images.cpu()


# ── Visualization ──

def save_image_grid(images, nrow, title, save_path):
    """Save a grid of images."""
    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=nrow, padding=2, normalize=True, value_range=(0, 1))
    ndarr = grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(nrow * 1.5, (len(images) // nrow + 1) * 1.5))
    ax.imshow(ndarr, cmap='gray' if ndarr.shape[2] == 1 else None)
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_all(vae, gan_G, vae_losses, vae_recon_l, vae_kl_l,
                  gan_gl, gan_dl, data, labels, latent_dim, device, save_dir):
    """Generate all comparison plots."""
    n_samples = 64
    img_size = data.size(2)

    # 1. Generated samples comparison
    with torch.no_grad():
        vae_samples = vae.sample(n_samples, device).cpu()
        gan_samples = gan_G(torch.randn(n_samples, latent_dim, device=device)).cpu()
    real_samples = data[:n_samples].cpu()

    save_image_grid(real_samples, 8, "Real Data", save_dir / "real_samples.png")
    save_image_grid(vae_samples, 8, "VAE Generated", save_dir / "vae_samples.png")
    save_image_grid(gan_samples, 8, "GAN Generated", save_dir / "gan_samples.png")

    # 2. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    window = min(50, len(vae_losses) // 10)
    if window > 1:
        vae_l_s = np.convolve(vae_losses, np.ones(window)/window, mode='valid')
        vae_r_s = np.convolve(vae_recon_l, np.ones(window)/window, mode='valid')
        vae_k_s = np.convolve(vae_kl_l, np.ones(window)/window, mode='valid')
    else:
        vae_l_s, vae_r_s, vae_k_s = vae_losses, vae_recon_l, vae_kl_l

    axes[0].plot(vae_l_s, label='Total', color='blue')
    axes[0].plot(vae_r_s, label='Reconstruction', color='green', alpha=0.7)
    axes[0].plot(vae_k_s, label='KL', color='red', alpha=0.7)
    axes[0].set_title("VAE Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if window > 1:
        gan_g_s = np.convolve(gan_gl, np.ones(window)/window, mode='valid')
        gan_d_s = np.convolve(gan_dl, np.ones(window)/window, mode='valid')
    else:
        gan_g_s, gan_d_s = gan_gl, gan_dl

    axes[1].plot(gan_g_s, label='Generator', color='blue')
    axes[1].plot(gan_d_s, label='Discriminator', color='red')
    axes[1].set_title("GAN Training Loss")
    axes[1].set_xlabel("Step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Loss stability comparison (variance of losses in second half)
    mid = len(vae_l_s) // 2
    vae_var = np.var(vae_l_s[mid:])
    mid_g = len(gan_g_s) // 2
    gan_var = np.var(gan_g_s[mid_g:])
    axes[2].bar(['VAE', 'GAN'], [vae_var, gan_var],
                color=['blue', 'red'], alpha=0.7)
    axes[2].set_title("Training Stability\n(Loss Variance, 2nd half)")
    axes[2].set_ylabel("Variance")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("VAE vs GAN: Training Dynamics", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()

    # 3. Latent space interpolation
    n_interp = 10
    vae_interp = latent_interpolation_vae(vae, n_interp, device)
    gan_interp = latent_interpolation_gan(gan_G, latent_dim, n_interp, device)

    fig, axes = plt.subplots(2, 1, figsize=(n_interp * 1.5, 3.5))
    from torchvision.utils import make_grid

    vae_grid = make_grid(vae_interp, nrow=n_interp, padding=2, normalize=True, value_range=(0, 1))
    axes[0].imshow(vae_grid.permute(1, 2, 0).numpy(), cmap='gray')
    axes[0].set_title("VAE: Latent Interpolation (smooth transitions)")
    axes[0].axis('off')

    gan_grid = make_grid(gan_interp, nrow=n_interp, padding=2, normalize=True, value_range=(0, 1))
    axes[1].imshow(gan_grid.permute(1, 2, 0).numpy(), cmap='gray')
    axes[1].set_title("GAN: Latent Interpolation (sharp transitions)")
    axes[1].axis('off')

    plt.suptitle("Latent Space Interpolation Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "latent_interpolation.png", dpi=150)
    plt.close()

    # 4. Latent space structure (t-SNE-like 2D visualization via first 2 dims)
    with torch.no_grad():
        vae_mu, _ = vae.encode(data[:500].to(device))
        vae_z = vae_mu.cpu().numpy()
        gan_z = torch.randn(500, latent_dim, device=device).cpu().numpy()

    l = labels[:500].cpu().numpy()
    mode_names = ['H-stripe', 'V-stripe', 'Circle', 'Diagonal']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode_id in range(4):
        mask = l == mode_id
        axes[0].scatter(vae_z[mask, 0], vae_z[mask, 1],
                        alpha=0.3, s=5, label=mode_names[mode_id])
    axes[0].set_title("VAE: Latent Space (mu, dim 0-1)")
    axes[0].set_xlabel("z[0]")
    axes[0].set_ylabel("z[1]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(gan_z[:, 0], gan_z[:, 1], alpha=0.2, s=5, color='gray')
    axes[1].set_title("GAN: Random Latent Samples (dim 0-1)")
    axes[1].set_xlabel("z[0]")
    axes[1].set_ylabel("z[1]")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Latent Space Structure", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "latent_space.png", dpi=150)
    plt.close()

    # 5. Metrics comparison bar chart
    with torch.no_grad():
        vae_s = vae.sample(500, device)
        gan_s = gan_G(torch.randn(500, latent_dim, device=device))
        real_500 = data[:500].to(device)

    vae_div = compute_diversity(vae_s)
    gan_div = compute_diversity(gan_s)
    vae_fid = compute_fid_simple(real_500, vae_s)
    gan_fid = compute_fid_simple(real_500, gan_s)
    vae_cov = compute_mode_coverage(vae_s, labels[:500])
    gan_cov = compute_mode_coverage(gan_s, labels[:500])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(['VAE', 'GAN'], [vae_div, gan_div], color=['blue', 'red'], alpha=0.7)
    axes[0].set_title("Sample Diversity\n(avg min distance)")
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(['VAE', 'GAN'], [vae_fid, gan_fid], color=['blue', 'red'], alpha=0.7)
    axes[1].set_title("FID-like Score\n(lower = better)")
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(['VAE', 'GAN'], [vae_cov, gan_cov], color=['blue', 'red'], alpha=0.7)
    axes[2].set_title("Mode Coverage\n(fraction of 4 modes)")
    axes[2].set_ylim(0, 1.3)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("VAE vs GAN: Quality & Diversity Metrics", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_comparison.png", dpi=150)
    plt.close()

    print(f"\n  Metrics Summary:")
    print(f"  {'Metric':<25} {'VAE':>10} {'GAN':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Diversity (avg min dist)':<25} {vae_div:>10.4f} {gan_div:>10.4f}")
    print(f"  {'FID-like score':<25} {vae_fid:>10.4f} {gan_fid:>10.4f}")
    print(f"  {'Mode coverage':<25} {vae_cov:>10.2%} {gan_cov:>10.2%}")
    print(f"  {'Loss stability (var)':<25} {vae_var:>10.4f} {gan_var:>10.4f}")


# ── Main ──

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results_dir = Path(__file__).parent / "results" / "80-vae-gan-compare"
    results_dir.mkdir(parents=True, exist_ok=True)

    latent_dim = 16
    img_size = 28
    n_epochs = 100

    # Generate data
    print("\n=== Generating Synthetic Pattern Data ===")
    data, labels = generate_pattern_dataset(5000, img_size, device)
    print(f"  Data shape: {data.shape}, Labels shape: {labels.shape}")

    # Train VAE
    print("\n=== Training VAE ===")
    vae = VAE(latent_dim, img_size).to(device)
    vae_losses, vae_recon_l, vae_kl_l = train_vae(
        vae, data, n_epochs=n_epochs, device=device
    )

    # Train GAN
    print("\n=== Training GAN ===")
    G = GAN_Generator(latent_dim, img_size).to(device)
    D = GAN_Discriminator(img_size).to(device)
    gan_gl, gan_dl = train_gan(
        G, D, data, n_epochs=n_epochs, latent_dim=latent_dim, device=device
    )

    # Visualize
    print("\n=== Generating Visualizations ===")
    visualize_all(vae, G, vae_losses, vae_recon_l, vae_kl_l,
                  gan_gl, gan_dl, data, labels, latent_dim, device, results_dir)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
