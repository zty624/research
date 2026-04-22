"""
Minimal VQ-VAE Reproduction
============================
Reproduces core ideas from VQ-VAE (1711.00937, van den Oord et al.):
1. Vector Quantization: discrete latent codes via nearest-neighbor codebook lookup
2. Straight-Through Estimator: gradient flow through discrete bottleneck
3. Commitment loss: keep encoder outputs close to codebook entries
4. Compare: standard VAE (continuous) vs VQ-VAE (discrete)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Vector Quantizer ──

class VectorQuantizer(nn.Module):
    """Vector Quantization with straight-through estimator."""
    def __init__(self, n_embeddings=64, embedding_dim=16, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / n_embeddings, 1.0 / n_embeddings
        )

    def forward(self, z):
        """Quantize continuous latent z to nearest codebook entry.
        z: (B, D)
        Returns: z_q, vq_loss, encoding_indices, perplexity
        """
        # Find nearest codebook entry
        distances = (z.unsqueeze(1) - self.embedding.weight.unsqueeze(0)).pow(2).sum(-1)  # (B, K)
        encoding_indices = distances.argmin(dim=1)  # (B,)
        z_q = self.embedding(encoding_indices)  # (B, D)

        # Losses
        commitment_loss = F.mse_loss(z, z_q.detach())  # encoder → codebook
        codebook_loss = F.mse_loss(z_q, z.detach())     # codebook → encoder
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradient from z_q to z
        z_q_st = z + (z_q - z).detach()

        # Perplexity (codebook utilization)
        avg_probs = F.one_hot(encoding_indices, self.n_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, encoding_indices, perplexity


class EMAVectorQuantizer(nn.Module):
    """VQ with Exponential Moving Average updates (more stable)."""
    def __init__(self, n_embeddings=64, embedding_dim=16, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_embeddings))
        self.register_buffer('ema_weight', embedding.clone())

    def forward(self, z):
        distances = (z.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1)
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding[encoding_indices]

        # EMA update (only during training)
        if self.training:
            one_hot = F.one_hot(encoding_indices, self.n_embeddings).float()
            count = one_hot.sum(0)
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * count
            weight = one_hot.T @ z
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * weight

            # Laplace smoothing
            n = self.ema_count.sum()
            count_smooth = (self.ema_count + self.epsilon) / (n + self.n_embeddings * self.epsilon) * n
            self.embedding = self.ema_weight / count_smooth.unsqueeze(-1)

        # Commitment loss only (no codebook loss with EMA)
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = commitment_loss

        # Straight-through
        z_q_st = z + (z_q - z).detach()

        # Perplexity
        avg_probs = F.one_hot(encoding_indices, self.n_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, encoding_indices, perplexity


# ── Encoder / Decoder ──

class Encoder(nn.Module):
    def __init__(self, in_dim=784, hidden=256, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden=256, out_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


# ── Models ──

class VAEModel(nn.Module):
    """Standard VAE with continuous latent space."""
    def __init__(self, in_dim=784, hidden=256, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = Decoder(latent_dim, hidden, in_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        # KL divergence
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1).mean()
        return x_recon, kl


class VQVAEModel(nn.Module):
    """VQ-VAE with discrete latent codes."""
    def __init__(self, in_dim=784, hidden=256, latent_dim=16, n_embeddings=64,
                 vq_type='standard'):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, latent_dim)
        self.decoder = Decoder(latent_dim, hidden, in_dim)
        if vq_type == 'ema':
            self.vq = EMAVectorQuantizer(n_embeddings, latent_dim)
        else:
            self.vq = VectorQuantizer(n_embeddings, latent_dim)
        self.n_embeddings = n_embeddings

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices, perplexity = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices, perplexity


# ── Training ──

def train_vae(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_losses = []
    kl_losses = []

    for epoch in range(n_epochs):
        epoch_recon = 0
        epoch_kl = 0
        for bx, _ in train_loader:
            bx = bx.view(bx.shape[0], -1).to(device)
            x_recon, kl = model(bx)
            recon = F.mse_loss(x_recon, bx)
            loss = recon + 0.001 * kl  # β=0.001 for good recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_kl += kl.item()

        recon_losses.append(epoch_recon / len(train_loader))
        kl_losses.append(epoch_kl / len(train_loader))
        print(f"  Epoch {epoch+1} | Recon: {recon_losses[-1]:.4f} | KL: {kl_losses[-1]:.2f}")

    return recon_losses, kl_losses


def train_vqvae(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_losses = []
    vq_losses = []
    perplexities = []

    for epoch in range(n_epochs):
        epoch_recon = 0
        epoch_vq = 0
        epoch_perp = 0
        n_batches = 0
        for bx, _ in train_loader:
            bx = bx.view(bx.shape[0], -1).to(device)
            x_recon, vq_loss, indices, perplexity = model(bx)
            recon = F.mse_loss(x_recon, bx)
            loss = recon + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_vq += vq_loss.item()
            epoch_perp += perplexity.item()
            n_batches += 1

        recon_losses.append(epoch_recon / n_batches)
        vq_losses.append(epoch_vq / n_batches)
        perplexities.append(epoch_perp / n_batches)
        print(f"  Epoch {epoch+1} | Recon: {recon_losses[-1]:.4f} | VQ: {vq_losses[-1]:.4f} | Perp: {perplexities[-1]:.1f}")

    return recon_losses, vq_losses, perplexities


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "25-vq-vae"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    n_epochs = 10
    n_embeddings = 64
    latent_dim = 16

    # 1. Train VAE
    print("=== Training VAE (continuous) ===")
    vae = VAEModel(latent_dim=latent_dim).to(device)
    vae_recon, vae_kl = train_vae(vae, train_loader, n_epochs, device=device)

    # 2. Train VQ-VAE (standard)
    print("\n=== Training VQ-VAE (standard VQ) ===")
    vqvae = VQVAEModel(latent_dim=latent_dim, n_embeddings=n_embeddings, vq_type='standard').to(device)
    vq_recon, vq_vq, vq_perp = train_vqvae(vqvae, train_loader, n_epochs, device=device)

    # 3. Train VQ-VAE (EMA)
    print("\n=== Training VQ-VAE (EMA) ===")
    vqvae_ema = VQVAEModel(latent_dim=latent_dim, n_embeddings=n_embeddings, vq_type='ema').to(device)
    ema_recon, ema_vq, ema_perp = train_vqvae(vqvae_ema, train_loader, n_epochs, device=device)

    # ── Evaluation ──
    print("\n=== Evaluating ===")
    models = {'VAE': vae, 'VQ-VAE': vqvae, 'VQ-VAE (EMA)': vqvae_ema}
    test_recons = {}
    codebook_usages = {}

    with torch.no_grad():
        test_batch = next(iter(test_loader))[0][:64].view(64, -1).to(device)

        for name, model in models.items():
            if name == 'VAE':
                recon, _ = model(test_batch)
                test_recons[name] = F.mse_loss(recon, test_batch).item()
            else:
                recon, vq_loss, indices, perp = model(test_batch)
                test_recons[name] = F.mse_loss(recon, test_batch).item()
                # Codebook usage
                unique_codes = indices.unique().numel()
                codebook_usages[name] = unique_codes / n_embeddings

    for name, recon_err in test_recons.items():
        usage = codebook_usages.get(name, None)
        print(f"  {name}: Recon MSE = {recon_err:.4f}" +
              (f", Codebook Usage = {usage:.1%}" if usage else ""))

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(vae_recon, label='VAE', color='red')
    axes[0].plot(vq_recon, label='VQ-VAE', color='blue')
    axes[0].plot(ema_recon, label='VQ-VAE (EMA)', color='green')
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(vq_vq, label='VQ-VAE', color='blue')
    axes[1].plot(ema_vq, label='VQ-VAE (EMA)', color='green')
    axes[1].set_title("VQ Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(vq_perp, label='VQ-VAE', color='blue')
    axes[2].plot(ema_perp, label='VQ-VAE (EMA)', color='green')
    axes[2].axhline(y=n_embeddings, color='red', linestyle='--', alpha=0.5, label=f'Max ({n_embeddings})')
    axes[2].set_title("Codebook Perplexity")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Perplexity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("VQ-VAE vs VAE: Training Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Reconstruction samples
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    with torch.no_grad():
        test_imgs = next(iter(test_loader))[0][:8]

        # Original
        for i in range(8):
            axes[0, i].imshow(test_imgs[i, 0].numpy(), cmap='gray')
            axes[0, i].axis('off')
        axes[0, 0].set_ylabel("Original", fontsize=10, fontweight='bold')

        # VAE
        for i in range(8):
            recon, _ = vae(test_imgs[i].view(1, -1).to(device))
            axes[1, i].imshow(recon.view(28, 28).cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
        axes[1, 0].set_ylabel("VAE", fontsize=10, fontweight='bold', color='red')

        # VQ-VAE
        for i in range(8):
            recon, _, _, _ = vqvae(test_imgs[i].view(1, -1).to(device))
            axes[2, i].imshow(recon.view(28, 28).cpu().numpy(), cmap='gray')
            axes[2, i].axis('off')
        axes[2, 0].set_ylabel("VQ-VAE", fontsize=10, fontweight='bold', color='blue')

        # VQ-VAE EMA
        for i in range(8):
            recon, _, _, _ = vqvae_ema(test_imgs[i].view(1, -1).to(device))
            axes[3, i].imshow(recon.view(28, 28).cpu().numpy(), cmap='gray')
            axes[3, i].axis('off')
        axes[3, 0].set_ylabel("VQ-VAE\n(EMA)", fontsize=10, fontweight='bold', color='green')

    plt.suptitle("VQ-VAE: Reconstruction Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "reconstructions.png", dpi=150)
    plt.close()

    # 3. Codebook usage
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    with torch.no_grad():
        for idx, (model, name, ax) in enumerate([
            (vqvae, 'VQ-VAE', axes[0]),
            (vqvae_ema, 'VQ-VAE (EMA)', axes[1])
        ]):
            all_indices = []
            for bx, _ in test_loader:
                bx = bx.view(bx.shape[0], -1).to(device)
                _, _, indices, _ = model(bx)
                all_indices.append(indices.cpu())
            all_indices = torch.cat(all_indices)

            counts = torch.bincount(all_indices, minlength=n_embeddings).float()
            counts_norm = counts / counts.sum()
            colors = ['blue' if c > 0 else 'lightgray' for c in counts_norm]
            ax.bar(range(n_embeddings), counts_norm.numpy(), color=colors, alpha=0.7)
            used = (counts > 0).sum().item()
            ax.set_title(f"{name}: Codebook Usage ({used}/{n_embeddings} used)")
            ax.set_xlabel("Code Index")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "codebook_usage.png", dpi=150)
    plt.close()

    # 4. Latent space visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    with torch.no_grad():
        test_batch = next(iter(test_loader))
        imgs = test_batch[0].view(256, -1).to(device)
        labels = test_batch[1]

        # VAE latent space
        h = vae.encoder(imgs)
        mu = vae.fc_mu(h)
        from sklearn.decomposition import PCA
        vae_pca = PCA(n_components=2).fit_transform(mu.cpu().numpy())
        axes[0].scatter(vae_pca[:, 0], vae_pca[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
        axes[0].set_title("VAE Latent (μ)")
        axes[0].grid(True, alpha=0.3)

        # VQ-VAE latent (use encoder output before quantization)
        z = vqvae.encoder(imgs)
        z_pca = PCA(n_components=2).fit_transform(z.cpu().numpy())
        _, _, indices, _ = vqvae.vq(z)
        axes[1].scatter(z_pca[:, 0], z_pca[:, 1], c=indices.cpu().numpy(), cmap='tab20', alpha=0.5, s=5)
        axes[1].set_title("VQ-VAE Pre-Quant (colored by code)")
        axes[1].grid(True, alpha=0.3)

        # VQ-VAE discrete codes
        axes[2].scatter(z_pca[:, 0], z_pca[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
        axes[2].set_title("VQ-VAE Pre-Quant (colored by digit)")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle("Latent Space: VAE (continuous) vs VQ-VAE (discrete)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "latent_space.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    texts = [
        ("VAE", "z ~ N(μ, σ²)\nContinuous latent\nKL divergence", 0.17, 'red'),
        ("VQ-VAE", "z_q = e_k nearest\nDiscrete codebook\nCommitment loss", 0.5, 'blue'),
        ("VQ-VAE + EMA", "EMA codebook\nupdate\nMore stable", 0.83, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.7, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=11, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("From Continuous to Discrete Latent Spaces", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "vq_vae_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
