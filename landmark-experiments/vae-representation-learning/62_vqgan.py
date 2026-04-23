"""
Minimal VQGAN Reproduction
===========================
Reproduces core ideas from "Taming Transformers for High-Resolution Image Synthesis"
(Esser, Rombach & Ommer, 2020 — arXiv 2012.09841):
1. VQ-VAE with EMA codebook updates: discrete latent tokens from learned codebook
2. Patch discriminator: adversarial loss for perceptually sharp reconstructions
3. Adaptive weight lambda: balance reconstruction vs GAN loss via gradient norm ratio
4. Combined objective: L_recon + lambda * L_GAN + commitment + codebook losses
5. Compare: VQ-VAE (L2 only) vs VQGAN (L2 + GAN) — sharper, more perceptual
6. Visualize: reconstructions, codebook usage, training curves, generated samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Vector Quantizer with EMA ──

class EMAVectorQuantizer(nn.Module):
    """VQ with Exponential Moving Average codebook updates (VQ-VAE-2 style)."""
    def __init__(self, n_embeddings=512, embedding_dim=64, decay=0.99, epsilon=1e-5,
                 commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        embedding = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_embeddings))
        self.register_buffer('ema_weight', embedding.clone())

    def forward(self, z):
        """z: (B, D, H, W) — spatial feature map from encoder.
        Returns: z_q, vq_loss, encoding_indices, perplexity
        """
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

        # Nearest codebook lookup
        distances = (z_flat.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1)
        encoding_indices = distances.argmin(dim=1)  # (B*H*W,)
        z_q_flat = self.embedding[encoding_indices]  # (B*H*W, D)

        # EMA update (training only)
        if self.training:
            one_hot = F.one_hot(encoding_indices, self.n_embeddings).float()
            count = one_hot.sum(0)
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * count
            weight = one_hot.T @ z_flat
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * weight

            # Laplace smoothing
            n = self.ema_count.sum()
            count_smooth = (self.ema_count + self.epsilon) / (n + self.n_embeddings * self.epsilon) * n
            self.embedding = self.ema_weight / count_smooth.unsqueeze(-1)

        # Losses (commitment only with EMA; no explicit codebook loss)
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_st = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back
        z_q_st = z_q_st.view(B, H, W, D).permute(0, 3, 1, 2)
        encoding_indices = encoding_indices.view(B, H, W)

        # Perplexity
        avg_probs = F.one_hot(encoding_indices.reshape(-1), self.n_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, encoding_indices, perplexity


# ── Encoder ──

class Encoder(nn.Module):
    """3 conv blocks with stride-2 downsampling. Channels: 3 -> 64 -> 128 -> 256."""
    def __init__(self, in_channels=3, hidden_channels=[64, 128, 256]):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Decoder ──

class Decoder(nn.Module):
    """Mirror of encoder with stride-2 upsampling. Channels: 256 -> 128 -> 64 -> 3."""
    def __init__(self, out_channels=3, hidden_channels=[256, 128, 64]):
        super().__init__()
        layers = []
        ch_in = hidden_channels[0]
        for ch_out in hidden_channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        # Final layer to output channels
        layers.extend([
            nn.ConvTranspose2d(ch_in, out_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ── Patch Discriminator ──

class PatchDiscriminator(nn.Module):
    """4-layer patch discriminator classifying 8x8 patches as real/fake.
    Output is a spatial map of patch-wise real/fake scores.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        layers = [
            # Layer 1: no BatchNorm on first layer
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: output 1-channel patch map
            nn.Conv2d(base_channels * 4, 1, 4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── VQ-VAE (L2 only) ──

class VQVAEModel(nn.Module):
    """VQ-VAE: encoder + EMA quantizer + decoder, trained with L2 reconstruction only."""
    def __init__(self, n_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels=3, hidden_channels=[64, 128, 256])
        # Project to embedding_dim
        self.quant_conv = nn.Conv2d(256, embedding_dim, 1)
        self.post_quant_conv = nn.Conv2d(embedding_dim, 256, 1)
        self.vq = EMAVectorQuantizer(n_embeddings, embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(out_channels=3, hidden_channels=[256, 128, 64])

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, z_q):
        h = self.post_quant_conv(z_q)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        z_q, vq_loss, indices, perplexity = self.vq(z)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices, perplexity


# ── VQGAN (L2 + GAN) ──

class VQGANModel(VQVAEModel):
    """VQGAN: same architecture as VQ-VAE but trained with additional GAN loss."""
    def __init__(self, n_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__(n_embeddings, embedding_dim, commitment_cost)


# ── Adaptive Weight Calculation ──

def compute_adaptive_weight(recon_loss, gan_g_loss, generator_last_layer, delta=1e-6):
    """Compute lambda = ||grad_G L_recon|| / (||grad_G L_GAN|| + delta).
    Balances reconstruction and adversarial losses so neither dominates.
    """
    # Gradient norm of reconstruction loss w.r.t. last generator layer
    recon_grads = torch.autograd.grad(
        recon_loss, generator_last_layer, retain_graph=True
    )[0]
    recon_norm = recon_grads.pow(2).sum().sqrt() + delta

    # Gradient norm of GAN generator loss w.r.t. last generator layer
    gan_grads = torch.autograd.grad(
        gan_g_loss, generator_last_layer, retain_graph=True
    )[0]
    gan_norm = gan_grads.pow(2).sum().sqrt() + delta

    return recon_norm / gan_norm


# ── Training ──

def train_vqvae(model, train_loader, n_epochs=15, lr=2e-4, device='cpu'):
    """Train VQ-VAE with L2 reconstruction loss only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    recon_losses, vq_losses, perplexities, lambdas = [], [], [], []

    for epoch in range(n_epochs):
        epoch_recon, epoch_vq, epoch_perp, n_batches = 0, 0, 0, 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            x_recon, vq_loss, indices, perplexity = model(imgs)
            recon_loss = F.mse_loss(x_recon, imgs)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_vq += vq_loss.item()
            epoch_perp += perplexity.item()
            n_batches += 1

        recon_losses.append(epoch_recon / n_batches)
        vq_losses.append(epoch_vq / n_batches)
        perplexities.append(epoch_perp / n_batches)
        lambdas.append(0.0)  # no adaptive lambda for VQ-VAE
        print(f"  Epoch {epoch+1:2d} | Recon: {recon_losses[-1]:.4f} | "
              f"VQ: {vq_losses[-1]:.4f} | Perp: {perplexities[-1]:.1f}")

    return recon_losses, vq_losses, perplexities, lambdas


def train_vqgan(model, disc, train_loader, n_epochs=15, lr=2e-4, disc_lr=2e-4,
                device='cpu', disc_start_epoch=0):
    """Train VQGAN with L2 + GAN loss + adaptive weight lambda."""
    g_opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()) +
        list(model.quant_conv.parameters()) + list(model.post_quant_conv.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    d_opt = torch.optim.Adam(disc.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    recon_losses, vq_losses, gan_g_losses, gan_d_losses = [], [], [], []
    perplexities, lambdas = [], []

    for epoch in range(n_epochs):
        epoch_recon, epoch_vq, epoch_gan_g, epoch_gan_d = 0, 0, 0, 0
        epoch_perp, epoch_lambda, n_batches = 0, 0, 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            use_gan = epoch >= disc_start_epoch

            # ── Generator (encoder + quantizer + decoder) ──
            x_recon, vq_loss, indices, perplexity = model(imgs)
            recon_loss = F.mse_loss(x_recon, imgs)

            if use_gan:
                # GAN generator loss (non-saturating)
                disc_fake = disc(x_recon)
                gan_g_loss = F.binary_cross_entropy_with_logits(
                    disc_fake, torch.ones_like(disc_fake)
                )

                # Adaptive weight
                last_layer = model.decoder.net[-2].weight  # last conv before Tanh
                lam = compute_adaptive_weight(recon_loss, gan_g_loss, last_layer)

                g_loss = recon_loss + lam.detach() * gan_g_loss + vq_loss
            else:
                gan_g_loss = torch.tensor(0.0)
                lam = torch.tensor(0.0)
                g_loss = recon_loss + vq_loss

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            # ── Discriminator ──
            if use_gan:
                with torch.no_grad():
                    x_recon_det = model(imgs)[0].detach()
                disc_real = disc(imgs)
                disc_fake = disc(x_recon_det)
                d_loss_real = F.binary_cross_entropy_with_logits(
                    disc_real, torch.ones_like(disc_real)
                )
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    disc_fake, torch.zeros_like(disc_fake)
                )
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()
            else:
                d_loss = torch.tensor(0.0)

            epoch_recon += recon_loss.item()
            epoch_vq += vq_loss.item()
            epoch_gan_g += gan_g_loss.item() if isinstance(gan_g_loss, torch.Tensor) else gan_g_loss
            epoch_gan_d += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
            epoch_perp += perplexity.item()
            epoch_lambda += lam.item() if isinstance(lam, torch.Tensor) else lam
            n_batches += 1

        recon_losses.append(epoch_recon / n_batches)
        vq_losses.append(epoch_vq / n_batches)
        gan_g_losses.append(epoch_gan_g / n_batches)
        gan_d_losses.append(epoch_gan_d / n_batches)
        perplexities.append(epoch_perp / n_batches)
        lambdas.append(epoch_lambda / n_batches)

        gan_str = f" | GAN_G: {gan_g_losses[-1]:.4f} | GAN_D: {gan_d_losses[-1]:.4f}" if use_gan else ""
        print(f"  Epoch {epoch+1:2d} | Recon: {recon_losses[-1]:.4f} | "
              f"VQ: {vq_losses[-1]:.4f} | Perp: {perplexities[-1]:.1f} | "
              f"Lambda: {lambdas[-1]:.2f}{gan_str}")

    return recon_losses, vq_losses, perplexities, lambdas


# ── Visualization helpers ──

def save_reconstructions(models, test_imgs, results_dir, device, n=8):
    """Save side-by-side original vs reconstruction comparison."""
    fig, axes = plt.subplots(len(models) + 1, n, figsize=(2 * n, 2 * (len(models) + 1)))
    with torch.no_grad():
        # Originals
        for i in range(n):
            img = (test_imgs[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
            axes[0, i].imshow(np.clip(img, 0, 1))
            axes[0, i].axis('off')
        axes[0, 0].set_ylabel("Original", fontsize=9, fontweight='bold')

        # Reconstructions
        for row, (name, model) in enumerate(models.items(), start=1):
            for i in range(n):
                recon = model(test_imgs[i:i+1].to(device))[0]
                img = (recon[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
                axes[row, i].imshow(np.clip(img, 0, 1))
                axes[row, i].axis('off')
            color = 'red' if 'VQ-VAE' == name else 'blue'
            axes[row, 0].set_ylabel(name, fontsize=9, fontweight='bold', color=color)

    plt.suptitle("VQ-VAE vs VQGAN: Reconstruction Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "reconstructions.png", dpi=150)
    plt.close()


def save_codebook_usage(models, test_loader, results_dir, device, n_embeddings=512):
    """Visualize codebook utilization for each model."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, (name, model) in zip(axes, models.items()):
            all_indices = []
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                _, _, indices, _ = model(imgs)
                all_indices.append(indices.cpu().reshape(-1))
            all_indices = torch.cat(all_indices)

            counts = torch.bincount(all_indices, minlength=n_embeddings).float()
            counts_norm = counts / counts.sum()
            used = (counts > 0).sum().item()
            colors = ['steelblue' if c > 0 else 'lightgray' for c in counts_norm]
            ax.bar(range(n_embeddings), counts_norm.numpy(), color=colors, alpha=0.7, width=1.0)
            ax.set_title(f"{name}: {used}/{n_embeddings} codes used")
            ax.set_xlabel("Code Index")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "codebook_usage.png", dpi=150)
    plt.close()


def save_training_curves(vqvae_metrics, vqgan_metrics, results_dir):
    """Plot training curves comparing VQ-VAE and VQGAN."""
    vqvae_recon, vqvae_vq, vqvae_perp, _ = vqvae_metrics
    vqgan_recon, vqgan_vq, vqgan_perp, vqgan_lam = vqgan_metrics

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reconstruction loss
    axes[0, 0].plot(vqvae_recon, label='VQ-VAE (L2 only)', color='red')
    axes[0, 0].plot(vqgan_recon, label='VQGAN (L2 + GAN)', color='blue')
    axes[0, 0].set_title("Reconstruction Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # VQ loss
    axes[0, 1].plot(vqvae_vq, label='VQ-VAE', color='red')
    axes[0, 1].plot(vqgan_vq, label='VQGAN', color='blue')
    axes[0, 1].set_title("VQ Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Perplexity
    axes[1, 0].plot(vqvae_perp, label='VQ-VAE', color='red')
    axes[1, 0].plot(vqgan_perp, label='VQGAN', color='blue')
    axes[1, 0].axhline(y=512, color='gray', linestyle='--', alpha=0.5, label='Max (512)')
    axes[1, 0].set_title("Codebook Perplexity")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Perplexity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Adaptive lambda
    axes[1, 1].plot(vqgan_lam, label='Lambda (VQGAN)', color='blue')
    axes[1, 1].set_title("Adaptive Weight Lambda")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Lambda")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("VQ-VAE vs VQGAN: Training Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()


def save_generated_samples(model, results_dir, device, n=64, n_embeddings=512):
    """Generate samples by sampling random codebook indices and decoding."""
    model.eval()
    # CIFAR-10: 32x32 -> encoder downsamples 8x -> spatial 4x4
    H, W = 4, 4
    with torch.no_grad():
        # Sample random indices
        indices = torch.randint(0, n_embeddings, (n, H, W), device=device)
        # Look up embeddings
        z_q = model.vq.embedding[indices]  # (n, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2)     # (n, D, H, W)
        # Decode
        samples = model.decode(z_q)
        samples = (samples + 1) / 2  # [-1,1] -> [0,1]

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            img = samples[idx].cpu().permute(1, 2, 0).numpy()
            axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].axis('off')

    plt.suptitle("VQGAN: Randomly Sampled Codebook Decodes", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "generated_samples.png", dpi=150)
    plt.close()


def save_concept_diagram(results_dir):
    """Draw concept diagram explaining VQGAN key ideas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax in axes:
        ax.axis('off')

    # Left: VQ-VAE vs VQGAN
    texts_left = [
        ("VQ-VAE", "L_recon only\nBlurry reconstructions\nL2 = pixel-level MSE", 0.25, 'red'),
        ("VQGAN", "L_recon + lambda * L_GAN\n+ L_VQ\nSharp, perceptual", 0.75, 'blue'),
    ]
    for name, desc, x_pos, color in texts_left:
        axes[0].text(x_pos, 0.7, name, fontsize=14, fontweight='bold',
                     ha='center', va='center', color=color)
        axes[0].text(x_pos, 0.32, desc, fontsize=10, ha='center', va='center',
                     fontfamily='monospace', color=color,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    axes[0].set_title("Loss Function Comparison", fontsize=13, fontweight='bold')

    # Right: Adaptive lambda
    lambda_text = (
        "Adaptive Weight Lambda:\n"
        "lambda = ||grad L_recon|| / (||grad L_GAN|| + delta)\n\n"
        "If GAN loss gradients >> recon gradients:\n"
        "  lambda decreases -> prioritize reconstruction\n\n"
        "If recon gradients >> GAN gradients:\n"
        "  lambda increases -> prioritize perceptual quality"
    )
    axes[1].text(0.5, 0.5, lambda_text, fontsize=10, ha='center', va='center',
                 fontfamily='monospace', color='darkblue',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9))
    axes[1].set_title("Adaptive Weight Balancing", fontsize=13, fontweight='bold')

    plt.suptitle("VQGAN: Key Ideas (2012.09841)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "concept_diagram.png", dpi=150)
    plt.close()


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "62-vqgan"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # map to [-1, 1]
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # Small subset for fast CPU training
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    n_epochs = 15
    n_embeddings = 512
    embedding_dim = 64

    # 1. Train VQ-VAE (L2 only)
    print("=== Training VQ-VAE (L2 only) ===")
    vqvae = VQVAEModel(n_embeddings=n_embeddings, embedding_dim=embedding_dim).to(device)
    vqvae_metrics = train_vqvae(vqvae, train_loader, n_epochs=n_epochs, device=device)

    # 2. Train VQGAN (L2 + GAN with adaptive lambda)
    print("\n=== Training VQGAN (L2 + GAN) ===")
    vqgan = VQGANModel(n_embeddings=n_embeddings, embedding_dim=embedding_dim).to(device)
    disc = PatchDiscriminator(in_channels=3, base_channels=64).to(device)
    vqgan_metrics = train_vqgan(
        vqgan, disc, train_loader, n_epochs=n_epochs,
        disc_start_epoch=3, device=device
    )

    # ── Evaluate ──
    print("\n=== Evaluating ===")
    models = {'VQ-VAE': vqvae, 'VQGAN': vqgan}
    test_recons = {}
    test_perps = {}

    with torch.no_grad():
        test_imgs = next(iter(test_loader))[0][:64].to(device)
        for name, model in models.items():
            x_recon, vq_loss, indices, perp = model(test_imgs)
            mse = F.mse_loss(x_recon, test_imgs).item()
            test_recons[name] = mse
            test_perps[name] = perp.item()
            print(f"  {name}: Test MSE = {mse:.4f}, Perplexity = {perp.item():.1f}")

    # ── Visualizations ──
    print("\n=== Saving visualizations ===")

    # 1. Reconstructions
    with torch.no_grad():
        test_vis = next(iter(test_loader))[0][:8].to(device)
    save_reconstructions(models, test_vis, results_dir, device)

    # 2. Codebook usage
    save_codebook_usage(models, test_loader, results_dir, device, n_embeddings)

    # 3. Training curves
    save_training_curves(vqvae_metrics, vqgan_metrics, results_dir)

    # 4. Generated samples
    save_generated_samples(vqgan, results_dir, device, n_embeddings=n_embeddings)

    # 5. Concept diagram
    save_concept_diagram(results_dir)

    # 6. Quantitative comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    names = list(test_recons.keys())
    mses = [test_recons[n] for n in names]
    perps = [test_perps[n] for n in names]
    colors = ['red', 'blue']

    axes[0].bar(names, mses, color=colors, alpha=0.7)
    axes[0].set_title("Test Reconstruction MSE")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mses):
        axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

    axes[1].bar(names, perps, color=colors, alpha=0.7)
    axes[1].set_title("Test Codebook Perplexity")
    axes[1].set_ylabel("Perplexity")
    axes[1].axhline(y=n_embeddings, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(perps):
        axes[1].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')

    plt.suptitle("VQ-VAE vs VQGAN: Quantitative Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "quantitative_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
