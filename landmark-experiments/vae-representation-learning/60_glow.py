"""
Minimal Glow Reproduction
==========================
Reproduces core ideas from Glow (1807.03039, Kingma & Dhariwal):
1. Invertible 1x1 Convolution: replaces channel permutation with learned
   invertible mixing via LU decomposition (W = PLU, log_det = sum(log|diag(U)|))
2. ActNorm: replaces BatchNorm with per-channel scale+bias, initialized from
   first batch statistics, then learned (forward: y = s*x + b, log_det = sum(log|s|))
3. Affine Coupling Layers (from RealNVP): split input, scale+shift one half
4. GlowStep = ActNorm -> Invertible1x1Conv -> AffineCoupling
5. Compare Glow vs simple RealNVP (no 1x1 conv, no actnorm) convergence speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── 2D Target Distributions ──

def sample_target(n, target='moons', device='cpu'):
    if target == 'moons':
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n, noise=0.05)
        return torch.tensor(data, dtype=torch.float32, device=device)
    elif target == 'circles':
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=n, noise=0.05, factor=0.5)
        return torch.tensor(data, dtype=torch.float32, device=device)
    elif target == 'mixture':
        n1 = n // 4
        n2 = n // 4
        n3 = n // 4
        n4 = n - n1 - n2 - n3
        c1 = torch.randn(n1, 2, device=device) * 0.2 + torch.tensor([2.0, 0.0], device=device)
        c2 = torch.randn(n2, 2, device=device) * 0.2 + torch.tensor([-2.0, 0.0], device=device)
        c3 = torch.randn(n3, 2, device=device) * 0.2 + torch.tensor([0.0, 2.0], device=device)
        c4 = torch.randn(n4, 2, device=device) * 0.2 + torch.tensor([0.0, -2.0], device=device)
        return torch.cat([c1, c2, c3, c4])
    else:  # pinwheel
        n_arms = 4
        n_per = n // n_arms
        arms = []
        for i in range(n_arms):
            angle = 2 * np.pi * i / n_arms
            r = torch.randn(n_per, device=device) * 0.3 + 1.0
            theta = torch.randn(n_per, device=device) * 0.3 + angle
            x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            arms.append(x)
        return torch.cat(arms)


# ── ActNorm ──

class ActNorm(nn.Module):
    """Activation Normalization (Glow's replacement for BatchNorm).
    Per-channel scale s and bias b, initialized from first batch statistics.
    Forward: y = s * x + b,  log_det = sum(log|s|) * spatial_size
    Inverse: x = (y - b) / s
    """
    def __init__(self, num_features):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(1, num_features, 1))
        self.b = nn.Parameter(torch.zeros(1, num_features, 1))
        self.register_buffer('initialized', torch.tensor(False))

    def initialize(self, x):
        """Initialize from first batch: s = 1/std, b = -mean/std."""
        with torch.no_grad():
            # x shape: (B, C, D) for our 2D case where D=1
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
            self.log_s.data.copy_(torch.log(1.0 / std))
            self.b.data.copy_(-mean / std)
            self.initialized.fill_(True)

    def forward(self, x):
        """x: (B, C, D). Returns (output, log_det)."""
        if not self.initialized:
            self.initialize(x)
        s = torch.exp(self.log_s)
        out = s * x + self.b
        # log_det per sample: sum over C and D dimensions
        log_det = self.log_s.sum() * x.shape[-1]  # spatial size = D
        log_det = log_det.expand(x.shape[0])
        return out, log_det

    def inverse(self, y):
        """Inverse: x = (y - b) / s."""
        s = torch.exp(self.log_s)
        return (y - self.b) / s


# ── Invertible 1x1 Convolution ──

class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 Convolution using LU decomposition (Glow).
    W = P @ L @ U where P is a fixed permutation, L is lower triangular
    with ones on diagonal, U is upper triangular.
    log_det = sum(log(|diag(U)|)) * spatial_size
    Inverse: W^{-1} = U^{-1} @ L^{-1} @ P^T
    """
    def __init__(self, num_features):
        super().__init__()
        # Random rotation matrix for initialization
        W = torch.linalg.qr(torch.randn(num_features, num_features))[0]
        # Extract permutation from sign of LU pivot
        P, L, U = torch.linalg.lu(W)
        # Register permutation as buffer (not learned)
        self.register_buffer('P', P)
        # Learnable parameters: L (lower triangular) and U (upper triangular)
        self.L = nn.Parameter(L)  # lower triangular
        self.U = nn.Parameter(U)  # upper triangular
        # Store L mask and U mask for enforcing triangular structure
        self.register_buffer('L_mask', torch.tril(torch.ones_like(L), diagonal=-1))
        self.register_buffer('U_mask', torch.triu(torch.ones_like(U), diagonal=1))
        # Diagonal of U stored in log space for numerical stability
        self.log_U_diag = nn.Parameter(torch.log(torch.abs(torch.diag(U)) + 1e-8))

    def _get_W(self):
        """Reconstruct W = P @ L @ U with triangular constraints."""
        L = self.L * self.L_mask + torch.eye(self.L.shape[0], device=self.L.device)
        U = self.U * self.U_mask + torch.diag(torch.exp(self.log_U_diag))
        return self.P @ L @ U

    def forward(self, x):
        """x: (B, C, D). Apply 1x1 conv: y = W @ x. Returns (output, log_det)."""
        W = self._get_W()
        # x: (B, C, D) -> conv along C dimension
        out = torch.einsum('cd,bde->be...', W, x.unsqueeze(-1)).squeeze(-1)
        # Reshape for clarity: (B, C, D) -> (B, D, C) -> matmul -> (B, D, C) -> (B, C, D)
        # Actually simpler: just use matmul on (B, C, D)
        # Let's do it directly: for each sample, W @ x[i] where x[i] is (C, D)
        out_unfold = (W @ x.reshape(x.shape[0], x.shape[1], -1))
        out = out_unfold.reshape_as(x)

        # log_det = sum(log(|diag(U)|)) * spatial_size
        log_det = self.log_U_diag.sum() * x.shape[-1]
        log_det = log_det.expand(x.shape[0])
        return out, log_det

    def inverse(self, y):
        """Inverse: x = W^{-1} @ y."""
        W = self._get_W()
        W_inv = torch.linalg.inv(W)
        x = W_inv @ y.reshape(y.shape[0], y.shape[1], -1)
        return x.reshape_as(y)


# ── Affine Coupling Layer ──

class AffineCoupling(nn.Module):
    """Affine Coupling Layer (from RealNVP, used in Glow).
    Split x into x1, x2 along channel dim.
    y1 = x1,  y2 = x2 * exp(s(x1)) + t(x1)
    log_det = sum(s(x1))
    """
    def __init__(self, num_features, hidden=64, flip=False):
        super().__init__()
        self.flip = flip
        half = num_features // 2
        self.net_s = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, half)
        )
        self.net_t = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, half)
        )
        # Initialize s near zero for stable training
        nn.init.zeros_(self.net_s[-1].weight)
        nn.init.zeros_(self.net_s[-1].bias)

    def forward(self, x):
        """x: (B, C, D). Returns (output, log_det)."""
        # Flatten spatial dims for the MLP, then reshape back
        B, C, D = x.shape
        x_flat = x.reshape(B, C * D)  # (B, C*D) for 2D case

        if self.flip:
            x1, x2 = x_flat[:, 1:], x_flat[:, :1]
        else:
            x1, x2 = x_flat[:, :1], x_flat[:, 1:]

        s = self.net_s(x1)
        t = self.net_t(x1)

        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        if self.flip:
            y_flat = torch.cat([y2, x1], dim=1)
        else:
            y_flat = torch.cat([x1, y2], dim=1)

        return y_flat.reshape(B, C, D), log_det

    def inverse(self, y):
        """Inverse: x2 = (y2 - t(y1)) / exp(s(y1))."""
        B, C, D = y.shape
        y_flat = y.reshape(B, C * D)

        if self.flip:
            y1, y2 = y_flat[:, 1:], y_flat[:, :1]
        else:
            y1, y2 = y_flat[:, :1], y_flat[:, 1:]

        s = self.net_s(y1)
        t = self.net_t(y1)

        x2 = (y2 - t) * torch.exp(-s)

        if self.flip:
            x_flat = torch.cat([x2, y1], dim=1)
        else:
            x_flat = torch.cat([y1, x2], dim=1)

        return x_flat.reshape(B, C, D)


# ── Glow Step and Model ──

class GlowStep(nn.Module):
    """One step of Glow: ActNorm -> Invertible1x1Conv -> AffineCoupling."""
    def __init__(self, num_features, hidden=64, flip=False):
        super().__init__()
        self.actnorm = ActNorm(num_features)
        self.conv1x1 = Invertible1x1Conv(num_features)
        self.coupling = AffineCoupling(num_features, hidden, flip)

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        x, ld = self.actnorm(x)
        log_det_total += ld
        x, ld = self.conv1x1(x)
        log_det_total += ld
        x, ld = self.coupling(x)
        log_det_total += ld
        return x, log_det_total

    def inverse(self, y):
        y = self.coupling.inverse(y)
        y = self.conv1x1.inverse(y)
        y = self.actnorm.inverse(y)
        return y


class GlowModel(nn.Module):
    """Glow model: sequence of GlowSteps with forward and inverse."""
    def __init__(self, dim=2, n_steps=6, hidden=64):
        super().__init__()
        self.dim = dim
        # For 2D data, we use C=2 channels with D=1 spatial dim
        # So input shape is (B, 2, 1)
        self.steps = nn.ModuleList([
            GlowStep(dim, hidden, flip=(i % 2 == 1))
            for i in range(n_steps)
        ])

    def forward(self, x):
        """x: (B, 2). Returns (z, log_det_total)."""
        # Reshape to (B, C, D) format
        x = x.unsqueeze(-1)  # (B, 2, 1)
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for step in self.steps:
            x, ld = step(x)
            log_det_total += ld
        z = x.squeeze(-1)  # (B, 2)
        return z, log_det_total

    def inverse(self, z):
        """z: (B, 2). Returns x: (B, 2)."""
        z = z.unsqueeze(-1)  # (B, 2, 1)
        for step in reversed(self.steps):
            z = step.inverse(z)
        return z.squeeze(-1)

    def log_prob(self, x):
        """Compute log p(x) using change of variables."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - self.dim * 0.5 * np.log(2 * np.pi)
        return log_pz + log_det

    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.dim, device=device)
        with torch.no_grad():
            return self.inverse(z)


# ── Baseline: Simple RealNVP (no ActNorm, no 1x1 Conv) ──

class SimpleRealNVPStep(nn.Module):
    """Simple RealNVP: just AffineCoupling, no ActNorm, no 1x1 Conv."""
    def __init__(self, dim=2, hidden=64, flip=False):
        super().__init__()
        self.dim = dim
        self.flip = flip
        half = dim // 2
        self.net_s = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, half)
        )
        self.net_t = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, half)
        )
        nn.init.zeros_(self.net_s[-1].weight)
        nn.init.zeros_(self.net_s[-1].bias)

    def forward(self, x):
        """x: (B, dim). Returns (output, log_det)."""
        if self.flip:
            x1, x2 = x[:, 1:], x[:, :1]
        else:
            x1, x2 = x[:, :1], x[:, 1:]

        s = self.net_s(x1)
        t = self.net_t(x1)

        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        if self.flip:
            y = torch.cat([y2, x1], dim=1)
        else:
            y = torch.cat([x1, y2], dim=1)

        return y, log_det

    def inverse(self, y):
        if self.flip:
            y1, y2 = y[:, 1:], y[:, :1]
        else:
            y1, y2 = y[:, :1], y[:, 1:]

        s = self.net_s(y1)
        t = self.net_t(y1)

        x2 = (y2 - t) * torch.exp(-s)

        if self.flip:
            x = torch.cat([x2, y1], dim=1)
        else:
            x = torch.cat([y1, x2], dim=1)

        return x


class SimpleRealNVP(nn.Module):
    """Simple RealNVP without ActNorm or 1x1 Convolution."""
    def __init__(self, dim=2, n_steps=6, hidden=64):
        super().__init__()
        self.dim = dim
        self.steps = nn.ModuleList([
            SimpleRealNVPStep(dim, hidden, flip=(i % 2 == 1))
            for i in range(n_steps)
        ])

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for step in self.steps:
            x, ld = step(x)
            log_det_total += ld
        return x, log_det_total

    def inverse(self, z):
        for step in reversed(self.steps):
            z = step.inverse(z)
        return z

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - self.dim * 0.5 * np.log(2 * np.pi)
        return log_pz + log_det

    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.dim, device=device)
        with torch.no_grad():
            return self.inverse(z)


# ── Training ──

def train_model(model, target='moons', n_steps=5000, batch_size=256, lr=1e-3, device='cpu'):
    """Train with negative log-likelihood: -log p(x) = -log p(z) - sum(log_det)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        x = sample_target(batch_size, target, device)
        log_px = model.log_prob(x)
        loss = -log_px.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | NLL: {loss.item():.4f}")

    return losses


# ── Density Evaluation ──

def compute_density(model, xlim=(-3, 3), ylim=(-3, 3), n_grid=100, device='cpu'):
    """Compute learned density on a grid using the inverse pass."""
    x1 = torch.linspace(xlim[0], xlim[1], n_grid, device=device)
    x2 = torch.linspace(ylim[0], ylim[1], n_grid, device=device)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.cat([X1.unsqueeze(-1), X2.unsqueeze(-1)], dim=-1).reshape(-1, 2)

    with torch.no_grad():
        log_px = model.log_prob(grid)
        density = log_px.exp().reshape(n_grid, n_grid).cpu().numpy()

    return density


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "60-glow"
    results_dir.mkdir(parents=True, exist_ok=True)

    target = 'moons'
    n_steps = 5000

    # Experiment 1: Glow vs Simple RealNVP on 2D density estimation
    print("=== Training Glow ===")
    glow = GlowModel(dim=2, n_steps=6, hidden=64).to(device)
    n_params_g = sum(p.numel() for p in glow.parameters())
    print(f"  Params: {n_params_g:,}")
    glow_losses = train_model(glow, target, n_steps, lr=1e-3, device=device)

    print("\n=== Training Simple RealNVP (no ActNorm, no 1x1 Conv) ===")
    realnvp = SimpleRealNVP(dim=2, n_steps=6, hidden=64).to(device)
    n_params_r = sum(p.numel() for p in realnvp.parameters())
    print(f"  Params: {n_params_r:,}")
    realnvp_losses = train_model(realnvp, target, n_steps, lr=1e-3, device=device)

    # Experiment 2: Invertibility check
    print("\n=== Invertibility Check ===")
    x_test = torch.randn(100, 2, device=device)
    with torch.no_grad():
        z_test, log_det_fwd = glow.forward(x_test)
        x_recon = glow.inverse(z_test)
        fwd_inv_err = (x_test - x_recon).abs().max().item()

        z_test2, log_det_fwd2 = realnvp.forward(x_test)
        x_recon2 = realnvp.inverse(z_test2)
        fwd_inv_err2 = (x_test - x_recon2).abs().max().item()

    print(f"  Glow roundtrip max error:        {fwd_inv_err:.2e}")
    print(f"  RealNVP roundtrip max error:      {fwd_inv_err2:.2e}")

    # Also check inverse-forward roundtrip
    z_rand = torch.randn(100, 2, device=device)
    with torch.no_grad():
        x_from_z = glow.inverse(z_rand)
        z_back, _ = glow.forward(x_from_z)
        inv_fwd_err = (z_rand - z_back).abs().max().item()
    print(f"  Glow inv->fwd roundtrip max error: {inv_fwd_err:.2e}")

    # Experiment 3: Train on different targets
    print("\n=== Training on Multiple Targets ===")
    target_results = {}
    for tgt in ['moons', 'circles', 'mixture']:
        print(f"\n  Target: {tgt}")
        g = GlowModel(dim=2, n_steps=6, hidden=64).to(device)
        losses = train_model(g, tgt, n_steps=3000, lr=1e-3, device=device)
        target_results[tgt] = {'model': g, 'losses': losses}

    # ── Visualization ──

    # 1. Training loss comparison: Glow vs RealNVP
    fig, ax = plt.subplots(figsize=(8, 5))
    window = 50
    glow_smooth = np.convolve(glow_losses, np.ones(window)/window, mode='valid')
    realnvp_smooth = np.convolve(realnvp_losses, np.ones(window)/window, mode='valid')

    ax.plot(glow_smooth, label='Glow (ActNorm + 1x1 Conv + Coupling)', color='blue', linewidth=2)
    ax.plot(realnvp_smooth, label='Simple RealNVP (Coupling only)', color='red', linewidth=2, alpha=0.7)
    ax.set_title("Training NLL: Glow vs Simple RealNVP", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("NLL (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # 2. Learned 2D density contours
    target_data = sample_target(2000, target, device).cpu().numpy()

    print("\nComputing density grids...")
    glow_density = compute_density(glow, device=device)
    realnvp_density = compute_density(realnvp, device=device)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(target_data[:, 0], target_data[:, 1], alpha=0.2, s=3)
    axes[0].set_title("Target Data (Moons)")
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    im1 = axes[1].imshow(glow_density.T, extent=[-3, 3, -3, 3], origin='lower',
                          cmap='viridis', aspect='equal')
    axes[1].set_title("Glow Learned Density")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(realnvp_density.T, extent=[-3, 3, -3, 3], origin='lower',
                          cmap='viridis', aspect='equal')
    axes[2].set_title("Simple RealNVP Learned Density")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle("2D Density Estimation: Glow vs RealNVP", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "density_comparison.png", dpi=150)
    plt.close()

    # 3. Density contours for different targets
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, tgt in enumerate(['moons', 'circles', 'mixture']):
        model = target_results[tgt]['model']
        data = sample_target(2000, tgt, device).cpu().numpy()
        density = compute_density(model, device=device)

        axes[0, i].scatter(data[:, 0], data[:, 1], alpha=0.2, s=3)
        axes[0, i].set_title(f"Target: {tgt}")
        axes[0, i].set_xlim(-3, 3); axes[0, i].set_ylim(-3, 3)
        axes[0, i].set_aspect('equal'); axes[0, i].grid(True, alpha=0.3)

        im = axes[1, i].imshow(density.T, extent=[-3, 3, -3, 3], origin='lower',
                                cmap='viridis', aspect='equal')
        axes[1, i].set_title(f"Glow: {tgt}")
        plt.colorbar(im, ax=axes[1, i], shrink=0.8)

    plt.suptitle("Glow: Density Estimation on Multiple Targets", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "multi_target_density.png", dpi=150)
    plt.close()

    # 4. Latent space interpolation
    print("Computing latent interpolations...")
    glow.eval()
    with torch.no_grad():
        x1 = sample_target(1, target, device)
        x2 = sample_target(1, target, device)
        z1, _ = glow.forward(x1)
        z2, _ = glow.forward(x2)

        n_interp = 10
        alphas = torch.linspace(0, 1, n_interp, device=device)
        interp_samples = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = glow.inverse(z_interp)
            interp_samples.append(x_interp.cpu().numpy().squeeze())

    fig, axes = plt.subplots(1, n_interp, figsize=(2 * n_interp, 2))
    for i, (ax, sample) in enumerate(zip(axes, interp_samples)):
        ax.scatter(target_data[:, 0], target_data[:, 1], alpha=0.05, s=1, color='gray')
        ax.scatter(sample[0], sample[1], color='red', s=100, zorder=5, edgecolors='black')
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f"alpha={alphas[i].item():.1f}", fontsize=9)
        ax.axis('off')

    plt.suptitle("Glow: Latent Space Interpolation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "latent_interpolation.png", dpi=150)
    plt.close()

    # 5. Sampling comparison
    with torch.no_grad():
        glow_samples = glow.sample(2000, device).cpu().numpy()
        realnvp_samples = realnvp.sample(2000, device).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(target_data[:, 0], target_data[:, 1], alpha=0.2, s=3)
    axes[0].set_title("Target (Moons)")
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(glow_samples[:, 0], glow_samples[:, 1], alpha=0.2, s=3, color='blue')
    axes[1].set_title("Glow Samples")
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(realnvp_samples[:, 0], realnvp_samples[:, 1], alpha=0.2, s=3, color='red')
    axes[2].set_title("Simple RealNVP Samples")
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Sampling Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "sampling_comparison.png", dpi=150)
    plt.close()

    # 6. Concept diagram: ActNorm + 1x1 Conv + Affine Coupling
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    # Draw flow of one GlowStep
    components = [
        ("ActNorm\n(Glow innovation)", "Replaces BatchNorm\ny = s*x + b (learned)\nInit from batch stats\nlog_det = sum(log|s|)\n-> Stable, invertible", 0.14, '#2196F3'),
        ("Invertible 1x1 Conv\n(Glow innovation)", "Replaces permutation\nW = PLU decomposition\nLearned channel mixing\nlog_det = sum(log|diag(U)|)\n-> Expressive, invertible", 0.5, '#FF9800'),
        ("Affine Coupling\n(from RealNVP)", "Split x -> x1, x2\ny1 = x1 (identity)\ny2 = x2*exp(s(x1))+t(x1)\nlog_det = sum(s)\n-> Easy inverse", 0.86, '#4CAF50'),
    ]

    for name, desc, x_pos, color in components:
        ax.text(x_pos, 0.78, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Arrow between components
    arrow_y = 0.78
    for (x_start, x_end) in [(0.27, 0.36), (0.63, 0.73)]:
        ax.annotate('', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(0.5, 0.02, "GlowStep = ActNorm -> Invertible1x1Conv -> AffineCoupling  |  Glow = stack of GlowSteps",
            fontsize=11, ha='center', va='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.9))

    ax.set_title("Glow (1807.03039): Key Innovations over RealNVP", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "glow_concept.png", dpi=150)
    plt.close()

    # 7. Invertibility verification plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    with torch.no_grad():
        # Check reconstruction error for many samples
        n_check = 500
        x_check = torch.randn(n_check, 2, device=device)
        z_check, _ = glow.forward(x_check)
        x_rec = glow.inverse(z_check)
        errors_glow = (x_check - x_rec).abs().sum(dim=-1).cpu().numpy()

        z_check2, _ = realnvp.forward(x_check)
        x_rec2 = realnvp.inverse(z_check2)
        errors_realnvp = (x_check - x_rec2).abs().sum(dim=-1).cpu().numpy()

    axes[0].hist(errors_glow, bins=50, alpha=0.7, color='blue', label='Glow')
    axes[0].hist(errors_realnvp, bins=50, alpha=0.7, color='red', label='RealNVP')
    axes[0].set_title("Forward-Inverse Roundtrip Error")
    axes[0].set_xlabel("L1 Error per Sample")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Check log_det consistency: log|det J_f| + log|det J_f^{-1}| should = 0
    with torch.no_grad():
        x_check2 = torch.randn(100, 2, device=device)
        z2, log_det_fwd = glow.forward(x_check2)
        # For inverse, log_det should be negative of forward
        z_rand2 = torch.randn(100, 2, device=device)
        x_from_z2 = glow.inverse(z_rand2)
        z_back2, log_det_inv = glow.forward(x_from_z2)
        # The roundtrip log_det should not cancel directly, but check
        # that forward-inverse is consistent
        x_roundtrip = glow.inverse(z2)
        roundtrip_err = (x_check2 - x_roundtrip).abs().max().item()

    summary_text = (
        f"Glow roundtrip max error: {fwd_inv_err:.2e}\n"
        f"RealNVP roundtrip max error: {fwd_inv_err2:.2e}\n"
        f"Inv-Fwd roundtrip max error: {inv_fwd_err:.2e}\n"
        f"Glow params: {n_params_g:,}\n"
        f"RealNVP params: {n_params_r:,}\n"
        f"Final NLL - Glow: {glow_losses[-1]:.4f}\n"
        f"Final NLL - RealNVP: {realnvp_losses[-1]:.4f}"
    )
    axes[1].text(0.1, 0.5, summary_text, fontsize=11, va='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    axes[1].set_title("Invertibility & Summary")
    axes[1].axis('off')

    plt.suptitle("Glow: Invertibility Verification", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "invertibility_check.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
