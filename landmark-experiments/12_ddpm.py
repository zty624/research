"""
Minimal DDPM (Denoising Diffusion Probabilistic Models) Reproduction
=====================================================================
Reproduces the core ideas from "Denoising Diffusion Probabilistic Models"
(Ho et al., 2020, 2006.11239):
1. Forward process: gradually add Gaussian noise to data
2. Reverse process: learn to denoise step by step
3. Noise schedule (linear, cosine)
4. Simplified training objective (predict noise, not mean)
5. DDPM vs DDIM sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Noise Schedule ──

def linear_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_schedule(timesteps, s=0.008):
    """Cosine noise schedule (improved DDPM)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


# ── Simple U-Net for 1D ──

class SimpleUNet1D(nn.Module):
    """Minimal 1D U-Net for denoising 1D signals."""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels + 1, base_channels, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(base_channels, base_channels*2, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(base_channels*2, base_channels*4, 3, stride=2, padding=1), nn.ReLU())

        # Decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec1 = nn.Conv1d(base_channels, in_channels, 3, padding=1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, base_channels*4)
        )

        # Time modulation layers
        self.time_mod3 = nn.Linear(base_channels*4, base_channels*4)

    def forward(self, x, t):
        """x: (B, C, L), t: (B,) normalized to [0, 1]"""
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))  # (B, base*4)
        t_mod = self.time_mod3(t_emb).unsqueeze(-1)  # (B, base*4, 1)

        # Concatenate time as extra channel
        t_channel = t.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.shape[-1])
        h = torch.cat([x, t_channel], dim=1)

        # Encode
        e1 = self.enc1(h)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Modulate bottleneck with time
        e3 = e3 * (1 + t_mod)

        # Decode with skip connections
        d3 = self.dec3(e3)
        # Handle size mismatch
        if d3.shape[-1] != e2.shape[-1]:
            d3 = d3[:, :, :e2.shape[-1]]
        d3 = d3 + e2

        d2 = self.dec2(d3)
        if d2.shape[-1] != e1.shape[-1]:
            d2 = d2[:, :, :e1.shape[-1]]
        d2 = d2 + e1

        return self.dec1(d2)


# ── 2D Point Cloud U-Net ──

class PointCloudDenoiser(nn.Module):
    """Simple MLP-based denoiser for 2D point clouds."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t):
        """x: (B, N, 2), t: (B,)"""
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, N, hidden)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


# ── DDPM Training & Sampling ──

class DDPM:
    """Denoising Diffusion Probabilistic Model."""
    def __init__(self, model, timesteps=1000, schedule='linear', device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        if schedule == 'linear':
            self.betas = linear_schedule(timesteps).to(device)
        else:
            self.betas = cosine_schedule(timesteps).to(device)

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def add_noise(self, x0, t, noise=None):
        """Forward process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1) if x0.dim() == 3 else self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1) if x0.dim() == 3 else self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)

        return sqrt_ab * x0 + sqrt_omab * noise, noise

    def train_loss(self, x0):
        """Simplified DDPM training objective: predict noise ε."""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        x_t, noise = self.add_noise(x0, t)

        # Normalize t for model input
        t_norm = t.float() / self.timesteps

        if x0.dim() == 3:
            # Point cloud: (B, N, 2)
            pred_noise = self.model(x_t, t_norm)
        else:
            # 1D signal: (B, C, L)
            pred_noise = self.model(x_t, t_norm)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_ddpm(self, shape, n_points=None):
        """Full DDPM reverse process (T steps)."""
        if n_points is not None:
            x = torch.randn(shape[0], n_points, 2, device=self.device)
        else:
            x = torch.randn(shape, device=self.device)

        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            t_norm = t.float() / self.timesteps

            pred_noise = self.model(x, t_norm)

            # DDPM reverse step
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]

            # x_{t-1} = (1/sqrt(α)) * (x_t - β/sqrt(1-α_bar) * ε) + σ*z
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = beta / torch.sqrt(1 - alpha_bar)
            x = coeff1 * (x - coeff2 * pred_noise)

            if t_idx > 0:
                sigma = torch.sqrt(beta)
                x = x + sigma * torch.randn_like(x)

        return x

    @torch.no_grad()
    def sample_ddim(self, shape, n_points=None, n_steps=50):
        """DDIM sampling (deterministic, fewer steps)."""
        # Subsample timesteps
        step_size = self.timesteps // n_steps
        timesteps = list(range(0, self.timesteps, step_size))

        if n_points is not None:
            x = torch.randn(shape[0], n_points, 2, device=self.device)
        else:
            x = torch.randn(shape, device=self.device)

        for i in reversed(range(1, len(timesteps))):
            t_idx = timesteps[i]
            t_prev_idx = timesteps[i - 1]

            t = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            t_norm = t.float() / self.timesteps

            pred_noise = self.model(x, t_norm)

            alpha_bar_t = self.alpha_bar[t_idx]
            alpha_bar_prev = self.alpha_bar[t_prev_idx]

            # DDIM: x_{t-1} = sqrt(α_{t-1}) * (x_t - sqrt(1-α_t) * ε) / sqrt(α_t) + sqrt(1-α_{t-1}) * ε
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * pred_noise

        return x


# ── 2D Data ──

def sample_2d_data(n, device='cpu'):
    """2D mixture of Gaussians."""
    n1 = n // 4
    n2 = n // 4
    n3 = n // 4
    n4 = n - n1 - n2 - n3
    c1 = torch.randn(n1, 2, device=device) * 0.2 + torch.tensor([2.0, 2.0], device=device)
    c2 = torch.randn(n2, 2, device=device) * 0.2 + torch.tensor([-2.0, 2.0], device=device)
    c3 = torch.randn(n3, 2, device=device) * 0.2 + torch.tensor([-2.0, -2.0], device=device)
    c4 = torch.randn(n4, 2, device=device) * 0.2 + torch.tensor([2.0, -2.0], device=device)
    return torch.cat([c1, c2, c3, c4])


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "12-ddpm"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Noise Schedule Comparison ──
    print("=== Comparing Noise Schedules ===")
    timesteps = 1000
    betas_linear = linear_schedule(timesteps)
    betas_cosine = cosine_schedule(timesteps)
    alpha_bar_linear = torch.cumprod(1 - betas_linear, dim=0)
    alpha_bar_cosine = torch.cumprod(1 - betas_cosine, dim=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(alpha_bar_linear.numpy(), label='Linear', color='red')
    axes[0].plot(alpha_bar_cosine.numpy(), label='Cosine', color='blue')
    axes[0].set_title("ᾱ(t) — Signal Remaining")
    axes[0].set_xlabel("Timestep t")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(betas_linear.numpy(), label='Linear', color='red')
    axes[1].plot(betas_cosine.numpy(), label='Cosine', color='blue')
    axes[1].set_title("β(t) — Noise Rate")
    axes[1].set_xlabel("Timestep t")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DDPM Noise Schedules: Linear vs Cosine", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "noise_schedules.png", dpi=150)
    plt.close()

    # ── Train DDPM on 2D Point Clouds ──
    print("\n=== Training DDPM on 2D Point Clouds ===")
    n_points = 32
    model = PointCloudDenoiser(hidden_dim=128).to(device)
    ddpm = DDPM(model, timesteps=200, schedule='cosine', device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses_linear = []
    losses_cosine = []

    # Train with cosine schedule
    for step in range(3000):
        data = sample_2d_data(256 * n_points, device).reshape(256, n_points, 2)
        loss = ddpm.train_loss(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_cosine.append(loss.item())

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.6f}")

    # Also train with linear schedule for comparison
    model_lin = PointCloudDenoiser(hidden_dim=128).to(device)
    ddpm_lin = DDPM(model_lin, timesteps=200, schedule='linear', device=device)
    optimizer_lin = torch.optim.AdamW(model_lin.parameters(), lr=1e-3)

    for step in range(3000):
        data = sample_2d_data(256 * n_points, device).reshape(256, n_points, 2)
        loss = ddpm_lin.train_loss(data)

        optimizer_lin.zero_grad()
        loss.backward()
        optimizer_lin.step()
        losses_linear.append(loss.item())

    # ── Visualization ──

    # 1. Training loss comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    window = 30
    loss_cos_s = np.convolve(losses_cosine, np.ones(window)/window, mode='valid')
    loss_lin_s = np.convolve(losses_linear, np.ones(window)/window, mode='valid')
    ax.plot(loss_cos_s, label='Cosine Schedule', color='blue')
    ax.plot(loss_lin_s, label='Linear Schedule', color='red')
    ax.set_title("DDPM Training: Cosine vs Linear Noise Schedule")
    ax.set_xlabel("Step")
    ax.set_ylabel("Noise Prediction Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "schedule_comparison.png", dpi=150)
    plt.close()

    # 2. Generated samples
    target = sample_2d_data(1000, device).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(target[:, 0], target[:, 1], alpha=0.2, s=3)
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # DDPM sampling (full 200 steps)
    samples_ddpm = ddpm.sample_ddpm((500,), n_points=n_points, ).reshape(-1, 2).cpu().numpy()
    axes[1].scatter(samples_ddpm[:, 0], samples_ddpm[:, 1], alpha=0.2, s=3, color='blue')
    axes[1].set_title("DDPM (200 steps)")
    axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    # DDIM sampling (20 steps)
    samples_ddim = ddpm.sample_ddim((500,), n_points=n_points, n_steps=20).reshape(-1, 2).cpu().numpy()
    axes[2].scatter(samples_ddim[:, 0], samples_ddim[:, 1], alpha=0.2, s=3, color='green')
    axes[2].set_title("DDIM (20 steps)")
    axes[2].set_xlim(-4, 4); axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("DDPM: Generated Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "generated_samples.png", dpi=150)
    plt.close()

    # 3. Forward diffusion visualization
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    data = sample_2d_data(300, device)
    timesteps_to_show = [0, 20, 50, 100, 150, 199]

    for ax, t_idx in zip(axes, timesteps_to_show):
        if t_idx == 0:
            noisy = data
        else:
            t = torch.full((300,), t_idx, device=device, dtype=torch.long)
            noisy, _ = ddpm.add_noise(data, t)

        noisy_np = noisy.cpu().numpy()
        ax.scatter(noisy_np[:, 0], noisy_np[:, 1], alpha=0.2, s=3, color='purple')
        ax.set_title(f"t = {t_idx}")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')

    plt.suptitle("Forward Diffusion: Adding Noise Step by Step", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "forward_diffusion.png", dpi=150)
    plt.close()

    # 4. DDPM vs DDIM efficiency
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    step_counts = [5, 10, 20, 50, 100, 200]
    ddim_losses = []
    ddpm_losses = []

    # Compute sample quality (MSE to target distribution mean)
    target_mean = torch.tensor([2.0, 2.0])  # One cluster center

    for n_steps in step_counts:
        with torch.no_grad():
            s_ddim = ddpm.sample_ddim((200,), n_points=n_points, n_steps=n_steps)
            ddim_center_dist = (s_ddim.mean(dim=1) - target_mean.to(device)).norm().item()
            ddim_losses.append(ddim_center_dist)

    for n_steps in step_counts:
        if n_steps > 200:
            continue
        with torch.no_grad():
            # Subsample DDPM steps
            step_size = max(1, 200 // n_steps)
            ddpm_t_sub = DDPM(model, timesteps=n_steps, schedule='cosine', device=device)
            s_ddpm = ddpm.sample_ddpm((200,), n_points=n_points)
            ddpm_center_dist = (s_ddpm.mean(dim=1) - target_mean.to(device)).norm().item()
            ddpm_losses.append(ddpm_center_dist)

    axes[0].plot(step_counts[:len(ddim_losses)], ddim_losses, 'o-', label='DDIM', color='green')
    axes[0].plot(step_counts[:len(ddpm_losses)], ddpm_losses, 's-', label='DDPM', color='blue')
    axes[0].set_xlabel("Sampling Steps")
    axes[0].set_ylabel("Distance to Target Center")
    axes[0].set_title("Sample Quality vs Sampling Steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sampling time comparison (theoretical)
    axes[1].bar(['DDPM\n(200 steps)', 'DDIM\n(20 steps)', 'DDIM\n(10 steps)'],
                [200, 20, 10], color=['blue', 'green', 'lightgreen'])
    axes[1].set_ylabel("Number of Model Evaluations")
    axes[1].set_title("Sampling Efficiency: DDPM vs DDIM")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "sampling_efficiency.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
