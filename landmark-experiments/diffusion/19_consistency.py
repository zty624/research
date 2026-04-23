"""
Minimal Consistency Model Reproduction
=======================================
Reproduces core ideas from "Consistency Models" (2303.01469, Song et al.):
1. Consistency function: map any point on ODE trajectory directly to x_0
2. Consistency training (CT): learn without pre-trained diffusion model
3. Single-step generation (vs multi-step DDPM/DDIM)
4. Compare: DDPM (50 steps) vs DDIM (10 steps) vs Consistency (1 step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Consistency Model Architecture ──

class ConsistencyMLP(nn.Module):
    """MLP-based consistency model for 2D point clouds.
    Key: output is in the same space as input (maps x_t → x_0).
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )
        # Zero initialization for the last layer (start close to identity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t, t):
        """Map noisy point x_t at time t to clean point x_0.
        x_t: (B, N, 2), t: (B,)
        """
        t_emb = self.time_embed(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        h = torch.cat([x_t, t_emb], dim=-1)
        return x_t + self.net(h)  # Residual: start as identity


# ── Point Cloud Denoiser (for DDPM baseline) ──

class DenoiserMLP(nn.Module):
    """Standard noise-predicting denoiser for DDPM baseline."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)


# ── Noise Schedule ──

def karras_schedule(t, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Karras et al. noise schedule used in consistency models."""
    return (sigma_min ** (1/rho) + t * (sigma_max ** (1/rho) - sigma_min ** (1/rho))) ** rho


# ── 2D Data ──

def sample_2d_data(n, device='cpu'):
    """2D mixture of Gaussians (4 modes)."""
    n1 = n // 4
    n2 = n // 4
    n3 = n // 4
    n4 = n - n1 - n2 - n3
    c1 = torch.randn(n1, 2, device=device) * 0.2 + torch.tensor([2.0, 2.0], device=device)
    c2 = torch.randn(n2, 2, device=device) * 0.2 + torch.tensor([-2.0, 2.0], device=device)
    c3 = torch.randn(n3, 2, device=device) * 0.2 + torch.tensor([-2.0, -2.0], device=device)
    c4 = torch.randn(n4, 2, device=device) * 0.2 + torch.tensor([2.0, -2.0], device=device)
    return torch.cat([c1, c2, c3, c4])


# ── Consistency Training ──

def consistency_training_loss(model, x_0, t_min=0.002, t_max=80.0, device='cpu'):
    """Consistency Training (CT) loss.

    Key idea: for an ODE trajectory x_t, the consistency function should satisfy:
    f_θ(x_t, t) = f_θ(x_{t'}, t') for all t, t' > 0

    Training: sample t_n > t_{n+1}, enforce:
    f_θ(x_{t_n}, t_n) ≈ f_{θ-}(x_{t_{n+1}}, t_{n+1})

    where θ- is the EMA target network.
    """
    B, N, D = x_0.shape

    # Sample two adjacent timesteps
    t_n = karras_schedule(torch.rand(B, device=device), t_min, t_max)
    t_n1 = karras_schedule(torch.rand(B, device=device), t_min, t_max)

    # Ensure t_n > t_n1
    t_n = torch.maximum(t_n, t_n1)
    t_n1 = torch.minimum(t_n, t_n1)
    # Make t_n1 slightly smaller
    t_n1 = t_n1 * 0.95 + t_min * 0.05

    # Add noise
    noise_n = torch.randn_like(x_0)
    noise_n1 = torch.randn_like(x_0)

    x_tn = x_0 + t_n.unsqueeze(-1).unsqueeze(-1) * noise_n
    x_tn1 = x_0 + t_n1.unsqueeze(-1).unsqueeze(-1) * noise_n1

    # Predict x_0 from both timesteps
    pred_0_from_tn = model(x_tn, t_n)
    with torch.no_grad():
        pred_0_from_tn1 = model(x_tn1, t_n1)

    # Consistency loss: outputs should match
    loss = F.mse_loss(pred_0_from_tn, pred_0_from_tn1.detach())

    return loss


def train_consistency_model(model, n_steps=5000, batch_size=256, n_points=32,
                             lr=1e-3, ema_decay=0.999, device='cpu'):
    """Train consistency model with EMA target."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create EMA target
    target_model = deepcopy(model)
    target_model.eval()

    losses = []

    for step in range(n_steps):
        data = sample_2d_data(batch_size * n_points, device).reshape(batch_size, n_points, 2)

        loss = consistency_training_loss(model, data, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # EMA update
        with torch.no_grad():
            for p_target, p_online in zip(target_model.parameters(), model.parameters()):
                p_target.data = ema_decay * p_target.data + (1 - ema_decay) * p_online.data

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.6f}")

    return losses, target_model


from copy import deepcopy


# ── DDPM Baseline ──

def train_ddpm_baseline(model, n_steps=5000, batch_size=256, n_points=32,
                         lr=1e-3, device='cpu'):
    """Train standard DDPM (noise prediction) baseline."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    timesteps = 200
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    losses = []

    for step in range(n_steps):
        data = sample_2d_data(batch_size * n_points, device).reshape(batch_size, n_points, 2)
        t_idx = torch.randint(0, timesteps, (batch_size,), device=device)
        noise = torch.randn_like(data)

        sqrt_ab = torch.sqrt(alpha_bar[t_idx]).unsqueeze(-1).unsqueeze(-1)
        sqrt_omab = torch.sqrt(1 - alpha_bar[t_idx]).unsqueeze(-1).unsqueeze(-1)
        x_t = sqrt_ab * data + sqrt_omab * noise

        t_norm = t_idx.float() / timesteps
        pred_noise = model(x_t, t_norm)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.6f}")

    return losses, betas, alphas, alpha_bar


@torch.no_grad()
def sample_ddpm(model, betas, alphas, alpha_bar, n_samples=500, n_points=32,
                 n_steps=50, device='cpu'):
    """DDIM-like sampling with n_steps."""
    x = torch.randn(n_samples, n_points, 2, device=device)
    timesteps = list(range(0, 200, 200 // n_steps))

    for i in reversed(range(1, len(timesteps))):
        t_idx = timesteps[i]
        t_prev = timesteps[i - 1]
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        t_norm = t.float() / 200

        pred_noise = model(x, t_norm)

        ab_t = alpha_bar[t_idx]
        ab_prev = alpha_bar[t_prev]
        x0_pred = (x - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * pred_noise

    return x


@torch.no_grad()
def sample_consistency(model, n_samples=500, n_points=32, sigma=80.0, device='cpu'):
    """Single-step sampling from consistency model!"""
    # Sample from N(0, σ²I)
    x_t = torch.randn(n_samples, n_points, 2, device=device) * sigma
    t = torch.full((n_samples,), sigma, device=device)
    # One-step denoising
    x_0 = model(x_t, t)
    return x_0


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "19-consistency"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_points = 32

    # Train Consistency Model
    print("=== Training Consistency Model (CT) ===")
    cm = ConsistencyMLP(hidden_dim=128).to(device)
    cm_losses, cm_target = train_consistency_model(cm, n_steps=5000, n_points=n_points, device=device)

    # Train DDPM baseline
    print("\n=== Training DDPM Baseline ===")
    ddpm = DenoiserMLP(hidden_dim=128).to(device)
    ddpm_losses, betas, alphas, alpha_bar = train_ddpm_baseline(
        ddpm, n_steps=5000, n_points=n_points, device=device
    )

    # ── Sample from all models ──
    print("\n=== Sampling ===")
    target = sample_2d_data(1000, device).cpu().numpy()

    # DDPM 50 steps
    ddpm_50 = sample_ddpm(ddpm, betas, alphas, alpha_bar, n_samples=500,
                           n_points=n_points, n_steps=50, device=device
                           ).reshape(-1, 2).cpu().numpy()

    # DDPM 10 steps
    ddpm_10 = sample_ddpm(ddpm, betas, alphas, alpha_bar, n_samples=500,
                           n_points=n_points, n_steps=10, device=device
                           ).reshape(-1, 2).cpu().numpy()

    # Consistency 1 step
    cm_1step = sample_consistency(cm_target, n_samples=500, n_points=n_points,
                                   sigma=80.0, device=device
                                   ).reshape(-1, 2).cpu().numpy()

    # ── Visualization ──

    # 1. Training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    window = 30
    cm_s = np.convolve(cm_losses, np.ones(window)/window, mode='valid')
    ddpm_s = np.convolve(ddpm_losses, np.ones(window)/window, mode='valid')
    ax.plot(cm_s, label='Consistency Model (CT)', color='green')
    ax.plot(ddpm_s, label='DDPM (noise pred)', color='blue')
    ax.set_title("Training Loss: Consistency Model vs DDPM")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # 2. Sample comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].scatter(target[:, 0], target[:, 1], alpha=0.2, s=3)
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ddpm_50[:, 0], ddpm_50[:, 1], alpha=0.2, s=3, color='blue')
    axes[1].set_title("DDPM (50 steps)")
    axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(ddpm_10[:, 0], ddpm_10[:, 1], alpha=0.2, s=3, color='orange')
    axes[2].set_title("DDIM (10 steps)")
    axes[2].set_xlim(-4, 4); axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    axes[3].scatter(cm_1step[:, 0], cm_1step[:, 1], alpha=0.2, s=3, color='green')
    axes[3].set_title("Consistency (1 step!)")
    axes[3].set_xlim(-4, 4); axes[3].set_ylim(-4, 4)
    axes[3].set_aspect('equal'); axes[3].grid(True, alpha=0.3)

    plt.suptitle("Consistency Model: Single-Step Generation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "sample_comparison.png", dpi=150)
    plt.close()

    # 3. Step count comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['DDPM\n(200 steps)', 'DDIM\n(50 steps)', 'DDIM\n(10 steps)',
               'Consistency\n(1 step)']
    steps = [200, 50, 10, 1]
    colors = ['blue', 'cyan', 'orange', 'green']
    ax.bar(methods, steps, color=colors, alpha=0.7)
    ax.set_ylabel("Number of Model Evaluations")
    ax.set_title("Sampling Efficiency: Steps Required")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "sampling_efficiency.png", dpi=150)
    plt.close()

    # 4. Consistency property visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Show how consistency model maps different noise levels to same x_0
    cm_target.eval()
    test_point = torch.tensor([[[2.0, 2.0]]], device=device)  # Target point

    with torch.no_grad():
        sigmas = [1.0, 5.0, 20.0, 50.0, 80.0]
        for idx, sigma in enumerate(sigmas[:3]):
            ax = axes[idx]
            noisy = test_point + torch.randn(100, 1, 2, device=device) * sigma
            t = torch.full((100,), float(sigma), device=device)
            denoised = cm_target(noisy, t).cpu().numpy()

            noisy_np = noisy.cpu().numpy().reshape(-1, 2)
            denoised_np = denoised.reshape(-1, 2)

            ax.scatter(noisy_np[:, 0], noisy_np[:, 1], alpha=0.3, s=10, color='red', label=f'x_t (σ={sigma})')
            ax.scatter(denoised_np[:, 0], denoised_np[:, 1], alpha=0.3, s=10, color='green', label='f(x_t) → x_0')
            ax.set_xlim(-5, 7); ax.set_ylim(-5, 7)
            ax.set_aspect('equal')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"σ = {sigma}")

    plt.suptitle("Consistency Property: All x_t Map to Same x_0", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "consistency_property.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
