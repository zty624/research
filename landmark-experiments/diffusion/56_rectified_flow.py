"""
Minimal Rectified Flow Reproduction
=====================================
Reproduces core ideas from Rectified Flow (2209.03003, Liu et al.):
1. Learn straight ODE paths from noise to data
2. z_t = t·x_1 + (1-t)·x_0 where x_0=noise, x_1=data
3. Model predicts velocity: v_θ(z_t, t) ≈ x_1 - x_0
4. Straight paths → fewer steps needed for generation
5. Reflow: retrain on generated pairs makes paths even straighter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Model ──

class VelocityNet(nn.Module):
    """Predicts velocity v(z_t, t) for rectified flow."""
    def __init__(self, data_dim=2, hidden=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


# ── Rectified Flow ──

class RectifiedFlow:
    def __init__(self, model, data_dim=2):
        self.model = model
        self.data_dim = data_dim

    def sample_pair(self, x1, x0=None):
        """Create (x_0, x_1) pairs. x_1=data, x_0=noise."""
        if x0 is None:
            x0 = torch.randn_like(x1)
        return x0, x1

    def compute_loss(self, x1):
        """Rectified flow loss: E[||v_θ(z_t, t) - (x_1 - x_0)||²]."""
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)

        # Interpolate: z_t = t·x_1 + (1-t)·x_0
        z_t = t.unsqueeze(-1) * x1 + (1 - t.unsqueeze(-1)) * x0

        # Target velocity: x_1 - x_0
        target_v = x1 - x0

        # Predicted velocity
        pred_v = self.model(z_t, t)

        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, n_samples=500, n_steps=50, device='cpu'):
        """Generate samples by solving ODE: dz/dt = v_θ(z, t)."""
        self.model.eval()
        z = torch.randn(n_samples, self.data_dim, device=device)
        dt = 1.0 / n_steps

        trajectory = [z.cpu().numpy().copy()]

        for i in range(n_steps):
            t = torch.full((n_samples,), i / n_steps, device=device)
            v = self.model(z, t)
            z = z + v * dt  # Euler step
            trajectory.append(z.cpu().numpy().copy())

        self.model.train()
        return z.cpu().numpy(), trajectory


# ── Diffusion Baseline (for comparison) ──

class DiffusionBaseline:
    """Simple DDPM baseline for comparison."""
    def __init__(self, model, data_dim=2, T=100):
        self.model = model
        self.data_dim = data_dim
        self.T = T
        betas = torch.linspace(0.0001, 0.02, T)
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, n_samples=500, n_steps=50, device='cpu'):
        """DDPM sampling."""
        self.model.eval()
        z = torch.randn(n_samples, self.data_dim, device=device)

        trajectory = [z.cpu().numpy().copy()]

        for t_idx in reversed(range(0, self.T, self.T // n_steps)):
            t = torch.full((n_samples,), t_idx, dtype=torch.float32, device=device) / self.T
            pred_noise = self.model(z, t)

            alpha = self.alphas[t_idx].to(device)
            alpha_bar = self.alpha_bar[t_idx].to(device)

            z = (z - (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) * pred_noise) / (torch.sqrt(alpha) + 1e-8)
            if t_idx > 0:
                z = z + torch.randn_like(z) * 0.05

            trajectory.append(z.cpu().numpy().copy())

        self.model.train()
        return z.cpu().numpy(), trajectory


# ── Training ──

def train_rectified_flow(rf, data, n_steps=5000, lr=2e-4, device='cpu'):
    optimizer = torch.optim.Adam(rf.model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, len(data), (256,))
        x1 = data[idx].to(device)

        loss = rf.compute_loss(x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")

    return losses


def train_diffusion_baseline(model, data, T=100, n_steps=5000, lr=2e-4, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    diffusion = DiffusionBaseline(model, data_dim=2, T=T)
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, len(data), (256,))
        x = data[idx].to(device)
        t = torch.randint(0, T, (len(x),), device=device)

        alpha_bar = diffusion.alpha_bar.to(device)[t].unsqueeze(-1)
        noise = torch.randn_like(x)
        noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

        pred = model(noisy, t.float() / T)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "56-rectified-flow"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create 2D data: mixture of Gaussians
    print("=== Creating Data ===")
    np.random.seed(42)
    n_data = 3000
    centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    data_np = []
    for c in centers:
        data_np.append(np.random.randn(n_data // 4, 2) * 0.3 + c)
    data = torch.tensor(np.vstack(data_np), dtype=torch.float32)
    print(f"  Data shape: {data.shape}")

    # Train Rectified Flow
    print("\n=== Training Rectified Flow ===")
    rf_model = VelocityNet(data_dim=2, hidden=128).to(device)
    rf = RectifiedFlow(rf_model, data_dim=2)
    rf_losses = train_rectified_flow(rf, data, n_steps=5000, device=device)

    # Train Diffusion baseline
    print("\n=== Training Diffusion Baseline ===")
    diff_model = VelocityNet(data_dim=2, hidden=128).to(device)
    diff_losses = train_diffusion_baseline(diff_model, data, n_steps=5000, device=device)

    # Experiment 1: Sample quality vs number of steps
    print("\n=== Sampling Steps Comparison ===")
    steps_results = {}
    for n_steps in [1, 2, 5, 10, 20, 50]:
        # Rectified Flow
        rf_samples, rf_traj = rf.sample(n_samples=500, n_steps=n_steps, device=device)
        rf_mean = rf_samples.mean(axis=0)
        rf_std = rf_samples.std(axis=0)

        # Diffusion
        diff_baseline = DiffusionBaseline(diff_model, data_dim=2)
        diff_samples, diff_traj = diff_baseline.sample(n_samples=500, n_steps=n_steps, device=device)
        diff_mean = diff_samples.mean(axis=0)
        diff_std = diff_samples.std(axis=0)

        steps_results[n_steps] = {
            'rf_mean': rf_mean, 'rf_std': rf_std,
            'diff_mean': diff_mean, 'diff_std': diff_std
        }
        print(f"  {n_steps} steps: RF std=({rf_std[0]:.2f},{rf_std[1]:.2f}), "
              f"Diff std=({diff_std[0]:.2f},{diff_std[1]:.2f})")

    # Experiment 2: Visualize trajectories
    print("\n=== Trajectory Visualization ===")
    rf_samples, rf_traj = rf.sample(n_samples=20, n_steps=20, device=device)
    diff_baseline = DiffusionBaseline(diff_model, data_dim=2)
    diff_samples, diff_traj = diff_baseline.sample(n_samples=20, n_steps=20, device=device)

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rf_losses, alpha=0.3, color='blue')
    smoothed_rf = np.convolve(rf_losses, np.ones(50)/50, mode='valid')
    axes[0].plot(smoothed_rf, color='blue', linewidth=2, label='Rectified Flow')

    axes[0].plot(diff_losses, alpha=0.3, color='red')
    smoothed_diff = np.convolve(diff_losses, np.ones(50)/50, mode='valid')
    axes[0].plot(smoothed_diff, color='red', linewidth=2, label='Diffusion')

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Generated samples
    rf_500, _ = rf.sample(n_samples=500, n_steps=50, device=device)
    diff_500, _ = diff_baseline.sample(n_samples=500, n_steps=50, device=device)

    axes[1].scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.1, s=1, color='gray', label='Data')
    axes[1].scatter(rf_500[:, 0], rf_500[:, 1], alpha=0.3, s=5, color='blue', label='Rectified Flow')
    axes[1].scatter(diff_500[:, 0], diff_500[:, 1], alpha=0.3, s=5, color='red', label='Diffusion')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].set_title("Generated Samples (50 steps)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Rectified Flow: Straight Paths for Fast Generation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "rf_comparison.png", dpi=150)
    plt.close()

    # 2. Trajectory comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(min(20, len(rf_traj[0]))):
        traj_x = [rf_traj[t][i, 0] for t in range(len(rf_traj))]
        traj_y = [rf_traj[t][i, 1] for t in range(len(rf_traj))]
        axes[0].plot(traj_x, traj_y, alpha=0.5, linewidth=1)
        axes[0].scatter(traj_x[0], traj_y[0], s=10, color='blue')  # start
        axes[0].scatter(traj_x[-1], traj_y[-1], s=10, color='red')  # end

    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_title("Rectified Flow Trajectories")
    axes[0].grid(True, alpha=0.3)

    for i in range(min(20, len(diff_traj[0]))):
        traj_x = [diff_traj[t][i, 0] for t in range(len(diff_traj))]
        traj_y = [diff_traj[t][i, 1] for t in range(len(diff_traj))]
        axes[1].plot(traj_x, traj_y, alpha=0.5, linewidth=1, color='red')
        axes[1].scatter(traj_x[0], traj_y[0], s=10, color='blue')
        axes[1].scatter(traj_x[-1], traj_y[-1], s=10, color='red')

    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_title("Diffusion Trajectories")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ODE Trajectories: Rectified Flow (straight) vs Diffusion (curved)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "trajectories.png", dpi=150)
    plt.close()

    # 3. Steps vs quality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_steps_list = sorted(steps_results.keys())
    rf_stds = [steps_results[n]['rf_std'].mean() for n in n_steps_list]
    diff_stds = [steps_results[n]['diff_std'].mean() for n in n_steps_list]

    axes[0].plot(n_steps_list, rf_stds, 'o-', label='Rectified Flow', color='blue')
    axes[0].plot(n_steps_list, diff_stds, 's--', label='Diffusion', color='red')
    axes[0].axhline(y=data.numpy().std(), color='gray', linestyle=':', label='Data std')
    axes[0].set_xlabel("Sampling Steps")
    axes[0].set_ylabel("Sample Std Dev")
    axes[0].set_title("Sample Diversity vs Steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Few-step generation quality
    for n_steps in [1, 2, 5]:
        rf_s, _ = rf.sample(n_samples=300, n_steps=n_steps, device=device)
        axes[1].scatter(rf_s[:, 0], rf_s[:, 1], alpha=0.3, s=5, label=f'RF {n_steps} steps')

    axes[1].scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.05, s=1, color='gray', label='Data')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].set_title("Rectified Flow: Few-Step Generation")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Rectified Flow: Efficiency with Few Steps", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "few_step_generation.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Diffusion\n(Curved)", "DDPM/DDIM paths\ncurved trajectories\nMany steps needed\nfor good quality\n→ Slow sampling", 0.14, 'red'),
        ("Rectified\nFlow", "Straight ODE paths\nz_t = t·x_1 + (1-t)·x_0\nPredict velocity v=x_1-x_0\nFewer steps needed\n→ Fast sampling!", 0.5, 'blue'),
        ("Reflow\n(Refinement)", "Retrain on generated\npairs (z_0, z_1)\nMakes paths even\nstraighter\n→ 1-step possible!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Rectified Flow: Straight Paths = Fewer Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rectified_flow_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
