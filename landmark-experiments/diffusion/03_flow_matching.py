"""
Minimal Flow Matching Reproduction
====================================
Reproduces the core ideas from "Flow Matching for Generative Modeling" (2210.02747):
1. Conditional Flow Matching (CFM) objective
2. OT path vs Diffusion (VP) path
3. Visualize trajectories on 2D point clouds
4. Compare training speed and path straightness
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Simple Vector Field Network ──

class VectorFieldNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1).expand_as(x[..., :1])  # (B, 1) or (B, D) with same t
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


# ── Data: 2D Mixture of Gaussians ──

def sample_data(n, device='cpu'):
    """Two moons-like mixture."""
    n1 = n // 2
    n2 = n - n1
    c1 = torch.randn(n1, 2, device=device) * 0.3 + torch.tensor([1.0, 0.5], device=device)
    c2 = torch.randn(n2, 2, device=device) * 0.3 + torch.tensor([-1.0, -0.5], device=device)
    return torch.cat([c1, c2])


# ── Flow Matching Training ──

def train_cfm(model, data_fn, n_steps=5000, batch_size=256, lr=1e-3,
              path_type='ot', sigma_min=0.001, device='cpu'):
    """
    Conditional Flow Matching.
    path_type: 'ot' (optimal transport) or 'vp' (variance preserving / diffusion)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        x1 = data_fn(batch_size, device)  # data samples
        x0 = torch.randn_like(x1)  # noise samples
        t = torch.rand(batch_size, device=device)

        if path_type == 'ot':
            # OT path: ψ_t(x0) = (1-(1-σ_min)*t)*x0 + t*x1
            # u_t(x|x1) = (x1 - (1-σ_min)*x0) / (1-(1-σ_min)*t)
            xt = (1 - (1 - sigma_min) * t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
            target = (x1 - (1 - sigma_min) * x0) / (1 - (1 - sigma_min) * t.unsqueeze(-1))
        elif path_type == 'vp':
            # VP (diffusion) path: α_t = 1 - t, σ_t = t (simplified)
            alpha_t = 1 - t.unsqueeze(-1)
            sigma_t = t.unsqueeze(-1)
            xt = alpha_t * x0 + sigma_t * x1
            # Target vector field for VP
            target = (x1 - alpha_t * x0) / (sigma_t + 1e-8)

        pred = model(xt, t)
        loss = nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"  [{path_type.upper()}] Step {step+1} | Loss: {loss.item():.6f}")

    return losses


# ── Sampling via Euler ODE solver ──

def sample_flow(model, n_samples=500, n_steps=100, device='cpu'):
    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    trajectory = [x.cpu().numpy()]

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=device)
            v = model(x, t)
            x = x + v * dt
            trajectory.append(x.cpu().numpy())

    return trajectory


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "03-flow-matching"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train OT Flow Matching
    print("=== Training Flow Matching (OT Path) ===")
    model_ot = VectorFieldNet().to(device)
    losses_ot = train_cfm(model_ot, sample_data, n_steps=5000, path_type='ot', device=device)

    # Train VP Flow Matching
    print("\n=== Training Flow Matching (VP Path) ===")
    model_vp = VectorFieldNet().to(device)
    losses_vp = train_cfm(model_vp, sample_data, n_steps=5000, path_type='vp', device=device)

    # Sample trajectories
    traj_ot = sample_flow(model_ot, n_samples=200, n_steps=50, device=device)
    traj_vp = sample_flow(model_vp, n_samples=200, n_steps=50, device=device)

    # ── Visualization ──

    # 1. Training loss comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    window = 50
    loss_ot_smooth = np.convolve(losses_ot, np.ones(window)/window, mode='valid')
    loss_vp_smooth = np.convolve(losses_vp, np.ones(window)/window, mode='valid')
    ax.plot(loss_ot_smooth, label='OT Path', color='blue')
    ax.plot(loss_vp_smooth, label='VP (Diffusion) Path', color='red')
    ax.set_title("Flow Matching Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "loss_comparison.png", dpi=150)
    plt.close()

    # 2. Trajectory visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    target = sample_data(500, device).cpu().numpy()

    for ax, traj, title in zip(axes, [traj_ot, traj_vp], ['OT Path', 'VP (Diffusion) Path']):
        # Show trajectories for a few samples
        n_show = 20
        for i in range(n_show):
            path = np.array([t[i] for t in traj])
            ax.plot(path[:, 0], path[:, 1], alpha=0.3, linewidth=0.5, color='steelblue')
            ax.scatter(path[0, 0], path[0, 1], s=10, color='red', zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], s=10, color='green', zorder=5)

        ax.scatter(target[:, 0], target[:, 1], alpha=0.1, s=5, color='green', label='Target data')
        ax.set_title(title)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(results_dir / "trajectories.png", dpi=150)
    plt.close()

    # 3. Generated samples
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, title in zip(axes, [model_ot, model_vp], ['OT Path', 'VP Path']):
        traj = sample_flow(model, n_samples=1000, n_steps=100, device=device)
        final = traj[-1]
        ax.scatter(final[:, 0], final[:, 1], alpha=0.3, s=5)
        ax.scatter(target[:, 0], target[:, 1], alpha=0.1, s=5, color='red', label='Real data')
        ax.set_title(f"Generated Samples ({title})")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(results_dir / "generated_samples.png", dpi=150)
    plt.close()

    # 4. Sampling efficiency: fewer steps
    fig, ax = plt.subplots(figsize=(8, 4))
    for n_steps in [5, 10, 20, 50, 100]:
        traj = sample_flow(model_ot, n_samples=500, n_steps=n_steps, device=device)
        final = traj[-1]
        ax.scatter(final[:, 0], final[:, 1], alpha=0.2, s=3, label=f'{n_steps} steps')
    ax.set_title("OT Flow: Sampling with Different Step Counts")
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(results_dir / "sampling_efficiency.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
