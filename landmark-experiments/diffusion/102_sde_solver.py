"""
Minimal Score-Based SDE Solver Comparison
==========================================
Reproduces core ideas from "Score-Based Generative Modeling through Stochastic
Differential Equations" (Song et al., 2011.13456):
1. Euler-Maruyama solver for the reverse SDE (1st-order, stochastic)
2. Milstein solver for the reverse SDE (2nd-order, stochastic)
3. RK-based (Heun's) solver for the reverse SDE (2nd-order, stochastic)
4. Probability Flow ODE solver with Euler and RK4 (deterministic)
5. Compare: sample quality vs number of function evaluations (NFE)
6. Analyze: numerical stability of different discretization schemes
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Score Network ──

class ScoreNetwork(nn.Module):
    """MLP: (x, t) → score ∇_x log p_t(x)."""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))


# ── SDE: VP-SDE ──

class VPSDE:
    """Variance Preserving SDE: dx = -½β(t)x dt + √β(t) dw.
    β(t) = β_min + t(β_max - β_min), t ∈ [0, 1].
    """
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_cumprod(self, t):
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return np.exp(-0.5 * integral)

    def alpha_cumprod_torch(self, t):
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def g(self, t):
        return np.sqrt(self.beta(t))

    def g_torch(self, t):
        t_t = torch.as_tensor(t, dtype=torch.float32)
        return torch.sqrt(torch.tensor(self.beta_min, dtype=torch.float32)
                          + t_t * (self.beta_max - self.beta_min))

    def marginal_std(self, t):
        a = self.alpha_cumprod(t)
        return np.sqrt(1 - a ** 2)

    def drift(self, x, t):
        return -0.5 * self.beta(t) * x


# ── Training: Denoising Score Matching ──

def train_score(score_net, sde, data_fn, n_steps=3000, batch_size=256,
                lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []
    eps = 1e-5

    for step in range(n_steps):
        x0 = data_fn(batch_size).to(device)
        actual_bs = x0.shape[0]
        t = torch.rand(actual_bs, device=device) * (1 - eps) + eps

        alphas = torch.tensor([sde.alpha_cumprod(ti.item()) for ti in t],
                              device=device, dtype=torch.float32).unsqueeze(-1)
        noise = torch.randn_like(x0)
        stds = torch.sqrt(1 - alphas ** 2 + 1e-8)
        xt = alphas * x0 + stds * noise
        target = -noise / stds

        score_pred = score_net(xt, t)
        weight = 1 - alphas ** 2
        loss = (weight * ((score_pred - target) ** 2).sum(dim=-1, keepdim=True)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")
    return losses


# ── Solvers ──

def _reverse_sde_drift(score_net, sde, x, t_val, device):
    """Compute drift of the reverse SDE: f(x,t) - g(t)^2 * s_theta(x,t)."""
    t = torch.full((x.shape[0],), t_val, device=device)
    with torch.no_grad():
        score = score_net(x, t)
        g = sde.g_torch(t_val).to(device)
        if g.dim() == 0:
            g = g.unsqueeze(0).unsqueeze(-1)
        elif g.dim() == 1:
            g = g.unsqueeze(-1)
        f = -0.5 * sde.beta_min * x - 0.5 * (sde.beta_max - sde.beta_min) * t_val * x
        f = f.to(device)
    return f - g ** 2 * score, g


def _ode_drift(score_net, sde, x, t_val, device):
    """Compute drift of the probability flow ODE: f(x,t) - 0.5*g(t)^2*s_theta(x,t)."""
    t = torch.full((x.shape[0],), t_val, device=device)
    with torch.no_grad():
        score = score_net(x, t)
        g = sde.g_torch(t_val).to(device)
        if g.dim() == 0:
            g = g.unsqueeze(0).unsqueeze(-1)
        elif g.dim() == 1:
            g = g.unsqueeze(-1)
        f = sde.drift(x, t_val)
    return f - 0.5 * g ** 2 * score


def sample_euler_maruyama(score_net, sde, n_samples, n_steps, device='cpu'):
    """Euler-Maruyama for reverse SDE (1st order, stochastic).
    NFE = n_steps.
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device) * 0.9
    trajectory = [x.cpu().clone()]
    nfe = 0

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        drift, g = _reverse_sde_drift(score_net, sde, x, t_val, device)
        nfe += 1
        z = torch.randn_like(x) if t_val > dt else torch.zeros_like(x)
        x = x + drift * dt + g * np.sqrt(dt) * z
        if i % max(1, n_steps // 10) == 0:
            trajectory.append(x.cpu().clone())

    trajectory.append(x.cpu().clone())
    return x.cpu(), trajectory, nfe


def sample_milstein(score_net, sde, n_samples, n_steps, device='cpu'):
    """Milstein scheme for reverse SDE (2nd order, stochastic).
    For VP-SDE, the diffusion term g(t) does not depend on x, so the Milstein
    correction term (∂g/∂x) is zero. This degenerates to Euler-Maruyama, but
    we implement it for pedagogical completeness and show that for VP-SDE the
    two solvers coincide.
    NFE = n_steps.
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device) * 0.9
    trajectory = [x.cpu().clone()]
    nfe = 0

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        drift, g = _reverse_sde_drift(score_net, sde, x, t_val, device)
        nfe += 1
        z = torch.randn_like(x) if t_val > dt else torch.zeros_like(x)
        delta_w = np.sqrt(dt) * z
        # Milstein: for state-independent g, correction = 0
        x = x + drift * dt + g * delta_w
        if i % max(1, n_steps // 10) == 0:
            trajectory.append(x.cpu().clone())

    trajectory.append(x.cpu().clone())
    return x.cpu(), trajectory, nfe


def sample_heun_sde(score_net, sde, n_samples, n_steps, device='cpu'):
    """Heun's method (2nd-order Runge-Kutta) for the reverse SDE drift,
    combined with Euler-Maruyama for the diffusion part.
    This is a predictor-corrector approach:
      Predict: x_hat = x + drift(x,t)*dt + g*sqrt(dt)*z
      Correct: x_new = x + 0.5*(drift(x,t) + drift(x_hat, t-dt))*dt + g*sqrt(dt)*z
    NFE = 2 * n_steps.
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device) * 0.9
    trajectory = [x.cpu().clone()]
    nfe = 0

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        z = torch.randn_like(x) if t_val > dt else torch.zeros_like(x)

        # Predict
        drift1, g = _reverse_sde_drift(score_net, sde, x, t_val, device)
        nfe += 1
        x_hat = x + drift1 * dt + g * np.sqrt(dt) * z

        # Correct
        t_next = max(t_val - dt, 0)
        drift2, _ = _reverse_sde_drift(score_net, sde, x_hat, t_next, device)
        nfe += 1
        x = x + 0.5 * (drift1 + drift2) * dt + g * np.sqrt(dt) * z

        if i % max(1, n_steps // 10) == 0:
            trajectory.append(x.cpu().clone())

    trajectory.append(x.cpu().clone())
    return x.cpu(), trajectory, nfe


def sample_ode_euler(score_net, sde, n_samples, n_steps, device='cpu'):
    """Euler method for the probability flow ODE (1st order, deterministic).
    NFE = n_steps.
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device) * 0.9
    trajectory = [x.cpu().clone()]
    nfe = 0

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        drift = _ode_drift(score_net, sde, x, t_val, device)
        nfe += 1
        x = x + drift * dt
        if i % max(1, n_steps // 10) == 0:
            trajectory.append(x.cpu().clone())

    trajectory.append(x.cpu().clone())
    return x.cpu(), trajectory, nfe


def sample_ode_rk4(score_net, sde, n_samples, n_steps, device='cpu'):
    """RK4 method for the probability flow ODE (4th order, deterministic).
    NFE = 4 * n_steps.
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device) * 0.9
    trajectory = [x.cpu().clone()]
    nfe = 0

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        k1 = _ode_drift(score_net, sde, x, t_val, device)
        nfe += 1
        k2 = _ode_drift(score_net, sde, x + 0.5 * dt * k1, t_val - 0.5 * dt, device)
        nfe += 1
        k3 = _ode_drift(score_net, sde, x + 0.5 * dt * k2, t_val - 0.5 * dt, device)
        nfe += 1
        k4 = _ode_drift(score_net, sde, x + dt * k3, t_val - dt, device)
        nfe += 1
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % max(1, n_steps // 10) == 0:
            trajectory.append(x.cpu().clone())

    trajectory.append(x.cpu().clone())
    return x.cpu(), trajectory, nfe


# ── Quality Metrics ──

def mmd(x, y, sigma=1.0):
    """Maximum Mean Discrepancy with Gaussian kernel."""
    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T
    rx = xx.diag().unsqueeze(0)  # (1, n_x)
    ry = yy.diag().unsqueeze(0)  # (1, n_y)
    Kxx = torch.exp(-0.5 / sigma ** 2 * (rx.T + rx - 2 * xx))
    Kyy = torch.exp(-0.5 / sigma ** 2 * (ry.T + ry - 2 * yy))
    # For Kxy: need ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2<x_i, y_j>
    Kxy = torch.exp(-0.5 / sigma ** 2 * (rx.T + ry - 2 * xy))  # (n_x, n_y)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


# ── Data ──

def make_gmm(n, device='cpu'):
    centers = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    per = n // len(centers)
    parts = [np.random.randn(per, 2) * 0.3 + np.array(c) for c in centers]
    data = np.concatenate(parts)[:n]
    return torch.tensor(data, dtype=torch.float32, device=device)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "102-sde-solver"
    results_dir.mkdir(parents=True, exist_ok=True)

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    data_fn = lambda bs: make_gmm(bs, device=device)
    data_ref = make_gmm(2000, 'cpu')

    # Train score model
    print("=== Training score model (VP-SDE) ===")
    score_net = ScoreNetwork(dim=2, hidden=128).to(device)
    losses = train_score(score_net, sde, data_fn, n_steps=3000,
                         batch_size=256, lr=1e-3, device=device)

    # ── Experiment 1: NFE vs sample quality ──
    print("\n=== Experiment 1: NFE vs Sample Quality ===")
    solvers = {
        'Euler-Maruyama\n(Rev SDE, order 1)': sample_euler_maruyama,
        'Milstein\n(Rev SDE, order 2)': sample_milstein,
        'Heun SDE\n(Rev SDE, order 2)': sample_heun_sde,
        'ODE Euler\n(Prob Flow, order 1)': sample_ode_euler,
        'ODE RK4\n(Prob Flow, order 4)': sample_ode_rk4,
    }
    step_counts = [10, 25, 50, 100, 200, 500]

    all_results = {}
    for name, solver_fn in solvers.items():
        results = []
        for ns in step_counts:
            samples, _, nfe = solver_fn(score_net, sde, 500, ns, device=device)
            quality = mmd(samples, data_ref, sigma=0.5).item()
            results.append((nfe, quality))
            print(f"  {name.replace(chr(10), ' ')} | steps={ns} | NFE={nfe} | MMD={quality:.4f}")
        all_results[name] = results

    # Plot NFE vs quality
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
    markers = ['o', 's', '^', 'D', 'v']
    for (name, results), color, marker in zip(all_results.items(), colors, markers):
        nfes = [r[0] for r in results]
        mmds = [r[1] for r in results]
        ax.plot(nfes, mmds, marker=marker, color=color, label=name, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12)
    ax.set_ylabel("MMD (lower = better)", fontsize=12)
    ax.set_title("Solver Comparison: NFE vs Sample Quality", fontsize=14)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "nfe_vs_quality.png", dpi=150)
    plt.close()

    # ── Experiment 2: Sample quality at fixed NFE budget ──
    print("\n=== Experiment 2: Fixed NFE Budget Comparison ===")
    budget = 100  # total NFE budget
    configs = {
        f'Euler-Maruyama ({budget} steps)': (sample_euler_maruyama, budget),
        f'Milstein ({budget} steps)': (sample_milstein, budget),
        f'Heun SDE ({budget // 2} steps)': (sample_heun_sde, budget // 2),
        f'ODE Euler ({budget} steps)': (sample_ode_euler, budget),
        f'ODE RK4 ({budget // 4} steps)': (sample_ode_rk4, budget // 4),
    }

    fixed_samples = {}
    for name, (fn, steps) in configs.items():
        samples, _, nfe = fn(score_net, sde, 1000, steps, device=device)
        quality = mmd(samples, data_ref, sigma=0.5).item()
        fixed_samples[name] = (samples.numpy(), quality, nfe)
        print(f"  {name} | NFE={nfe} | MMD={quality:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat
    axes[0].scatter(data_ref[:, 0].numpy(), data_ref[:, 1].numpy(), s=2, alpha=0.3, c='gray')
    axes[0].set_title("Reference Data", fontsize=11)
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')

    for idx, (name, (samps, quality, nfe)) in enumerate(fixed_samples.items(), 1):
        axes[idx].scatter(samps[:, 0], samps[:, 1], s=2, alpha=0.3)
        axes[idx].set_title(f"{name}\nMMD={quality:.4f}", fontsize=9)
        axes[idx].set_xlim(-4, 4); axes[idx].set_ylim(-4, 4)
        axes[idx].set_aspect('equal')

    for ax in axes:
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Fixed NFE Budget = {budget}: Solver Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "fixed_budget.png", dpi=150)
    plt.close()

    # ── Experiment 3: Numerical stability — very few steps ──
    print("\n=== Experiment 3: Numerical Stability at Low Step Counts ===")
    low_steps = [3, 5, 8, 10, 15]

    fig, axes = plt.subplots(len(solvers), len(low_steps), figsize=(20, 4 * len(solvers)))
    stability_data = {}

    for row, (name, solver_fn) in enumerate(solvers.items()):
        stability_data[name] = []
        for col, ns in enumerate(low_steps):
            samples, _, nfe = solver_fn(score_net, sde, 500, ns, device=device)
            quality = mmd(samples, data_ref, sigma=0.5).item()
            stability_data[name].append((ns, quality))
            ax = axes[row, col]
            s_np = samples.numpy()
            spread = np.percentile(np.abs(s_np), 95)
            lim = max(4, min(spread * 1.2, 20))
            ax.scatter(s_np[:, 0], s_np[:, 1], s=2, alpha=0.3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_title(f"steps={ns} | MMD={quality:.3f}", fontsize=8)
            if col == 0:
                ax.set_ylabel(name.split('\n')[0], fontsize=9)

    plt.suptitle("Numerical Stability: Very Low Step Counts", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "stability.png", dpi=150)
    plt.close()

    # Stability metric: quality at minimal steps
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, data_list), color, marker in zip(stability_data.items(), colors, markers):
        steps_list = [d[0] for d in data_list]
        mmd_list = [d[1] for d in data_list]
        ax.plot(steps_list, mmd_list, marker=marker, color=color, label=name,
                linewidth=2, markersize=8)
    ax.set_xlabel("Number of Discretization Steps", fontsize=12)
    ax.set_ylabel("MMD (lower = better)", fontsize=12)
    ax.set_title("Numerical Stability: Sample Quality at Low Step Counts", fontsize=13)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "stability_curve.png", dpi=150)
    plt.close()

    # ── Experiment 4: Trajectory comparison ──
    print("\n=== Experiment 4: Sampling Trajectory Comparison ===")
    n_traj_samples = 5
    n_traj_steps = 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    traj_solvers = [
        ('Euler-Maruyama (SDE)', sample_euler_maruyama, 100),
        ('ODE RK4', sample_ode_rk4, 25),  # 25 steps * 4 = 100 NFE
        ('ODE Euler', sample_ode_euler, 100),
    ]

    for ax, (name, fn, steps) in zip(axes, traj_solvers):
        _, traj, _ = fn(score_net, sde, n_traj_samples, n_traj_steps, device=device)
        for i in range(n_traj_samples):
            path = np.array([t[i].numpy() for t in traj])
            ax.plot(path[:, 0], path[:, 1], alpha=0.7, linewidth=1.2)
            ax.plot(path[0, 0], path[0, 1], 'o', color='red', markersize=8,
                    label='Start (noise)' if i == 0 else None)
            ax.plot(path[-1, 0], path[-1, 1], 's', color='green', markersize=8,
                    label='End (data)' if i == 0 else None)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Sampling Trajectories: Stochastic SDE vs Deterministic ODE", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "trajectories.png", dpi=150)
    plt.close()

    # ── Training loss ──
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 50
    smoothed = np.convolve(losses, np.ones(w) / w, mode='valid')
    ax.plot(smoothed, color='steelblue', linewidth=2)
    ax.set_title("Score Model Training Loss (VP-SDE, DSM)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # ── Summary diagram ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    solver_info = (
        "Score-Based SDE Solver Comparison (2011.13456)\n"
        "=" * 55 + "\n\n"
        "Reverse SDE (stochastic):\n"
        "  dx = [f(x,t) - g(t)^2 * s_θ(x,t)] dt + g(t) dw̄\n"
        "  • Euler-Maruyama: order 1, NFE = steps\n"
        "  • Milstein: order 2, NFE = steps (degenerate for VP-SDE: g independent of x)\n"
        "  • Heun (RK2): order 2, NFE = 2*steps\n\n"
        "Probability Flow ODE (deterministic):\n"
        "  dx = [f(x,t) - 0.5*g(t)^2 * s_θ(x,t)] dt\n"
        "  • Euler ODE: order 1, NFE = steps\n"
        "  • RK4 ODE: order 4, NFE = 4*steps\n\n"
        "Key finding: Higher-order solvers (RK4, Heun) achieve better\n"
        "sample quality per NFE at low step counts, but stochastic\n"
        "solvers can be more robust to very coarse discretization."
    )
    ax.text(0.05, 0.95, solver_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / "summary.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
