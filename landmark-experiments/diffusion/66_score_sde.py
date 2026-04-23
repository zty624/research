"""
Minimal Score-Based SDE Reproduction
=====================================
Reproduces core ideas from Score-Based Generative Modeling through SDEs
(2011.13456, Song et al.):
1. VE-SDE (variance exploding, corresponds to SMLD): dx = sqrt(d[σ²(t)]/dt) dw
2. VP-SDE (variance preserving, corresponds to DDPM): dx = -½β(t)x dt + sqrt(β(t)) dw
3. Reverse SDE: dx = [f(x,t) - g(t)²∇_x log p_t(x)] dt + g(t) dw̄
4. Probability Flow ODE: dx = [f(x,t) - ½g(t)²∇_x log p_t(x)] dt (deterministic)
5. Predictor-Corrector sampling: SDE predictor step + Langevin corrector step
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Score Network ──

class ScoreNetwork(nn.Module):
    """MLP that takes (x, t) → score ∇_x log p_t(x)."""
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # Initialize last layer near zero for stable training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        # t: (B,) in [0, 1]
        t_emb = t.unsqueeze(-1)  # (B, 1)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


# ── SDE Definitions ──

class VESDE:
    """Variance Exploding SDE: dx = sqrt(d[σ²(t)]/dt) dw.

    σ(t) = σ_min * (σ_max/σ_min)^t
    g(t) = σ(t) * sqrt(2 * log(σ_max/σ_min))
    """
    def __init__(self, sigma_min=0.01, sigma_max=5.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = np.log(sigma_max / sigma_min)

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def g(self, t):
        """Diffusion coefficient g(t)."""
        return self.sigma(t) * np.sqrt(2 * self.log_ratio)

    def g_torch(self, t):
        """Diffusion coefficient g(t) as tensor."""
        s = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return s * np.sqrt(2 * self.log_ratio)

    def marginal_prob_std(self, t):
        """Standard deviation of the marginal distribution p_t(x|x_0)."""
        return self.sigma(t)

    def perturb(self, x0, t, noise=None):
        """Add noise to x0 at time t: x_t = x0 + σ(t) * noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        std = self.marginal_prob_std(t.item()) if t.dim() == 0 else torch.tensor(self.marginal_prob_std(t.item()))
        # Handle batch of t
        if t.dim() > 0:
            std = torch.tensor([self.marginal_prob_std(ti.item()) for ti in t],
                               device=x0.device).unsqueeze(-1)
        else:
            std = torch.tensor(self.marginal_prob_std(t.item()), device=x0.device)
        return x0 + std * noise, noise, std

    def drift(self, x, t):
        """Drift f(x,t) = 0 for VE-SDE."""
        return torch.zeros_like(x)


class VPSDE:
    """Variance Preserving SDE: dx = -½β(t)x dt + sqrt(β(t)) dw.

    β(t) linearly interpolated from β_min to β_max.
    Marginal: x_t = α(t)x_0 + sqrt(1 - α(t)²) noise
    where α(t) = exp(-½ ∫₀ᵗ β(s) ds)
    """
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_cumprod(self, t):
        """∫₀ᵗ β(s) ds = β_min*t + ½(β_max - β_min)*t²"""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return np.exp(-0.5 * integral)

    def marginal_prob_std(self, t):
        """Standard deviation of marginal: sqrt(1 - α(t)²)."""
        a = self.alpha_cumprod(t)
        return np.sqrt(1 - a ** 2)

    def g(self, t):
        return np.sqrt(self.beta(t))

    def g_torch(self, t):
        return torch.sqrt(torch.tensor(self.beta(t.item()), device=t.device)) if t.dim() == 0 else torch.sqrt(self.beta(t))

    def perturb(self, x0, t, noise=None):
        """x_t = α(t)x_0 + sqrt(1-α(t)²) noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_cumprod(t.item()) if t.dim() == 0 else torch.tensor([self.alpha_cumprod(ti.item()) for ti in t], device=x0.device).unsqueeze(-1)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=x0.device)
        std = torch.sqrt(1 - a ** 2 + 1e-8)
        return a * x0 + std * noise, noise, std

    def drift(self, x, t):
        """f(x,t) = -½β(t)x."""
        b = self.beta(t.item()) if not isinstance(t, torch.Tensor) else self.beta(t)
        return -0.5 * b * x


# ── Training: Denoising Score Matching ──

def train_score_model(score_net, sde, data_fn, n_steps=2500, batch_size=256, lr=1e-3, device='cpu'):
    """Train score network with denoising score matching loss.

    Loss = E_{t~U[ε,1]} E_{x_0~p_data} E_{noise} [ λ(t) || s_θ(x_t, t) - noise/σ(t) ||² ]
    where λ(t) = σ(t)² (likelihood weighting).
    """
    optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    eps = 1e-5  # avoid t=0 singularity

    for step in range(n_steps):
        x0 = data_fn(batch_size).to(device)
        # Sample t uniformly
        t = torch.rand(batch_size, device=device) * (1 - eps) + eps

        # Add noise according to SDE
        if isinstance(sde, VESDE):
            # VE: x_t = x_0 + σ(t) * noise, score target = -noise / σ(t)
            stds = torch.tensor([sde.marginal_prob_std(ti.item()) for ti in t],
                                device=device).unsqueeze(-1)  # (B, 1)
            noise = torch.randn_like(x0)
            xt = x0 + stds * noise
            target = -noise / stds  # score = ∇ log p(x_t | x_0) = -(x_t - x_0)/σ²
        else:
            # VP: x_t = α(t)x_0 + sqrt(1-α²)noise, score target = -noise / sqrt(1-α²)
            alphas = torch.tensor([sde.alpha_cumprod(ti.item()) for ti in t],
                                  device=device).unsqueeze(-1)
            noise = torch.randn_like(x0)
            stds = torch.sqrt(1 - alphas ** 2 + 1e-8)
            xt = alphas * x0 + stds * noise
            target = -noise / stds

        # Predict score
        score_pred = score_net(xt, t)

        # Loss with likelihood weighting: λ(t) = σ(t)² for VE, 1-α² for VP
        if isinstance(sde, VESDE):
            weight = stds ** 2
        else:
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


# ── Sampling ──

def sample_reverse_sde(score_net, sde, n_samples=1000, n_steps=500, device='cpu'):
    """Euler-Maruyama sampling of the reverse SDE.

    Reverse SDE: dx = [f(x,t) - g(t)² s_θ(x,t)] dt + g(t) dw̄
    Discretized: x_{t-Δt} = x_t + [f(x,t) - g(t)² s_θ(x,t)] Δt + g(t) sqrt(Δt) z
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device)  # start from N(0, I) for VE or adjusted for VP

    if isinstance(sde, VPSDE):
        # For VP, start from approximately N(0, I)
        x = x * 0.9  # scale down slightly

    trajectory = [x.detach().cpu().numpy()]

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        t = torch.full((n_samples,), t_val, device=device)

        with torch.no_grad():
            score = score_net(x, t)
            g = sde.g_torch(t).unsqueeze(-1) if callable(sde.g_torch) else sde.g_torch(t)

            if isinstance(sde, VESDE):
                # f(x,t) = 0 for VE-SDE
                drift = -g ** 2 * score
            else:
                # f(x,t) = -½β(t)x for VP-SDE
                f = sde.drift(x, t_val)
                drift = f - g ** 2 * score

            z = torch.randn_like(x) if t_val > dt else torch.zeros_like(x)
            x = x + drift * dt + g * np.sqrt(dt) * z

        if i % 50 == 0:
            trajectory.append(x.detach().cpu().numpy())

    trajectory.append(x.detach().cpu().numpy())
    return x.detach().cpu().numpy(), trajectory


def sample_probability_flow_ode(score_net, sde, n_samples=1000, n_steps=500, device='cpu'):
    """RK4 sampling of the probability flow ODE.

    ODE: dx = [f(x,t) - ½g(t)² s_θ(x,t)] dt
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device)

    if isinstance(sde, VPSDE):
        x = x * 0.9

    trajectory = [x.detach().cpu().numpy()]

    def ode_rhs(x, t_val):
        t = torch.full((x.shape[0],), t_val, device=device)
        with torch.no_grad():
            score = score_net(x, t)
            g = sde.g_torch(t).unsqueeze(-1) if callable(sde.g_torch) else sde.g_torch(t)

            if isinstance(sde, VESDE):
                return -0.5 * g ** 2 * score
            else:
                f = sde.drift(x, t_val)
                return f - 0.5 * g ** 2 * score

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        # RK4
        k1 = ode_rhs(x, t_val)
        k2 = ode_rhs(x + 0.5 * dt * k1, t_val - 0.5 * dt)
        k3 = ode_rhs(x + 0.5 * dt * k2, t_val - 0.5 * dt)
        k4 = ode_rhs(x + dt * k3, t_val - dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if i % 50 == 0:
            trajectory.append(x.detach().cpu().numpy())

    trajectory.append(x.detach().cpu().numpy())
    return x.detach().cpu().numpy(), trajectory


def sample_pc(score_net, sde, n_samples=1000, n_steps=500, n_corrector_steps=3,
              snr=0.1, device='cpu'):
    """Predictor-Corrector sampling.

    Predictor: one reverse SDE step (Euler-Maruyama)
    Corrector: K steps of Langevin dynamics: x = x + (ε/2) s_θ(x,t) + sqrt(ε) z
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device)

    if isinstance(sde, VPSDE):
        x = x * 0.9

    trajectory = [x.detach().cpu().numpy()]

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        t = torch.full((n_samples,), t_val, device=device)

        with torch.no_grad():
            score = score_net(x, t)
            g = sde.g_torch(t).unsqueeze(-1) if callable(sde.g_torch) else sde.g_torch(t)

            # Predictor step (reverse SDE)
            if isinstance(sde, VESDE):
                drift = -g ** 2 * score
            else:
                f = sde.drift(x, t_val)
                drift = f - g ** 2 * score

            z_pred = torch.randn_like(x) if t_val > dt else torch.zeros_like(x)
            x = x + drift * dt + g * np.sqrt(dt) * z_pred

            # Corrector steps (Langevin dynamics)
            for _ in range(n_corrector_steps):
                score_c = score_net(x, t)
                # Langevin step size from SNR: ε = 2 * snr / ||s_θ||²
                score_norm_sq = (score_c ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
                step_size = 2 * snr / score_norm_sq
                z_corr = torch.randn_like(x)
                x = x + step_size * score_c + torch.sqrt(step_size) * z_corr

        if i % 50 == 0:
            trajectory.append(x.detach().cpu().numpy())

    trajectory.append(x.detach().cpu().numpy())
    return x.detach().cpu().numpy(), trajectory


# ── 2D Toy Data ──

def make_mixture_of_gaussians(n, device='cpu'):
    """Mixture of 5 Gaussians arranged in a circle + center."""
    centers = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    samples = []
    per_center = n // len(centers)
    for cx, cy in centers:
        x = np.random.randn(per_center, 2) * 0.3 + np.array([cx, cy])
        samples.append(x)
    samples = np.concatenate(samples, axis=0)[:n]
    return torch.tensor(samples, dtype=torch.float32, device=device)


def make_swiss_roll(n, device='cpu'):
    """2D Swiss roll."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t) / 21.0
    y = t * np.sin(t) / 21.0
    samples = np.stack([x, y], axis=1) + np.random.randn(n, 2) * 0.03
    return torch.tensor(samples, dtype=torch.float32, device=device)


def make_moons(n, device='cpu'):
    """Two moons."""
    n_half = n // 2
    theta = np.linspace(0, np.pi, n_half)
    x1 = np.cos(theta) + np.random.randn(n_half) * 0.1
    y1 = np.sin(theta) + np.random.randn(n_half) * 0.1
    x2 = 1 - np.cos(theta) + np.random.randn(n_half) * 0.1
    y2 = 1 - np.sin(theta) + np.random.randn(n_half) * 0.1 - 0.5
    samples = np.concatenate([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y2], axis=1)
    ], axis=0)[:n]
    return torch.tensor(samples, dtype=torch.float32, device=device)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "66-score-sde"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_train = 256   # batch size per step
    n_steps = 2500  # training steps
    n_samples = 1000

    # ── Experiment 1: VE-SDE vs VP-SDE on mixture of Gaussians ──
    print("=== Experiment 1: VE-SDE vs VP-SDE (Mixture of Gaussians) ===")

    data_gmm = make_mixture_of_gaussians(5000, device=device)
    data_fn_gmm = lambda bs: make_mixture_of_gaussians(bs, device=device)
    data_gmm_np = data_gmm.cpu().numpy()

    ve_sde = VESDE(sigma_min=0.01, sigma_max=5.0)
    vp_sde = VPSDE(beta_min=0.1, beta_max=20.0)

    # Train VE score model
    print("\n  Training VE-SDE score model:")
    score_ve = ScoreNetwork(dim=2, hidden=128).to(device)
    ve_losses = train_score_model(score_ve, ve_sde, data_fn_gmm, n_steps=n_steps,
                                  batch_size=n_train, lr=1e-3, device=device)

    # Train VP score model
    print("\n  Training VP-SDE score model:")
    score_vp = ScoreNetwork(dim=2, hidden=128).to(device)
    vp_losses = train_score_model(score_vp, vp_sde, data_fn_gmm, n_steps=n_steps,
                                  batch_size=n_train, lr=1e-3, device=device)

    # Sample from both
    print("\n  Sampling VE-SDE (reverse SDE)...")
    ve_sde_samples, ve_sde_traj = sample_reverse_sde(score_ve, ve_sde, n_samples, device=device)
    print("  Sampling VP-SDE (reverse SDE)...")
    vp_sde_samples, vp_sde_traj = sample_reverse_sde(score_vp, vp_sde, n_samples, device=device)

    # ── Visualization 1: VE-SDE vs VP-SDE comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(data_gmm_np[:, 0], data_gmm_np[:, 1], s=2, alpha=0.3, c='blue')
    axes[0].set_title("Training Data", fontsize=12)
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')

    axes[1].scatter(ve_sde_samples[:, 0], ve_sde_samples[:, 1], s=2, alpha=0.3, c='red')
    axes[1].set_title("VE-SDE Samples", fontsize=12)
    axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')

    axes[2].scatter(vp_sde_samples[:, 0], vp_sde_samples[:, 1], s=2, alpha=0.3, c='green')
    axes[2].set_title("VP-SDE Samples", fontsize=12)
    axes[2].set_xlim(-4, 4); axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal')

    plt.suptitle("Score-Based SDE: VE-SDE vs VP-SDE (Mixture of Gaussians)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "ve_vs_vp.png", dpi=150)
    plt.close()

    # ── Visualization 2: Sampling trajectories ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, traj, title in [(axes[0], ve_sde_traj, "VE-SDE Trajectories"),
                             (axes[1], vp_sde_traj, "VP-SDE Trajectories")]:
        n_show = 5  # show 5 sample trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(traj)))
        for i in range(n_show):
            points = np.array([traj[j][i] for j in range(len(traj))])
            ax.plot(points[:, 0], points[:, 1], '-', alpha=0.6, linewidth=1.5)
            ax.plot(points[0, 0], points[0, 1], 'o', color='red', markersize=6, label='Start (noise)' if i == 0 else None)
            ax.plot(points[-1, 0], points[-1, 1], 's', color='blue', markersize=6, label='End (data)' if i == 0 else None)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Reverse SDE Sampling: Noise → Data Trajectories", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "sampling_trajectories.png", dpi=150)
    plt.close()

    # ── Experiment 2: Reverse SDE vs Probability Flow ODE ──
    print("\n=== Experiment 2: Reverse SDE vs Probability Flow ODE ===")

    print("  Sampling with Probability Flow ODE (VE)...")
    ve_ode_samples, ve_ode_traj = sample_probability_flow_ode(score_ve, ve_sde, n_samples, device=device)
    print("  Sampling with Probability Flow ODE (VP)...")
    vp_ode_samples, vp_ode_traj = sample_probability_flow_ode(score_vp, vp_sde, n_samples, device=device)

    # ── Visualization 3: Reverse SDE vs Prob Flow ODE ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].scatter(ve_sde_samples[:, 0], ve_sde_samples[:, 1], s=2, alpha=0.3, c='red')
    axes[0, 0].set_title("VE-SDE: Reverse SDE", fontsize=11)
    axes[0, 1].scatter(ve_ode_samples[:, 0], ve_ode_samples[:, 1], s=2, alpha=0.3, c='darkred')
    axes[0, 1].set_title("VE-SDE: Prob Flow ODE", fontsize=11)

    axes[1, 0].scatter(vp_sde_samples[:, 0], vp_sde_samples[:, 1], s=2, alpha=0.3, c='green')
    axes[1, 0].set_title("VP-SDE: Reverse SDE", fontsize=11)
    axes[1, 1].scatter(vp_ode_samples[:, 0], vp_ode_samples[:, 1], s=2, alpha=0.3, c='darkgreen')
    axes[1, 1].set_title("VP-SDE: Prob Flow ODE", fontsize=11)

    for ax in axes.flat:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Reverse SDE vs Probability Flow ODE", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "reverse_sde_vs_ode.png", dpi=150)
    plt.close()

    # ── Experiment 3: PC Sampling vs Pure Predictor ──
    print("\n=== Experiment 3: PC Sampling vs Pure Predictor ===")

    print("  Sampling with PC (VE)...")
    ve_pc_samples, ve_pc_traj = sample_pc(score_ve, ve_sde, n_samples, n_steps=500,
                                          n_corrector_steps=3, snr=0.1, device=device)
    print("  Sampling with PC (VP)...")
    vp_pc_samples, vp_pc_traj = sample_pc(score_vp, vp_sde, n_samples, n_steps=500,
                                           n_corrector_steps=3, snr=0.1, device=device)

    # ── Visualization 4: PC vs pure predictor ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].scatter(ve_sde_samples[:, 0], ve_sde_samples[:, 1], s=2, alpha=0.3, c='red')
    axes[0, 0].set_title("VE: Pure Predictor (Reverse SDE)", fontsize=10)
    axes[0, 1].scatter(ve_pc_samples[:, 0], ve_pc_samples[:, 1], s=2, alpha=0.3, c='orangered')
    axes[0, 1].set_title("VE: Predictor-Corrector", fontsize=10)

    axes[1, 0].scatter(vp_sde_samples[:, 0], vp_sde_samples[:, 1], s=2, alpha=0.3, c='green')
    axes[1, 0].set_title("VP: Pure Predictor (Reverse SDE)", fontsize=10)
    axes[1, 1].scatter(vp_pc_samples[:, 0], vp_pc_samples[:, 1], s=2, alpha=0.3, c='limegreen')
    axes[1, 1].set_title("VP: Predictor-Corrector", fontsize=10)

    for ax in axes.flat:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Predictor-Corrector vs Pure Predictor Sampling", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "pc_vs_predictor.png", dpi=150)
    plt.close()

    # ── Visualization 5: Score field visualization ──
    print("\n  Generating score field visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    t_values = [0.1, 0.5, 0.9]  # early, mid, late noise levels
    grid_n = 20
    xx = np.linspace(-4, 4, grid_n)
    yy = np.linspace(-4, 4, grid_n)
    XX, YY = np.meshgrid(xx, yy)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)

    for idx, t_val in enumerate(t_values):
        t_batch = torch.full((grid_n ** 2,), t_val, device=device)
        with torch.no_grad():
            score_vals = score_ve(grid_tensor, t_batch).cpu().numpy()

        U = score_vals[:, 0].reshape(grid_n, grid_n)
        V = score_vals[:, 1].reshape(grid_n, grid_n)
        magnitude = np.sqrt(U ** 2 + V ** 2 + 1e-8)
        # Normalize arrows for visibility
        U_norm = U / magnitude
        V_norm = V / magnitude

        ax = axes[idx]
        ax.quiver(XX, YY, U_norm, V_norm, magnitude, cmap='coolwarm', alpha=0.7)
        ax.set_title(f"Score Field (VE-SDE, t={t_val})", fontsize=11)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.suptitle("Score Field ∇_x log p_t(x) at Different Noise Levels", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "score_field.png", dpi=150)
    plt.close()

    # ── Visualization 6: Training curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 50
    for losses, label, color in [(ve_losses, 'VE-SDE', 'red'),
                                  (vp_losses, 'VP-SDE', 'green')]:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=label, color=color, linewidth=2)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("DSM Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Swiss roll experiment
    print("\n=== Experiment 4: Swiss Roll with VE-SDE ===")
    data_sr = make_swiss_roll(5000, device=device)
    data_fn_sr = lambda bs: make_swiss_roll(bs, device=device)
    data_sr_np = data_sr.cpu().numpy()

    score_sr = ScoreNetwork(dim=2, hidden=128).to(device)
    sr_losses = train_score_model(score_sr, ve_sde, data_fn_sr, n_steps=2500,
                                  batch_size=n_train, lr=1e-3, device=device)

    sr_sde_samples, _ = sample_reverse_sde(score_sr, ve_sde, n_samples, device=device)
    sr_ode_samples, _ = sample_probability_flow_ode(score_sr, ve_sde, n_samples, device=device)

    axes[1].scatter(data_sr_np[:, 0], data_sr_np[:, 1], s=2, alpha=0.2, c='blue', label='Data')
    axes[1].scatter(sr_sde_samples[:, 0], sr_sde_samples[:, 1], s=2, alpha=0.2, c='red', label='Generated')
    axes[1].set_title("Swiss Roll: Data vs Generated", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Score-Based SDE: Training & Generation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_and_swiss_roll.png", dpi=150)
    plt.close()

    # ── Visualization 7: Concept diagram ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    concepts = [
        ("Forward SDE\n(Noise Injection)",
         "dx = f(x,t)dt + g(t)dw\n\n"
         "VE: dx = √(dσ²/dt) dw\n"
         "VP: dx = -½β(t)x dt + √β(t) dw\n\n"
         "Data → Noise",
         0.14, 'gray'),
        ("Reverse SDE\n(Generative)",
         "dx = [f - g²∇log p_t] dt + g dw̄\n\n"
         "Only needs score ∇_x log p_t(x)\n"
         "Stochastic: same marginals\n\n"
         "Noise → Data (stochastic)",
         0.5, 'blue'),
        ("Prob Flow ODE\n(Deterministic)",
         "dx = [f - ½g²∇log p_t] dt\n\n"
         "Same marginals as reverse SDE\n"
         "Exact likelihood via ODE\n\n"
         "Noise → Data (deterministic)",
         0.86, 'green'),
    ]

    for name, desc, x_pos, color in concepts:
        ax.text(x_pos, 0.78, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Arrows between concepts
    ax.annotate('', xy=(0.32, 0.55), xytext=(0.24, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.68, 0.55), xytext=(0.6, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(0.28, 0.6, 'reverse\ntime', fontsize=8, ha='center', color='gray')
    ax.text(0.64, 0.6, 'remove\nnoise', fontsize=8, ha='center', color='gray')

    ax.set_title("Score-Based SDE: Unified Framework (2011.13456)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "concept_diagram.png", dpi=150)
    plt.close()

    # ── Visualization 8: Noising process (forward SDE) ──
    print("  Generating noising process visualization...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    t_noise = [0.0, 0.1, 0.3, 0.6, 1.0]
    sample_data = data_gmm_np[:500]

    for idx, t_val in enumerate(t_noise):
        if t_val == 0.0:
            noisy = sample_data
        else:
            t_tensor = torch.full((500,), t_val)
            std = ve_sde.marginal_prob_std(t_val)
            noise = np.random.randn(500, 2)
            noisy = sample_data + std * noise

        axes[idx].scatter(noisy[:, 0], noisy[:, 1], s=3, alpha=0.3, c='blue')
        axes[idx].set_title(f"t = {t_val}", fontsize=11)
        axes[idx].set_xlim(-6, 6); axes[idx].set_ylim(-6, 6)
        axes[idx].set_aspect('equal')

    plt.suptitle("Forward VE-SDE: Data → Noise", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "noising_process.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
