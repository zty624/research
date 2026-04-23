"""
Minimal Diffusion Policy Reproduction
======================================
Reproduces core ideas from Diffusion Policy (2303.04137, Chi et al.):
1. Use diffusion model for action generation (not just images!)
2. Condition on observations (state) via cross-attention or concatenation
3. Denoise action trajectories rather than single actions
4. Compare: deterministic policy vs diffusion policy on 1D control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Environment: 1D Target Tracking ──

class TargetTrackingEnv:
    """1D target tracking: agent must follow a moving target.
    State: [agent_pos, target_pos, target_vel]
    Action: agent velocity in [-1, 1]
    Reward: -|agent - target|
    """
    def __init__(self, horizon=50):
        self.horizon = horizon
        self.dt = 0.1

    def generate_trajectory(self, n_trajectories=1):
        """Generate expert trajectories using an oracle controller."""
        states = []
        actions = []
        rewards = []

        for _ in range(n_trajectories):
            agent_pos = 0.0
            target_pos = 0.0
            target_vel = 0.5

            traj_states = []
            traj_actions = []

            for t in range(self.horizon):
                # Target follows sinusoidal path
                target_vel = 2.0 * np.sin(0.3 * t * self.dt)
                target_pos += target_vel * self.dt

                state = np.array([agent_pos, target_pos, target_vel], dtype=np.float32)
                traj_states.append(state)

                # Oracle: move toward target with some smoothing
                error = target_pos - agent_pos
                action = np.clip(error * 2.0, -1.0, 1.0)
                traj_actions.append(action)

                agent_pos += action * self.dt

            states.append(traj_states)
            actions.append(traj_actions)

        return np.array(states), np.array(actions)


# ── Deterministic Policy ──

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


# ── Diffusion Policy ──

class DiffusionPolicy(nn.Module):
    """Diffusion-based policy: denoise action from noise, conditioned on state."""
    def __init__(self, state_dim=3, action_dim=1, hidden=128, n_diffusion_steps=20):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_diffusion_steps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        # State conditioning
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        # Denoising network
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim)
        )

        # Noise schedule (cosine)
        self.register_buffer('betas', self._cosine_schedule(n_diffusion_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bar', torch.cumprod(1 - self.betas, dim=0))

    def _cosine_schedule(self, T, s=0.008):
        steps = torch.arange(T + 1)
        f = torch.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return beta.clamp(0, 0.999)

    def forward(self, noisy_action, t, state):
        """Predict noise given noisy action, timestep, and state."""
        t_emb = self.time_embed(t.unsqueeze(-1))
        s_emb = self.state_embed(state)
        h = torch.cat([noisy_action, t_emb, s_emb], dim=-1)
        return self.net(h)

    def training_loss(self, state, action):
        """DDPM training loss: predict noise."""
        B = action.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=action.device)
        noise = torch.randn_like(action)

        # Add noise
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).unsqueeze(-1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).unsqueeze(-1)
        noisy_action = sqrt_ab * action + sqrt_omab * noise

        # Predict noise
        pred_noise = self.forward(noisy_action, t.float() / self.n_steps, state)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, state, n_steps=None):
        """Sample action by denoising from pure noise."""
        n_steps = n_steps or self.n_steps
        B = state.shape[0]

        # Start from noise
        x = torch.randn(B, self.action_dim, device=state.device)

        for t_idx in reversed(range(n_steps)):
            t = torch.full((B,), t_idx, device=state.device, dtype=torch.float32) / self.n_steps

            pred_noise = self.forward(x, t, state)

            # DDPM update
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]

            if t_idx > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t_idx])
            else:
                noise = torch.zeros_like(x)
                sigma = 0

            # x_{t-1} = (1/sqrt(α)) * (x - β/sqrt(1-ᾱ) * ε) + σ * z
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) + sigma * noise

        return x.clamp(-1, 1)


# ── Training ──

def train_deterministic(policy, states, actions, n_epochs=50, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []

    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.FloatTensor(actions).to(device)
    N = len(states_t)

    for epoch in range(n_epochs):
        idx = torch.randint(0, N, (256,))
        s = states_t[idx]
        a = actions_t[idx]

        pred = policy(s)
        loss = F.mse_loss(pred, a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Loss: {loss.item():.6f}")

    return losses


def train_diffusion(policy, states, actions, n_epochs=200, lr=1e-4, device='cpu'):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []

    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.FloatTensor(actions).to(device)
    N = len(states_t)

    for epoch in range(n_epochs):
        idx = torch.randint(0, N, (256,))
        s = states_t[idx]
        a = actions_t[idx]

        loss = policy.training_loss(s, a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Loss: {loss.item():.6f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "33-diffusion-policy"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = TargetTrackingEnv(horizon=50)

    # Generate expert data
    print("=== Generating Expert Data ===")
    states, actions = env.generate_trajectory(n_trajectories=200)
    states_flat = states.reshape(-1, 3)  # (N*T, 3)
    actions_flat = actions.reshape(-1, 1)  # (N*T, 1)
    print(f"  Expert data: {len(states_flat)} state-action pairs")

    # Train deterministic policy
    print("\n=== Training Deterministic Policy ===")
    det_policy = DeterministicPolicy().to(device)
    det_losses = train_deterministic(det_policy, states_flat, actions_flat, n_epochs=100, device=device)

    # Train diffusion policy
    print("\n=== Training Diffusion Policy ===")
    diff_policy = DiffusionPolicy(n_diffusion_steps=20).to(device)
    diff_losses = train_diffusion(diff_policy, states_flat, actions_flat, n_epochs=500, device=device)

    # ── Evaluate ──
    print("\n=== Evaluating ===")
    # Test on new trajectories
    test_states, test_actions = env.generate_trajectory(n_trajectories=10)
    test_s = torch.FloatTensor(test_states.reshape(-1, 3)).to(device)
    test_a = torch.FloatTensor(test_actions.reshape(-1, 1)).to(device)

    with torch.no_grad():
        det_pred = det_policy(test_s)
        det_mse = F.mse_loss(det_pred, test_a).item()

        diff_pred = diff_policy.sample(test_s)
        diff_mse = F.mse_loss(diff_pred, test_a).item()

    print(f"  Deterministic MSE: {det_mse:.6f}")
    print(f"  Diffusion MSE:     {diff_mse:.6f}")

    # Multi-modal evaluation
    print("\n=== Multi-Modal Evaluation ===")
    # Create a state where there are two equally good actions
    # Agent at 0, target at 0.5 going right: both "go right" and "go right faster" are OK
    multimodal_state = torch.FloatTensor([[0.0, 0.5, 1.0]]).to(device)

    det_action = det_policy(multimodal_state).item()
    diff_actions = []
    for _ in range(100):
        a = diff_policy.sample(multimodal_state).item()
        diff_actions.append(a)

    print(f"  Deterministic action: {det_action:.4f}")
    print(f"  Diffusion actions: mean={np.mean(diff_actions):.4f}, std={np.std(diff_actions):.4f}")

    # ── Visualization ──

    # 1. Training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 5
    det_s = np.convolve(det_losses, np.ones(window)/window, mode='valid')
    axes[0].plot(det_s, color='blue')
    axes[0].set_title("Deterministic Policy: Training MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (smoothed)")
    axes[0].grid(True, alpha=0.3)

    diff_s = np.convolve(diff_losses, np.ones(window)/window, mode='valid')
    axes[1].plot(diff_s, color='green')
    axes[1].set_title("Diffusion Policy: Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Noise Prediction Loss (smoothed)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Policy Training: Deterministic vs Diffusion", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Trajectory comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Expert trajectory
    traj_states, traj_actions = env.generate_trajectory(1)
    t_range = np.arange(len(traj_actions[0]))

    axes[0].plot(t_range, traj_actions[0], label='Expert', color='black', linewidth=2)
    axes[0].set_title("Expert Actions")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Action")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Deterministic
    with torch.no_grad():
        s_t = torch.FloatTensor(traj_states[0]).to(device)
        det_a = det_policy(s_t).cpu().numpy().flatten()
    axes[1].plot(t_range, traj_actions[0], label='Expert', color='black', alpha=0.3, linewidth=2)
    axes[1].plot(t_range, det_a, label='Deterministic', color='blue')
    axes[1].set_title("Deterministic Policy")
    axes[1].set_xlabel("Timestep")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Diffusion (with uncertainty band)
    with torch.no_grad():
        s_t = torch.FloatTensor(traj_states[0]).to(device)
        diff_samples = []
        for _ in range(20):
            a_sample = diff_policy.sample(s_t).cpu().numpy().flatten()
            diff_samples.append(a_sample)
        diff_samples = np.array(diff_samples)
        diff_mean = diff_samples.mean(axis=0)
        diff_std = diff_samples.std(axis=0)

    axes[2].plot(t_range, traj_actions[0], label='Expert', color='black', alpha=0.3, linewidth=2)
    axes[2].plot(t_range, diff_mean, label='Diffusion (mean)', color='green')
    axes[2].fill_between(t_range, diff_mean - diff_std, diff_mean + diff_std,
                        alpha=0.2, color='green', label='±1 std')
    axes[2].set_title("Diffusion Policy")
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Trajectory Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "trajectory_comparison.png", dpi=150)
    plt.close()

    # 3. Action distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Expert action distribution
    axes[0].hist(actions_flat.flatten(), bins=30, alpha=0.7, color='black')
    axes[0].set_title("Expert Action Distribution")
    axes[0].set_xlabel("Action")
    axes[0].grid(True, alpha=0.3)

    # Diffusion: multi-sample distribution
    with torch.no_grad():
        sample_states = torch.FloatTensor(states_flat[:500]).to(device)
        diff_acts = diff_policy.sample(sample_states).cpu().numpy().flatten()
    axes[1].hist(diff_acts, bins=30, alpha=0.7, color='green')
    axes[1].set_title("Diffusion Policy Actions")
    axes[1].set_xlabel("Action")
    axes[1].grid(True, alpha=0.3)

    # Multi-modal case
    axes[2].hist(diff_actions, bins=30, alpha=0.7, color='green')
    axes[2].axvline(det_action, color='blue', linewidth=2, label='Deterministic')
    axes[2].set_title("Action Distribution at Single State")
    axes[2].set_xlabel("Action")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Diffusion Policy: Action Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "action_distribution.png", dpi=150)
    plt.close()

    # 4. Denoising visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 3))
    with torch.no_grad():
        test_state = torch.FloatTensor([[0.0, 0.5, 1.0]]).to(device)
        x = torch.randn(1, 1, device=device)
        steps_to_show = [20, 15, 10, 5, 0]

        for idx, t_step in enumerate(steps_to_show):
            # Denoise from t_step
            for t_idx in reversed(range(t_step, 20)):
                t = torch.full((1,), t_idx / 20.0, device=device)
                pred_noise = diff_policy(x, t, test_state)
                alpha = diff_policy.alphas[t_idx]
                alpha_bar = diff_policy.alpha_bar[t_idx]
                if t_idx > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(diff_policy.betas[t_idx])
                else:
                    noise = torch.zeros_like(x)
                    sigma = 0
                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) + sigma * noise

            axes[idx].hist(x.cpu().numpy().flatten() if x.shape[0] > 1 else [x.item()],
                          bins=20, alpha=0.7, color='green')
            axes[idx].set_title(f"t={t_step}")
            axes[idx].set_xlim(-3, 3)

    plt.suptitle("Diffusion Policy: Denoising Process", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "denoising_process.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Deterministic\nPolicy", "π(a|s) = f(s)\nSingle action\nNo uncertainty\n→ Fails multi-modal", 0.17, 'blue'),
        ("Diffusion\nPolicy", "π(a|s) = denoise(a_t, s)\nStochastic samples\nCaptures multi-modal\n→ Better exploration", 0.5, 'green'),
        ("Key Idea", "Replace single action\nwith denoising process\nAction = sample from\nlearned distribution", 0.83, 'purple'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Diffusion Policy: From Deterministic to Stochastic Actions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "diffusion_policy_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
