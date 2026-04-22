"""
Minimal PPO (Proximal Policy Optimization) Reproduction
========================================================
Reproduces core ideas from "Proximal Policy Optimization Algorithms"
(Schulman et al., 2017, 1707.06347):
1. Clipped surrogate objective
2. Value function baseline
3. Advantage estimation (GAE-like)
4. Compare PPO vs vanilla policy gradient vs REINFORCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Environment: Simple Bandit with Structure ──

class StructuredBandit:
    """A structured multi-armed bandit where actions have ordinal meaning.
    Action 0-9, reward is highest for action 5, with Gaussian noise.
    This tests whether the policy can learn the structure, not just explore.
    """
    def __init__(self, n_actions=10, optimal=5, noise=0.5):
        self.n_actions = n_actions
        self.optimal = optimal
        self.noise = noise

    def step(self, actions):
        """Return rewards for a batch of actions."""
        rewards = -0.1 * (actions.float() - self.optimal) ** 2 / self.optimal ** 2
        rewards = rewards + torch.randn_like(rewards) * self.noise
        return rewards


class CartpoleLite:
    """Simplified cart-pole-like environment (1D balance).
    State: position (continuous)
    Action: push left (0) or right (1)
    Reward: -|position|
    """
    def __init__(self, max_steps=20):
        self.max_steps = max_steps

    def rollout(self, policy_fn, batch_size=16, device='cpu'):
        """Generate rollouts using the policy."""
        states = torch.zeros(batch_size, self.max_steps, 1, device=device)
        actions = torch.zeros(batch_size, self.max_steps, dtype=torch.long, device=device)
        rewards = torch.zeros(batch_size, self.max_steps, device=device)
        log_probs = torch.zeros(batch_size, self.max_steps, device=device)
        values = torch.zeros(batch_size, self.max_steps, device=device)

        pos = torch.zeros(batch_size, 1, device=device)
        vel = torch.zeros(batch_size, 1, device=device)

        for t in range(self.max_steps):
            states[:, t] = pos.squeeze(-1).clone().unsqueeze(-1) if pos.dim() == 2 else pos.clone()

            # Get policy action
            logits, value = policy_fn(pos)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)

            actions[:, t] = action
            log_probs[:, t] = log_prob
            values[:, t] = value.squeeze(-1)

            # Dynamics: push = ±0.3
            push = (action.float() - 0.5) * 0.6  # 0 → -0.3, 1 → +0.3
            vel = vel + push.unsqueeze(-1) * 0.1 - 0.01 * pos  # damping
            pos = pos + vel
            pos = pos.clamp(-2, 2)

            rewards[:, t] = -pos.abs().squeeze(-1) - 0.01 * vel.abs().squeeze(-1)

        return states, actions, rewards, log_probs, values


# ── Policy Network ──

class PolicyValueNet(nn.Module):
    """Actor-Critic: shared backbone with policy and value heads."""
    def __init__(self, state_dim=1, n_actions=2, hidden=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state):
        h = self.backbone(state)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


# ── RL Algorithms ──

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE (Generalized Advantage Estimation)."""
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * last_adv
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def train_ppo(env, n_iters=500, batch_size=16, lr=3e-4, clip_eps=0.2,
              n_epochs=4, device='cpu'):
    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_rewards = []
    all_losses = []

    for iteration in range(n_iters):
        # Collect rollout
        with torch.no_grad():
            states, actions, rewards, old_log_probs, values = env.rollout(
                lambda s: model(s), batch_size=batch_size, device=device
            )

        # Compute advantages
        advantages, returns = compute_advantages(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update (multiple epochs)
        for _ in range(n_epochs):
            logits, value_pred = model(states)
            new_log_probs = F.log_softmax(logits, dim=-1).gather(2, actions.unsqueeze(-1)).squeeze(-1)
            new_values = value_pred.squeeze(-1)

            # Clipped surrogate
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns)

            # Entropy bonus
            entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        avg_reward = rewards.sum(dim=1).mean().item()
        all_rewards.append(avg_reward)
        all_losses.append(loss.item())

        if (iteration + 1) % 100 == 0:
            print(f"  Iter {iteration+1} | Avg Reward: {avg_reward:.3f}")

    return all_rewards, all_losses


def train_reinforce(env, n_iters=500, batch_size=16, lr=3e-4, device='cpu'):
    """REINFORCE with baseline."""
    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_rewards = []
    all_losses = []

    for iteration in range(n_iters):
        states, actions, rewards, log_probs, values = env.rollout(
            lambda s: model(s), batch_size=batch_size, device=device
        )

        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + 0.99 * running_return
            returns[:, t] = running_return

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # REINFORCE loss
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = rewards.sum(dim=1).mean().item()
        all_rewards.append(avg_reward)
        all_losses.append(loss.item())

        if (iteration + 1) % 100 == 0:
            print(f"  Iter {iteration+1} | Avg Reward: {avg_reward:.3f}")

    return all_rewards, all_losses


def train_vanilla_pg(env, n_iters=500, batch_size=16, lr=3e-4, device='cpu'):
    """Vanilla policy gradient (no baseline, no clipping)."""
    model = PolicyValueNet().to(device)
    # Only use policy head
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_rewards = []
    all_losses = []

    for iteration in range(n_iters):
        states, actions, rewards, log_probs, _ = env.rollout(
            lambda s: model(s), batch_size=batch_size, device=device
        )

        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + 0.99 * running_return
            returns[:, t] = running_return

        # Vanilla PG: no baseline
        policy_loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        avg_reward = rewards.sum(dim=1).mean().item()
        all_rewards.append(avg_reward)
        all_losses.append(policy_loss.item())

        if (iteration + 1) % 100 == 0:
            print(f"  Iter {iteration+1} | Avg Reward: {avg_reward:.3f}")

    return all_rewards, all_losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "16-ppo"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = CartpoleLite(max_steps=20)

    # Train PPO
    print("=== Training PPO ===")
    ppo_rewards, ppo_losses = train_ppo(env, n_iters=500, device=device)

    # Train REINFORCE with baseline
    print("\n=== Training REINFORCE + Baseline ===")
    reinforce_rewards, reinforce_losses = train_reinforce(env, n_iters=500, device=device)

    # Train vanilla PG
    print("\n=== Training Vanilla PG ===")
    vpg_rewards, vpg_losses = train_vanilla_pg(env, n_iters=500, device=device)

    # ── Visualization ──
    window = 20

    # 1. Reward curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ppo_s = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
    reinf_s = np.convolve(reinforce_rewards, np.ones(window)/window, mode='valid')
    vpg_s = np.convolve(vpg_rewards, np.ones(window)/window, mode='valid')

    axes[0].plot(ppo_s, label='PPO', color='blue')
    axes[0].plot(reinf_s, label='REINFORCE + Baseline', color='orange')
    axes[0].plot(vpg_s, label='Vanilla PG', color='red')
    axes[0].set_title("Reward During Training")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Avg Episode Reward (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Loss curves
    ppo_l = np.convolve(ppo_losses, np.ones(window)/window, mode='valid')
    reinf_l = np.convolve(reinforce_losses, np.ones(window)/window, mode='valid')
    vpg_l = np.convolve(vpg_losses, np.ones(window)/window, mode='valid')

    axes[1].plot(ppo_l, label='PPO', color='blue')
    axes[1].plot(reinf_l, label='REINFORCE', color='orange')
    axes[1].plot(vpg_l, label='Vanilla PG', color='red')
    axes[1].set_title("Policy Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("PPO vs REINFORCE vs Vanilla Policy Gradient", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "algorithm_comparison.png", dpi=150)
    plt.close()

    # 3. PPO clipping visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ratios = np.linspace(0.5, 1.5, 100)
    clip_eps = 0.2

    # Clipped surrogate objective
    adv_pos = 1.0  # Positive advantage
    adv_neg = -1.0  # Negative advantage

    surr_pos = ratios * adv_pos
    clipped_pos = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * adv_pos
    ppo_obj_pos = np.minimum(surr_pos, clipped_pos)

    surr_neg = ratios * adv_neg
    clipped_neg = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * adv_neg
    ppo_obj_neg = np.minimum(surr_neg, clipped_neg)

    axes[0].plot(ratios, surr_pos, '--', label='Unclipped (A>0)', color='blue', alpha=0.5)
    axes[0].plot(ratios, clipped_pos, ':', label='Clipped (A>0)', color='blue', alpha=0.3)
    axes[0].plot(ratios, ppo_obj_pos, '-', label='PPO objective (A>0)', color='blue', linewidth=2)
    axes[0].axvline(1 - clip_eps, color='gray', linestyle='--', alpha=0.3)
    axes[0].axvline(1 + clip_eps, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlabel("Probability Ratio r(θ)")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("PPO Clipping (Positive Advantage)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ratios, surr_neg, '--', label='Unclipped (A<0)', color='red', alpha=0.5)
    axes[1].plot(ratios, clipped_neg, ':', label='Clipped (A<0)', color='red', alpha=0.3)
    axes[1].plot(ratios, ppo_obj_neg, '-', label='PPO objective (A<0)', color='red', linewidth=2)
    axes[1].axvline(1 - clip_eps, color='gray', linestyle='--', alpha=0.3)
    axes[1].axvline(1 + clip_eps, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_xlabel("Probability Ratio r(θ)")
    axes[1].set_ylabel("Objective")
    axes[1].set_title("PPO Clipping (Negative Advantage)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("PPO: Clipped Surrogate Objective", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "ppo_clipping.png", dpi=150)
    plt.close()

    # 4. Clip epsilon sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
    final_rewards = []

    for eps in epsilons:
        rewards, _ = train_ppo(env, n_iters=300, clip_eps=eps, device=device)
        final_rewards.append(np.mean(rewards[-50:]))

    ax.plot(epsilons, final_rewards, 'o-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel("Clip ε")
    ax.set_ylabel("Final Avg Reward (last 50 iters)")
    ax.set_title("PPO: Effect of Clip Epsilon")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "clip_epsilon.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
