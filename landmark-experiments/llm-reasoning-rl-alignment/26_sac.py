"""
Minimal SAC (Soft Actor-Critic) Reproduction
=============================================
Reproduces core ideas from SAC (1801.01290, Haarnoja et al.):
1. Maximum entropy RL: maximize reward + entropy (exploration)
2. Soft Q-function and soft value function
3. Automatic temperature tuning (α)
4. Compare: SAC vs DDPG vs REINFORCE on continuous control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import random


# ── Environment: Simple Pendulum ──

class PendulumEnv:
    """Simple pendulum: continuous action space [-2, 2].
    State: [cos(θ), sin(θ), ω]
    Action: torque in [-2, 2]
    Reward: -(θ² + 0.1*ω² + 0.001*torque²)
    """
    def __init__(self, dt=0.05, max_speed=8.0):
        self.dt = dt
        self.max_speed = max_speed
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0

    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.omega = np.random.uniform(-1, 1)
        return self._get_state()

    def _get_state(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.omega], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -2.0, 2.0)
        cost = self.theta**2 + 0.1 * self.omega**2 + 0.001 * action**2
        reward = -cost

        self.omega += ((-self.g / self.l) * np.sin(self.theta) - 0.05 * self.omega + action / (self.m * self.l**2)) * self.dt
        self.omega = np.clip(self.omega, -self.max_speed, self.max_speed)
        self.theta += self.omega * self.dt
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi

        done = False
        return self._get_state(), reward, done

    @property
    def state_dim(self):
        return 3

    @property
    def action_dim(self):
        return 1


# ── Replay Buffer ──

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── Networks ──

class Actor(nn.Module):
    """Gaussian policy network."""
    def __init__(self, state_dim=3, action_dim=1, hidden=64, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, std

    def sample(self, state, deterministic=False):
        mean, std = self.forward(state)
        if deterministic:
            return torch.tanh(mean), None
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization
        action = torch.tanh(x_t)
        # Log prob with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# ── SAC Agent ──

class SAC:
    def __init__(self, state_dim=3, action_dim=1, hidden=64, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True):
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        self.actor = Actor(state_dim, action_dim, hidden)
        self.critic = Critic(state_dim, action_dim, hidden)
        self.critic_target = Critic(state_dim, action_dim, hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic temperature tuning
        if auto_alpha:
            self.target_entropy = -action_dim  # -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def update(self, batch):
        states = torch.FloatTensor(batch[0])
        actions = torch.FloatTensor(batch[1])
        rewards = torch.FloatTensor(batch[2]).unsqueeze(-1)
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4]).unsqueeze(-1)

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # --- Soft update target ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), self.alpha.item() if self.auto_alpha else self.alpha


# ── DDPG Baseline ──

class DDPGActor(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * 2  # scale to [-2, 2]


class DDPGCritic(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class DDPG:
    def __init__(self, state_dim=3, action_dim=1, hidden=64, lr=1e-3, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.actor = DDPGActor(state_dim, action_dim, hidden)
        self.critic = DDPGCritic(state_dim, action_dim, hidden)
        self.actor_target = DDPGActor(state_dim, action_dim, hidden)
        self.critic_target = DDPGCritic(state_dim, action_dim, hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, batch):
        states = torch.FloatTensor(batch[0])
        actions = torch.FloatTensor(batch[1])
        rewards = torch.FloatTensor(batch[2]).unsqueeze(-1)
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4]).unsqueeze(-1)

        # Critic
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            q_target = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, target_actions)
        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for t, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)
        for t, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)

        return critic_loss.item(), actor_loss.item(), 0.0  # no alpha


# ── REINFORCE Baseline ──

class REINFORCEActor(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        h = self.net(state)
        mean = torch.tanh(self.mean_head(h)) * 2
        std = self.log_std.exp().clamp(1e-3, 2.0)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


def train_reinforce(env, n_episodes=500, lr=1e-3, gamma=0.99):
    actor = REINFORCEActor(env.state_dim, env.action_dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    rewards_history = []

    for ep in range(n_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        for _ in range(200):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = actor.sample(state_t)
            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            if done:
                break

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(sum(rewards))

    return rewards_history


# ── Training Loop ──

def train_offpolicy(agent, env, n_steps=10000, batch_size=64, start_steps=1000,
                    eval_every=500):
    buffer = ReplayBuffer(50000)
    rewards_history = []
    eval_rewards = []
    alphas = []

    state = env.reset()
    episode_reward = 0
    episode_rewards = []
    step_in_episode = 0

    for step in range(n_steps):
        # Collect experience
        if step < start_steps:
            action = np.random.uniform(-2, 2, size=(1,))
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                if isinstance(agent, SAC):
                    action, _ = agent.actor.sample(state_t, deterministic=False)
                else:
                    action = agent.actor(state_t)
                action = action.numpy().flatten()

        next_state, reward, done = env.step(action[0] if action.ndim > 0 else action)
        buffer.push(state, action, reward, next_state, float(done))
        episode_reward += reward
        step_in_episode += 1
        state = next_state

        # End episode after 200 steps
        if done or step_in_episode >= 200:
            episode_rewards.append(episode_reward)
            state = env.reset()
            episode_reward = 0
            step_in_episode = 0

        # Update
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            cl, al, alpha = agent.update(batch)
            alphas.append(alpha)

        # Eval
        if (step + 1) % eval_every == 0:
            if episode_rewards:
                avg_reward = np.mean(episode_rewards[-10:])
                rewards_history.append(avg_reward)
                print(f"  Step {step+1} | Avg Reward: {avg_reward:.2f} | α: {alphas[-1]:.3f}" if alphas else f"  Step {step+1} | Avg Reward: {avg_reward:.2f}")

    return rewards_history, alphas


# ── Main ──

def main():
    results_dir = Path(__file__).parent / "results" / "26-sac"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = PendulumEnv()
    n_steps = 15000

    # 1. SAC (auto alpha)
    print("=== Training SAC (auto α) ===")
    sac = SAC(state_dim=env.state_dim, action_dim=env.action_dim, auto_alpha=True)
    sac_rewards, sac_alphas = train_offpolicy(sac, env, n_steps)

    # 2. SAC (fixed alpha)
    print("\n=== Training SAC (fixed α=0.2) ===")
    sac_fixed = SAC(state_dim=env.state_dim, action_dim=env.action_dim, auto_alpha=False, alpha=0.2)
    sac_fixed_rewards, _ = train_offpolicy(sac_fixed, env, n_steps)

    # 3. DDPG
    print("\n=== Training DDPG ===")
    ddpg = DDPG(state_dim=env.state_dim, action_dim=env.action_dim)
    ddpg_rewards, _ = train_offpolicy(ddpg, env, n_steps)

    # 4. REINFORCE
    print("\n=== Training REINFORCE ===")
    reinforce_rewards = train_reinforce(env, n_episodes=300)

    # ── Visualization ──

    # 1. Reward comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(sac_rewards, label='SAC (auto α)', color='blue')
    axes[0].plot(sac_fixed_rewards, label='SAC (fixed α)', color='cyan')
    axes[0].plot(ddpg_rewards, label='DDPG', color='red')
    axes[0].set_title("Off-Policy: SAC vs DDPG")
    axes[0].set_xlabel("Evaluation")
    axes[0].set_ylabel("Avg Reward (last 10 episodes)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # REINFORCE
    window = 20
    rf_s = np.convolve(reinforce_rewards, np.ones(window)/window, mode='valid')
    axes[1].plot(rf_s, label='REINFORCE', color='green')
    axes[1].set_title("REINFORCE (on-policy)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("SAC: Soft Actor-Critic for Continuous Control", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "reward_comparison.png", dpi=150)
    plt.close()

    # 2. Alpha evolution
    if sac_alphas:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sac_alphas, color='blue', alpha=0.5)
        window = 100
        if len(sac_alphas) > window:
            alpha_s = np.convolve(sac_alphas, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(sac_alphas)), alpha_s, color='blue', linewidth=2, label='Smoothed')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Fixed α=0.2')
        ax.set_title("SAC: Automatic Temperature (α) Tuning")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("α (temperature)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "alpha_evolution.png", dpi=150)
        plt.close()

    # 3. Policy visualization
    print("\n=== Visualizing Policies ===")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for idx, (agent, name, color) in enumerate([
        (sac, 'SAC', 'blue'),
        (sac_fixed, 'SAC (fixed α)', 'cyan'),
        (ddpg, 'DDPG', 'red'),
        (None, 'REINFORCE', 'green')
    ]):
        ax = axes[idx]
        n_angles = 50
        thetas = np.linspace(-np.pi, np.pi, n_angles)
        actions = []
        stds = []

        for theta in thetas:
            state = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                if isinstance(agent, SAC):
                    mean, std = agent.actor(state_t)
                    actions.append(mean.item())
                    stds.append(std.item())
                elif isinstance(agent, DDPG):
                    a = agent.actor(state_t)
                    actions.append(a.item())
                    stds.append(0)
                else:
                    # REINFORCE
                    ra = REINFORCEActor(env.state_dim, env.action_dim)
                    mean, std = ra(state_t)
                    actions.append(mean.item())
                    stds.append(std.item())

        thetas_deg = np.degrees(thetas)
        ax.plot(thetas_deg, actions, color=color, linewidth=2)
        if any(s > 0 for s in stds):
            actions_arr = np.array(actions)
            stds_arr = np.array(stds)
            ax.fill_between(thetas_deg, actions_arr - stds_arr, actions_arr + stds_arr,
                          alpha=0.2, color=color)
        ax.set_title(f"{name} Policy")
        ax.set_xlabel("θ (degrees)")
        ax.set_ylabel("Action (torque)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)

    plt.suptitle("Learned Policies: Pendulum Control", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "policy_visualization.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    texts = [
        ("Maximum Entropy", "max π E[Σ r + α·H(π)]\nStochastic policy\nRobust exploration", 0.2, 'blue'),
        ("Twin Q-Networks", "min(Q1, Q2) prevents\noverestimation\n→ stable learning", 0.5, 'orange'),
        ("Auto Temperature", "α adapts to maintain\nentropy target\n→ no hyperparam tuning", 0.8, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.7, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=11, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("SAC: Three Key Ingredients", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "sac_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
