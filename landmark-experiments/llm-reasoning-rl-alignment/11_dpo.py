"""
Minimal DPO (Direct Preference Optimization) Reproduction
==========================================================
Reproduces the core ideas from "Direct Preference Optimization:
Your Language Model is Secretly a Reward Model" (2305.18290):
1. DPO eliminates the need for a separate reward model
2. Closed-form optimal policy under KL-constrained reward maximization
3. Compare DPO vs RLHF (PPO-style) training dynamics
4. Preference learning from pairwise comparisons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Policy Model (small sequence model) ──

class SmallPolicy(nn.Module):
    """A small autoregressive model for simple token sequences."""
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.head(self.norm(h))

    def get_log_probs(self, sequences):
        """Get log probability of each token in the sequence."""
        logits = self.forward(sequences[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probs for the actual tokens
        token_log_probs = log_probs.gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=1)  # Sum over sequence length


# ── Reward Model (for RLHF baseline) ──

class RewardModel(nn.Module):
    """Simple reward model that scores sequences."""
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # Scalar reward

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        # Use mean pooling for sequence-level reward
        return self.head(h.mean(dim=1)).squeeze(-1)


# ── Preference Data Generation ──

class PreferenceEnvironment:
    """Simple environment that generates preference pairs.
    Task: generate sequences where higher digit sums are preferred.
    """
    def __init__(self, vocab_size=12, seq_len=8):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # Tokens: 0=vocab_start ... digits represent quality
        # Higher tokens = "better" responses
        self.digit_offset = 2  # First 2 tokens are special

    def generate_preference_pair(self, n=1, device='cpu'):
        """Generate (chosen, rejected) pairs where chosen has higher quality."""
        chosen = torch.zeros(n, self.seq_len, dtype=torch.long, device=device)
        rejected = torch.zeros(n, self.seq_len, dtype=torch.long, device=device)

        for i in range(n):
            for j in range(self.seq_len):
                # Chosen: higher digits on average
                chosen[i, j] = torch.randint(self.digit_offset + 4, self.vocab_size, (1,)).item()
                # Rejected: lower digits on average
                rejected[i, j] = torch.randint(self.digit_offset, self.vocab_size - 3, (1,)).item()

        return chosen, rejected

    def true_reward(self, sequences):
        """Ground truth reward: sum of token values (higher = better)."""
        return sequences.float().sum(dim=1) / self.seq_len


# ── DPO Training ──

def train_dpo(policy, ref_policy, pref_env, n_steps=2000, beta=0.1,
              batch_size=32, lr=1e-4, device='cpu'):
    """Direct Preference Optimization.

    Key equation:
    L_DPO = -E[log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

    where y_w = chosen (winner), y_l = rejected (loser), σ = sigmoid.
    """
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    losses = []
    reward_margins = []

    for step in range(n_steps):
        chosen, rejected = pref_env.generate_preference_pair(batch_size, device)

        # Get log probs from policy and reference
        log_pi_chosen = policy.get_log_probs(chosen)
        log_pi_rejected = policy.get_log_probs(rejected)

        with torch.no_grad():
            log_ref_chosen = ref_policy.get_log_probs(chosen)
            log_ref_rejected = ref_policy.get_log_probs(rejected)

        # DPO loss
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected

        # L = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        logits = beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # Track implicit reward margin
        with torch.no_grad():
            margin = (log_pi_chosen - log_ref_chosen - log_pi_rejected + log_ref_rejected).mean().item()
            reward_margins.append(margin)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Reward Margin: {margin:.4f}")

    return losses, reward_margins


# ── RLHF (PPO-style) Training ──

def train_rlhf(policy, reward_model, pref_env, n_steps=2000, kl_coeff=0.1,
               batch_size=32, lr=1e-4, clip_eps=0.2, device='cpu'):
    """Simplified RLHF with PPO-style updates.

    Two-stage:
    1. Train reward model on preference data
    2. Optimize policy with PPO using learned reward + KL penalty
    """
    # Stage 1: Train reward model
    print("  Training reward model...")
    rm_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)
    for step in range(500):
        chosen, rejected = pref_env.generate_preference_pair(batch_size, device)
        r_chosen = reward_model(chosen)
        r_rejected = reward_model(rejected)

        # Bradley-Terry loss: P(chosen > rejected) = σ(r_chosen - r_rejected)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        rm_optimizer.zero_grad()
        loss.backward()
        rm_optimizer.step()

    # Stage 2: PPO-style policy optimization
    print("  Training policy with PPO...")
    ref_policy = SmallPolicy(pref_env.vocab_size, d_model=64, n_heads=2,
                            n_layers=2, max_len=pref_env.seq_len).to(device)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    losses = []
    reward_margins = []

    for step in range(n_steps):
        # Generate samples
        prompts = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)

        with torch.no_grad():
            # Simple generation: sample from policy
            logits = policy(prompts)
            # Generate sequence (simplified: just sample all tokens)
            sequences = torch.zeros(batch_size, pref_env.seq_len, dtype=torch.long, device=device)
            x = prompts
            for t in range(pref_env.seq_len - 1):
                logits_t = policy(x)
                probs = F.softmax(logits_t[:, -1, :], dim=-1)
                next_tok = torch.multinomial(probs, 1)
                sequences[:, t] = next_tok.squeeze(-1)
                x = torch.cat([x, next_tok], dim=1)
                if x.shape[1] >= policy.max_len:
                    break

        # Get rewards from learned reward model
        with torch.no_grad():
            rewards = reward_model(sequences)

        # Compute old log probs
        with torch.no_grad():
            old_log_probs = policy.get_log_probs(sequences)
            ref_log_probs = ref_policy.get_log_probs(sequences)

        # Advantage = reward - KL penalty
        kl = old_log_probs - ref_log_probs
        advantages = rewards - kl_coeff * kl

        # PPO update
        new_log_probs = policy.get_log_probs(sequences)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        losses.append(ppo_loss.item())
        reward_margins.append(advantages.mean().item())

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {ppo_loss.item():.4f} | "
                  f"Avg Advantage: {advantages.mean().item():.4f}")

    return losses, reward_margins


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "11-dpo"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 12
    seq_len = 8
    d_model = 64
    n_heads = 2
    n_layers = 2
    n_steps = 2000

    pref_env = PreferenceEnvironment(vocab_size, seq_len)

    # ── Train DPO ──
    print("=== Training with DPO ===")
    policy_dpo = SmallPolicy(vocab_size, d_model, n_heads, n_layers, seq_len).to(device)
    ref_policy = SmallPolicy(vocab_size, d_model, n_heads, n_layers, seq_len).to(device)
    ref_policy.eval()

    dpo_losses, dpo_margins = train_dpo(
        policy_dpo, ref_policy, pref_env, n_steps=n_steps, beta=0.1, device=device
    )

    # ── Train RLHF ──
    print("\n=== Training with RLHF (PPO) ===")
    policy_rlhf = SmallPolicy(vocab_size, d_model, n_heads, n_layers, seq_len).to(device)
    reward_model = RewardModel(vocab_size, d_model, n_heads, n_layers, seq_len).to(device)

    rlhf_losses, rlhf_margins = train_rlhf(
        policy_rlhf, reward_model, pref_env, n_steps=n_steps, kl_coeff=0.1, device=device
    )

    # ── Visualization ──
    window = 30

    # 1. Loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dpo_loss_s = np.convolve(dpo_losses, np.ones(window)/window, mode='valid')
    rlhf_loss_s = np.convolve(rlhf_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(dpo_loss_s, label='DPO', color='blue')
    axes[0].plot(rlhf_loss_s, label='RLHF (PPO)', color='red')
    axes[0].set_title("Training Loss: DPO vs RLHF")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Reward margin
    dpo_margin_s = np.convolve(dpo_margins, np.ones(window)/window, mode='valid')
    rlhf_margin_s = np.convolve(rlhf_margins, np.ones(window)/window, mode='valid')

    axes[1].plot(dpo_margin_s, label='DPO', color='blue')
    axes[1].plot(rlhf_margin_s, label='RLHF (PPO)', color='red')
    axes[1].set_title("Implicit Reward Margin (chosen - rejected)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Log-ratio difference (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle("DPO vs RLHF: Direct Preference Optimization", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "dpo_vs_rlhf.png", dpi=150)
    plt.close()

    # 3. DPO key insight diagram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    steps = [
        ("RLHF Pipeline", "Prompt → Policy → Samples\n→ Reward Model → Score\n→ PPO Update (4 models!)", 0.17, 'red'),
        ("", "vs", 0.5, 'gray'),
        ("DPO Pipeline", "Prompt + (chosen, rejected)\n→ Direct policy update\n→ Only 2 models (π + π_ref)", 0.83, 'blue'),
    ]

    for name, desc, x_pos, color in steps:
        if name:
            ax.text(x_pos, 0.7, name, fontsize=14, fontweight='bold',
                    ha='center', va='center', color=color)
            ax.text(x_pos, 0.35, desc, fontsize=11, ha='center', va='center',
                    fontfamily='monospace', color=color,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(x_pos, 0.5, desc, fontsize=20, fontweight='bold',
                    ha='center', va='center', color=color)

    ax.set_title("RLHF vs DPO: Eliminating the Reward Model", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "dpo_insight.png", dpi=150)
    plt.close()

    # 4. Beta sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    betas = [0.01, 0.05, 0.1, 0.5, 1.0]
    final_margins = []

    for beta in betas:
        policy = SmallPolicy(vocab_size, d_model, n_heads, n_layers, seq_len).to(device)
        _, margins = train_dpo(
            policy, ref_policy, pref_env, n_steps=1000, beta=beta, device=device
        )
        final_margins.append(np.mean(margins[-50:]))

    ax.plot(betas, final_margins, 'o-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel("β (KL penalty strength)")
    ax.set_ylabel("Final Reward Margin")
    ax.set_title("DPO: Effect of β on Learning")
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "beta_sensitivity.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
