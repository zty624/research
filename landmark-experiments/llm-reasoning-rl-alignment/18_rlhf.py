"""
Minimal RLHF Full Pipeline Reproduction
========================================
Reproduces the complete RLHF training pipeline from InstructGPT (2203.02155):
1. Supervised Fine-Tuning (SFT)
2. Reward Model training from preference data
3. PPO with reward model + KL penalty
4. Compare: SFT-only vs SFT+RM vs full RLHF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── Models ──

class PolicyModel(nn.Module):
    """Small autoregressive model."""
    def __init__(self, vocab_size=16, d_model=64, n_heads=2, n_layers=2, max_len=12):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
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

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.head(self.norm(h))

    def get_log_probs(self, sequences):
        logits = self.forward(sequences[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=1)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=8, temperature=1.0, device='cpu'):
        x = prompt.clone()
        for _ in range(max_new_tokens):
            if x.shape[1] >= self.max_len:
                break
            logits = self.forward(x)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token], dim=1)
        return x


class RewardModel(nn.Module):
    """Reward model that scores sequences."""
    def __init__(self, vocab_size=16, d_model=64, n_heads=2, n_layers=2, max_len=12):
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
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


# ── Environment ──

class PreferenceEnv:
    """Simple environment: sequences with higher tokens = better.
    Tokens 0-3: special/low quality, 4-15: content tokens.
    Reward = mean token value + bonus for pattern matching.
    """
    def __init__(self, vocab_size=16, seq_len=8):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pad_id = 0
        self.bos_id = 1

    def true_reward(self, sequences):
        """Ground truth reward function."""
        # Higher token values = better
        content = sequences[:, 1:].float()  # Skip BOS
        base_reward = content.mean(dim=1) / self.vocab_size

        # Bonus for having a "structured" pattern (ascending tokens)
        diffs = content[:, 1:] - content[:, :-1]
        structure_bonus = (diffs >= 0).float().mean(dim=1) * 0.3

        return base_reward + structure_bonus

    def generate_preference_pairs(self, n_pairs, device='cpu'):
        """Generate (chosen, rejected) pairs for reward model training."""
        # Generate two random sequences, compare with true reward
        seq1 = torch.randint(2, self.vocab_size, (n_pairs, self.seq_len), device=device)
        seq1[:, 0] = self.bos_id  # BOS token
        seq2 = torch.randint(2, self.vocab_size, (n_pairs, self.seq_len), device=device)
        seq2[:, 0] = self.bos_id

        r1 = self.true_reward(seq1)
        r2 = self.true_reward(seq2)

        # Higher reward = chosen
        chosen = torch.where((r1 >= r2).unsqueeze(-1).expand_as(seq1), seq1, seq2)
        rejected = torch.where((r1 < r2).unsqueeze(-1).expand_as(seq1), seq1, seq2)

        return chosen, rejected


# ── Training Functions ──

def train_sft(model, env, n_steps=2000, batch_size=64, lr=1e-3, device='cpu'):
    """Step 1: Supervised Fine-Tuning on 'good' sequences."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        # Generate "good" sequences (higher token values)
        targets = torch.randint(6, env.vocab_size, (batch_size, env.seq_len), device=device)
        targets[:, 0] = env.bos_id

        # Input = targets shifted right (with BOS)
        inputs = targets.clone()
        inputs[:, 1:] = targets[:, :-1]
        inputs[:, 0] = env.bos_id

        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1),
                               ignore_index=env.pad_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            print(f"  SFT Step {step+1} | Loss: {loss.item():.4f}")

    return losses


def train_reward_model(rm, env, n_steps=2000, batch_size=64, lr=1e-3, device='cpu'):
    """Step 2: Train reward model on preference pairs."""
    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr)
    losses = []
    accs = []

    for step in range(n_steps):
        chosen, rejected = env.generate_preference_pairs(batch_size, device=device)

        r_chosen = rm(chosen)
        r_rejected = rm(rejected)

        # Bradley-Terry loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        acc = (r_chosen > r_rejected).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  RM Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    return losses, accs


def train_ppo_rlhf(policy, ref_policy, reward_model, env, n_iters=500,
                    kl_coeff=0.1, clip_eps=0.2, lr=1e-4, batch_size=32, device='cpu'):
    """Step 3: RLHF with PPO."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    rewards_history = []
    kl_history = []

    for iteration in range(n_iters):
        # Generate sequences from current policy
        prompts = torch.full((batch_size, 1), env.bos_id, dtype=torch.long, device=device)
        sequences = policy.generate(prompts, max_new_tokens=env.seq_len-1,
                                     temperature=1.0, device=device)

        # Pad or truncate to seq_len
        if sequences.shape[1] < env.seq_len:
            padding = torch.zeros(batch_size, env.seq_len - sequences.shape[1],
                                  dtype=torch.long, device=device)
            sequences = torch.cat([sequences, padding], dim=1)
        sequences = sequences[:, :env.seq_len]

        # Get rewards from reward model
        with torch.no_grad():
            rewards = reward_model(sequences)

        # Compute old log probs
        with torch.no_grad():
            old_log_probs = policy.get_log_probs(sequences)
            ref_log_probs = ref_policy.get_log_probs(sequences)
            kl = (old_log_probs - ref_log_probs).mean().item()

        # Advantage = reward - KL penalty
        with torch.no_grad():
            kl_per_sample = policy.get_log_probs(sequences) - ref_policy.get_log_probs(sequences)
            advantages = rewards - kl_coeff * kl_per_sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        new_log_probs = policy.get_log_probs(sequences)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        rewards_history.append(rewards.mean().item())
        kl_history.append(kl)

        if (iteration + 1) % 100 == 0:
            print(f"  PPO Iter {iteration+1} | Avg Reward: {rewards.mean().item():.3f} | KL: {kl:.4f}")

    return rewards_history, kl_history


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "18-rlhf"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 16
    seq_len = 8
    d_model = 64

    env = PreferenceEnv(vocab_size, seq_len)

    # Step 1: SFT
    print("=== Step 1: Supervised Fine-Tuning (SFT) ===")
    sft_model = PolicyModel(vocab_size, d_model).to(device)
    sft_losses = train_sft(sft_model, env, n_steps=2000, device=device)

    # Step 2: Train Reward Model
    print("\n=== Step 2: Reward Model Training ===")
    reward_model = RewardModel(vocab_size, d_model).to(device)
    rm_losses, rm_accs = train_reward_model(reward_model, env, n_steps=2000, device=device)

    # Step 3: RLHF with PPO
    print("\n=== Step 3: RLHF (PPO + RM) ===")
    rlhf_policy = deepcopy(sft_model).to(device)
    ref_policy = deepcopy(sft_model).to(device)
    ref_policy.eval()
    ppo_rewards, ppo_kl = train_ppo_rlhf(
        rlhf_policy, ref_policy, reward_model, env,
        n_iters=500, kl_coeff=0.1, device=device
    )

    # Baselines
    # SFT-only performance
    print("\n=== Evaluating Models ===")
    n_eval = 200

    with torch.no_grad():
        prompts = torch.full((n_eval, 1), env.bos_id, dtype=torch.long, device=device)

        sft_seqs = sft_model.generate(prompts, max_new_tokens=seq_len-1, temperature=1.0, device=device)
        if sft_seqs.shape[1] < seq_len:
            sft_seqs = torch.cat([sft_seqs, torch.zeros(n_eval, seq_len-sft_seqs.shape[1],
                                                          dtype=torch.long, device=device)], dim=1)
        sft_seqs = sft_seqs[:, :seq_len]
        sft_true_reward = env.true_reward(sft_seqs).mean().item()
        sft_rm_reward = reward_model(sft_seqs).mean().item()

        rlhf_seqs = rlhf_policy.generate(prompts, max_new_tokens=seq_len-1, temperature=1.0, device=device)
        if rlhf_seqs.shape[1] < seq_len:
            rlhf_seqs = torch.cat([rlhf_seqs, torch.zeros(n_eval, seq_len-rlhf_seqs.shape[1],
                                                             dtype=torch.long, device=device)], dim=1)
        rlhf_seqs = rlhf_seqs[:, :seq_len]
        rlhf_true_reward = env.true_reward(rlhf_seqs).mean().item()
        rlhf_rm_reward = reward_model(rlhf_seqs).mean().item()

    print(f"  SFT:   True reward = {sft_true_reward:.3f}, RM reward = {sft_rm_reward:.3f}")
    print(f"  RLHF:  True reward = {rlhf_true_reward:.3f}, RM reward = {rlhf_rm_reward:.3f}")

    # ── Visualization ──
    window = 20

    # 1. SFT training
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sft_s = np.convolve(sft_losses, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(sft_s, color='blue')
    axes[0, 0].set_title("Step 1: SFT Training Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Cross-Entropy Loss (smoothed)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reward Model training
    rm_l = np.convolve(rm_losses, np.ones(window)/window, mode='valid')
    rm_a = np.convolve(rm_accs, np.ones(window)/window, mode='valid')
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    ax1.plot(rm_l, color='red', label='Loss')
    ax2.plot(rm_a, color='green', label='Accuracy')
    ax1.set_title("Step 2: Reward Model Training")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Bradley-Terry Loss", color='red')
    ax2.set_ylabel("Pairwise Accuracy", color='green')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 3. PPO reward
    ppo_r_s = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
    axes[1, 0].plot(ppo_r_s, color='blue')
    axes[1, 0].set_title("Step 3: PPO Reward (from RM)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Avg Reward (smoothed)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. KL divergence
    ppo_kl_s = np.convolve(ppo_kl, np.ones(window)/window, mode='valid')
    axes[1, 1].plot(ppo_kl_s, color='orange')
    axes[1, 1].set_title("Step 3: KL(π || π_ref)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("KL Divergence (smoothed)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Full RLHF Pipeline: SFT → RM → PPO", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "rlhf_pipeline.png", dpi=150)
    plt.close()

    # 5. Final comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['SFT', 'RLHF (SFT+RM+PPO)']
    true_r = [sft_true_reward, rlhf_true_reward]
    rm_r = [sft_rm_reward, rlhf_rm_reward]

    x = np.arange(len(methods))
    width = 0.3
    axes[0].bar(x - width/2, true_r, width, label='True Reward', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, rm_r, width, label='RM Reward', color='orange', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("SFT vs RLHF: Reward Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 6. Pipeline diagram
    axes[1].axis('off')
    steps = [
        ("Step 1: SFT", "Fine-tune on\ndemonstrations", 0.15, 'blue'),
        ("→", "", 0.33, 'gray'),
        ("Step 2: RM", "Train reward model\non preferences", 0.5, 'orange'),
        ("→", "", 0.67, 'gray'),
        ("Step 3: PPO", "Optimize policy\nwith RL + KL", 0.83, 'green'),
    ]
    for name, desc, x_pos, color in steps:
        if name == '→':
            axes[1].text(x_pos, 0.5, name, fontsize=24, ha='center', va='center', color=color)
        else:
            axes[1].text(x_pos, 0.7, name, fontsize=13, fontweight='bold',
                        ha='center', va='center', color=color)
            axes[1].text(x_pos, 0.35, desc, fontsize=10, ha='center', va='center',
                        fontfamily='monospace', color=color,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    axes[1].set_title("RLHF Pipeline (InstructGPT)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / "rlhf_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
