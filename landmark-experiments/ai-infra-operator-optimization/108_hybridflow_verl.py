"""
Minimal HybridFlow/veRL Framework Reproduction
===============================================
Reproduces core ideas from "HybridFlow: A Flexible and Efficient RLHF
Framework" (2409.19256, Shao et al.):
1. Actor-Worker separation: policy model (actor) vs reward/reference models (workers)
2. Hybrid parallelism: combine DP (data), TP (tensor), PP (pipeline) for different models
3. Experience collection: rollout → reward → advantage computation
4. Compare: sequential vs overlapping actor/worker execution
5. Show: throughput scaling with parallelism strategy
6. Demonstrate: memory-efficient 3D parallelism assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict


# ── Models ──

class PolicyActor(nn.Module):
    """Policy model (actor) for RLHF — generates responses."""
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2, max_len=24):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=12, temperature=1.0):
        """Autoregressive generation from prompts."""
        B = prompts.shape[0]
        current = prompts.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(current)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            current = torch.cat([current, next_tok], dim=1)
            if current.shape[1] >= self.max_len:
                break
        return current


class RewardModel(nn.Module):
    """Reward model: scores sequences."""
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2, max_len=24):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h[:, -1, :]).squeeze(-1)  # scalar reward per sequence


class ReferenceModel(nn.Module):
    """Reference (KL constraint) model — copy of initial policy."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, x):
        return self.policy(x)


# ── Experience Collection ──

def collect_experience(actor, ref_model, reward_model, prompts, max_new_tokens=12,
                       kl_coef=0.1, device='cpu'):
    """Collect experience: rollout → reward → advantage.

    This simulates the HybridFlow actor-worker pattern:
    1. Actor generates responses (rollout)
    2. Reward model scores responses
    3. Reference model provides KL penalty
    """
    # Step 1: Rollout (Actor)
    sequences = actor.generate(prompts, max_new_tokens=max_new_tokens)

    # Step 2: Reward scoring (Worker)
    rewards = reward_model(sequences)

    # Step 3: KL penalty (Reference Worker)
    with torch.no_grad():
        actor_logits = actor(sequences)
        ref_logits = ref_model(sequences)

        actor_logp = F.log_softmax(actor_logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)

        # Per-token KL divergence
        kl = (actor_logp.exp() * (actor_logp - ref_logp)).sum(dim=-1)  # (B, T)
        kl_penalty = -kl_coef * kl.sum(dim=-1)  # (B,)

    # Total reward = task reward + KL penalty
    total_reward = rewards + kl_penalty

    # Step 4: Compute advantages (simple: center rewards)
    advantages = total_reward - total_reward.mean()
    if total_reward.std() > 1e-8:
        advantages = advantages / (total_reward.std() + 1e-8)

    return sequences, advantages, rewards, kl_penalty


# ── RLHF Training Step ──

def rlhf_step(actor, ref_model, reward_model, prompts, optimizer,
              kl_coef=0.1, clip_eps=0.2, device='cpu'):
    """Single RLHF (PPO-style) training step."""
    sequences, advantages, rewards, kl_penalty = collect_experience(
        actor, ref_model, reward_model, prompts, kl_coef=kl_coef, device=device)

    # Policy gradient with clipping
    logits = actor(sequences[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    actions = sequences[:, 1:]

    # Gather log probs of taken actions
    action_logp = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Old log probs (from rollout, detached)
    with torch.no_grad():
        old_logp = action_logp.clone()

    # PPO ratio
    ratio = torch.exp(action_logp - old_logp)

    # Expand advantages to (B, T-1)
    adv_expanded = advantages.unsqueeze(1).expand_as(action_logp)

    # Clipped surrogate
    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_expanded
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    return loss.item(), rewards.mean().item(), kl_penalty.mean().item()


# ── Parallelism Simulation ──

def simulate_parallelism(n_gpus, model_sizes, strategy='hybrid'):
    """Simulate parallelism strategies for HybridFlow.

    model_sizes: dict with 'actor', 'reward', 'ref' → parameter counts
    Returns: estimated throughput and memory per GPU
    """
    total_mem = sum(model_sizes.values())

    if strategy == 'dp':
        # Data parallelism: replicate all models on all GPUs
        mem_per_gpu = total_mem
        throughput = n_gpus  # linear scaling
    elif strategy == 'tp':
        # Tensor parallelism: split each model across GPUs
        mem_per_gpu = total_mem / n_gpus
        throughput = n_gpus * 0.85  # communication overhead
    elif strategy == 'pp':
        # Pipeline parallelism: stage each model across GPUs
        mem_per_gpu = total_mem / n_gpus
        throughput = n_gpus * 0.7  # pipeline bubbles
    elif strategy == 'hybrid':
        # HybridFlow: DP for reward/ref, TP+DP for actor
        actor_ratio = model_sizes['actor'] / total_mem
        # Assign more GPUs to actor (larger, more compute)
        actor_gpus = max(1, int(n_gpus * actor_ratio * 1.2))
        worker_gpus = max(1, n_gpus - actor_gpus)

        actor_mem = model_sizes['actor'] / actor_gpus
        worker_mem = (model_sizes['reward'] + model_sizes['ref']) / worker_gpus
        mem_per_gpu = max(actor_mem, worker_mem)

        # Overlap actor and worker: throughput ~ max(actor_time, worker_time)
        actor_throughput = actor_gpus * 0.9
        worker_throughput = worker_gpus * 0.95  # smaller model, less overhead
        throughput = min(actor_throughput, worker_throughput) * 1.3  # overlap bonus
    else:
        mem_per_gpu = total_mem
        throughput = 1.0

    return throughput, mem_per_gpu


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "108-hybridflow-verl"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    max_len = 24
    prompt_len = 4

    # Initialize models
    actor = PolicyActor(vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=max_len).to(device)
    ref_model = ReferenceModel(PolicyActor(vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=max_len).to(device))
    reward_model = RewardModel(vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=max_len).to(device)

    # Pre-train reward model
    print("=== Pre-training Reward Model ===")
    rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    for step in range(500):
        # Synthetic: prefer sequences with more diverse tokens
        seqs = torch.randint(0, vocab_size, (32, max_len), device=device)
        # Reward = number of unique tokens (diversity)
        diversity = torch.tensor([len(s.unique()) for s in seqs], dtype=torch.float32, device=device)
        pred = reward_model(seqs)
        loss = F.mse_loss(pred, diversity / max_len * 5)
        rm_optimizer.zero_grad()
        loss.backward()
        rm_optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"  Step {step+1} | RM Loss: {loss.item():.4f}")

    # ── RLHF Training ──
    print("\n=== RLHF Training ===")
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    n_steps = 1000
    metrics = {'loss': [], 'reward': [], 'kl': [], 'diversity': []}

    for step in range(n_steps):
        prompts = torch.randint(0, vocab_size, (16, prompt_len), device=device)
        loss, reward, kl = rlhf_step(actor, ref_model, reward_model, prompts,
                                      optimizer, kl_coef=0.1, device=device)

        # Measure diversity
        with torch.no_grad():
            sample = actor.generate(prompts[:4], max_new_tokens=8)
            div = sum(len(s.unique()) for s in sample) / (4 * max_len)

        metrics['loss'].append(loss)
        metrics['reward'].append(reward)
        metrics['kl'].append(kl)
        metrics['diversity'].append(div)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1} | Loss: {loss:.4f} | Reward: {reward:.3f} | "
                  f"KL: {kl:.3f} | Diversity: {div:.3f}")

    # ── Parallelism Simulation ──
    print("\n=== Parallelism Strategy Simulation ===")
    actor_params = sum(p.numel() for p in actor.parameters())
    reward_params = sum(p.numel() for p in reward_model.parameters())
    ref_params = sum(p.numel() for p in ref_model.parameters())

    model_sizes = {
        'actor': actor_params,
        'reward': reward_params,
        'ref': ref_params,
    }

    strategies = ['dp', 'tp', 'pp', 'hybrid']
    gpu_counts = [1, 2, 4, 8, 16, 32]

    parallel_results = {}
    for strategy in strategies:
        throughputs = []
        mems = []
        for n_gpu in gpu_counts:
            tp, mem = simulate_parallelism(n_gpu, model_sizes, strategy)
            throughputs.append(tp)
            mems.append(mem)
        parallel_results[strategy] = (throughputs, mems)
        print(f"  {strategy}: throughput@8GPU={throughputs[2]:.2f}x, "
              f"mem/GPU={mems[2]:.0f} params")

    # ── Overlapping execution simulation ──
    print("\n=== Actor-Worker Overlap Simulation ===")
    n_gpus = 8
    batch_sizes = [4, 8, 16, 32, 64]

    sequential_times = []
    overlapping_times = []
    for bs in batch_sizes:
        # Sequential: actor → reward → ref
        t_actor = bs * 0.1  # ms per sample
        t_reward = bs * 0.05
        t_ref = bs * 0.05
        seq_time = t_actor + t_reward + t_ref

        # Overlapping: actor runs in parallel with reward+ref
        overlap_time = max(t_actor, t_reward + t_ref) + bs * 0.02  # sync overhead

        sequential_times.append(seq_time)
        overlapping_times.append(overlap_time)
        speedup = seq_time / overlap_time
        print(f"  BS={bs}: sequential={seq_time:.2f}ms, overlap={overlap_time:.2f}ms, "
              f"speedup={speedup:.2f}x")

    # ── Visualization ──

    # 1. RLHF training metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    w = 30

    for ax, key, title, color in [
        (axes[0, 0], 'loss', 'PPO Loss', 'blue'),
        (axes[0, 1], 'reward', 'Reward', 'green'),
        (axes[1, 0], 'kl', 'KL Penalty', 'red'),
        (axes[1, 1], 'diversity', 'Response Diversity', 'purple'),
    ]:
        smoothed = np.convolve(metrics[key], np.ones(w)/w, mode='valid')
        ax.plot(smoothed, color=color)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.suptitle('HybridFlow RLHF Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'rlhf_training.png', dpi=150)
    plt.close()

    # 2. Parallelism scaling
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strategy_colors = {'dp': '#e74c3c', 'tp': '#3498db', 'pp': '#2ecc71', 'hybrid': '#9b59b6'}

    for strategy in strategies:
        tp, mem = parallel_results[strategy]
        axes[0].plot(gpu_counts, tp, marker='o', label=strategy.upper(),
                     color=strategy_colors[strategy], linewidth=2)
    axes[0].set_xlabel("Number of GPUs")
    axes[0].set_ylabel("Relative Throughput")
    axes[0].set_title("Throughput Scaling")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for strategy in strategies:
        _, mem = parallel_results[strategy]
        axes[1].plot(gpu_counts, [m / 1e6 for m in mem], marker='o', label=strategy.upper(),
                     color=strategy_colors[strategy], linewidth=2)
    axes[1].set_xlabel("Number of GPUs")
    axes[1].set_ylabel("Memory per GPU (M params)")
    axes[1].set_title("Memory Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('HybridFlow Parallelism Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'parallelism.png', dpi=150)
    plt.close()

    # 3. Sequential vs Overlapping
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(batch_sizes))
    width = 0.35
    ax.bar(x_pos - width/2, sequential_times, width, label='Sequential',
           color='#e74c3c', alpha=0.7)
    ax.bar(x_pos + width/2, overlapping_times, width, label='Overlapping',
           color='#3498db', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"BS={bs}" for bs in batch_sizes])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Actor-Worker Execution: Sequential vs Overlapping")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (s, o) in enumerate(zip(sequential_times, overlapping_times)):
        ax.text(i, max(s, o) + 0.5, f"{s/o:.2f}x", ha='center', fontweight='bold', color='green')
    plt.tight_layout()
    plt.savefig(results_dir / 'overlap_comparison.png', dpi=150)
    plt.close()

    # 4. Timeline diagram
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    # Sequential timeline
    ax = axes[0]
    ax.barh(['Actor', 'Reward', 'Reference'], [40, 20, 20], left=[0, 40, 60],
            color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time")
    ax.set_title("Sequential Execution: Actor → Reward → Reference")
    ax.axvline(80, color='gray', linestyle='--', alpha=0.5)

    # Overlapping timeline
    ax = axes[1]
    ax.barh(['Actor', 'Reward+Ref'], [40, 40], left=[0, 0],
            color=['#e74c3c', '#3498db'], alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time")
    ax.set_title("Overlapping Execution: Actor ‖ Reward+Ref (HybridFlow)")
    ax.axvline(42, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('HybridFlow: Actor-Worker Overlap (2409.19256)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'timeline.png', dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    concept = (
        "HybridFlow / veRL: Flexible RLHF Framework (2409.19256)\n"
        "=" * 60 + "\n\n"
        "Architecture:\n"
        "  ┌─────────────────┐     ┌─────────────────┐\n"
        "  │  Actor Worker    │     │  Critic Worker   │\n"
        "  │  (Policy Model)  │     │  (Reward + Ref)  │\n"
        "  │  TP + DP         │     │  DP              │\n"
        "  └────────┬─────────┘     └────────┬─────────┘\n"
        "           │  experience    reward  │\n"
        "           └──────────┬─────────────┘\n"
        "                      │\n"
        "              ┌───────┴───────┐\n"
        "              │  Controller   │\n"
        "              │  (HybridFlow) │\n"
        "              └───────────────┘\n\n"
        "Key innovations:\n"
        "  • 3D Hybrid Parallelism: TP for actor, DP for workers\n"
        "  • Actor-Worker overlap: hide reward computation behind generation\n"
        "  • Automatic parallelism assignment based on model sizes\n"
        "  • Supports PPO, GRPO, REINFORCE variants"
    )
    ax.text(0.05, 0.95, concept, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
