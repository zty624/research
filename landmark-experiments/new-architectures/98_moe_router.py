"""
MoE Router Load Balancing Reproduction
=======================================
Reproduces core ideas from Mixtral (2401.04088, Jiang et al.) and
Switch Transformer (2101.03961, Fedus et al.):
1. Top-k gating with noise for exploration during training
2. Load balancing auxiliary loss to prevent expert collapse
3. Expert capacity management (drop tokens when capacity exceeded)
4. Compare: naive routing vs load-balanced routing
5. Show: expert utilization distribution, load balance loss effect, routing entropy
6. Demonstrate: how auxiliary loss prevents expert collapse and improves specialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Expert ──

class Expert(nn.Module):
    """A single FFN expert."""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ── Router Variants ──

class NaiveRouter(nn.Module):
    """Top-k router without load balancing — suffers from expert collapse."""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        """x: (B, T, D) -> (weights, indices, gate_logits)"""
        logits = self.gate(x)  # (B, T, n_experts)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices, logits


class NoisyRouter(nn.Module):
    """Top-k router with Gaussian noise for exploration (Switch Transformer style)."""
    def __init__(self, d_model, n_experts, top_k=2, noise_std=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Noise projection (like Switch Transformer)
        self.noise_proj = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        """x: (B, T, D) -> (weights, indices, gate_logits)"""
        logits = self.gate(x)  # (B, T, n_experts)
        if self.training:
            # Add noise during training for exploration
            noise = torch.randn_like(logits) * F.softplus(self.noise_proj(x))
            logits = logits + noise
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices, logits


# ── Load Balancing Loss ──

def load_balance_loss(gate_logits, n_experts, top_k=2):
    """Auxiliary load balancing loss (Switch Transformer / GShard style).

    L_balance = n_experts * sum_i (f_i * P_i)
    where:
      f_i = fraction of tokens routed to expert i (based on top-1)
      P_i = mean router probability for expert i

    When balanced: f_i = 1/n_experts, P_i = 1/n_experts
    -> L = n_experts * n_experts * (1/n_experts)^2 = 1.0 (minimum)
    """
    B, T, _ = gate_logits.shape
    n_tokens = B * T

    # Fraction of tokens routed to each expert (top-1 routing)
    top1_indices = gate_logits.argmax(dim=-1)  # (B, T)
    f = torch.zeros(n_experts, device=gate_logits.device)
    for i in range(n_experts):
        f[i] = (top1_indices == i).float().mean()

    # Mean router probability per expert
    P = F.softmax(gate_logits, dim=-1).mean(dim=(0, 1))  # (n_experts,)

    balance_loss = n_experts * (f * P).sum()
    return balance_loss


# ── MoE Layer with Capacity ──

class MoELayer(nn.Module):
    """Mixture of Experts layer with configurable router and capacity."""
    def __init__(self, d_model, n_experts=8, top_k=2, capacity_factor=1.0,
                 router_type='noisy', balance_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.balance_weight = balance_weight

        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])

        if router_type == 'naive':
            self.router = NaiveRouter(d_model, n_experts, top_k)
        elif router_type == 'noisy':
            self.router = NoisyRouter(d_model, n_experts, top_k)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

        self.router_type = router_type
        # Track expert usage for analysis
        self.expert_counts = torch.zeros(n_experts)

    def forward(self, x):
        """x: (B, T, D) -> (output, balance_loss)"""
        B, T, D = x.shape
        weights, indices, gate_logits = self.router(x)

        output = torch.zeros_like(x)
        x_flat = x.view(-1, D)  # (B*T, D)
        weights_flat = weights.view(-1, self.top_k)
        indices_flat = indices.view(-1, self.top_k)

        # Expert capacity: max tokens per expert
        n_tokens = B * T
        capacity = int(self.capacity_factor * n_tokens / self.n_experts)

        # Track expert usage
        expert_counts = torch.zeros(self.n_experts, device=x.device)
        tokens_dropped = 0

        for k in range(self.top_k):
            # Count tokens per expert for capacity management
            expert_token_count = torch.zeros(self.n_experts, device=x.device, dtype=torch.long)
            # First pass: count
            for expert_idx in range(self.n_experts):
                mask = (indices_flat[:, k] == expert_idx)
                expert_token_count[expert_idx] = mask.sum()

            # Second pass: route with capacity
            for expert_idx in range(self.n_experts):
                mask = (indices_flat[:, k] == expert_idx)
                if not mask.any():
                    continue

                token_indices = mask.nonzero(as_tuple=True)[0]
                # Apply capacity limit
                if self.capacity_factor < float('inf') and len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]
                    tokens_dropped += mask.sum().item() - capacity

                expert_input = x_flat[token_indices]
                expert_output = self.experts[expert_idx](expert_input)

                output.view(-1, D)[token_indices] += (
                    weights_flat[token_indices, k].unsqueeze(-1) * expert_output
                )
                expert_counts[expert_idx] += len(token_indices)

        self.expert_counts = expert_counts.detach().cpu()

        # Load balancing loss
        bl = load_balance_loss(gate_logits, self.n_experts, self.top_k)

        return output, bl


# ── Full MoE Model ──

class MoEModel(nn.Module):
    """Simple MoE model for sequence classification."""
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 n_experts=8, top_k=2, router_type='noisy',
                 balance_weight=0.01, capacity_factor=1.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)

        self.attn_layers = nn.ModuleList()
        self.moe_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            )
            self.moe_layers.append(
                MoELayer(d_model, n_experts, top_k, capacity_factor,
                         router_type, balance_weight)
            )
            self.norms1.append(nn.LayerNorm(d_model))
            self.norms2.append(nn.LayerNorm(d_model))

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.n_experts = n_experts

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))

        total_bl = 0
        expert_counts_all = []
        for attn, moe, n1, n2 in zip(self.attn_layers, self.moe_layers,
                                       self.norms1, self.norms2):
            # Self-attention
            h_norm = n1(h)
            attn_out, _ = attn(h_norm, h_norm, h_norm)
            h = h + attn_out
            # MoE FFN
            moe_out, bl = moe(n2(h))
            h = h + moe_out
            total_bl += bl
            expert_counts_all.append(moe.expert_counts)

        return self.head(self.norm_out(h)), total_bl, expert_counts_all


# ── Data ──

def generate_synthetic_data(n_samples=10000, seq_len=32, vocab_size=50, n_patterns=4):
    """Generate synthetic sequence data with distinct patterns that should
    route to different experts."""
    sequences = torch.zeros(n_samples, seq_len + 1, dtype=torch.long)
    labels = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        pattern_id = i % n_patterns
        labels[i] = pattern_id
        if pattern_id == 0:
            # Ascending sequence
            base = torch.randint(1, vocab_size // 2, (1,)).item()
            for j in range(seq_len + 1):
                sequences[i, j] = (base + j) % vocab_size
        elif pattern_id == 1:
            # Descending sequence
            base = torch.randint(vocab_size // 2, vocab_size, (1,)).item()
            for j in range(seq_len + 1):
                sequences[i, j] = (base - j) % vocab_size
        elif pattern_id == 2:
            # Periodic sequence
            period = np.random.randint(2, 5)
            pattern = torch.randint(1, vocab_size, (period,))
            for j in range(seq_len + 1):
                sequences[i, j] = pattern[j % period]
        else:
            # Random sequence
            sequences[i] = torch.randint(1, vocab_size, (seq_len + 1,))

    return sequences[:, :-1], sequences[:, 1:]


# ── Training ──

def train_model(model, n_steps=2000, lr=1e-3, batch_size=64,
                vocab_size=50, seq_len=32, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    balance_losses = []
    expert_util_history = []

    for step in range(n_steps):
        x, y = generate_synthetic_data(batch_size, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)

        logits, bl, expert_counts = model(x)
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        # Find the balance_weight from the MoE layer
        balance_weight = model.moe_layers[0].balance_weight
        loss = ce_loss + balance_weight * bl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(ce_loss.item())
        balance_losses.append(bl.item())

        # Record expert utilization periodically
        if step % 100 == 0:
            expert_util_history.append(expert_counts[0].numpy().copy())

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | CE: {ce_loss.item():.4f} | BL: {bl.item():.4f}")

    return losses, balance_losses, expert_util_history


def compute_routing_entropy(model, n_batches=10, batch_size=64,
                             seq_len=32, vocab_size=50, device='cpu'):
    """Compute routing entropy: higher = more balanced expert usage."""
    model.eval()
    all_gate_probs = []

    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = generate_synthetic_data(batch_size, seq_len, vocab_size)
            x = x.to(device)

            B, T = x.shape
            h = model.emb(x) + model.pos_emb(torch.arange(T, device=device))

            for attn, moe, n1, n2 in zip(model.attn_layers, model.moe_layers,
                                           model.norms1, model.norms2):
                h_norm = n1(h)
                attn_out, _ = attn(h_norm, h_norm, h_norm)
                h = h + attn_out
                # Get gate logits from router
                _, _, gate_logits = moe.router(n2(h))
                gate_probs = F.softmax(gate_logits, dim=-1)  # (B, T, n_experts)
                all_gate_probs.append(gate_probs.mean(dim=(0, 1)))  # mean over batch & seq

                moe_out, _ = moe(n2(h))
                h = h + moe_out

    # Average gate probabilities
    avg_probs = torch.stack(all_gate_probs).mean(dim=0)  # (n_experts,)
    # Entropy: H = -sum(p * log(p))
    entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum().item()
    # Max entropy (uniform): log(n_experts)
    max_entropy = math.log(model.n_experts)
    # Normalized entropy
    norm_entropy = entropy / max_entropy

    return entropy, max_entropy, norm_entropy, avg_probs.cpu().numpy()


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "98-moe-router"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 50
    seq_len = 32
    d_model = 64
    n_heads = 4
    n_layers = 2
    n_experts = 8
    top_k = 2
    n_steps = 2000

    # ── Experiment 1: Naive vs Noisy Router with Load Balancing ──

    print("=== Naive Router (no balance loss) ===")
    naive_model = MoEModel(
        vocab_size, d_model, n_heads, n_layers, n_experts, top_k,
        router_type='naive', balance_weight=0.0
    ).to(device)
    naive_params = sum(p.numel() for p in naive_model.parameters())
    print(f"  Params: {naive_params:,}")
    naive_losses, naive_bl, naive_util = train_model(
        naive_model, n_steps, device=device, vocab_size=vocab_size, seq_len=seq_len
    )

    print("\n=== Noisy Router (with balance loss) ===")
    balanced_model = MoEModel(
        vocab_size, d_model, n_heads, n_layers, n_experts, top_k,
        router_type='noisy', balance_weight=0.01
    ).to(device)
    balanced_params = sum(p.numel() for p in balanced_model.parameters())
    print(f"  Params: {balanced_params:,}")
    balanced_losses, balanced_bl, balanced_util = train_model(
        balanced_model, n_steps, device=device, vocab_size=vocab_size, seq_len=seq_len
    )

    # ── Experiment 2: Balance weight sensitivity ──

    print("\n=== Balance Weight Sensitivity ===")
    bw_results = {}
    for bw in [0.0, 0.001, 0.01, 0.1]:
        print(f"  balance_weight={bw}:")
        m = MoEModel(
            vocab_size, d_model, n_heads, n_layers, n_experts, top_k,
            router_type='noisy', balance_weight=bw
        ).to(device)
        l, bl, util = train_model(
            m, n_steps=1500, device=device, vocab_size=vocab_size, seq_len=seq_len
        )
        bw_results[bw] = {'losses': l, 'bl': bl, 'util': util}

    # ── Experiment 3: Routing entropy analysis ──

    print("\n=== Routing Entropy ===")
    naive_ent, naive_max_ent, naive_norm_ent, naive_probs = compute_routing_entropy(
        naive_model, device=device, vocab_size=vocab_size, seq_len=seq_len
    )
    print(f"  Naive router: H={naive_ent:.3f}, H_max={naive_max_ent:.3f}, "
          f"H_norm={naive_norm_ent:.3f}")

    balanced_ent, balanced_max_ent, balanced_norm_ent, balanced_probs = compute_routing_entropy(
        balanced_model, device=device, vocab_size=vocab_size, seq_len=seq_len
    )
    print(f"  Balanced router: H={balanced_ent:.3f}, H_max={balanced_max_ent:.3f}, "
          f"H_norm={balanced_norm_ent:.3f}")

    # ── Experiment 4: Expert utilization at different training stages ──

    print("\n=== Expert Utilization Over Training ===")
    # Train a new model and record utilization every 200 steps
    track_model = MoEModel(
        vocab_size, d_model, n_heads, n_layers, n_experts, top_k,
        router_type='noisy', balance_weight=0.01
    ).to(device)
    _, _, tracked_util = train_model(
        track_model, n_steps=2000, device=device, vocab_size=vocab_size, seq_len=seq_len
    )

    # ── Visualization ──

    window = 30

    # 1. Training loss: naive vs balanced
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    smoothed_naive = np.convolve(naive_losses, np.ones(window)/window, mode='valid')
    smoothed_balanced = np.convolve(balanced_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(smoothed_naive, label='Naive (no balance)', color='red')
    axes[0].plot(smoothed_balanced, label='Noisy + Balance', color='blue')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("CE Loss (smoothed)")
    axes[0].set_title("Training Loss: Naive vs Balanced Routing")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Balance loss over time
    smoothed_naive_bl = np.convolve(naive_bl, np.ones(window)/window, mode='valid')
    smoothed_balanced_bl = np.convolve(balanced_bl, np.ones(window)/window, mode='valid')

    axes[1].plot(smoothed_naive_bl, label='Naive', color='red')
    axes[1].plot(smoothed_balanced_bl, label='Noisy + Balance', color='blue')
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Balance Loss (smoothed)")
    axes[1].set_title("Load Balancing Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MoE Router: Naive vs Load-Balanced Routing", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Expert utilization distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Naive: final expert utilization
    if len(naive_util) > 0:
        final_naive = naive_util[-1]
        total = final_naive.sum()
        fracs = final_naive / total if total > 0 else final_naive
        axes[0].bar(range(n_experts), fracs, color='red', alpha=0.7)
        axes[0].axhline(y=1.0/n_experts, color='green', linestyle='--', alpha=0.5,
                        label='Uniform')
        axes[0].set_xlabel("Expert Index")
        axes[0].set_ylabel("Fraction of Tokens")
        axes[0].set_title("Naive Router: Expert Utilization")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

    # Balanced: final expert utilization
    if len(balanced_util) > 0:
        final_balanced = balanced_util[-1]
        total = final_balanced.sum()
        fracs = final_balanced / total if total > 0 else final_balanced
        axes[1].bar(range(n_experts), fracs, color='blue', alpha=0.7)
        axes[1].axhline(y=1.0/n_experts, color='green', linestyle='--', alpha=0.5,
                        label='Uniform')
        axes[1].set_xlabel("Expert Index")
        axes[1].set_ylabel("Fraction of Tokens")
        axes[1].set_title("Balanced Router: Expert Utilization")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Expert Utilization: Expert Collapse vs Balanced Routing", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "expert_utilization.png", dpi=150)
    plt.close()

    # 3. Balance weight sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['red', 'orange', 'blue', 'green']
    for (bw, data), color in zip(bw_results.items(), colors):
        smoothed = np.convolve(data['losses'], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=f'bw={bw}', color=color)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("CE Loss (smoothed)")
    axes[0].set_title("Training Loss vs Balance Weight")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for (bw, data), color in zip(bw_results.items(), colors):
        smoothed = np.convolve(data['bl'], np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed, label=f'bw={bw}', color=color)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Balance Loss (smoothed)")
    axes[1].set_title("Balance Loss vs Balance Weight")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MoE: Balance Weight Sensitivity Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "balance_weight_sensitivity.png", dpi=150)
    plt.close()

    # 4. Routing entropy visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_experts = np.arange(n_experts)
    axes[0].bar(x_experts - 0.2, naive_probs, 0.4, label='Naive', color='red', alpha=0.7)
    axes[0].bar(x_experts + 0.2, balanced_probs, 0.4, label='Balanced', color='blue', alpha=0.7)
    axes[0].axhline(y=1.0/n_experts, color='green', linestyle='--', alpha=0.5,
                    label='Uniform')
    axes[0].set_xlabel("Expert Index")
    axes[0].set_ylabel("Mean Gate Probability")
    axes[0].set_title("Gate Probability Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Entropy bar chart
    entropies = [naive_norm_ent, balanced_norm_ent]
    labels = ['Naive\nRouter', 'Balanced\nRouter']
    colors_bar = ['red', 'blue']
    bars = axes[1].bar(labels, entropies, color=colors_bar, alpha=0.7)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max (uniform)')
    axes[1].set_ylabel("Normalized Entropy (H/H_max)")
    axes[1].set_title("Routing Entropy")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, ent in zip(bars, entropies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{ent:.3f}', ha='center', fontsize=10)

    plt.suptitle("MoE: Routing Entropy Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "routing_entropy.png", dpi=150)
    plt.close()

    # 5. Expert utilization evolution over training
    fig, ax = plt.subplots(figsize=(12, 6))
    if len(tracked_util) > 0:
        n_snapshots = len(tracked_util)
        util_matrix = np.stack(tracked_util)  # (n_snapshots, n_experts)
        # Normalize each row
        row_sums = util_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        util_norm = util_matrix / row_sums

        im = ax.imshow(util_norm.T, aspect='auto', cmap='Blues',
                       interpolation='nearest')
        ax.set_xlabel("Training Snapshot (every 100 steps)")
        ax.set_ylabel("Expert Index")
        ax.set_title("Expert Utilization Over Training (Balanced Router)")
        plt.colorbar(im, ax=ax, label="Fraction of Tokens")
    plt.tight_layout()
    plt.savefig(results_dir / "utilization_evolution.png", dpi=150)
    plt.close()

    # 6. Concept diagram: naive vs balanced routing
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (title, desc, color) in enumerate([
        ("Naive\nRouting",
         "Tokens greedily\nassigned to\nbest expert\n-> Expert collapse\n-> Wasted capacity",
         'red'),
        ("Noisy\nRouter",
         "Gaussian noise\nadded to logits\n-> Exploration\n-> Some diversity\nbut not enough",
         'orange'),
        ("Load-Balanced\nRouting",
         "Auxiliary loss:\nn*sum(f_i*P_i)\n+ Noise for\nexploration\n-> All experts used\n-> Specialization",
         'blue'),
    ]):
        ax = axes[ax_idx]
        ax.axis('off')
        ax.text(0.5, 0.7, title, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("MoE Router: From Expert Collapse to Balanced Specialization",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "routing_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()