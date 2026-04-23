"""
Minimal RWKV Reproduction
===========================
Reproduces core ideas from RWKV (2305.13048, Peng et al.):
1. Linear attention: O(N) complexity instead of O(N²)
2. Time-mixing: WKV attention with channel-wise decay (like attention without softmax)
3. Time-shift: shift tokens to enable positional information flow
4. RNN-mode at inference: constant memory per step (unlike Transformer's O(N) KV cache)
5. Combines Transformer parallel training + RNN efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── RWKV Time Mixing ──

class RWKVTimeMixing(nn.Module):
    """RWKV time-mixing block: linear attention with channel-wise decay."""
    def __init__(self, d_model, n_heads, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps

        # Time-mixing projections
        self.time_mix_k = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_v = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_r = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out = nn.Linear(d_model, d_model)

        # Per-head learnable decay (initialized to different rates)
        # μ = e^(-e^ω) where ω is learnable
        self.omega = nn.Parameter(torch.randn(n_heads) * 0.3 - 5.0)

        # Group norm for output
        self.ln_x = nn.GroupNorm(n_heads, d_model)

    def get_decay(self):
        """Compute decay factor μ from ω: μ = e^(-e^ω)."""
        return torch.exp(-torch.exp(self.omega))  # (H,)

    def forward(self, x):
        """x: (B, T, D) — parallel mode for training."""
        B, T, D = x.shape
        H = self.n_heads

        # Time-shift: mix current and previous token
        x_shift = torch.cat([x[:, :1], x[:, :-1]], dim=1)

        # Compute k, v, r with time mixing
        k = self.time_mix_k(x_shift).reshape(B, T, H, self.d_head)  # (B, T, H, d)
        v = self.time_mix_v(x_shift).reshape(B, T, H, self.d_head)
        r = self.time_mix_r(x).reshape(B, T, H, self.d_head)

        # WKV computation: linear attention with decay
        # wkv_t = (Σ_{i=1}^{t-1} μ^{t-1-i} · v_i · σ(k_i)) + μ^t · wkv_{t-1}
        # This can be computed in parallel using cumulative sums
        decay = self.get_decay()  # (H,)

        # For parallel computation, use the chunked approach
        # Simplified: compute weighted cumulative sum
        # σ(k) as "attention weights" (no softmax, just sigmoid)
        w = torch.sigmoid(k)  # (B, T, H, d)

        # Compute WKV using cumulative product of decay
        # wkv_t = Σ_{i=1}^{t} (Π_{j=i+1}^{t} μ) * w_i * v_i
        # = μ * wkv_{t-1} + w_t * v_t (recurrence)

        # Parallel computation via prefix sum
        # log-space: log(wkv_t) = log(μ * wkv_{t-1} + w_t * v_t)
        # Use standard recurrence (sequential but efficient on GPU with scan)
        wkv = torch.zeros_like(v)
        acc = torch.zeros(B, H, self.d_head, device=x.device)

        decay_expanded = decay.unsqueeze(0).unsqueeze(2)  # (1, H, 1)
        for t in range(T):
            acc = decay_expanded * acc + w[:, t] * v[:, t]
            wkv[:, t] = acc

        # Output: σ(r) * wkv
        out = torch.sigmoid(r) * wkv  # (B, T, H, d)
        out = out.reshape(B, T, D)

        # Group norm + output projection
        out = out.transpose(1, 2)  # (B, D, T) for group norm
        out = self.ln_x(out)
        out = out.transpose(1, 2)  # back to (B, T, D)
        return self.out(out)


# ── RWKV Channel Mixing ──

class RWKVChannelMixing(nn.Module):
    """RWKV channel-mixing block: FFN with time-shift and squared ReLU."""
    def __init__(self, d_model, expand_factor=4):
        super().__init__()
        hidden = d_model * expand_factor

        self.time_mix_k = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_r = nn.Linear(d_model, d_model, bias=False)

        self.key = nn.Linear(d_model, hidden, bias=False)
        self.value = nn.Linear(hidden, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # Time-shift
        x_shift = torch.cat([x[:, :1], x[:, :-1]], dim=1)

        k = self.key(self.time_mix_k(x_shift))
        r = torch.sigmoid(self.receptance(self.time_mix_r(x)))
        v = self.value(torch.square(torch.relu(k)))  # Squared ReLU

        return r * v


# ── RWKV Block ──

class RWKVBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix = RWKVTimeMixing(d_model, n_heads)
        self.channel_mix = RWKVChannelMixing(d_model)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ── RWKV Model ──

class RWKVModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            RWKVBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.ln_in(self.emb(x))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_out(h))


# ── Baseline: Standard Transformer ──

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            h = block(h, src_mask=mask)
        return self.head(self.ln(h))


# ── Data ──

def generate_periodic_data(batch_size, seq_len, vocab_size=20):
    """Periodic sequence: predict next token given pattern."""
    sequences = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    for i in range(batch_size):
        period = np.random.randint(2, 6)
        pattern = torch.randint(1, vocab_size, (period,))
        for j in range(seq_len + 1):
            sequences[i, j] = pattern[j % period]
    return sequences[:, :-1], sequences[:, 1:]


# ── Training ──

def train_model(model, vocab_size, seq_len, n_steps=3000, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    for step in range(n_steps):
        x, y = generate_periodic_data(64, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")

    return losses


# ── Inference Speed Benchmark ──

def benchmark_inference(model, seq_len, n_steps=50, device='cpu'):
    """Benchmark inference speed."""
    import time
    model.eval()

    x = torch.randint(0, 20, (1, 8), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.randint(0, 20, (1, seq_len), device=device)
            t0 = time.time()
            _ = model(x)
            times.append(time.time() - t0)

    model.train()
    return np.mean(times) * 1000  # ms


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "53-rwkv"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 20
    seq_len = 64

    # Experiment 1: RWKV vs Transformer training
    print("=== RWKV vs Transformer Training ===")

    print("\n  RWKV:")
    rwkv = RWKVModel(vocab_size, d_model=64, n_heads=4, n_layers=2).to(device)
    n_params_rwkv = sum(p.numel() for p in rwkv.parameters())
    print(f"    Params: {n_params_rwkv:,}")
    rwkv_losses = train_model(rwkv, vocab_size, seq_len, n_steps=3000, device=device)

    print("\n  Transformer:")
    tf = TransformerModel(vocab_size, d_model=64, n_heads=4, n_layers=2).to(device)
    n_params_tf = sum(p.numel() for p in tf.parameters())
    print(f"    Params: {n_params_tf:,}")
    tf_losses = train_model(tf, vocab_size, seq_len, n_steps=3000, device=device)

    # Experiment 2: Different sequence lengths
    print("\n=== Sequence Length Scaling ===")
    len_results = {}
    for sl in [32, 64, 128, 256]:
        print(f"\n  Len={sl}:")

        r = RWKVModel(vocab_size, d_model=64, n_heads=4, n_layers=2).to(device)
        t = TransformerModel(vocab_size, d_model=64, n_heads=4, n_layers=2).to(device)

        rl = train_model(r, vocab_size, sl, n_steps=2000, device=device)
        tl = train_model(t, vocab_size, sl, n_steps=2000, device=device)

        len_results[sl] = {'rwkv_loss': rl[-1], 'tf_loss': tl[-1]}
        print(f"    RWKV: {rl[-1]:.4f}, Transformer: {tl[-1]:.4f}")

    # Experiment 3: RWKV decay visualization
    print("\n=== RWKV Decay Analysis ===")
    rwkv_block = rwkv.blocks[0].time_mix
    decay = rwkv_block.get_decay().detach().cpu().numpy()
    print(f"  Per-head decay: {decay}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rwkv_losses, label='RWKV', color='blue')
    axes[0].plot(tf_losses, label='Transformer', color='red')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sequence length scaling
    lens = sorted(len_results.keys())
    rwkv_l = [len_results[s]['rwkv_loss'] for s in lens]
    tf_l = [len_results[s]['tf_loss'] for s in lens]

    axes[1].plot(lens, rwkv_l, 'o-', label='RWKV', color='blue')
    axes[1].plot(lens, tf_l, 's--', label='Transformer', color='red')
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Loss vs Sequence Length")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("RWKV: Linear Attention with Trainable Decay", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "rwkv_comparison.png", dpi=150)
    plt.close()

    # 2. WKV attention pattern (decay effect)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, head_decay in enumerate(decay[:4]):
        ax = axes[idx]
        T = 32
        # Show how much each past position contributes to current wkv
        contributions = np.zeros(T)
        for t in range(T):
            contributions[t] = head_decay ** (T - 1 - t)

        ax.bar(range(T), contributions, color='blue', alpha=0.7)
        ax.set_title(f"Head {idx}: μ={head_decay:.4f}")
        ax.set_xlabel("Position offset")
        ax.set_ylabel("Contribution weight")

    plt.suptitle("RWKV: Per-Head Temporal Decay Patterns", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "rwkv_decay.png", dpi=150)
    plt.close()

    # 3. Complexity comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    seq_lens_plot = np.array([16, 32, 64, 128, 256, 512, 1024])
    # Transformer: O(N²) memory
    tf_mem = seq_lens_plot ** 2
    # RWKV: O(N) memory (constant per step in RNN mode)
    rwkv_mem = seq_lens_plot * 64  # linear

    ax.plot(seq_lens_plot, tf_mem / 1000, 'o-', label='Transformer O(N²)', color='red')
    ax.plot(seq_lens_plot, rwkv_mem / 1000, 's--', label='RWKV O(N)', color='blue')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Relative Memory (K units)")
    ax.set_title("Memory Complexity: Transformer vs RWKV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(results_dir / "memory_complexity.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Transformer\nAttention", "O(N²) compute & memory\nKV cache grows linearly\nGreat training\nExpensive inference\n→ O(N) per step", 0.14, 'red'),
        ("RWKV\nLinear Attn", "WKV with channel decay\nμ^t decay over time\nO(N) memory\nRNN-mode inference\n→ O(1) per step!", 0.5, 'blue'),
        ("Best of\nBoth Worlds", "Parallel training\n(like Transformer)\nEfficient inference\n(like RNN)\n→ Linear Attn!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("RWKV: Transformer-Quality Training + RNN-Efficient Inference", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rwkv_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
