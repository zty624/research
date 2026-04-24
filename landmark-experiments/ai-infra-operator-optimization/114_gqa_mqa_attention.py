"""
Minimal GQA/MQA Attention Reproduction
========================================
Reproduces core ideas from Multi-Query Attention (MQA, 1911.02150, Shazeer)
and Grouped-Query Attention (GQA, 2305.13245, Ainslie et al.):
1. Multi-Head Attention (MHA): each head has its own K,V projections
2. Multi-Query Attention (MQA): all heads share single K,V projections
3. Grouped-Query Attention (GQA): groups of heads share K,V projections
4. Compare: memory (KV cache), speed, and quality tradeoffs
5. Show: KV cache size reduction vs quality degradation
6. Demonstrate: GQA as a sweet spot between MHA and MQA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── Attention Implementations ──

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA): each head has own K,V."""
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """x: (B, T, D) → (B, T, D). Returns output and new KV cache."""
        B, T, D = x.shape
        h = self.n_heads
        d = self.d_head

        q = self.q_proj(x).reshape(B, T, h, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).reshape(B, T, h, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, h, d).transpose(1, 2)

        # KV cache: append to existing
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache = (k, v)

        # Attention
        scale = d ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, T_kv)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), new_cache

    def kv_cache_size(self, seq_len, batch_size=1, dtype_bytes=2):
        """KV cache memory in bytes: 2 * B * T * H * d * bytes."""
        return 2 * batch_size * seq_len * self.n_heads * self.d_head * dtype_bytes


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA): all heads share single K,V."""
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.d_head)  # single head K
        self.v_proj = nn.Linear(d_model, self.d_head)  # single head V
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        h = self.n_heads
        d = self.d_head

        q = self.q_proj(x).reshape(B, T, h, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).unsqueeze(1)  # (B, 1, T, d) — broadcast to all heads
        v = self.v_proj(x).unsqueeze(1)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache = (k, v)

        # Expand K,V to match heads
        k_exp = k.expand(-1, h, -1, -1)  # (B, H, T_kv, d)
        v_exp = v.expand(-1, h, -1, -1)

        scale = d ** 0.5
        attn = (q @ k_exp.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_exp).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), new_cache

    def kv_cache_size(self, seq_len, batch_size=1, dtype_bytes=2):
        return 2 * batch_size * seq_len * 1 * self.d_head * dtype_bytes  # 1 head for KV


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA): groups of heads share K,V."""
    def __init__(self, d_model=256, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        h = self.n_heads
        kv_h = self.n_kv_heads
        d = self.d_head

        q = self.q_proj(x).reshape(B, T, h, d).transpose(1, 2)   # (B, H, T, d)
        k = self.k_proj(x).reshape(B, T, kv_h, d).transpose(1, 2)  # (B, kv_H, T, d)
        v = self.v_proj(x).reshape(B, T, kv_h, d).transpose(1, 2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache = (k, v)

        # Expand KV heads to match query heads via repeat
        # k: (B, kv_H, T, d) → (B, H, T, d)
        k_exp = k.repeat_interleave(self.group_size, dim=1)
        v_exp = v.repeat_interleave(self.group_size, dim=1)

        scale = d ** 0.5
        attn = (q @ k_exp.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_exp).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), new_cache

    def kv_cache_size(self, seq_len, batch_size=1, dtype_bytes=2):
        return 2 * batch_size * seq_len * self.n_kv_heads * self.d_head * dtype_bytes


# ── Transformer Block ──

class TransformerBlock(nn.Module):
    def __init__(self, attn_module, d_model=256):
        super().__init__()
        self.attn = attn_module
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ── Benchmarking ──

def benchmark_attention(attn_cls, d_model=256, n_heads=8, n_kv_heads=None,
                         seq_len=64, batch_size=4, n_warmup=10, n_iter=50, device='cpu'):
    """Benchmark attention module forward pass."""
    kwargs = {'d_model': d_model, 'n_heads': n_heads}
    if n_kv_heads is not None:
        kwargs['n_kv_heads'] = n_kv_heads

    attn = attn_cls(**kwargs).to(device)
    block = TransformerBlock(attn, d_model).to(device)
    block.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _, _ = block(x)

    torch.cuda.synchronize() if device != 'cpu' else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            _, _ = block(x)
    torch.cuda.synchronize() if device != 'cpu' else None
    t1 = time.perf_counter()

    ms = (t1 - t0) / n_iter * 1000
    params = sum(p.numel() for p in block.parameters())
    cache_size = attn.kv_cache_size(seq_len, batch_size)

    return ms, params, cache_size


# ── Quality Test: Copy Task ──

def train_copy_task(attn_cls, d_model=128, n_heads=4, n_kv_heads=None,
                     n_steps=1500, seq_len=16, batch_size=32, lr=1e-3, device='cpu'):
    """Train on a copy task: given input, reproduce it after a delay."""
    kwargs = {'d_model': d_model, 'n_heads': n_heads}
    if n_kv_heads is not None:
        kwargs['n_kv_heads'] = n_kv_heads

    attn = attn_cls(**kwargs).to(device)
    block = TransformerBlock(attn, d_model).to(device)

    # Simple: embed input → transformer → predict input
    embed = nn.Linear(1, d_model).to(device)
    head = nn.Linear(d_model, 1).to(device)
    model = nn.Sequential()

    optimizer = torch.optim.AdamW(
        list(block.parameters()) + list(embed.parameters()) + list(head.parameters()),
        lr=lr
    )

    losses = []
    for step in range(n_steps):
        # Input: random sine waves
        x_raw = torch.randn(batch_size, seq_len, 1, device=device)
        x_in = embed(x_raw)
        out, _ = block(x_in)
        pred = head(out)

        loss = F.mse_loss(pred, x_raw)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "114-gqa-mqa-attention"
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = 256
    n_heads = 8

    # ── Experiment 1: KV Cache Size Comparison ──
    print("=== Experiment 1: KV Cache Size ===")
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    batch_size = 1

    mha = MultiHeadAttention(d_model, n_heads)
    mqa = MultiQueryAttention(d_model, n_heads)
    gqa_configs = [
        ('GQA-2', 2), ('GQA-4', 4), ('GQA-6', 6),
    ]

    cache_data = {'MHA': [], 'MQA': []}
    for name, kv_h in gqa_configs:
        cache_data[name] = []

    for sl in seq_lengths:
        cache_data['MHA'].append(mha.kv_cache_size(sl, batch_size) / 1e6)  # MB
        cache_data['MQA'].append(mqa.kv_cache_size(sl, batch_size) / 1e6)
        for name, kv_h in gqa_configs:
            g = GroupedQueryAttention(d_model, n_heads, kv_h)
            cache_data[name].append(g.kv_cache_size(sl, batch_size) / 1e6)

    for name in cache_data:
        print(f"  {name} @ T=4096: {cache_data[name][-1]:.2f} MB")

    # ── Experiment 2: Parameter Count ──
    print("\n=== Experiment 2: Parameter Count ===")
    configs = [
        ('MHA', MultiHeadAttention, None),
        ('GQA-2', GroupedQueryAttention, 2),
        ('GQA-4', GroupedQueryAttention, 4),
        ('GQA-6', GroupedQueryAttention, 6),
        ('MQA', MultiQueryAttention, None),
    ]

    param_data = {}
    for name, cls, kv_h in configs:
        kwargs = {'d_model': d_model, 'n_heads': n_heads}
        if kv_h is not None:
            kwargs['n_kv_heads'] = kv_h
        m = cls(**kwargs)
        p = sum(p.numel() for p in m.parameters())
        param_data[name] = p
        print(f"  {name}: {p:,} params")

    # ── Experiment 3: Speed Benchmark ──
    print("\n=== Experiment 3: Speed Benchmark ===")
    speed_data = {}
    for name, cls, kv_h in configs:
        try:
            ms, p, cs = benchmark_attention(cls, d_model, n_heads, kv_h,
                                             seq_len=128, device=device)
            speed_data[name] = ms
            print(f"  {name}: {ms:.2f} ms/forward")
        except Exception as e:
            print(f"  {name}: failed ({e})")
            speed_data[name] = 0

    # ── Experiment 4: Copy Task Quality ──
    print("\n=== Experiment 4: Copy Task Quality ===")
    quality_data = {}
    copy_n_heads = 4
    for name, cls, kv_h in configs:
        # n_kv_heads cannot exceed n_heads
        copy_kv_h = min(kv_h, copy_n_heads) if kv_h is not None else None
        losses = train_copy_task(cls, d_model=128, n_heads=copy_n_heads,
                                  n_kv_heads=copy_kv_h, n_steps=1000, device=device)
        quality_data[name] = losses
        print(f"  {name}: final loss = {np.mean(losses[-50:]):.4f}")

    # ── Visualization ──

    # 1. KV Cache size
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'MHA': '#e74c3c', 'GQA-2': '#3498db', 'GQA-4': '#2ecc71',
              'GQA-6': '#f39c12', 'MQA': '#9b59b6'}
    for name in ['MHA', 'GQA-2', 'GQA-4', 'GQA-6', 'MQA']:
        ax.plot(seq_lengths, cache_data[name], marker='o', label=name,
                color=colors[name], linewidth=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("KV Cache Size (MB)")
    ax.set_title("KV Cache Memory: MHA vs GQA vs MQA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'kv_cache_size.png', dpi=150)
    plt.close()

    # 2. Parameter count
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(param_data.keys())
    vals = list(param_data.values())
    bar_colors = [colors[n] for n in names]
    ax.bar(names, vals, color=bar_colors, alpha=0.7)
    ax.set_ylabel("Parameters")
    ax.set_title("Attention Parameter Count")
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(vals):
        ax.text(i, v + 500, f"{v:,}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / 'param_count.png', dpi=150)
    plt.close()

    # 3. Speed comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(speed_data.keys())
    vals = list(speed_data.values())
    bar_colors = [colors[n] for n in names]
    ax.bar(names, vals, color=bar_colors, alpha=0.7)
    ax.set_ylabel("Time per forward (ms)")
    ax.set_title("Inference Speed")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / 'speed.png', dpi=150)
    plt.close()

    # 4. Quality comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 30
    for name, losses in quality_data.items():
        smoothed = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax.plot(smoothed, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("Copy Task: Quality Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'quality.png', dpi=150)
    plt.close()

    # 5. Pareto: quality vs cache
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, losses in quality_data.items():
        final_loss = np.mean(losses[-50:])
        cache_mb = cache_data[name][2]  # at seq_len=512
        ax.scatter(cache_mb, final_loss, s=150, color=colors[name], zorder=5)
        ax.annotate(name, (cache_mb, final_loss), fontsize=10,
                    xytext=(10, 5), textcoords='offset points')
    ax.set_xlabel("KV Cache Size at T=512 (MB)")
    ax.set_ylabel("Final Copy Task Loss")
    ax.set_title("Pareto: Quality vs Memory")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'pareto.png', dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    concepts = [
        ("MHA\n(Multi-Head)", "Each head has its own\nK, V projection\n\nKV cache: H × T × d\n(largest memory)", '#e74c3c'),
        ("GQA\n(Grouped-Query)", "Groups of heads share\nK, V projections\n\nKV cache: G × T × d\n(G < H, sweet spot)", '#3498db'),
        ("MQA\n(Multi-Query)", "All heads share single\nK, V projection\n\nKV cache: 1 × T × d\n(smallest memory)", '#9b59b6'),
    ]
    for ax, (title, desc, color) in zip(axes, concepts):
        ax.axis('off')
        ax.text(0.5, 0.7, title, transform=ax.transAxes, fontsize=16,
                ha='center', va='center', fontweight='bold', color=color)
        ax.text(0.5, 0.3, desc, transform=ax.transAxes, fontsize=11,
                ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('MHA → GQA → MQA: Quality-Memory Tradeoff', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
