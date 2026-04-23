"""
FlashAttention Memory-Efficient Attention Implementation
========================================================
Reproduces core ideas from "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness" (arxiv 2205.14135, Dao et al.):
1. Tiled attention: process QK^T in blocks that fit in SRAM
2. Online softmax: running max + running sum trick (no full NxN materialization)
3. O(N) memory for attention (vs O(N^2) standard)
4. Numerical equivalence: tiled produces same result as standard attention
5. Causal masking support via block-level skip + intra-block masking

This experiment focuses on the algorithmic implementation and memory analysis,
complementing the existing 07_flash_attention.py which covers IO-awareness concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── Standard Attention (baseline: materializes full NxN) ──

def standard_attention(Q, K, V, causal=False):
    """Standard attention. Materializes full S = QK^T and P = softmax(S).
    Memory footprint: O(N^2) for the attention matrix.
    """
    d = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / np.sqrt(d)

    if causal:
        T = S.shape[-2]
        mask = torch.triu(torch.ones(T, T, device=S.device), diagonal=1).bool()
        S = S.masked_fill(mask, float('-inf'))

    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O, P


# ── Tiled (FlashAttention-style) Attention with Online Softmax ──

def flash_attention_tiled(Q, K, V, block_size=64, causal=False):
    """FlashAttention-style tiled attention with online softmax.

    Online softmax algorithm (per query row i):
      Initialize: m_i = -inf, l_i = 0, O_i = 0
      For each key block j:
        1. Compute S_ij = Q_i @ K_j^T / sqrt(d)
        2. m_new = max(m_i, max(S_ij))
        3. alpha = exp(m_i - m_new)    # rescale old accumulation
        4. beta  = exp(S_ij - m_new)   # new block contributions
        5. l_i = alpha * l_i + sum(beta)
        6. O_i = alpha * O_i + beta @ V_j
        7. m_i = m_new
      Final: O_i = O_i / l_i

    Key property: never materializes the full NxN attention matrix.
    Only block_size x block_size intermediates exist at any time.
    """
    B, N, d = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(B, N, 1, device=Q.device)       # running sum of exp
    m = torch.full((B, N, 1), float('-inf'), device=Q.device)  # running max

    n_blocks = (N + block_size - 1) // block_size

    for j in range(n_blocks):
        j_s = j * block_size
        j_e = min(j_s + block_size, N)
        K_j = K[:, j_s:j_e]
        V_j = V[:, j_s:j_e]

        for i in range(n_blocks):
            i_s = i * block_size
            i_e = min(i_s + block_size, N)

            # Causal: skip blocks entirely above the diagonal
            if causal and j_s >= i_e:
                continue

            Q_i = Q[:, i_s:i_e]

            # Block of attention scores
            S_ij = Q_i @ K_j.transpose(-2, -1) / np.sqrt(d)

            if causal:
                row_idx = torch.arange(i_s, i_e, device=Q.device)
                col_idx = torch.arange(j_s, j_e, device=Q.device)
                causal_mask = col_idx.unsqueeze(0) > row_idx.unsqueeze(1)
                S_ij = S_ij.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

            # Online softmax update
            m_ij = S_ij.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m[:, i_s:i_e], m_ij)

            alpha = torch.exp(m[:, i_s:i_e] - m_new)
            beta = torch.exp(S_ij - m_new)

            l[:, i_s:i_e] = alpha * l[:, i_s:i_e] + beta.sum(dim=-1, keepdim=True)
            O[:, i_s:i_e] = alpha * O[:, i_s:i_e] + beta @ V_j
            m[:, i_s:i_e] = m_new

    O = O / l.clamp(min=1e-10)
    return O, l.squeeze(-1)  # return l for analysis


# ── PyTorch Module Wrapping Tiled Attention ──

class FlashAttentionLayer(nn.Module):
    """Multi-head attention using our tiled FlashAttention implementation."""
    def __init__(self, d_model, n_heads, block_size=64, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size
        self.causal = causal
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Reshape to (B*H, T, D_head) for our flash function
        BH = B * self.n_heads
        q = q.reshape(BH, T, self.d_head)
        k = k.reshape(BH, T, self.d_head)
        v = v.reshape(BH, T, self.d_head)

        o, _ = flash_attention_tiled(q, k, v, self.block_size, self.causal)

        # Reshape back
        o = o.reshape(B, self.n_heads, T, self.d_head).transpose(1, 2).reshape(B, T, D)
        return self.out(o)


class StandardAttentionLayer(nn.Module):
    """Multi-head attention using standard attention (baseline)."""
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        d = self.d_head
        S = (q @ k.transpose(-2, -1)) / np.sqrt(d)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            S = S.masked_fill(mask, float('-inf'))
        P = torch.softmax(S, dim=-1)
        o = (P @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(o)


# ── Memory Estimation ──

def memory_standard(N, d, dtype_bytes=4):
    """Standard attention memory: Q,K,V + S(NxN) + P(NxN) + O."""
    return (4 * N * d + 2 * N * N) * dtype_bytes


def memory_flash(N, d, block_size, dtype_bytes=4):
    """Flash attention memory: Q,K,V + O + block_sized S + l,m vectors."""
    return (4 * N * d + block_size * block_size + 2 * N) * dtype_bytes


# ── Numerical Equivalence Verification ──

def verify_equivalence(N, d, block_size, causal=False, device='cpu'):
    """Verify tiled attention matches standard attention numerically."""
    torch.manual_seed(42)
    B = 2
    Q = torch.randn(B, N, d, device=device)
    K = torch.randn(B, N, d, device=device)
    V = torch.randn(B, N, d, device=device)

    O_std, _ = standard_attention(Q, K, V, causal=causal)
    O_flash, _ = flash_attention_tiled(Q, K, V, block_size=block_size, causal=causal)

    max_diff = (O_std - O_flash).abs().max().item()
    rel_diff = (O_std - O_flash).abs().norm() / (O_std.abs().norm() + 1e-10)
    return max_diff, rel_diff.item()


# ── Online Softmax Step-by-Step Trace ──

def trace_online_softmax(Q, K, V, block_size, row_idx=0):
    """Trace the online softmax computation for a single query row.
    Returns per-step running max, running sum, and partial output norms.
    """
    B, N, d = Q.shape
    n_blocks = (N + block_size - 1) // block_size

    m_i = float('-inf')
    l_i = 0.0
    o_norm = 0.0

    trace = []
    q_row = Q[0, row_idx]  # (d,)

    for j in range(n_blocks):
        j_s = j * block_size
        j_e = min(j_s + block_size, N)
        k_block = K[0, j_s:j_e]  # (block, d)
        v_block = V[0, j_s:j_e]  # (block, d)

        s_block = (q_row @ k_block.T) / np.sqrt(d)  # (block,)
        m_new = max(m_i, s_block.max().item())
        alpha = np.exp(m_i - m_new) if m_i != float('-inf') else 0.0
        beta = torch.exp(s_block - m_new)
        l_i = alpha * l_i + beta.sum().item()
        o_norm = alpha * o_norm + (beta.unsqueeze(0) @ v_block).norm().item()
        m_i = m_new

        trace.append({
            'step': j,
            'block': (j_s, j_e),
            'm_i': m_i,
            'l_i': l_i,
            'o_norm': o_norm,
        })

    return trace


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "95-flash-attn-impl"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Experiment 1: Numerical Equivalence ──
    print("=== Experiment 1: Numerical Equivalence ===")
    configs = [
        (64, 16, 32, False),
        (64, 16, 32, True),
        (128, 32, 64, False),
        (128, 32, 64, True),
        (256, 64, 64, False),
        (256, 64, 64, True),
        (512, 64, 128, False),
        (512, 64, 128, True),
    ]
    equiv_results = []
    for N, d, bs, causal in configs:
        max_d, rel_d = verify_equivalence(N, d, bs, causal, device)
        tag = "causal" if causal else "bidir"
        equiv_results.append({'N': N, 'd': d, 'bs': bs, 'causal': causal,
                              'max_diff': max_d, 'rel_diff': rel_d})
        print(f"  N={N:4d} d={d:2d} bs={bs:3d} {tag:6s} | max_diff={max_d:.2e} rel_diff={rel_d:.2e}")

    # ── Experiment 2: Memory Scaling ──
    print("\n=== Experiment 2: Memory Scaling ===")
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    d = 64
    block_size = 64

    mem_std = [memory_standard(N, d) / 1e6 for N in seq_lengths]
    mem_flash = [memory_flash(N, d, block_size) / 1e6 for N in seq_lengths]
    ratios = [s / f for s, f in zip(mem_std, mem_flash)]

    for N, ms, mf, r in zip(seq_lengths, mem_std, mem_flash, ratios):
        print(f"  N={N:6d} | Standard={ms:10.2f} MB | Flash={mf:8.2f} MB | Ratio={r:.1f}x")

    # ── Experiment 3: Actual Peak Memory Measurement ──
    print("\n=== Experiment 3: Peak Memory (torch.cuda.max_memory_allocated) ===")
    measured_std = {}
    measured_flash = {}

    if device.type == 'cuda':
        test_sizes = [256, 512, 1024, 2048, 4096]
        for N in test_sizes:
            Q = torch.randn(2, N, d, device=device)
            K = torch.randn(2, N, d, device=device)
            V = torch.randn(2, N, d, device=device)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            _ = standard_attention(Q, K, V)
            torch.cuda.synchronize()
            measured_std[N] = torch.cuda.max_memory_allocated() / 1e6

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            _ = flash_attention_tiled(Q, K, V, block_size=64)
            torch.cuda.synchronize()
            measured_flash[N] = torch.cuda.max_memory_allocated() / 1e6

            print(f"  N={N:5d} | Standard={measured_std[N]:8.1f} MB | Flash={measured_flash[N]:8.1f} MB")
    else:
        # CPU: use theoretical estimates only
        test_sizes = [256, 512, 1024, 2048, 4096]
        for N in test_sizes:
            measured_std[N] = memory_standard(N, d) / 1e6
            measured_flash[N] = memory_flash(N, d, block_size) / 1e6
        print("  (CPU mode — showing theoretical estimates only)")

    # ── Experiment 4: Online Softmax Trace ──
    print("\n=== Experiment 4: Online Softmax Trace ===")
    torch.manual_seed(42)
    N_trace = 32
    d_trace = 16
    bs_trace = 8
    Q_t = torch.randn(1, N_trace, d_trace)
    K_t = torch.randn(1, N_trace, d_trace)
    V_t = torch.randn(1, N_trace, d_trace)

    trace = trace_online_softmax(Q_t, K_t, V_t, bs_trace, row_idx=0)
    for step in trace:
        print(f"  Step {step['step']}: block={step['block']}  m_i={step['m_i']:.4f}  "
              f"l_i={step['l_i']:.4f}  ||O||={step['o_norm']:.4f}")

    # ── Visualization ──

    # Plot 1: Memory scaling comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(seq_lengths, mem_std, 'o-', label='Standard O(N^2)', color='red', linewidth=2)
    axes[0].plot(seq_lengths, mem_flash, 's-', label='FlashAttention O(N)', color='blue', linewidth=2)
    axes[0].set_xlabel("Sequence Length N")
    axes[0].set_ylabel("Memory (MB)")
    axes[0].set_title("Memory Scaling: Standard vs FlashAttention")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].bar(range(len(seq_lengths)), ratios, color='green', alpha=0.7)
    axes[1].set_xticks(range(len(seq_lengths)))
    axes[1].set_xticklabels([str(N) for N in seq_lengths], rotation=45)
    axes[1].set_xlabel("Sequence Length N")
    axes[1].set_ylabel("Memory Reduction Ratio (std/flash)")
    axes[1].set_title("FlashAttention Memory Savings")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("FlashAttention: Memory-Efficient Exact Attention", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "memory_scaling.png", dpi=150)
    plt.close()

    # Plot 2: Numerical equivalence heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by causal/bidir
    for ax_idx, (causal, label) in enumerate([(False, "Bidirectional"), (True, "Causal")]):
        ax = axes[ax_idx]
        subset = [r for r in equiv_results if r['causal'] == causal]
        x_labels = [f"N={r['N']},bs={r['bs']}" for r in subset]
        max_diffs = [r['max_diff'] for r in subset]
        rel_diffs = [r['rel_diff'] for r in subset]

        x = np.arange(len(subset))
        width = 0.35
        ax.bar(x - width/2, max_diffs, width, label='Max abs diff', color='coral', alpha=0.8)
        ax.bar(x + width/2, rel_diffs, width, label='Relative diff', color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Difference")
        ax.set_yscale('log')
        ax.set_title(f"{label} Attention")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Numerical Equivalence: Standard vs Tiled Attention", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "numerical_equivalence.png", dpi=150)
    plt.close()

    # Plot 3: Online softmax step-by-step
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    steps = [s['step'] for s in trace]
    m_vals = [s['m_i'] for s in trace]
    l_vals = [s['l_i'] for s in trace]
    o_norms = [s['o_norm'] for s in trace]

    axes[0].plot(steps, m_vals, 'o-', color='red')
    axes[0].set_xlabel("Block step j")
    axes[0].set_ylabel("Running max m_i")
    axes[0].set_title("Online Softmax: Running Max")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, l_vals, 'o-', color='blue')
    axes[1].set_xlabel("Block step j")
    axes[1].set_ylabel("Running sum l_i")
    axes[1].set_title("Online Softmax: Running Sum of exp")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, o_norms, 'o-', color='green')
    axes[2].set_xlabel("Block step j")
    axes[2].set_ylabel("||O_i||")
    axes[2].set_title("Online Softmax: Output Norm Accumulation")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Online Softmax: Incremental Computation (row 0, N=32, bs=8)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "online_softmax_trace.png", dpi=150)
    plt.close()

    # Plot 4: Measured peak memory (or theoretical on CPU)
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = sorted(measured_std.keys())
    std_vals = [measured_std[N] for N in sizes]
    flash_vals = [measured_flash[N] for N in sizes]

    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, std_vals, width, label='Standard', color='red', alpha=0.7)
    ax.bar(x + width/2, flash_vals, width, label='Flash (tiled)', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in sizes])
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Peak Memory (MB)")
    label = "Measured" if device.type == 'cuda' else "Theoretical"
    ax.set_title(f"{label} Peak Memory Usage (d=64, B=2)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(results_dir / "peak_memory.png", dpi=150)
    plt.close()

    # Plot 5: Attention weight comparison for a small case
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    torch.manual_seed(42)
    N_viz = 16
    d_viz = 8
    Q_v = torch.randn(1, N_viz, d_viz)
    K_v = torch.randn(1, N_viz, d_viz)
    V_v = torch.randn(1, N_viz, d_viz)

    _, P_std = standard_attention(Q_v, K_v, V_v)
    O_flash, _ = flash_attention_tiled(Q_v, K_v, V_v, block_size=4)
    O_std, _ = standard_attention(Q_v, K_v, V_v)

    im0 = axes[0].imshow(P_std[0].numpy(), cmap='Blues', aspect='auto')
    axes[0].set_title("Standard Attention Weights")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    # Show difference between standard and flash output
    diff = (O_std - O_flash).abs()[0].numpy()
    im1 = axes[1].imshow(diff, cmap='Reds', aspect='auto')
    axes[1].set_title("|O_std - O_flash| (should be ~0)")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Position")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    plt.suptitle("Attention Weights & Output Equivalence", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "attention_comparison.png", dpi=150)
    plt.close()

    # Plot 6: Block size tradeoff
    fig, ax = plt.subplots(figsize=(8, 5))

    N = 4096
    block_sizes = [16, 32, 64, 128, 256, 512, 1024]
    mems = [memory_flash(N, d, bs) / 1e6 for bs in block_sizes]

    ax.plot(block_sizes, mems, 'o-', color='purple', linewidth=2)
    ax.axhline(y=memory_standard(N, d) / 1e6, color='red', linestyle='--',
               label=f'Standard attention ({memory_standard(N, d)/1e6:.1f} MB)')
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"FlashAttention Memory vs Block Size (N={N}, d={d})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(results_dir / "block_size_tradeoff.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
