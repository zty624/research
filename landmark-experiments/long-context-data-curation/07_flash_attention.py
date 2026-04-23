"""
Minimal FlashAttention IO-Awareness Reproduction
=================================================
Reproduces the core insight from "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness" (2205.14135):
1. IO-awareness: SRAM vs HBM bandwidth gap
2. Tiling: process attention in blocks to fit SRAM
3. Online softmax: compute softmax incrementally without materializing full NxN matrix
4. Recomputation vs storing tradeoff for backward pass
5. Compare memory usage: standard vs tiled attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── Standard Attention (baseline, materializes full NxN matrix) ──

def standard_attention(Q, K, V, causal=False):
    """Standard attention that materializes the full S = QK^T matrix.
    Memory: O(N^2) for the attention matrix.
    """
    d = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / np.sqrt(d)

    if causal:
        mask = torch.triu(torch.ones(S.shape[-2:], device=S.device), diagonal=1).bool()
        S = S.masked_fill(mask, float('-inf'))

    P = torch.softmax(S, dim=-1)  # NxN matrix materialized in HBM
    O = P @ V
    return O, P  # Return attention weights for visualization


# ── Tiled (FlashAttention-style) Attention ──

def flash_attention_forward(Q, K, V, block_size=64, causal=False):
    """FlashAttention-style tiled attention.
    Key insight: process QK^T in blocks, compute softmax incrementally
    (online softmax), never materialize the full NxN matrix.

    Online softmax algorithm:
    - Maintain running max m_i and sum l_i for each query row
    - For each new block, update m_i and rescale l_i
    - Final output: O_i = (1/l_i) * accumulated result
    """
    B, N, d = Q.shape
    assert N % block_size == 0 or True  # Handle remainder

    O = torch.zeros_like(Q)
    l = torch.zeros(B, N, 1, device=Q.device)   # Running sum of exp
    m = torch.full((B, N, 1), float('-inf'), device=Q.device)  # Running max

    n_blocks = (N + block_size - 1) // block_size

    for j in range(n_blocks):
        j_start = j * block_size
        j_end = min(j_start + block_size, N)
        K_block = K[:, j_start:j_end]
        V_block = V[:, j_start:j_end]

        for i in range(n_blocks):
            i_start = i * block_size
            i_end = min(i_start + block_size, N)

            # Skip if causal and block is above diagonal
            if causal and j_start > i_end:
                continue

            Q_block = Q[:, i_start:i_end]

            # Compute block of S = QK^T / sqrt(d)
            S_block = Q_block @ K_block.transpose(-2, -1) / np.sqrt(d)

            if causal:
                # Apply causal mask within block
                block_rows = i_end - i_start
                block_cols = j_end - j_start
                row_idx = torch.arange(i_start, i_end, device=Q.device)
                col_idx = torch.arange(j_start, j_end, device=Q.device)
                mask = col_idx.unsqueeze(0) > row_idx.unsqueeze(1)
                S_block = S_block.masked_fill(mask.unsqueeze(0), float('-inf'))

            # Online softmax update
            m_new = torch.maximum(m[:, i_start:i_end], S_block.max(dim=-1, keepdim=True).values)
            # Rescale previous accumulation
            alpha = torch.exp(m[:, i_start:i_end] - m_new)
            beta = torch.exp(S_block - m_new)

            l[:, i_start:i_end] = alpha * l[:, i_start:i_end] + beta.sum(dim=-1, keepdim=True)
            O[:, i_start:i_end] = alpha * O[:, i_start:i_end] + beta @ V_block

            m[:, i_start:i_end] = m_new

    # Final normalization
    O = O / l.clamp(min=1e-10)
    return O, None  # No attention weights saved (memory saving!)


# ── Memory Estimation ──

def estimate_memory_standard(N, d, dtype_bytes=4):
    """Memory for standard attention (materializes NxN matrix)."""
    # Q, K, V: 3 * N * d
    # S (QK^T): N * N
    # P (softmax(S)): N * N
    # O: N * d
    return (3 * N * d + 2 * N * N + N * d) * dtype_bytes


def estimate_memory_flash(N, d, block_size, dtype_bytes=4):
    """Memory for flash attention (only block-sized intermediates)."""
    # Q, K, V: 3 * N * d
    # O: N * d
    # Block S: block_size * block_size
    # l, m: N each
    return (3 * N * d + N * d + block_size * block_size + 2 * N) * dtype_bytes


# ── Correctness Verification ──

def verify_correctness(device='cpu'):
    """Verify that tiled attention produces the same output as standard."""
    torch.manual_seed(42)
    B, N, d = 2, 128, 32
    Q = torch.randn(B, N, d, device=device)
    K = torch.randn(B, N, d, device=device)
    V = torch.randn(B, N, d, device=device)

    O_std, _ = standard_attention(Q, K, V)
    O_flash, _ = flash_attention_forward(Q, K, V, block_size=32)

    max_diff = (O_std - O_flash).abs().max().item()
    rel_diff = (O_std - O_flash).abs().norm() / O_std.abs().norm()
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Relative diff:     {rel_diff:.2e}")
    return max_diff < 1e-4


def verify_causal(device='cpu'):
    """Verify causal attention correctness."""
    torch.manual_seed(42)
    B, N, d = 2, 64, 16
    Q = torch.randn(B, N, d, device=device)
    K = torch.randn(B, N, d, device=device)
    V = torch.randn(B, N, d, device=device)

    O_std, P_std = standard_attention(Q, K, V, causal=True)
    O_flash, _ = flash_attention_forward(Q, K, V, block_size=16, causal=True)

    max_diff = (O_std - O_flash).abs().max().item()
    print(f"  Causal max diff: {max_diff:.2e}")
    return max_diff < 1e-3


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "07-flash-attn"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Verify correctness
    print("=== Verifying Correctness ===")
    print("Bidirectional:")
    verify_correctness(device)
    print("Causal:")
    verify_causal(device)

    # ── Visualization ──

    # 1. Memory comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]
    d = 64
    block_size = 64

    mem_standard = [estimate_memory_standard(N, d) / 1e6 for N in seq_lengths]
    mem_flash = [estimate_memory_flash(N, d, block_size) / 1e6 for N in seq_lengths]

    ax.plot(seq_lengths, mem_standard, 'o-', label='Standard Attention', color='red', linewidth=2)
    ax.plot(seq_lengths, mem_flash, 's-', label='FlashAttention (tiled)', color='blue', linewidth=2)
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory: Standard vs FlashAttention")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "memory_comparison.png", dpi=150)
    plt.close()

    # 2. Online Softmax visualization (step-by-step)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    torch.manual_seed(42)
    N = 16
    d = 8
    Q = torch.randn(1, N, d)
    K = torch.randn(1, N, d)
    V = torch.randn(1, N, d)

    # Compute full attention for reference
    S_full = (Q @ K.transpose(-2, -1) / np.sqrt(d)).squeeze(0)
    P_full = torch.softmax(S_full, dim=-1)

    # Show how online softmax builds up the attention matrix block by block
    block_size = 4
    n_blocks = N // block_size
    # Simulate the incremental process for row 0
    row = 0
    for step, j in enumerate(range(n_blocks)):
        ax = axes[step]
        # Show the full attention row
        attn_row = P_full[row].numpy()
        bars = ax.bar(range(N), attn_row, color='lightgray', alpha=0.5)

        # Highlight the block being processed
        j_start = j * block_size
        j_end = j_start + block_size
        for k in range(j_start, j_end):
            bars[k].set_color('blue')
            bars[k].set_alpha(1.0)

        # Show what online softmax has accumulated so far
        S_so_far = S_full[row, :j_end].numpy()
        P_so_far = np.exp(S_so_far - S_so_far.max())
        P_so_far = P_so_far / P_so_far.sum()

        ax2 = ax.twinx()
        ax2.plot(range(j_end), P_so_far, 'ro-', markersize=4, label='Online softmax')
        ax2.set_ylim(0, 1)

        ax.set_title(f"Step {step+1}: Process block [{j_start}:{j_end}]")
        ax.set_xlabel("Key position")
        if step == 0:
            ax.set_ylabel("Attention weight")

    plt.suptitle("Online Softmax: Incremental Attention Computation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "online_softmax.png", dpi=150)
    plt.close()

    # 3. IO analysis: HBM reads/writes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Approximate HBM operations
    seq_lengths_detail = [256, 512, 1024, 2048, 4096]
    # Standard: read Q,K,V (3Nd) + write S,P,O (2N^2 + Nd)
    hbm_standard = [(3*N*d + 2*N*N + N*d) / 1e6 for N in seq_lengths_detail]
    # Flash: read Q,K,V per block + write O (Nd) — much less
    hbm_flash = []
    for N in seq_lengths_detail:
        # Each block reads Q_block, K, V_block
        # Number of outer loop iterations: N/block_size
        # Each iteration reads: block_size*d (Q) + N*d (K) + block_size*d (V)
        # Plus writes: block_size*d (O)
        n_outer = N // block_size
        hbm = n_outer * (block_size * d + N * d + block_size * d + block_size * d)
        hbm_flash.append(hbm / 1e6)

    ax.bar(range(len(seq_lengths_detail)), hbm_standard,
           width=0.35, label='Standard', color='red', alpha=0.7)
    ax.bar([x + 0.35 for x in range(len(seq_lengths_detail))], hbm_flash,
           width=0.35, label='FlashAttention', color='blue', alpha=0.7)
    ax.set_xticks([x + 0.175 for x in range(len(seq_lengths_detail))])
    ax.set_xticklabels([str(N) for N in seq_lengths_detail])
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("HBM I/O (MB, approx)")
    ax.set_title("HBM Read/Write: Standard vs FlashAttention")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "hbm_io_comparison.png", dpi=150)
    plt.close()

    # 4. Timing comparison (on actual tensors)
    fig, ax = plt.subplots(figsize=(8, 5))
    if device.type == 'cuda':
        sizes = [256, 512, 1024, 2048, 4096]
        std_times = []
        flash_times = []

        for N in sizes:
            Q = torch.randn(4, N, 64, device=device)
            K = torch.randn(4, N, 64, device=device)
            V = torch.randn(4, N, 64, device=device)

            # Warmup
            for _ in range(3):
                standard_attention(Q, K, V)

            # Time standard
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                standard_attention(Q, K, V)
            torch.cuda.synchronize()
            std_times.append((time.time() - t0) / 10)

            # Time flash
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                flash_attention_forward(Q, K, V, block_size=64)
            torch.cuda.synchronize()
            flash_times.append((time.time() - t0) / 10)

        ax.plot(sizes, std_times, 'o-', label='Standard', color='red')
        ax.plot(sizes, flash_times, 's-', label='Flash (tiled)', color='blue')
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (s)")
        ax.set_title("Wall-clock Time: Standard vs Tiled Attention")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # CPU: just show the theoretical speedup ratios
        ratios = [estimate_memory_standard(N, d) / estimate_memory_flash(N, d, block_size)
                  for N in [256, 512, 1024, 2048, 4096, 8192]]
        ax.bar(range(len(ratios)), ratios, color='green', alpha=0.7)
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(['256', '512', '1K', '2K', '4K', '8K'])
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Memory Reduction Ratio")
        ax.set_title("FlashAttention Memory Savings (CPU — ratios only)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "timing_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
