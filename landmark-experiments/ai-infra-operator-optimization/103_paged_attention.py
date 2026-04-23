"""
Minimal PagedAttention / vLLM Reproduction
============================================
Reproduces core ideas from PagedAttention (2309.06180, Kwon et al.):
1. KV cache managed like virtual memory: logical blocks → physical blocks via block table
2. Non-contiguous physical memory eliminates fragmentation from variable-length sequences
3. Compare: contiguous allocation vs paged allocation for KV cache
4. Simulate: memory usage with different sequence lengths and batch sizes
5. Show: memory fragmentation reduction, max batch size improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── KV Cache Allocation Strategies ──

class ContiguousKVCache:
    """Naive KV cache: pre-allocate max_seq_len contiguous slots per sequence.

    Wastes memory when sequences are shorter than max length.
    Cannot share memory across sequences after prefill finishes.
    """

    def __init__(self, n_layers, n_heads, head_dim, max_batch, max_seq_len, dtype=torch.float16):
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocate full [max_batch, max_seq_len] for every slot
        self.k_cache = torch.zeros(n_layers, max_batch, max_seq_len, n_heads, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(n_layers, max_batch, max_seq_len, n_heads, head_dim, dtype=dtype)

        # Track actual lengths per sequence
        self.seq_lens = [0] * max_batch
        self.allocated = [False] * max_batch

    def allocate_slot(self):
        """Find a free slot. Returns slot index or -1 if full."""
        for i in range(self.max_batch):
            if not self.allocated[i]:
                self.allocated[i] = True
                self.seq_lens[i] = 0
                return i
        return -1

    def free_slot(self, slot):
        self.allocated[slot] = False
        self.seq_lens[slot] = 0

    def update(self, layer_idx, slot, pos, k, v):
        self.k_cache[layer_idx, slot, pos] = k
        self.v_cache[layer_idx, slot, pos] = v
        self.seq_lens[slot] = max(self.seq_lens[slot], pos + 1)

    def get_kv(self, layer_idx, slot, seq_len):
        return self.k_cache[layer_idx, slot, :seq_len], self.v_cache[layer_idx, slot, :seq_len]

    def total_memory_mb(self):
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        # Both K and V caches
        return (self.k_cache.numel() + self.v_cache.numel()) * elem_bytes / (1024 ** 2)

    def used_memory_mb(self):
        """Only count actually-used positions."""
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        total_tokens = sum(self.seq_lens)
        # 2 for K and V, n_layers for layers, n_heads * head_dim per token
        return total_tokens * self.n_layers * 2 * self.n_heads * self.head_dim * elem_bytes / (1024 ** 2)

    def wasted_memory_mb(self):
        return self.total_memory_mb() - self.used_memory_mb()


class PagedKVCache:
    """Paged KV cache: allocate in fixed-size blocks, map logical → physical via block table.

    Inspired by OS virtual memory paging:
    - Block size = fixed number of tokens (e.g. 16)
    - Block table[seq_id][logical_block] → physical_block
    - Physical blocks are non-contiguous → eliminates external fragmentation
    - freed blocks return to pool for reuse
    """

    def __init__(self, n_layers, n_heads, head_dim, n_physical_blocks, block_size=16, dtype=torch.float16):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype

        # Physical block pool: [n_layers, n_physical_blocks, block_size, n_heads, head_dim]
        self.k_pool = torch.zeros(n_layers, n_physical_blocks, block_size, n_heads, head_dim, dtype=dtype)
        self.v_pool = torch.zeros(n_layers, n_physical_blocks, block_size, n_heads, head_dim, dtype=dtype)

        self.n_physical_blocks = n_physical_blocks
        self.free_blocks = set(range(n_physical_blocks))

        # Per-sequence metadata
        self.block_tables = {}   # seq_id → [physical_block_ids]
        self.seq_lens = {}       # seq_id → current length

    def allocate_sequence(self, seq_id, prefill_len):
        """Allocate blocks for a new sequence of given prefill length."""
        n_blocks_needed = (prefill_len + self.block_size - 1) // self.block_size

        if n_blocks_needed > len(self.free_blocks):
            return False  # Out of memory

        phys_blocks = []
        for _ in range(n_blocks_needed):
            phys_block = self.free_blocks.pop()
            phys_blocks.append(phys_block)

        self.block_tables[seq_id] = phys_blocks
        self.seq_lens[seq_id] = prefill_len
        return True

    def append_token(self, seq_id):
        """Append one token: allocate new block if current block is full."""
        self.seq_lens[seq_id] += 1
        cur_len = self.seq_lens[seq_id]

        # Check if we need a new block
        n_blocks_needed = (cur_len + self.block_size - 1) // self.block_size
        n_blocks_have = len(self.block_tables[seq_id])

        if n_blocks_needed > n_blocks_have:
            if len(self.free_blocks) == 0:
                self.seq_lens[seq_id] -= 1
                return False  # OOM
            phys_block = self.free_blocks.pop()
            self.block_tables[seq_id].append(phys_block)

        return True

    def free_sequence(self, seq_id):
        """Release all blocks for a sequence back to the pool."""
        for phys_block in self.block_tables[seq_id]:
            self.free_blocks.add(phys_block)
        del self.block_tables[seq_id]
        del self.seq_lens[seq_id]

    def update(self, layer_idx, seq_id, pos, k, v):
        """Write K, V at position `pos` for sequence `seq_id`."""
        logical_block = pos // self.block_size
        offset = pos % self.block_size
        phys_block = self.block_tables[seq_id][logical_block]
        self.k_pool[layer_idx, phys_block, offset] = k
        self.v_pool[layer_idx, phys_block, offset] = v

    def get_kv(self, layer_idx, seq_id):
        """Gather KV from non-contiguous physical blocks."""
        seq_len = self.seq_lens[seq_id]
        blocks = self.block_tables[seq_id]
        n_full_blocks = seq_len // self.block_size
        remainder = seq_len % self.block_size

        k_parts = []
        v_parts = []
        for i, phys_block in enumerate(blocks):
            end = self.block_size if i < n_full_blocks else remainder
            if end == 0:
                break
            k_parts.append(self.k_pool[layer_idx, phys_block, :end])
            v_parts.append(self.v_pool[layer_idx, phys_block, :end])

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def total_memory_mb(self):
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        # Both K and V pools
        return (self.k_pool.numel() + self.v_pool.numel()) * elem_bytes / (1024 ** 2)

    def used_memory_mb(self):
        """Memory actually used by allocated blocks (not free blocks)."""
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        allocated = self.n_physical_blocks - len(self.free_blocks)
        # Per allocated block: n_layers * 2 (K+V) * block_size * n_heads * head_dim
        return allocated * self.n_layers * 2 * self.block_size * self.n_heads * self.head_dim * elem_bytes / (1024 ** 2)

    def wasted_in_blocks_mb(self):
        """Internal fragmentation: unused slots within allocated blocks."""
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        waste_tokens = 0
        for seq_id in self.seq_lens:
            remainder = self.seq_lens[seq_id] % self.block_size
            if remainder > 0:
                waste_tokens += (self.block_size - remainder)
        return waste_tokens * self.n_layers * 2 * self.n_heads * self.head_dim * elem_bytes / (1024 ** 2)


# ── Simulation: Memory Comparison ──

def simulate_contiguous_cache(seq_lengths, n_layers, n_heads, head_dim, dtype=torch.float16):
    """Simulate contiguous allocation. Return (total_mb, used_mb, wasted_mb)."""
    max_seq = max(seq_lengths)
    max_batch = len(seq_lengths)
    cache = ContiguousKVCache(n_layers, n_heads, head_dim, max_batch, max_seq, dtype)

    for i, seq_len in enumerate(seq_lengths):
        slot = cache.allocate_slot()
        for pos in range(seq_len):
            k = torch.randn(n_heads, head_dim, dtype=dtype)
            v = torch.randn(n_heads, head_dim, dtype=dtype)
            for layer in range(n_layers):
                cache.update(layer, slot, pos, k, v)

    return cache.total_memory_mb(), cache.used_memory_mb(), cache.wasted_memory_mb()


def simulate_paged_cache(seq_lengths, n_layers, n_heads, head_dim, block_size=16, dtype=torch.float16):
    """Simulate paged allocation. Return (total_mb, used_mb, wasted_mb, internal_frag_mb)."""
    max_total_tokens = sum(seq_lengths)
    # Generously over-provision physical blocks
    n_physical_blocks = (max_total_tokens * 2 + block_size - 1) // block_size
    cache = PagedKVCache(n_layers, n_heads, head_dim, n_physical_blocks, block_size, dtype)

    for i, seq_len in enumerate(seq_lengths):
        ok = cache.allocate_sequence(i, seq_len)
        if not ok:
            return None, None, None, None
        for pos in range(seq_len):
            k = torch.randn(n_heads, head_dim, dtype=dtype)
            v = torch.randn(n_heads, head_dim, dtype=dtype)
            for layer in range(n_layers):
                cache.update(layer, i, pos, k, v)

    return cache.total_memory_mb(), cache.used_memory_mb(), cache.wasted_in_blocks_mb(), cache.wasted_in_blocks_mb()


# ── Simulation: Max Batch Size ──

def find_max_batch_contiguous(gpu_memory_mb, max_seq_len, n_layers, n_heads, head_dim, dtype=torch.float16):
    """Find max batch size under contiguous allocation."""
    elem_bytes = 2 if dtype == torch.float16 else 4
    # Per sequence: n_layers * 2 * max_seq_len * n_heads * head_dim * elem_bytes
    per_seq_mb = n_layers * 2 * max_seq_len * n_heads * head_dim * elem_bytes / (1024 ** 2)
    return int(gpu_memory_mb / per_seq_mb)


def find_max_batch_paged(gpu_memory_mb, max_seq_len, avg_seq_len, n_layers, n_heads, head_dim,
                         block_size=16, dtype=torch.float16):
    """Find max batch size under paged allocation.
    Uses avg_seq_len for actual memory, not max_seq_len.
    """
    elem_bytes = 2 if dtype == torch.float16 else 4
    # Per sequence: n_layers * 2 * avg_seq_len * n_heads * head_dim * elem_bytes (approx)
    # With block overhead: ceil(avg_seq_len / block_size) * block_size
    effective_len = ((avg_seq_len + block_size - 1) // block_size) * block_size
    per_seq_mb = n_layers * 2 * effective_len * n_heads * head_dim * elem_bytes / (1024 ** 2)
    return int(gpu_memory_mb / per_seq_mb)


# ── Simulation: Fragmentation Over Time ──

def simulate_fragmentation_over_time(n_layers=2, n_heads=8, head_dim=64, block_size=16,
                                     n_steps=200, dtype=torch.float16):
    """Simulate sequences arriving and completing; track fragmentation."""
    max_seq_len = 512
    max_batch = 64
    n_physical_blocks = 4096

    contiguous = ContiguousKVCache(n_layers, n_heads, head_dim, max_batch, max_seq_len, dtype)
    paged = PagedKVCache(n_layers, n_heads, head_dim, n_physical_blocks, block_size, dtype)

    contiguous_frag = []
    paged_frag = []
    contiguous_used = []
    paged_used = []

    active_seqs_cont = []
    active_seqs_paged = []
    next_seq_id = 0

    rng = np.random.RandomState(42)

    for step in range(n_steps):
        # Randomly: either add a new sequence or remove one
        if rng.random() < 0.6 or len(active_seqs_cont) == 0:
            # Add sequence with random length
            seq_len = rng.randint(32, max_seq_len + 1)

            # Contiguous
            slot = contiguous.allocate_slot()
            if slot >= 0:
                for pos in range(seq_len):
                    for layer in range(n_layers):
                        contiguous.update(layer, slot, pos,
                                          torch.randn(n_heads, head_dim, dtype=dtype),
                                          torch.randn(n_heads, head_dim, dtype=dtype))
                active_seqs_cont.append((slot, seq_len))

            # Paged
            ok = paged.allocate_sequence(next_seq_id, seq_len)
            if ok:
                for pos in range(seq_len):
                    for layer in range(n_layers):
                        paged.update(layer, next_seq_id, pos,
                                     torch.randn(n_heads, head_dim, dtype=dtype),
                                     torch.randn(n_heads, head_dim, dtype=dtype))
                active_seqs_paged.append((next_seq_id, seq_len))

            next_seq_id += 1
        else:
            # Remove a random active sequence
            if active_seqs_cont:
                idx = rng.randint(0, len(active_seqs_cont))
                slot, _ = active_seqs_cont.pop(idx)
                contiguous.free_slot(slot)

            if active_seqs_paged:
                idx = rng.randint(0, len(active_seqs_paged))
                sid, _ = active_seqs_paged.pop(idx)
                paged.free_sequence(sid)

        contiguous_frag.append(contiguous.wasted_memory_mb())
        paged_frag.append(paged.wasted_in_blocks_mb())
        contiguous_used.append(contiguous.used_memory_mb())
        paged_used.append(paged.used_memory_mb())

    return contiguous_frag, paged_frag, contiguous_used, paged_used


# ── Minimal Attention Computation with Paged KV ──

def paged_attention_forward(q, paged_cache, layer_idx, seq_id, n_heads, head_dim):
    """Demonstrate attention computation using paged KV cache.
    q: (1, 1, n_heads, head_dim) — single query token
    """
    k_all, v_all = paged_cache.get_kv(layer_idx, seq_id)  # (seq_len, n_heads, head_dim)

    # Scaled dot-product attention
    seq_len = k_all.shape[0]
    q = q.squeeze(0).squeeze(0)  # (n_heads, head_dim)
    k_all = k_all  # (seq_len, n_heads, head_dim)
    v_all = v_all  # (seq_len, n_heads, head_dim)

    # Per-head attention
    scores = torch.einsum('hd,sd->hs', q, k_all.reshape(seq_len, -1))
    # More correctly per head:
    attn_out = torch.zeros(n_heads, head_dim, dtype=q.dtype)
    for h in range(n_heads):
        s = torch.matmul(q[h], k_all[:, h].T) / (head_dim ** 0.5)  # (seq_len,)
        s = F.softmax(s, dim=0)
        attn_out[h] = torch.matmul(s, v_all[:, h])

    return attn_out


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "103-paged-attention"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_layers = 2
    n_heads = 8
    head_dim = 64
    block_size = 16
    dtype = torch.float16

    # ── Experiment 1: Memory usage vs sequence length distribution ──
    print("=== Experiment 1: Memory Usage vs Sequence Lengths ===")

    seq_len_configs = {
        'Uniform (64-512)': lambda n: [np.random.randint(64, 513) for _ in range(n)],
        'Short-heavy (32-128)': lambda n: [np.random.randint(32, 129) for _ in range(n)],
        'Long-heavy (256-1024)': lambda n: [np.random.randint(256, 1025) for _ in range(n)],
        'Bimodal (64 or 1024)': lambda n: [np.random.choice([64, 1024]) for _ in range(n)],
    }

    batch_sizes = [4, 8, 16, 32, 48, 64]

    mem_results = {}
    for config_name, gen_fn in seq_len_configs.items():
        cont_waste = []
        paged_waste = []
        cont_total = []
        paged_total = []

        for bs in batch_sizes:
            rng = np.random.RandomState(42)
            seqs = gen_fn(bs)

            ct, cu, cw = simulate_contiguous_cache(seqs, n_layers, n_heads, head_dim, dtype)
            pt, pu, pw, _ = simulate_paged_cache(seqs, n_layers, n_heads, head_dim, block_size, dtype)

            cont_waste.append(cw)
            paged_waste.append(pw)
            cont_total.append(ct)
            paged_total.append(pt)

        mem_results[config_name] = {
            'cont_waste': cont_waste, 'paged_waste': paged_waste,
            'cont_total': cont_total, 'paged_total': paged_total,
        }

        # Report for largest batch
        print(f"  {config_name}: Batch={batch_sizes[-1]} | "
              f"Contiguous waste={cont_waste[-1]:.1f}MB, Paged waste={paged_waste[-1]:.1f}MB")

    # ── Experiment 2: Max batch size comparison ──
    print("\n=== Experiment 2: Max Batch Size Under Memory Budget ===")

    gpu_memory_mb = 2048  # Simulated 2GB for KV cache
    max_seq_len = 1024

    avg_ratios = [0.25, 0.5, 0.75, 1.0]  # avg_seq / max_seq
    max_batch_cont = find_max_batch_contiguous(gpu_memory_mb, max_seq_len, n_layers, n_heads, head_dim, dtype)
    print(f"  Contiguous max batch (max_seq={max_seq_len}): {max_batch_cont}")

    max_batch_paged_list = []
    for ratio in avg_ratios:
        avg_len = int(max_seq_len * ratio)
        mb = find_max_batch_paged(gpu_memory_mb, max_seq_len, avg_len, n_layers, n_heads, head_dim, block_size, dtype)
        max_batch_paged_list.append(mb)
        print(f"  Paged max batch (avg/max={ratio:.0%}, avg={avg_len}): {mb}  ({mb/max_batch_cont:.1f}x)")

    # ── Experiment 3: Fragmentation over time ──
    print("\n=== Experiment 3: Fragmentation Over Time ===")
    cont_frag, paged_frag, cont_used, paged_used = simulate_fragmentation_over_time(
        n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, block_size=block_size, n_steps=200
    )
    print(f"  Avg contiguous fragmentation: {np.mean(cont_frag):.2f} MB")
    print(f"  Avg paged fragmentation: {np.mean(paged_frag):.2f} MB")
    print(f"  Fragmentation reduction: {1 - np.mean(paged_frag) / max(np.mean(cont_frag), 1e-6):.1%}")

    # ── Experiment 4: Block size trade-off ──
    print("\n=== Experiment 4: Block Size Trade-off ===")
    block_sizes = [4, 8, 16, 32, 64, 128]
    rng = np.random.RandomState(42)
    test_seqs = [rng.randint(32, 513) for _ in range(32)]

    block_frag = []
    block_internal = []
    for bs in block_sizes:
        _, _, _, internal = simulate_paged_cache(test_seqs, n_layers, n_heads, head_dim, bs, dtype)
        block_frag.append(internal)
        # Internal fragmentation ratio
        total_allocated_tokens = sum(((s + bs - 1) // bs) * bs for s in test_seqs)
        actual_tokens = sum(test_seqs)
        block_internal.append((total_allocated_tokens - actual_tokens) / total_allocated_tokens * 100)
        print(f"  Block size={bs:>3d}: Internal frag={block_internal[-1]:.1f}%, "
              f"Wasted memory={internal:.2f}MB")

    # ── Experiment 5: Attention correctness with paged cache ──
    print("\n=== Experiment 5: Attention Correctness (Paged vs Contiguous) ===")
    cache_contiguous = ContiguousKVCache(1, n_heads, head_dim, 1, 128, dtype)
    cache_paged = PagedKVCache(1, n_heads, head_dim, 512, block_size, dtype)

    seq_len = 50
    slot = cache_contiguous.allocate_slot()
    cache_paged.allocate_sequence(0, seq_len)

    for pos in range(seq_len):
        k = torch.randn(n_heads, head_dim, dtype=dtype)
        v = torch.randn(n_heads, head_dim, dtype=dtype)
        cache_contiguous.update(0, slot, pos, k, v)
        cache_paged.update(0, 0, pos, k, v)

    k_cont, v_cont = cache_contiguous.get_kv(0, slot, seq_len)
    k_paged, v_paged = cache_paged.get_kv(0, 0)

    kv_diff = (k_cont - k_paged).abs().max().item()
    vv_diff = (v_cont - v_paged).abs().max().item()
    print(f"  Max K difference: {kv_diff:.6f}")
    print(f"  Max V difference: {vv_diff:.6f}")
    print(f"  Paged attention produces identical KV cache: {kv_diff == 0 and vv_diff == 0}")

    # ── Visualization ──

    # 1. Memory waste comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for config_name, data in mem_results.items():
        short = config_name.split('(')[0].strip()
        axes[0].plot(batch_sizes, data['cont_waste'], '--', label=f'{short} (contiguous)', alpha=0.6)
        axes[0].plot(batch_sizes, data['paged_waste'], '-', label=f'{short} (paged)', alpha=0.8)

    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Wasted Memory (MB)")
    axes[0].set_title("Memory Waste: Contiguous vs Paged")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Waste ratio
    for config_name, data in mem_results.items():
        short = config_name.split('(')[0].strip()
        ratio = [p / max(c, 1e-6) for p, c in zip(data['paged_waste'], data['cont_waste'])]
        axes[1].plot(batch_sizes, ratio, '-o', label=short, markersize=4)

    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Paged Waste / Contiguous Waste")
    axes[1].set_title("Fragmentation Ratio (lower = better)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.5)

    plt.suptitle("PagedAttention: Memory Fragmentation Reduction", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "memory_comparison.png", dpi=150)
    plt.close()

    # 2. Max batch size improvement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ['Contiguous\n(max_seq)'] + [f'Paged\n(avg/max={r:.0%})' for r in avg_ratios]
    values = [max_batch_cont] + max_batch_paged_list
    colors = ['gray', 'blue', 'green', 'orange', 'red']

    axes[0].bar(labels, values, color=colors[:len(values)], alpha=0.7)
    axes[0].set_ylabel("Max Batch Size")
    axes[0].set_title(f"Max Batch Under {gpu_memory_mb}MB KV Cache Budget")
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.5, f'{v}x', ha='center', fontsize=9)

    # Speedup
    speedups = [v / max_batch_cont for v in values]
    axes[1].bar(labels, speedups, color=colors[:len(values)], alpha=0.7)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel("Batch Size Improvement (x)")
    axes[1].set_title("Throughput Improvement vs Contiguous")
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(speedups):
        axes[1].text(i, v + 0.05, f'{v:.1f}x', ha='center', fontsize=9)

    plt.suptitle("PagedAttention: Max Batch Size Improvement", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "max_batch_improvement.png", dpi=150)
    plt.close()

    # 3. Fragmentation over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(len(cont_frag))
    window = 10

    cont_smooth = np.convolve(cont_frag, np.ones(window)/window, mode='valid')
    paged_smooth = np.convolve(paged_frag, np.ones(window)/window, mode='valid')

    axes[0].plot(cont_smooth, label='Contiguous (external frag)', color='red', alpha=0.8)
    axes[0].plot(paged_smooth, label='Paged (internal frag)', color='blue', alpha=0.8)
    axes[0].set_xlabel("Simulation Step")
    axes[0].set_ylabel("Fragmentation (MB)")
    axes[0].set_title("Fragmentation Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cont_used_smooth = np.convolve(cont_used, np.ones(window)/window, mode='valid')
    paged_used_smooth = np.convolve(paged_used, np.ones(window)/window, mode='valid')
    axes[1].plot(cont_used_smooth, label='Contiguous', color='red', alpha=0.8)
    axes[1].plot(paged_used_smooth, label='Paged', color='blue', alpha=0.8)
    axes[1].set_xlabel("Simulation Step")
    axes[1].set_ylabel("Used Memory (MB)")
    axes[1].set_title("Effective Memory Usage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("PagedAttention: Dynamic Memory Management", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "fragmentation_over_time.png", dpi=150)
    plt.close()

    # 4. Block size trade-off
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar([str(bs) for bs in block_sizes], block_internal, color='steelblue', alpha=0.7)
    axes[0].set_xlabel("Block Size (tokens)")
    axes[0].set_ylabel("Internal Fragmentation (%)")
    axes[0].set_title("Internal Fragmentation vs Block Size")
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(block_internal):
        axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8)

    # Wasted memory
    axes[1].bar([str(bs) for bs in block_sizes], block_frag, color='coral', alpha=0.7)
    axes[1].set_xlabel("Block Size (tokens)")
    axes[1].set_ylabel("Wasted Memory (MB)")
    axes[1].set_title("Wasted Memory vs Block Size")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("PagedAttention: Block Size Trade-off", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "block_size_tradeoff.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Draw contiguous allocation (top)
    ax.text(0.02, 0.92, "Contiguous Allocation", fontsize=13, fontweight='bold', color='red',
            transform=ax.transAxes)
    for i in range(4):
        # Full block (reserved max_len)
        x_start = 0.05 + i * 0.12
        rect = plt.Rectangle((x_start, 0.72), 0.10, 0.15, linewidth=2,
                              edgecolor='red', facecolor='lightyellow', alpha=0.8,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        # Used portion
        used_frac = [0.3, 0.6, 0.4, 0.8][i]
        rect_used = plt.Rectangle((x_start, 0.72), 0.10 * used_frac, 0.15, linewidth=0,
                                  facecolor='red', alpha=0.3, transform=ax.transAxes)
        ax.add_patch(rect_used)
        ax.text(x_start + 0.05, 0.795, f'Seq {i}', ha='center', fontsize=9, transform=ax.transAxes)
        ax.text(x_start + 0.05, 0.74, f'{used_frac:.0%} used', ha='center', fontsize=8,
                color='red', transform=ax.transAxes)

    ax.text(0.55, 0.86, "Wasted space (gray) = external fragmentation",
            fontsize=9, color='gray', transform=ax.transAxes, style='italic')

    # Draw paged allocation (bottom)
    ax.text(0.02, 0.55, "Paged Allocation", fontsize=13, fontweight='bold', color='blue',
            transform=ax.transAxes)

    # Physical blocks pool
    colors_blocks = ['lightblue', 'lightgreen', 'lightsalmon', 'plum']
    seq_colors = ['blue', 'green', 'orange', 'purple']

    # Block table
    ax.text(0.02, 0.42, "Block Table:", fontsize=10, fontweight='bold', transform=ax.transAxes)
    for i in range(4):
        ax.text(0.15 + i * 0.10, 0.42,
                f'Seq{i}: [{i*2},{i*2+1}]', fontsize=8, color=seq_colors[i],
                fontfamily='monospace', transform=ax.transAxes)

    # Physical memory (non-contiguous)
    ax.text(0.02, 0.30, "Physical Blocks:", fontsize=10, fontweight='bold', transform=ax.transAxes)
    for i in range(8):
        x_start = 0.05 + i * 0.11
        seq_idx = i // 2
        rect = plt.Rectangle((x_start, 0.15), 0.09, 0.12, linewidth=1.5,
                              edgecolor='black', facecolor=colors_blocks[seq_idx], alpha=0.7,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x_start + 0.045, 0.21, f'P{i}', ha='center', fontsize=8, transform=ax.transAxes)
        ax.text(x_start + 0.045, 0.17, f'Seq{seq_idx}', ha='center', fontsize=7,
                color=seq_colors[seq_idx], transform=ax.transAxes)

    # Key insight
    ax.text(0.5, 0.03, "Key Insight: Physical blocks are non-contiguous → no external fragmentation\n"
            "Logical-to-physical mapping (block table) enables flexible memory sharing",
            fontsize=10, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            fontfamily='monospace')

    ax.set_title("PagedAttention: Virtual Memory for KV Cache (vLLM, 2309.06180)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "paged_attention_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
