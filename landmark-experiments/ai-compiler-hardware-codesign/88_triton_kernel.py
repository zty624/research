"""
Triton-style Custom Kernel Analysis
=====================================
Reproduces core ideas from OpenAI Triton:
1. Fused kernel operations (e.g., fused softmax + dropout)
2. Compare: separate ops vs fused kernel memory access patterns
3. Analytical memory access model (not actual GPU execution)
4. Show: memory access count comparison, operation graph visualization
5. Demonstrates kernel fusion benefits: reduced global memory round-trips
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


# ── Memory Access Model ──

@dataclass
class MemoryAccess:
    """Track a single memory access pattern."""
    name: str
    bytes_read: int
    bytes_write: int
    description: str


class MemoryModel:
    """Analytical model for GPU memory access counting.
    Assumes float32 (4 bytes per element) and models:
    - Global memory (HBM) reads/writes (expensive)
    - SRAM (shared memory) reads/writes (cheap, not counted)
    - Register file accesses (free, not counted)
    """
    BYTES_PER_ELEM = 4  # float32

    @staticmethod
    def bytes_for(shape: Tuple[int, ...]) -> int:
        n = 1
        for d in shape:
            n *= d
        return n * MemoryModel.BYTES_PER_ELEM


# ── Separate Operations: Softmax + Dropout ──

class SeparateSoftmaxDropout:
    """Model memory accesses for separate softmax then dropout."""
    def __init__(self, batch=128, seq_len=512, head_dim=64, n_heads=8):
        self.B = batch
        self.S = seq_len
        self.D = head_dim
        self.H = n_heads
        self.input_shape = (batch, n_heads, seq_len, head_dim)

    def count_accesses(self) -> List[MemoryAccess]:
        accesses = []
        n_elems = self.B * self.H * self.S * self.D
        n_bytes = MemoryModel.bytes_for(self.input_shape)

        # Op 1: Softmax
        # Read: input tensor from HBM
        accesses.append(MemoryAccess(
            "softmax_read", n_bytes, 0,
            f"Read input [{self.B}x{self.H}x{self.S}x{self.D}]"
        ))
        # Write: softmax output to HBM (intermediate tensor)
        accesses.append(MemoryAccess(
            "softmax_write", 0, n_bytes,
            f"Write softmax output [{self.B}x{self.H}x{self.S}x{self.D}]"
        ))

        # Op 2: Dropout
        # Read: softmax output from HBM
        accesses.append(MemoryAccess(
            "dropout_read", n_bytes, 0,
            f"Read softmax output [{self.B}x{self.H}x{self.S}x{self.D}]"
        ))
        # Write: dropout output to HBM
        accesses.append(MemoryAccess(
            "dropout_write", 0, n_bytes,
            f"Write dropout output [{self.B}x{self.H}x{self.S}x{self.D}]"
        ))

        return accesses

    def total_hbm_bytes(self) -> Tuple[int, int]:
        accesses = self.count_accesses()
        total_read = sum(a.bytes_read for a in accesses)
        total_write = sum(a.bytes_write for a in accesses)
        return total_read, total_write


# ── Fused Kernel: Softmax + Dropout ──

class FusedSoftmaxDropout:
    """Model memory accesses for fused softmax+dropout kernel.
    Key insight: the intermediate softmax output stays in SRAM/registers,
    never written back to HBM.
    """
    def __init__(self, batch=128, seq_len=512, head_dim=64, n_heads=8):
        self.B = batch
        self.S = seq_len
        self.D = head_dim
        self.H = n_heads
        self.input_shape = (batch, n_heads, seq_len, head_dim)

    def count_accesses(self) -> List[MemoryAccess]:
        accesses = []
        n_bytes = MemoryModel.bytes_for(self.input_shape)

        # Fused kernel: read input from HBM, compute softmax in SRAM,
        # apply dropout in SRAM, write final output to HBM
        accesses.append(MemoryAccess(
            "fused_read", n_bytes, 0,
            f"Read input [{self.B}x{self.H}x{self.S}x{self.D}]"
        ))
        # Softmax intermediate stays in SRAM — no HBM write!
        accesses.append(MemoryAccess(
            "softmax_sram", 0, 0,
            "Softmax intermediate: SRAM only (no HBM access)"
        ))
        accesses.append(MemoryAccess(
            "fused_write", 0, n_bytes,
            f"Write fused output [{self.B}x{self.H}x{self.S}x{self.D}]"
        ))

        return accesses

    def total_hbm_bytes(self) -> Tuple[int, int]:
        accesses = self.count_accesses()
        total_read = sum(a.bytes_read for a in accesses)
        total_write = sum(a.bytes_write for a in accesses)
        return total_read, total_write


# ── More Fusion Examples ──

class SeparateLayerNormGELU:
    """Separate LayerNorm then GELU."""
    def __init__(self, batch=128, seq_len=512, hidden_dim=768):
        self.B = batch
        self.S = seq_len
        self.D = hidden_dim
        self.shape = (batch, seq_len, hidden_dim)

    def count_accesses(self) -> List[MemoryAccess]:
        accesses = []
        n_bytes = MemoryModel.bytes_for(self.shape)

        # LayerNorm: read input, write output
        accesses.append(MemoryAccess("layernorm_read", n_bytes, 0, f"Read input"))
        accesses.append(MemoryAccess("layernorm_write", 0, n_bytes, f"Write LN output"))
        # GELU: read LN output, write final
        accesses.append(MemoryAccess("gelu_read", n_bytes, 0, f"Read LN output"))
        accesses.append(MemoryAccess("gelu_write", 0, n_bytes, f"Write GELU output"))

        return accesses


class FusedLayerNormGELU:
    """Fused LayerNorm + GELU."""
    def __init__(self, batch=128, seq_len=512, hidden_dim=768):
        self.B = batch
        self.S = seq_len
        self.D = hidden_dim
        self.shape = (batch, seq_len, hidden_dim)

    def count_accesses(self) -> List[MemoryAccess]:
        accesses = []
        n_bytes = MemoryModel.bytes_for(self.shape)

        accesses.append(MemoryAccess("fused_read", n_bytes, 0, f"Read input"))
        # LN intermediate in SRAM
        accesses.append(MemoryAccess("ln_sram", 0, 0, "LN intermediate: SRAM only"))
        accesses.append(MemoryAccess("fused_write", 0, n_bytes, f"Write fused output"))

        return accesses


class SeparateMatmulBiasGELU:
    """Separate: matmul output → write HBM → add bias → write HBM → GELU → write HBM."""
    def __init__(self, M=512, N=512, K=512):
        self.M, self.N, self.K = M, N, K

    def count_accesses(self) -> List[MemoryAccess]:
        accesses = []
        a_bytes = MemoryModel.bytes_for((self.M, self.K))
        b_bytes = MemoryModel.bytes_for((self.K, self.N))
        out_bytes = MemoryModel.bytes_for((self.M, self.N))

        # Matmul: read A, B; write output
        accesses.append(MemoryAccess("matmul_read_A", a_bytes, 0, "Read A"))
        accesses.append(MemoryAccess("matmul_read_B", b_bytes, 0, "Read B"))
        accesses.append(MemoryAccess("matmul_write", 0, out_bytes, "Write matmul output"))

        # Bias add: read matmul output + bias; write output
        bias_bytes = MemoryModel.bytes_for((self.N,))
        accesses.append(MemoryAccess("bias_read", out_bytes, 0, "Read matmul output"))
        accesses.append(MemoryAccess("bias_read_b", bias_bytes, 0, "Read bias"))
        accesses.append(MemoryAccess("bias_write", 0, out_bytes, "Write biased output"))

        # GELU: read biased output; write output
        accesses.append(MemoryAccess("gelu_read", out_bytes, 0, "Read biased output"))
        accesses.append(MemoryAccess("gelu_write", 0, out_bytes, "Write GELU output"))

        return accesses


class FusedMatmulBiasGELU:
    """Fused: matmul tile → add bias in regs → GELU in regs → write output."""
    def __init__(self, M=512, N=512, K=512):
        self.M, self.N, self.K = M, N, K

    def count_accesses(self) -> List[MemoryAccess]:
        accesses = []
        a_bytes = MemoryModel.bytes_for((self.M, self.K))
        b_bytes = MemoryModel.bytes_for((self.K, self.N))
        out_bytes = MemoryModel.bytes_for((self.M, self.N))
        bias_bytes = MemoryModel.bytes_for((self.N,))

        # Read A, B (same as separate)
        accesses.append(MemoryAccess("fused_read_A", a_bytes, 0, "Read A"))
        accesses.append(MemoryAccess("fused_read_B", b_bytes, 0, "Read B"))
        accesses.append(MemoryAccess("fused_read_bias", bias_bytes, 0, "Read bias"))
        # Tile output stays in registers: bias + GELU applied in-place
        accesses.append(MemoryAccess("regs_bias_gelu", 0, 0, "Bias + GELU in registers"))
        # Single write of final result
        accesses.append(MemoryAccess("fused_write", 0, out_bytes, "Write final output"))

        return accesses


# ── Actual PyTorch comparison (timing) ──

def benchmark_separate_vs_fused(device='cpu', B=64, S=256, D=512, n_warmup=10, n_iter=100):
    """Time separate vs fused PyTorch operations."""
    x = torch.randn(B, S, D, device=device)

    # Separate: LayerNorm → GELU
    ln = nn.LayerNorm(D, device=device)

    # Warmup
    for _ in range(n_warmup):
        h = ln(x)
        h = F.gelu(h)

    # Time separate
    start = torch.cuda.Event(enable_timing=True) if device != 'cpu' else None
    end = torch.cuda.Event(enable_timing=True) if device != 'cpu' else None

    if device != 'cpu':
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_iter):
            h = ln(x)
            h = F.gelu(h)
        end.record()
        torch.cuda.synchronize()
        separate_ms = start.elapsed_time(end) / n_iter
    else:
        import time
        t0 = time.perf_counter()
        for _ in range(n_iter):
            h = ln(x)
            h = F.gelu(h)
        t1 = time.perf_counter()
        separate_ms = (t1 - t0) / n_iter * 1000

    # Fused via JIT: LayerNorm → GELU in one pass
    class FusedLNGELU(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)

        @torch.jit.export
        def forward_fused(self, x):
            return F.gelu(self.ln(x))

    fused_model = FusedLNGELU(D).to(device)
    # Warmup
    for _ in range(n_warmup):
        _ = fused_model.forward_fused(x)

    if device != 'cpu':
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_iter):
            _ = fused_model.forward_fused(x)
        end.record()
        torch.cuda.synchronize()
        fused_ms = start.elapsed_time(end) / n_iter
    else:
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = fused_model.forward_fused(x)
        t1 = time.perf_counter()
        fused_ms = (t1 - t0) / n_iter * 1000

    return separate_ms, fused_ms


# ── Main ──

def main():
    results_dir = Path(__file__).parent / "results" / "88-triton-kernel"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Triton-style Kernel Fusion Analysis ===\n")

    # 1. Softmax + Dropout fusion
    B, S, D, H = 128, 512, 64, 8
    print(f"--- Softmax + Dropout (B={B}, H={H}, S={S}, D={D}) ---")
    sep_sd = SeparateSoftmaxDropout(B, S, D, H)
    fused_sd = FusedSoftmaxDropout(B, S, D, H)

    sep_r, sep_w = sep_sd.total_hbm_bytes()
    fused_r, fused_w = fused_sd.total_hbm_bytes()
    total_sep = sep_r + sep_w
    total_fused = fused_r + fused_w
    savings = (1 - total_fused / total_sep) * 100

    print(f"  Separate:  read={sep_r/1e6:.1f} MB, write={sep_w/1e6:.1f} MB, total={total_sep/1e6:.1f} MB")
    print(f"  Fused:     read={fused_r/1e6:.1f} MB, write={fused_w/1e6:.1f} MB, total={total_fused/1e6:.1f} MB")
    print(f"  Savings:   {savings:.1f}%\n")

    # 2. LayerNorm + GELU fusion
    B2, S2, D2 = 128, 512, 768
    print(f"--- LayerNorm + GELU (B={B2}, S={S2}, D={D2}) ---")
    sep_ln = SeparateLayerNormGELU(B2, S2, D2)
    fused_ln = FusedLayerNormGELU(B2, S2, D2)

    def total_bytes(ops):
        acc = ops.count_accesses()
        return sum(a.bytes_read for a in acc), sum(a.bytes_write for a in acc)

    sep_r2, sep_w2 = total_bytes(sep_ln)
    fused_r2, fused_w2 = total_bytes(fused_ln)
    total_sep2 = sep_r2 + sep_w2
    total_fused2 = fused_r2 + fused_w2
    savings2 = (1 - total_fused2 / total_sep2) * 100

    print(f"  Separate:  read={sep_r2/1e6:.1f} MB, write={sep_w2/1e6:.1f} MB, total={total_sep2/1e6:.1f} MB")
    print(f"  Fused:     read={fused_r2/1e6:.1f} MB, write={fused_w2/1e6:.1f} MB, total={total_fused2/1e6:.1f} MB")
    print(f"  Savings:   {savings2:.1f}%\n")

    # 3. Matmul + Bias + GELU fusion
    M3, N3, K3 = 512, 512, 512
    print(f"--- Matmul + Bias + GELU (M={M3}, N={N3}, K={K3}) ---")
    sep_mm = SeparateMatmulBiasGELU(M3, N3, K3)
    fused_mm = FusedMatmulBiasGELU(M3, N3, K3)

    sep_r3, sep_w3 = total_bytes(sep_mm)
    fused_r3, fused_w3 = total_bytes(fused_mm)
    total_sep3 = sep_r3 + sep_w3
    total_fused3 = fused_r3 + fused_w3
    savings3 = (1 - total_fused3 / total_sep3) * 100

    print(f"  Separate:  read={sep_r3/1e6:.1f} MB, write={sep_w3/1e6:.1f} MB, total={total_sep3/1e6:.1f} MB")
    print(f"  Fused:     read={fused_r3/1e6:.1f} MB, write={fused_w3/1e6:.1f} MB, total={total_fused3/1e6:.1f} MB")
    print(f"  Savings:   {savings3:.1f}%\n")

    # 4. PyTorch timing benchmark
    print("--- PyTorch Timing: Separate vs Fused LayerNorm+GELU ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sep_ms, fused_ms = benchmark_separate_vs_fused(device=device)
    speedup = sep_ms / fused_ms if fused_ms > 0 else float('inf')
    print(f"  Device:    {device}")
    print(f"  Separate:  {sep_ms:.3f} ms")
    print(f"  Fused:     {fused_ms:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x\n")

    # 5. Scaling analysis: memory savings vs tensor size
    print("--- Scaling Analysis ---")
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    savings_by_op = {'Softmax+Dropout': [], 'LayerNorm+GELU': [], 'Matmul+Bias+GELU': []}

    for sl in seq_lengths:
        # Softmax+Dropout
        s1 = SeparateSoftmaxDropout(128, sl, 64, 8)
        f1 = FusedSoftmaxDropout(128, sl, 64, 8)
        r1, w1 = s1.total_hbm_bytes(); rf1, wf1 = f1.total_hbm_bytes()
        savings_by_op['Softmax+Dropout'].append(
            (1 - (rf1 + wf1) / (r1 + w1)) * 100
        )

        # LayerNorm+GELU
        s2 = SeparateLayerNormGELU(128, sl, 768)
        f2 = FusedLayerNormGELU(128, sl, 768)
        r2, w2 = total_bytes(s2); rf2, wf2 = total_bytes(f2)
        savings_by_op['LayerNorm+GELU'].append(
            (1 - (rf2 + wf2) / (r2 + w2)) * 100
        )

        # Matmul+Bias+GELU
        s3 = SeparateMatmulBiasGELU(sl, sl, sl)
        f3 = FusedMatmulBiasGELU(sl, sl, sl)
        r3, w3 = total_bytes(s3); rf3, wf3 = total_bytes(f3)
        savings_by_op['Matmul+Bias+GELU'].append(
            (1 - (rf3 + wf3) / (r3 + w3)) * 100
        )

    for op_name, svgs in savings_by_op.items():
        print(f"  {op_name}: savings = {svgs[0]:.1f}% → {svgs[-1]:.1f}% (S={seq_lengths[0]}→{seq_lengths[-1]})")

    # ── Visualization ──

    # 1. Memory access comparison (bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fusion_pairs = [
        ("Softmax+Dropout", sep_sd, fused_sd),
        ("LayerNorm+GELU", sep_ln, fused_ln),
        ("Matmul+Bias+GELU", sep_mm, fused_mm),
    ]

    for ax, (name, sep_op, fused_op) in zip(axes, fusion_pairs):
        sr, sw = total_bytes(sep_op)
        fr, fw = total_bytes(fused_op)

        x = np.array([0, 1])
        width = 0.3

        # Separate
        ax.bar(x - width/2, [sr/1e6, sw/1e6], width, label='Separate (Read)',
               color='steelblue', alpha=0.8)
        ax.bar(x - width/2 + 1*0, [0, sw/1e6], width, bottom=[sr/1e6, 0],
               color='lightblue', alpha=0.8)  # stacked
        # Actually, let's just show total read/write side by side
        ax.bar(x - width/2, [sr/1e6, sw/1e6], width, label='Separate',
               color=['steelblue', 'lightcoral'], alpha=0.8)
        ax.bar(x + width/2, [fr/1e6, fw/1e6], width, label='Fused',
               color=['royalblue', 'salmon'], alpha=0.8)

        sav = (1 - (fr + fw) / (sr + sw)) * 100
        ax.set_xticks(x)
        ax.set_xticklabels(['Read', 'Write'])
        ax.set_ylabel("MB")
        ax.set_title(f"{name}\nSavings: {sav:.0f}%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Memory Access: Separate vs Fused Kernels", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "memory_access_comparison.png", dpi=150)
    plt.close()

    # 2. Operation graph visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax_row, (name, sep_op, fused_op) in zip(
        [axes], [("Softmax+Dropout", sep_sd, fused_sd)]
    ):
        # Separate ops graph
        ax = ax_row[0]
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        boxes = [
            (0.08, 0.4, "Input\n(HBM)", 'lightyellow'),
            (0.3, 0.4, "Softmax\n(compute)", 'lightblue'),
            (0.52, 0.4, "Intermediate\n(HBM write!)", 'lightcoral'),
            (0.72, 0.4, "Dropout\n(compute)", 'lightblue'),
            (0.92, 0.4, "Output\n(HBM)", 'lightyellow'),
        ]
        for x, y, txt, color in boxes:
            ax.add_patch(plt.Rectangle((x-0.06, y-0.12), 0.12, 0.24,
                         facecolor=color, edgecolor='black', lw=1.5))
            ax.text(x, y, txt, ha='center', va='center', fontsize=8)

        for x1, x2 in [(0.14, 0.24), (0.36, 0.46), (0.58, 0.66), (0.78, 0.86)]:
            ax.annotate('', xy=(x2, 0.4), xytext=(x1, 0.4),
                        arrowprops=dict(arrowstyle='->', lw=1.5))

        ax.set_title("Separate Operations: HBM round-trip for intermediate", fontsize=12, fontweight='bold')

        # Fused ops graph
        ax = ax_row[1]
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        boxes = [
            (0.1, 0.4, "Input\n(HBM)", 'lightyellow'),
            (0.35, 0.55, "Softmax\n(SRAM)", 'lightgreen'),
            (0.55, 0.55, "Dropout\n(SRAM)", 'lightgreen'),
            (0.9, 0.4, "Output\n(HBM)", 'lightyellow'),
        ]
        for x, y, txt, color in boxes:
            ax.add_patch(plt.Rectangle((x-0.07, y-0.12), 0.14, 0.24,
                         facecolor=color, edgecolor='black', lw=1.5))
            ax.text(x, y, txt, ha='center', va='center', fontsize=8)

        # Fused kernel boundary
        ax.add_patch(plt.Rectangle((0.22, 0.3), 0.46, 0.45,
                     facecolor='none', edgecolor='green', lw=2, linestyle='--'))
        ax.text(0.45, 0.78, "Fused Kernel (single launch)", ha='center', fontsize=10,
                color='green', fontweight='bold')

        for x1, x2, y1, y2 in [(0.17, 0.28, 0.4, 0.55), (0.9, 0.82, 0.4, 0.4)]:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=1.5))

        ax.set_title("Fused Kernel: Intermediate stays in SRAM", fontsize=12, fontweight='bold')

    plt.suptitle("Triton-style Kernel Fusion: Operation Graphs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "operation_graph.png", dpi=150)
    plt.close()

    # 3. Scaling analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    for op_name, svgs in savings_by_op.items():
        ax.plot(seq_lengths, svgs, 'o-', label=op_name, markersize=5)

    ax.set_xlabel("Sequence Length / Dimension")
    ax.set_ylabel("Memory Savings (%)")
    ax.set_title("Kernel Fusion: Memory Savings vs Tensor Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels(seq_lengths)
    plt.tight_layout()
    plt.savefig(results_dir / "scaling_savings.png", dpi=150)
    plt.close()

    # 4. HBM traffic breakdown
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, sep_op, fused_op) in zip(axes, fusion_pairs):
        sr, sw = total_bytes(sep_op)
        fr, fw = total_bytes(fused_op)

        labels = ['Separate\nTotal', 'Fused\nTotal']
        reads = [sr/1e6, fr/1e6]
        writes = [sw/1e6, fw/1e6]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, reads, width, label='Read', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, writes, width, label='Write', color='coral', alpha=0.8)

        sav = (1 - (fr + fw) / (sr + sw)) * 100
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("MB")
        ax.set_title(f"{name}\nTotal savings: {sav:.0f}%")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("HBM Traffic Breakdown: Read vs Write", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "hbm_traffic_breakdown.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("CUDA Kernels", "Each op = separate\nkernel launch\nIntermediate → HBM\n→ read again next op", 0.14, 'gray'),
        ("Triton\nFusion", "Fuse ops in one\nkernel launch\nIntermediate → SRAM\n→ no HBM round-trip", 0.5, 'blue'),
        ("Key\nInsight", "HBM: ~1 TB/s\nSRAM: ~19 TB/s\n→ Keep data on-chip\n→ 10-20x faster", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    for x1, x2 in [(0.28, 0.36), (0.64, 0.72)]:
        ax.annotate('', xy=(x2, 0.55), xytext=(x1, 0.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_title("Triton: Kernel Fusion for Memory-Efficient GPU Programming", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "triton_concept.png", dpi=150)
    plt.close()

    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
