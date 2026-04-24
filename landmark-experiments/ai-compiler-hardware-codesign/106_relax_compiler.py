"""
Minimal Relax Compiler Abstractions Reproduction
=================================================
Reproduces core ideas from "Relax: Composable Abstractions for End-to-End
ML" (2311.02103, Lian et al.):
1. Graph-level IR: represent ML models as computation graphs (not operator soup)
2. Dynamic shape support: symbolic shape variables in the IR
3. Function-level transformations: compose passes on graph IR
4. Compare: eager execution vs traced graph vs compiled graph
5. Show: fusion opportunities exposed by graph-level view
6. Demonstrate: symbolic shape inference and dynamic dispatch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import time


# ── Graph IR: Minimal Computation Graph ──

class Var:
    """Symbolic variable (tensor or shape)."""
    def __init__(self, name, shape=None, dtype='float32'):
        self.name = name
        self.shape = shape  # tuple of int or str (symbolic)
        self.dtype = dtype

    def __repr__(self):
        return f"Var({self.name}, shape={self.shape})"


class Op:
    """Operation node in computation graph."""
    def __init__(self, name, inputs, attrs=None):
        self.name = name
        self.inputs = inputs  # list of Var or Op
        self.attrs = attrs or {}
        self.output = Var(f"{name}_out")

    def __repr__(self):
        return f"Op({self.name}, inputs=[{', '.join(str(i) for i in self.inputs)}])"


class Function:
    """Relax-style function: named parameters → body → return."""
    def __init__(self, name, params, body, return_var):
        self.name = name
        self.params = params  # list of Var
        self.body = body      # list of Op
        self.return_var = return_var

    def __repr__(self):
        lines = [f"def {self.name}({', '.join(str(p) for p in self.params)}):"]
        for op in self.body:
            lines.append(f"  {op.output} = {op}")
        lines.append(f"  return {self.return_var}")
        return '\n'.join(lines)


class IRModule:
    """Module containing multiple functions (like Relax IRModule)."""
    def __init__(self):
        self.functions = {}

    def add(self, func):
        self.functions[func.name] = func

    def __repr__(self):
        return '\n\n'.join(str(f) for f in self.functions.values())


# ── Graph Builder ──

def build_transformer_block_ir(batch='B', seq='S', d_model=128, n_heads=4):
    """Build IR for a transformer block with symbolic shapes."""
    d_head = d_model // n_heads
    params = [
        Var('x', shape=(batch, seq, d_model)),
        Var('Wq', shape=(d_model, d_model)),
        Var('Wk', shape=(d_model, d_model)),
        Var('Wv', shape=(d_model, d_model)),
        Var('Wo', shape=(d_model, d_model)),
        Var('W1', shape=(d_model, d_model * 4)),
        Var('W2', shape=(d_model * 4, d_model)),
    ]

    ops = []
    # QKV projections
    ops.append(Op('matmul', [params[0], params[1]], attrs={'shape': (batch, seq, d_model)}))
    q = ops[-1].output
    ops.append(Op('matmul', [params[0], params[2]], attrs={'shape': (batch, seq, d_model)}))
    k = ops[-1].output
    ops.append(Op('matmul', [params[0], params[3]], attrs={'shape': (batch, seq, d_model)}))
    v = ops[-1].output

    # Attention: reshape → bmm → softmax → bmm
    ops.append(Op('reshape', [q], attrs={'shape': (batch, seq, n_heads, d_head)}))
    ops.append(Op('transpose', [ops[-1].output], attrs={'dims': (0, 2, 1, 3)}))
    q_t = ops[-1].output
    ops.append(Op('reshape', [k], attrs={'shape': (batch, seq, n_heads, d_head)}))
    ops.append(Op('transpose', [ops[-1].output], attrs={'dims': (0, 2, 1, 3)}))
    k_t = ops[-1].output
    ops.append(Op('reshape', [v], attrs={'shape': (batch, seq, n_heads, d_head)}))
    ops.append(Op('transpose', [ops[-1].output], attrs={'dims': (0, 2, 1, 3)}))
    v_t = ops[-1].output

    ops.append(Op('bmm', [q_t, k_t], attrs={'shape': (batch, n_heads, seq, seq)}))
    ops.append(Op('softmax', [ops[-1].output], attrs={'axis': -1}))
    ops.append(Op('bmm', [ops[-1].output, v_t], attrs={'shape': (batch, n_heads, seq, d_head)}))
    ops.append(Op('transpose', [ops[-1].output], attrs={'dims': (0, 2, 1, 3)}))
    ops.append(Op('reshape', [ops[-1].output], attrs={'shape': (batch, seq, d_model)}))

    # Output projection + residual
    ops.append(Op('matmul', [ops[-1].output, params[4]], attrs={'shape': (batch, seq, d_model)}))
    ops.append(Op('add', [ops[-1].output, params[0]], attrs={'shape': (batch, seq, d_model)}))
    attn_out = ops[-1].output

    # FFN + residual
    ops.append(Op('matmul', [attn_out, params[5]], attrs={'shape': (batch, seq, d_model * 4)}))
    ops.append(Op('gelu', [ops[-1].output], attrs={'shape': (batch, seq, d_model * 4)}))
    ops.append(Op('matmul', [ops[-1].output, params[6]], attrs={'shape': (batch, seq, d_model)}))
    ops.append(Op('add', [ops[-1].output, attn_out], attrs={'shape': (batch, seq, d_model)}))

    return Function('transformer_block', params, ops, ops[-1].output)


# ── Graph Passes (Transformations) ──

class GraphPass:
    """Base class for graph-level passes."""
    def __call__(self, func):
        return self.transform(func)

    def transform(self, func):
        raise NotImplementedError


class FuseMatmulGelu(GraphPass):
    """Fuse matmul + gelu into a single fused op."""
    def transform(self, func):
        new_body = []
        skip = set()
        for i, op in enumerate(func.body):
            if i in skip:
                continue
            if (op.name == 'matmul' and i + 1 < len(func.body)
                    and func.body[i + 1].name == 'gelu'):
                fused = Op('fused_matmul_gelu', op.inputs,
                           attrs={**op.attrs, 'fused': True})
                new_body.append(fused)
                skip.add(i + 1)
                # Update references
                gelu_out = func.body[i + 1].output
                for j, later_op in enumerate(func.body[i + 2:], i + 2):
                    if gelu_out in later_op.inputs:
                        later_op.inputs = [fused.output if x is gelu_out else x for x in later_op.inputs]
            else:
                new_body.append(op)
        return Function(func.name, func.params, new_body, func.return_var)


class FuseMatmulBiasAdd(GraphPass):
    """Fuse matmul + add (residual) into fused_matmul_add."""
    def transform(self, func):
        new_body = []
        skip = set()
        for i, op in enumerate(func.body):
            if i in skip:
                continue
            if (op.name == 'matmul' and i + 1 < len(func.body)
                    and func.body[i + 1].name == 'add'):
                fused = Op('fused_matmul_add', op.inputs + func.body[i + 1].inputs,
                           attrs={**op.attrs, 'fused': True})
                new_body.append(fused)
                skip.add(i + 1)
                add_out = func.body[i + 1].output
                for j, later_op in enumerate(func.body[i + 2:], i + 2):
                    if add_out in later_op.inputs:
                        later_op.inputs = [fused.output if x is add_out else x for x in later_op.inputs]
            else:
                new_body.append(op)
        return Function(func.name, func.params, new_body, func.return_var)


class SymbolicShapeInference(GraphPass):
    """Propagate symbolic shapes through the graph."""
    def transform(self, func):
        shape_map = {}
        for p in func.params:
            shape_map[p.name] = p.shape
        for op in func.body:
            if op.attrs.get('shape'):
                shape_map[op.output.name] = op.attrs['shape']
        inferred = {}
        for op in func.body:
            inferred[op.name] = {
                'input_shapes': [shape_map.get(inp.name if hasattr(inp, 'name') else str(inp), '?')
                                 for inp in op.inputs],
                'output_shape': shape_map.get(op.output.name, '?'),
            }
        return func, inferred


# ── Eager vs Traced vs Compiled Benchmarks ──

class EagerTransformerBlock(nn.Module):
    """Standard PyTorch eager transformer block."""
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = self.norm1(x + self.out_proj(out))
        ffn = self.ff2(F.gelu(self.ff1(x)))
        x = self.norm2(x + ffn)
        return x


def benchmark_eager(model, x, n_warmup=10, n_iter=100):
    """Benchmark eager execution."""
    for _ in range(n_warmup):
        _ = model(x)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(x)
    torch.cuda.synchronize() if x.is_cuda else None
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


def benchmark_traced(model, x, n_warmup=10, n_iter=100):
    """Benchmark torch.compile (graph-traced) execution."""
    compiled = torch.compile(model, mode='reduce-overhead')
    for _ in range(n_warmup):
        _ = compiled(x)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = compiled(x)
    torch.cuda.synchronize() if x.is_cuda else None
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


# ── Dynamic Shape Demonstration ──

def dynamic_shape_demo():
    """Demonstrate symbolic shape handling for variable-length inputs."""
    shape_env = {}

    def resolve_shape(shape, batch_size, seq_len):
        resolved = []
        for dim in shape:
            if isinstance(dim, str):
                if dim == 'B':
                    resolved.append(batch_size)
                elif dim == 'S':
                    resolved.append(seq_len)
                else:
                    resolved.append(shape_env.get(dim, 1))
            else:
                resolved.append(dim)
        return tuple(resolved)

    # Build IR with symbolic shapes
    func = build_transformer_block_ir(batch='B', seq='S', d_model=128, n_heads=4)

    # Test with different concrete shapes
    shapes = [(1, 16), (4, 32), (8, 64), (16, 128), (32, 256)]
    results = []
    for B, S in shapes:
        concrete_shapes = {}
        for op in func.body:
            if op.attrs.get('shape'):
                concrete = resolve_shape(op.attrs['shape'], B, S)
                concrete_shapes[op.name] = concrete
        results.append((B, S, len(concrete_shapes), concrete_shapes))
    return results


# ── Memory Estimation ──

def estimate_memory(func, batch_size, seq_len, d_model=128):
    """Estimate memory footprint from graph IR."""
    bytes_per_float = 4
    total = 0
    op_memory = {}
    for op in func.body:
        shape = op.attrs.get('shape', None)
        if shape:
            concrete = []
            for dim in shape:
                if isinstance(dim, str):
                    if dim == 'B':
                        concrete.append(batch_size)
                    elif dim == 'S':
                        concrete.append(seq_len)
                    else:
                        concrete.append(1)
                else:
                    concrete.append(dim)
            n_elements = 1
            for d in concrete:
                n_elements *= d
            mem = n_elements * bytes_per_float
            op_memory[op.name] = (concrete, mem)
            total += mem
    return total, op_memory


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "106-relax-compiler"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Experiment 1: Build and display graph IR ──
    print("=== Experiment 1: Building Transformer Block IR ===")
    func = build_transformer_block_ir(batch='B', seq='S', d_model=128, n_heads=4)
    print(f"  IR has {len(func.body)} operations")

    # Count op types
    op_counts = defaultdict(int)
    for op in func.body:
        op_counts[op.name] += 1
    print(f"  Op types: {dict(op_counts)}")

    # ── Experiment 2: Apply graph passes (fusion) ──
    print("\n=== Experiment 2: Graph Passes (Fusion) ===")
    original_ops = len(func.body)

    func_fused = FuseMatmulGelu()(func)
    fused1_ops = len(func_fused.body)
    print(f"  After Matmul+Gelu fusion: {original_ops} → {fused1_ops} ops")

    func_fused2 = FuseMatmulBiasAdd()(func_fused)
    fused2_ops = len(func_fused2.body)
    print(f"  After Matmul+Add fusion: {fused1_ops} → {fused2_ops} ops")

    # Check fused ops
    fused_ops = [op for op in func_fused2.body if 'fused' in op.name]
    print(f"  Fused operations: {[op.name for op in fused_ops]}")

    # ── Experiment 3: Symbolic shape inference ──
    print("\n=== Experiment 3: Symbolic Shape Inference ===")
    _, inferred = SymbolicShapeInference()(func)
    for op_name, shapes in list(inferred.items())[:5]:
        print(f"  {op_name}: in={shapes['input_shapes']} → out={shapes['output_shape']}")

    # ── Experiment 4: Dynamic shape dispatch ──
    print("\n=== Experiment 4: Dynamic Shape Dispatch ===")
    dyn_results = dynamic_shape_demo()
    for B, S, n_ops, shapes in dyn_results:
        print(f"  B={B}, S={S}: {n_ops} ops resolved")

    # ── Experiment 5: Memory estimation ──
    print("\n=== Experiment 5: Memory Estimation from IR ===")
    batch_sizes = [1, 4, 8, 16, 32]
    seq_lengths = [16, 32, 64, 128, 256]
    mem_heatmap = np.zeros((len(batch_sizes), len(seq_lengths)))

    for i, B in enumerate(batch_sizes):
        for j, S in enumerate(seq_lengths):
            total_mem, _ = estimate_memory(func, B, S)
            mem_heatmap[i, j] = total_mem / (1024 ** 2)  # MB

    # ── Experiment 6: Eager vs Compiled benchmark ──
    print("\n=== Experiment 6: Eager vs Compiled Benchmark ===")
    model = EagerTransformerBlock(d_model=128, n_heads=4).to(device)
    model.eval()

    bench_shapes = [(4, 64), (8, 128), (16, 256)]
    eager_times = []
    compiled_times = []
    speedups = []

    for B, S in bench_shapes:
        x = torch.randn(B, S, 128, device=device)
        t_eager = benchmark_eager(model, x, n_warmup=5, n_iter=50)
        eager_times.append(t_eager)
        print(f"  Shape ({B},{S}): eager={t_eager:.2f}ms")

        try:
            t_compiled = benchmark_traced(model, x, n_warmup=5, n_iter=50)
            compiled_times.append(t_compiled)
            speedups.append(t_eager / t_compiled)
            print(f"  Shape ({B},{S}): compiled={t_compiled:.2f}ms, speedup={t_eager/t_compiled:.2f}x")
        except Exception as e:
            compiled_times.append(t_eager)
            speedups.append(1.0)
            print(f"  Shape ({B},{S}): compile failed ({e}), fallback to eager")

    # ── Visualization ──

    # 1. Op count before/after fusion
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ['Original', 'After\nMatmul+Gelu\nFusion', 'After\nMatmul+Add\nFusion']
    counts = [original_ops, fused1_ops, fused2_ops]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    axes[0].bar(labels, counts, color=colors, alpha=0.8)
    axes[0].set_ylabel("Number of Operations")
    axes[0].set_title("Graph Pass: Operation Fusion")
    for i, c in enumerate(counts):
        axes[0].text(i, c + 0.3, str(c), ha='center', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Op type distribution before/after fusion
    def count_types(body):
        c = defaultdict(int)
        for op in body:
            c[op.name] += 1
        return dict(c)

    orig_types = count_types(func.body)
    fused_types = count_types(func_fused2.body)
    all_types = sorted(set(list(orig_types.keys()) + list(fused_types.keys())))
    x_pos = np.arange(len(all_types))
    width = 0.35
    orig_vals = [orig_types.get(t, 0) for t in all_types]
    fused_vals = [fused_types.get(t, 0) for t in all_types]
    axes[1].bar(x_pos - width/2, orig_vals, width, label='Original', color='#e74c3c', alpha=0.7)
    axes[1].bar(x_pos + width/2, fused_vals, width, label='After Fusion', color='#2ecc71', alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(all_types, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Operation Type Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Relax Compiler: Graph-Level IR & Fusion (2311.02103)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'graph_fusion.png', dpi=150)
    plt.close()

    # 2. Memory heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mem_heatmap, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([str(s) for s in seq_lengths])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Batch Size")
    ax.set_title("Memory Footprint from IR (MB)")
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lengths)):
            ax.text(j, i, f"{mem_heatmap[i,j]:.1f}", ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='MB')
    plt.tight_layout()
    plt.savefig(results_dir / 'memory_heatmap.png', dpi=150)
    plt.close()

    # 3. Eager vs Compiled
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(bench_shapes))
    width = 0.35
    ax.bar(x_pos - width/2, eager_times, width, label='Eager', color='#e74c3c', alpha=0.7)
    ax.bar(x_pos + width/2, compiled_times, width, label='Compiled', color='#3498db', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"B={b},S={s}" for b, s in bench_shapes])
    ax.set_ylabel("Time per inference (ms)")
    ax.set_title("Eager vs Compiled (torch.compile) Execution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (e, c, s) in enumerate(zip(eager_times, compiled_times, speedups)):
        ax.text(i, max(e, c) + 0.5, f"{s:.2f}x", ha='center', fontweight='bold', color='green')
    plt.tight_layout()
    plt.savefig(results_dir / 'eager_vs_compiled.png', dpi=150)
    plt.close()

    # 4. IR concept diagram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Eager execution
    ax = axes[0]
    ax.axis('off')
    ax.set_title("Eager Execution", fontsize=13, fontweight='bold')
    eager_text = (
        "x → Linear → reshape → bmm → softmax → bmm\n"
        "  → reshape → Linear → add → Linear → GELU\n"
        "  → Linear → add\n\n"
        "• No global view of computation\n"
        "• Each op dispatched independently\n"
        "• No fusion opportunity visible\n"
        "• Memory: every intermediate saved"
    )
    ax.text(0.05, 0.95, eager_text, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe0e0', alpha=0.9))

    # Graph IR (Relax)
    ax = axes[1]
    ax.axis('off')
    ax.set_title("Relax Graph IR", fontsize=13, fontweight='bold')
    ir_text = (
        "IRModule {\n"
        "  def block(x, Wq, Wk, ...):\n"
        "    q = matmul(x, Wq)    # shape: (B,S,D)\n"
        "    k = matmul(x, Wk)    # shape: (B,S,D)\n"
        "    attn = softmax(bmm(q,k.T))\n"
        "    out = bmm(attn, v)\n"
        "    ffn = fused_matmul_gelu(out, W1)\n"
        "    ...\n"
        "}\n\n"
        "• Symbolic shapes: B, S remain variables\n"
        "• Passes see the full graph\n"
        "• Fusion: matmul+gelu → single kernel"
    )
    ax.text(0.05, 0.95, ir_text, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0f0ff', alpha=0.9))

    # Compiled
    ax = axes[2]
    ax.axis('off')
    ax.set_title("After Compilation", fontsize=13, fontweight='bold')
    compiled_text = (
        "Fused Kernels:\n"
        "  1. fused_qkv_proj(x, Wq, Wk, Wv)\n"
        "  2. flash_attention(q, k, v)\n"
        "  3. fused_matmul_gelu(x, W1)\n"
        "  4. fused_matmul_add(ffn, W2, residual)\n\n"
        "• Fewer kernel launches\n"
        "• Less memory traffic\n"
        "• Dynamic shapes: recompile only\n"
        "  when shape changes"
    )
    ax.text(0.05, 0.95, compiled_text, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0ffe0', alpha=0.9))

    plt.suptitle('Relax: Composable Abstractions for ML Compilation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'compiler_concept.png', dpi=150)
    plt.close()

    # 5. Dynamic shape resolution
    fig, ax = plt.subplots(figsize=(10, 5))
    concrete_counts = [r[2] for r in dyn_results]
    shape_labels = [f"B={r[0]},S={r[1]}" for r in dyn_results]
    ax.bar(shape_labels, concrete_counts, color='steelblue', alpha=0.7)
    ax.set_ylabel("Resolved Operations")
    ax.set_xlabel("Concrete Shape")
    ax.set_title("Dynamic Shape Resolution: Same IR, Different Concrete Shapes")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / 'dynamic_shape.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
