"""
Minimal KV Cache & Quantization Reproduction
=============================================
Reproduces core ideas from several LLM inference optimization papers:
1. KV Cache: caching key-value pairs across autoregressive steps (2301.08243 GQA context)
2. KV Cache Quantization: INT8/INT4 quantization of KV cache (2305.14314 KIVI)
3. Grouped-Query Attention (GQA): sharing K/V heads across Q heads
4. Memory savings analysis for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Standard Multi-Head Attention (MHA) ──

class MHAWithCache(nn.Module):
    """Multi-Head Attention with KV cache support."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Append to cache
        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K], dim=2)
            V = torch.cat([V_prev, V], dim=2)

        new_cache = (K, V) if use_cache else None

        S = T if kv_cache is None else K.shape[2]
        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Causal mask
        if kv_cache is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        else:
            # When using cache, query only attends to all cached + current keys
            pass  # No mask needed for single query token

        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), new_cache


# ── Grouped-Query Attention (GQA) ──

class GQAWithCache(nn.Module):
    """Grouped-Query Attention: share K/V heads across groups of Q heads.
    n_kv_heads < n_heads → fewer K/V projections → smaller KV cache.
    When n_kv_heads = 1: Multi-Query Attention (MQA)
    When n_kv_heads = n_heads: Standard MHA
    """
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads share one KV head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K], dim=2)
            V = torch.cat([V_prev, V], dim=2)

        new_cache = (K, V) if use_cache else None

        # Expand K/V to match Q heads
        K = K.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(
            B, self.n_heads, -1, self.head_dim)
        V = V.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(
            B, self.n_heads, -1, self.head_dim)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), new_cache


# ── KV Cache Quantization ──

def quantize_kv_int8(tensor):
    """Quantize KV cache to INT8 with per-channel scaling."""
    # tensor shape: (B, n_heads, S, head_dim)
    scale = tensor.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)
    q_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q_tensor, scale


def dequantize_kv_int8(q_tensor, scale):
    """Dequantize INT8 KV cache back to float."""
    return q_tensor.float() * scale


def quantize_kv_int4(tensor):
    """Simulate INT4 quantization (store as int8 but with 4-bit range)."""
    scale = tensor.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-8)
    q_tensor = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
    return q_tensor, scale


# ── Memory Analysis ──

def kv_cache_memory(seq_len, n_layers, n_heads, head_dim, n_kv_heads=None,
                    dtype_bytes=2, batch_size=1):
    """Calculate KV cache memory in bytes.
    FP16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads
    # 2 (K+V) * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes


# ── Correctness Test ──

def test_kv_cache_correctness(device='cpu'):
    """Verify that incremental generation with KV cache matches full recomputation."""
    d_model, n_heads, head_dim = 64, 4, 16
    mha = MHAWithCache(d_model, n_heads).to(device).eval()

    # Generate full sequence at once
    torch.manual_seed(42)
    x_full = torch.randn(1, 8, d_model, device=device)
    with torch.no_grad():
        out_full, _ = mha(x_full)

    # Generate incrementally with KV cache
    out_incremental = []
    kv_cache = None
    with torch.no_grad():
        for t in range(8):
            x_t = x_full[:, t:t+1]
            out_t, kv_cache = mha(x_t, kv_cache=kv_cache, use_cache=True)
            out_incremental.append(out_t)

    out_incr = torch.cat(out_incremental, dim=1)
    max_diff = (out_full - out_incr).abs().max().item()
    print(f"  KV Cache vs Full: max diff = {max_diff:.2e}")
    return max_diff < 1e-5


def test_quantization_error(device='cpu'):
    """Test KV cache quantization quality."""
    torch.manual_seed(42)
    tensor = torch.randn(1, 8, 128, 64, device=device)

    # INT8
    q8, s8 = quantize_kv_int8(tensor)
    dq8 = dequantize_kv_int8(q8, s8)
    int8_err = (tensor - dq8).abs().mean().item()

    # INT4
    q4, s4 = quantize_kv_int4(tensor)
    dq4 = dequantize_kv_int8(q4, s4)  # Same dequant
    int4_err = (tensor - dq4).abs().mean().item()

    print(f"  INT8 avg error: {int8_err:.4f}")
    print(f"  INT4 avg error: {int4_err:.4f}")
    return int8_err, int4_err


# ── Main ──

def main():
    device = 'cpu'
    results_dir = Path(__file__).parent / "results" / "10-kv-cache"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Testing KV Cache Correctness ===")
    test_kv_cache_correctness(device)

    print("\n=== Testing Quantization Error ===")
    int8_err, int4_err = test_quantization_error(device)

    # ── Visualization ──

    # 1. KV Cache Memory: MHA vs GQA vs MQA
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    n_layers = 32
    n_heads = 32
    head_dim = 128

    configs = [
        ('MHA (32 KV heads)', n_heads, 2, 'red'),
        ('GQA (8 KV heads)', 8, 2, 'orange'),
        ('GQA (4 KV heads)', 4, 2, 'blue'),
        ('MQA (1 KV head)', 1, 2, 'green'),
        ('MHA + INT8', n_heads, 1, 'red'),
        ('GQA-8 + INT8', 8, 1, 'orange'),
        ('GQA-8 + INT4', 8, 0.5, 'purple'),
    ]

    for name, n_kv, dtype_bytes, color in configs:
        mems = [kv_cache_memory(N, n_layers, n_heads, head_dim, n_kv,
                                dtype_bytes=dtype_bytes) / 1e9
                for N in seq_lengths]
        style = '--' if dtype_bytes < 2 else '-'
        axes[0].plot(seq_lengths, mems, style, label=name, color=color, linewidth=2)

    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("KV Cache Memory (GB)")
    axes[0].set_title("KV Cache Memory: MHA vs GQA vs MQA + Quantization")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 2. Memory at fixed seq_len = 8192, varying batch size
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    for name, n_kv, dtype_bytes, color in configs:
        mems = [kv_cache_memory(8192, n_layers, n_heads, head_dim, n_kv,
                                dtype_bytes=dtype_bytes, batch_size=B) / 1e9
                for B in batch_sizes]
        style = '--' if dtype_bytes < 2 else '-'
        axes[1].plot(batch_sizes, mems, style, label=name, color=color, linewidth=2)

    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("KV Cache Memory (GB)")
    axes[1].set_title("KV Cache Memory at Seq Len 8192")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "kv_cache_memory.png", dpi=150)
    plt.close()

    # 3. GQA Accuracy: compare MHA vs GQA on a simple task
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    torch.manual_seed(42)
    d_model = 64
    n_heads = 8
    head_dim = d_model // n_heads
    seq_len = 32
    n_steps = 2000

    configs_train = [
        ('MHA (8 KV heads)', n_heads),
        ('GQA (4 KV heads)', 4),
        ('GQA (2 KV heads)', 2),
        ('MQA (1 KV head)', 1),
    ]

    for name, n_kv in configs_train:
        print(f"\nTraining {name}...")
        model = GQAWithCache(d_model, n_heads, n_kv).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []

        for step in range(n_steps):
            x = torch.randn(16, seq_len, d_model, device=device)
            # Target: copy task (output should match input)
            out, _ = model(x)
            loss = F.mse_loss(out, x)  # Simple auto-encoding task

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        losses_s = np.convolve(losses, np.ones(30)/30, mode='valid')
        axes[0].plot(losses_s, label=name)

    axes[0].set_title("GQA Training: Copy Task")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 4. Quantization error distribution
    torch.manual_seed(42)
    tensor = torch.randn(1, 16, 512, 64)

    q8, s8 = quantize_kv_int8(tensor)
    dq8 = dequantize_kv_int8(q8, s8)
    err8 = (tensor - dq8).flatten().numpy()

    q4, s4 = quantize_kv_int4(tensor)
    dq4 = dequantize_kv_int8(q4, s4)
    err4 = (tensor - dq4).flatten().numpy()

    axes[1].hist(err8, bins=100, alpha=0.5, label=f'INT8 (MAE={np.abs(err8).mean():.4f})',
                 density=True, color='blue')
    axes[1].hist(err4, bins=100, alpha=0.5, label=f'INT4 (MAE={np.abs(err4).mean():.4f})',
                 density=True, color='red')
    axes[1].set_title("KV Cache Quantization Error Distribution")
    axes[1].set_xlabel("Quantization Error")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "gqa_and_quantization.png", dpi=150)
    plt.close()

    # 5. KV Cache speed comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    gen_lengths = [1, 10, 50, 100, 200, 500]
    # Theoretical FLOPs comparison: with cache O(S*d^2) vs without cache O(S^2*d^2)
    d = 4096  # typical d_model for LLM
    flops_no_cache = [s * s * d * d * 2 for s in gen_lengths]  # O(S^2)
    flops_with_cache = [s * d * d * 2 for s in gen_lengths]  # O(S)

    ax.plot(gen_lengths, [f/1e9 for f in flops_no_cache], 'o-', label='Without KV Cache', color='red')
    ax.plot(gen_lengths, [f/1e9 for f in flops_with_cache], 's-', label='With KV Cache', color='blue')
    ax.set_xlabel("Generation Length")
    ax.set_ylabel("FLOPs (GFLOPs, theoretical)")
    ax.set_title("Generation Cost: With vs Without KV Cache")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "generation_cost.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
