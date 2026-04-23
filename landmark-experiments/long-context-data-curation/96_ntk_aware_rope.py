"""
NTK-Aware RoPE Scaling for Context Length Extension
====================================================
Reproduces core ideas from the NTK-aware scaling thread and YaRN
(2309.00071, Peng et al.) for extending RoPE-based model context lengths:

1. RoPE encodes position via rotation; frequencies theta_i = base^(-2i/d)
2. Naive linear scaling (divide positions by factor) distorts high-freq dimensions
3. NTK-aware scaling: scale the base (not positions) — high frequencies
   change less, low frequencies change more — preserving local structure
4. YaRN: combines NTK scaling with attention temperature adjustment for
   mitigating attention entropy changes at long contexts
5. Comparison: no scaling, linear scaling, NTK-aware scaling, YaRN
6. Demonstrate how different methods preserve relative position information
   at extended context lengths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── RoPE Implementations with Different Scaling ──

class RotaryEmbedding(nn.Module):
    """Base Rotary Position Embedding (no scaling)."""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, seq_len):
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),  # (1, 1, T, D/2)
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0),
        )


class LinearScaledRoPE(RotaryEmbedding):
    """Linear scaling: divide positions by scale_factor.
    Simple but distorts all frequency bands equally.
    Position n → n / scale_factor
    """
    def __init__(self, dim, max_seq_len=2048, base=10000, scale_factor=8):
        self.scale_factor = scale_factor
        super().__init__(dim, max_seq_len * scale_factor, base)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float() / self.scale_factor
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)


class NTKAwareRoPE(RotaryEmbedding):
    """NTK-aware scaling: scale the base instead of positions.
    base_new = base * scale_factor ^ (dim / (dim - 2))
    This stretches low frequencies more than high frequencies,
    preserving local positional resolution.

    Key insight from the NTK-aware perspective:
    - High-frequency dimensions (small i) barely change → local structure preserved
    - Low-frequency dimensions (large i) stretch more → global position extended
    - This matches the intuition that nearby tokens need precise position,
      while distant tokens only need coarse relative position
    """
    def __init__(self, dim, max_seq_len=2048, base=10000, scale_factor=8):
        self.scale_factor = scale_factor
        # NTK-aware base scaling
        base_new = base * (scale_factor ** (dim / (dim - 2)))
        super().__init__(dim, max_seq_len, base_new)


class YaRNRoPE(RotaryEmbedding):
    """YaRN: NTK-aware scaling + attention temperature adjustment.

    From "YaRN: Efficient Context Window Extension of Large Language Models"
    (2309.00071):

    1. Uses NTK-aware base scaling for the frequency bands
    2. Splits dimensions into:
       - Low-frequency (wavelength > original context): use scaled freq
       - High-frequency (wavelength < original context): use original freq
       - Middle: smooth interpolation
    3. Adjusts attention scores by temperature sqrt(scale_factor)
       to compensate for increased attention entropy at long contexts
    """
    def __init__(self, dim, max_seq_len=2048, base=10000,
                 scale_factor=8, original_seq_len=2048):
        self.scale_factor = scale_factor
        self.original_seq_len = original_seq_len

        # Call super first to initialize nn.Module properly
        super().__init__(dim, max_seq_len, base)

        # Compute wavelength for each frequency
        # wavelength_i = 2*pi / theta_i = 2*pi * base^(2i/d)
        inv_freq_orig = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        wavelengths = 2 * math.pi / inv_freq_orig

        # NTK-scaled frequencies
        base_new = base * (scale_factor ** (dim / (dim - 2)))
        inv_freq_scaled = 1.0 / (base_new ** (torch.arange(0, dim, 2).float() / dim))

        # Blend: use original for high-freq (short wavelength),
        #         scaled for low-freq (long wavelength)
        # Low freq: wavelength > original_seq_len → fully scaled
        # High freq: wavelength < original_seq_len/scale_factor → fully original
        # Middle: linear interpolation
        low_mask = wavelengths >= original_seq_len
        high_mask = wavelengths < original_seq_len / scale_factor

        smooth = torch.ones_like(inv_freq_orig)
        smooth[low_mask] = 0.0   # fully scaled
        smooth[high_mask] = 1.0  # fully original
        # Middle: linear interpolation
        mid_mask = ~low_mask & ~high_mask
        if mid_mask.any():
            mid_wavelengths = wavelengths[mid_mask]
            smooth[mid_mask] = (original_seq_len / mid_wavelengths - 1) / (scale_factor - 1)

        # Blended inverse frequencies — override the one from super()
        inv_freq = smooth * inv_freq_orig + (1 - smooth) * inv_freq_scaled
        self.inv_freq.copy_(inv_freq)  # overwrite buffer in-place
        self._build_cache(max_seq_len)  # rebuild cos/sin cache with new freqs

        # Attention temperature factor
        self.temp_factor = math.sqrt(scale_factor)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding. x: (..., D), cos/sin: (..., D/2)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ── Attention with Configurable RoPE ──

class RoPEAttention(nn.Module):
    """Multi-head attention with configurable RoPE scaling."""
    def __init__(self, d_model, n_heads, rope_emb, use_yarn_temp=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.rope = rope_emb
        self.use_yarn_temp = use_yarn_temp
        if use_yarn_temp:
            self.temp_factor = rope_emb.temp_factor
        else:
            self.temp_factor = 1.0

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(T)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # YaRN temperature scaling
        if self.temp_factor != 1.0:
            attn = attn / self.temp_factor

        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out), attn


# ── Analysis Utilities ──

def compute_rope_similarity(rope_emb, seq_len, device='cpu'):
    """Compute the dot-product similarity between positions under RoPE.
    For a fixed vector x, compute R(m)^T x · R(n)^T x for all m, n.
    This shows how well relative position is encoded.
    """
    d = rope_emb.dim
    x = torch.randn(d, device=device)

    cos, sin = rope_emb(seq_len)
    cos = cos.squeeze(0).squeeze(0)  # (T, D/2)
    sin = sin.squeeze(0).squeeze(0)  # (T, D/2)

    # Apply RoPE to x at each position
    x1, x2 = x[:d//2], x[d//2:]
    rotated = torch.zeros(seq_len, d, device=device)
    for m in range(seq_len):
        c, s = cos[m], sin[m]
        rotated[m, :d//2] = x1 * c - x2 * s
        rotated[m, d//2:] = x1 * s + x2 * c

    # Similarity matrix: rotated[m] · rotated[n]
    sim = rotated @ rotated.T
    return sim


def compute_attention_entropy(attn_weights):
    """Compute entropy of attention distribution for each query position.
    Higher entropy = more diffuse attention (potentially degraded).
    """
    # attn_weights: (B, H, T, T) or (H, T, T)
    if attn_weights.dim() == 4:
        attn = attn_weights[0].mean(0)  # average over batch, (H, T, T) -> (T, T)
        attn = attn.mean(0)  # average over heads -> (T, T)
    elif attn_weights.dim() == 3:
        attn = attn_weights.mean(0)  # (T, T)

    # Entropy per query position
    eps = 1e-10
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    return entropy


def frequency_analysis(rope_emb, scale_factor):
    """Analyze how each frequency band is affected by scaling."""
    d = rope_emb.dim
    inv_freq = rope_emb.inv_freq.cpu().numpy()

    # Original frequencies for reference
    inv_freq_orig = 1.0 / (10000.0 ** (np.arange(0, d, 2) / d))

    # Ratios
    ratios = inv_freq / inv_freq_orig
    return inv_freq_orig, inv_freq, ratios


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "96-ntk-rope"
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = 64
    n_heads = 4
    d_head = d_model // n_heads
    original_seq_len = 256
    scale_factor = 8
    extended_seq_len = original_seq_len * scale_factor  # 2048

    # ── Experiment 1: Frequency Analysis ──
    print("=== Experiment 1: Frequency Band Analysis ===")
    methods = {
        'no_scaling': RotaryEmbedding(d_head, max_seq_len=extended_seq_len, base=10000),
        'linear': LinearScaledRoPE(d_head, max_seq_len=extended_seq_len, base=10000, scale_factor=scale_factor),
        'ntk_aware': NTKAwareRoPE(d_head, max_seq_len=extended_seq_len, base=10000, scale_factor=scale_factor),
        'yarn': YaRNRoPE(d_head, max_seq_len=extended_seq_len, base=10000,
                         scale_factor=scale_factor, original_seq_len=original_seq_len),
    }

    freq_results = {}
    for name, rope in methods.items():
        orig, scaled, ratios = frequency_analysis(rope, scale_factor)
        freq_results[name] = {'orig': orig, 'scaled': scaled, 'ratios': ratios}
        # Show first and last frequency ratios
        print(f"  {name:12s}: freq[0] ratio={ratios[0]:.4f}  freq[-1] ratio={ratios[-1]:.4f}  "
              f"mean_ratio={ratios.mean():.4f}")

    # ── Experiment 2: Position Similarity Matrix ──
    print("\n=== Experiment 2: Position Similarity Under Different Scaling ===")
    sim_matrices = {}
    # Use a shorter seq_len for similarity matrix (visual clarity)
    sim_len = 512
    for name, rope in methods.items():
        rope_d = rope.to(device)
        sim = compute_rope_similarity(rope_d, sim_len, device)
        sim_matrices[name] = sim.cpu().numpy()
        # Diagonal should be high, off-diagonal should decay
        diag_mean = np.diag(sim_matrices[name]).mean()
        print(f"  {name:12s}: diagonal mean={diag_mean:.4f}")

    # ── Experiment 3: Attention Pattern at Extended Length ──
    print("\n=== Experiment 3: Attention Patterns at Extended Context ===")
    torch.manual_seed(42)

    attn_results = {}
    test_lengths = [256, 512, 1024, 2048]

    for name, rope_cls_name in [('no_scaling', 'no_scaling'), ('linear', 'linear'),
                                 ('ntk_aware', 'ntk_aware'), ('yarn', 'yarn')]:
        rope_emb = methods[rope_cls_name].to(device)
        use_temp = (rope_cls_name == 'yarn')
        attn_layer = RoPEAttention(d_model, n_heads, rope_emb, use_yarn_temp=use_temp).to(device)
        attn_layer.eval()

        length_entropy = {}
        for T in test_lengths:
            with torch.no_grad():
                x = torch.randn(1, T, d_model, device=device)
                _, attn_w = attn_layer(x)
                entropy = compute_attention_entropy(attn_w)
                length_entropy[T] = {
                    'mean': entropy.mean().item(),
                    'std': entropy.std().item(),
                    'max': entropy.max().item(),
                }
                print(f"  {name:12s} T={T:5d}: entropy mean={entropy.mean():.4f} "
                      f"std={entropy.std():.4f} max={entropy.max():.4f}")

        attn_results[name] = length_entropy

    # ── Experiment 4: Relative Position Preservation ──
    print("\n=== Experiment 4: Relative Position Preservation ===")
    # For positions m, n in extended range, check if the RoPE dot product
    # still encodes relative distance properly
    rel_pos_results = {}
    for name, rope in methods.items():
        rope_d = rope.to(device)
        sim = compute_rope_similarity(rope_d, 256, device).cpu().numpy()

        # For each relative distance k, collect all sim[m, m+k] values
        from collections import defaultdict
        dist_sims = defaultdict(list)
        for m in range(256):
            for n in range(m, min(m + 50, 256)):
                dist_sims[n - m].append(sim[m, n])

        dists = sorted(dist_sims.keys())
        mean_sims = [np.mean(dist_sims[d]) for d in dists]
        std_sims = [np.std(dist_sims[d]) for d in dists]

        rel_pos_results[name] = {'dists': dists, 'mean': mean_sims, 'std': std_sims}

        # Variance of similarity at same distance (should be low if rel. pos. preserved)
        avg_std = np.mean(std_sims[:20])
        print(f"  {name:12s}: avg std of sim at same distance (k<20) = {avg_std:.4f}")

    # ── Visualization ──

    # Plot 1: Frequency ratio comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dim_indices = np.arange(len(freq_results['no_scaling']['ratios']))
    colors = {'no_scaling': 'gray', 'linear': 'red', 'ntk_aware': 'blue', 'yarn': 'green'}
    labels = {'no_scaling': 'No Scaling', 'linear': 'Linear Scaling',
              'ntk_aware': 'NTK-Aware', 'yarn': 'YaRN'}

    for name in ['no_scaling', 'linear', 'ntk_aware', 'yarn']:
        axes[0].plot(dim_indices, freq_results[name]['ratios'],
                     'o-', markersize=3, label=labels[name], color=colors[name])
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("Frequency dimension index i")
    axes[0].set_ylabel("Scaled freq / Original freq")
    axes[0].set_title("Frequency Band Scaling Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Show actual frequencies (log scale)
    for name in ['no_scaling', 'linear', 'ntk_aware', 'yarn']:
        axes[1].semilogy(dim_indices, freq_results[name]['scaled'],
                         'o-', markersize=3, label=labels[name], color=colors[name])
    axes[1].set_xlabel("Frequency dimension index i")
    axes[1].set_ylabel("Inverse frequency theta_i")
    axes[1].set_title("Inverse Frequencies After Scaling")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("NTK-Aware RoPE: Frequency Band Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "frequency_analysis.png", dpi=150)
    plt.close()

    # Plot 2: Position similarity matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_len = 128  # show subset for clarity

    for idx, (name, sim) in enumerate(sim_matrices.items()):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(sim[:plot_len, :plot_len], cmap='RdBu_r', aspect='auto',
                       vmin=-sim[:plot_len, :plot_len].max(),
                       vmax=sim[:plot_len, :plot_len].max())
        ax.set_title(f"{labels[name]}")
        ax.set_xlabel("Position n")
        ax.set_ylabel("Position m")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("RoPE Position Similarity: x^T R(m)^T R(n) x (128x128 subset)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "similarity_matrices.png", dpi=150)
    plt.close()

    # Plot 3: Attention entropy vs sequence length
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in ['no_scaling', 'linear', 'ntk_aware', 'yarn']:
        lengths = test_lengths
        entropies = [attn_results[name][T]['mean'] for T in lengths]
        ax.plot(lengths, entropies, 'o-', label=labels[name], color=colors[name], linewidth=2)

    ax.axvline(x=original_seq_len, color='gray', linestyle='--', alpha=0.5,
               label=f'Original context ({original_seq_len})')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title(f"Attention Entropy at Extended Context (scale={scale_factor}x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "attention_entropy.png", dpi=150)
    plt.close()

    # Plot 4: Relative position preservation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name in ['no_scaling', 'linear', 'ntk_aware', 'yarn']:
        r = rel_pos_results[name]
        dists = r['dists'][:30]
        means = r['mean'][:30]
        stds = r['std'][:30]
        axes[0].plot(dists, means, 'o-', markersize=3, label=labels[name], color=colors[name])

    axes[0].set_xlabel("Relative Distance (n - m)")
    axes[0].set_ylabel("Mean dot product similarity")
    axes[0].set_title("Similarity vs Relative Distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Variance at same distance (lower = better relative position preservation)
    for name in ['no_scaling', 'linear', 'ntk_aware', 'yarn']:
        r = rel_pos_results[name]
        dists = r['dists'][:30]
        stds = r['std'][:30]
        axes[1].plot(dists, stds, 'o-', markersize=3, label=labels[name], color=colors[name])

    axes[1].set_xlabel("Relative Distance (n - m)")
    axes[1].set_ylabel("Std of dot product at same distance")
    axes[1].set_title("Position Encoding Consistency (lower = better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Relative Position Preservation Under Scaling", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "relative_position.png", dpi=150)
    plt.close()

    # Plot 5: Visual comparison of attention patterns (small case)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    T_viz = 128

    torch.manual_seed(42)
    x_viz = torch.randn(1, T_viz, d_model, device=device)

    for idx, (name, rope_cls_name) in enumerate([('no_scaling', 'no_scaling'),
                                                    ('linear', 'linear'),
                                                    ('ntk_aware', 'ntk_aware'),
                                                    ('yarn', 'yarn')]):
        rope_emb = methods[rope_cls_name].to(device)
        use_temp = (rope_cls_name == 'yarn')
        attn_layer = RoPEAttention(d_model, n_heads, rope_emb, use_yarn_temp=use_temp).to(device)
        attn_layer.eval()

        with torch.no_grad():
            _, attn_w = attn_layer(x_viz)

        # Average attention across heads
        attn_avg = attn_w[0].mean(0).cpu().numpy()

        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(attn_avg, cmap='Blues', aspect='auto', vmin=0, vmax=attn_avg.max())
        ax.set_title(f"{labels[name]} (T={T_viz})")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("Attention Patterns Under Different RoPE Scaling", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "attention_patterns.png", dpi=150)
    plt.close()

    # Plot 6: Concept diagram
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    concepts = [
        ("Linear\nScaling", "pos -> pos / s\nAll freqs scaled equally\nHigh-freq resolution lost\n-> Local structure\n   destroyed", 0.1, 'red'),
        ("NTK-Aware\nScaling", "base -> base * s^(d/(d-2))\nHigh freq: barely changed\nLow freq: stretched more\n-> Local structure\n   preserved!", 0.37, 'blue'),
        ("YaRN", "NTK scaling +\nfreq band blending\n+ attention temp adj.\n-> Best of both worlds\n   for long context", 0.63, 'green'),
        ("Key\nInsight", "Nearby tokens need\nprecise position info\n(high freq matters)\nDistant tokens need\ncoarse info only", 0.9, 'purple'),
    ]

    for name, desc, x_pos, color in concepts:
        ax.text(x_pos, 0.78, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Arrows
    for x1, x2 in [(0.22, 0.27), (0.48, 0.53), (0.74, 0.79)]:
        ax.annotate('', xy=(x2, 0.55), xytext=(x1, 0.55),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_title("RoPE Context Extension: From Linear to NTK-Aware to YaRN", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "concept_diagram.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
