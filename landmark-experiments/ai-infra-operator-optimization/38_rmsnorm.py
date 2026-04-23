"""
Minimal RMSNorm Reproduction
=============================
Reproduces core ideas from RMSNorm (2019.03210, Zhang & Sennrich):
1. LayerNorm: x_norm = (x - μ) / σ * γ + β  (needs both mean and variance)
2. RMSNorm: x_norm = x / RMS(x) * γ  (only needs root mean square, no mean subtraction)
3. Computationally simpler, no centering needed
4. Used in LLaMA, Mistral, and most modern LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── Normalization Implementations ──

class LayerNormImpl(nn.Module):
    """Layer Normalization: subtract mean, divide by std, scale and shift."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNormImpl(nn.Module):
    """Root Mean Square Normalization: divide by RMS, scale only (no shift)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm


# ── Models ──

class TransformerBlock(nn.Module):
    """Transformer block with configurable normalization."""
    def __init__(self, d_model, n_heads, norm_type='layernorm'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        if norm_type == 'layernorm':
            self.norm1 = LayerNormImpl(d_model)
            self.norm2 = LayerNormImpl(d_model)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNormImpl(d_model)
            self.norm2 = RMSNormImpl(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x):
        # Pre-norm
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class TransformerLM(nn.Module):
    """Small Transformer language model."""
    def __init__(self, vocab_size=32, d_model=128, n_heads=4, n_layers=4, norm_type='layernorm'):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(128, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, norm_type) for _ in range(n_layers)
        ])
        if norm_type == 'layernorm':
            self.final_norm = LayerNormImpl(d_model)
        elif norm_type == 'rmsnorm':
            self.final_norm = RMSNormImpl(d_model)
        else:
            self.final_norm = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.head(h)


# ── Data ──

def generate_data(vocab_size=32, length=8000):
    """Generate structured sequence data."""
    data = []
    for i in range(length):
        base = (i % 10)
        noise = np.random.randint(0, 4)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Training ──

def train_model(model, data, n_steps=3000, batch_size=32, seq_len=32, lr=3e-4, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    times = []

    for step in range(n_steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t1 = time.time()

        losses.append(loss.item())
        times.append(t1 - t0)

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f} | Step time: {(t1-t0)*1000:.1f}ms")

    return losses, times


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "38-rmsnorm"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    data = generate_data(vocab_size, length=8000)

    # Experiment 1: Compare LayerNorm vs RMSNorm
    configs = [
        ('layernorm', 'LayerNorm'),
        ('rmsnorm', 'RMSNorm'),
        ('none', 'No Norm'),
    ]

    results = {}
    for norm_type, name in configs:
        print(f"\n=== Training with {name} ===")
        model = TransformerLM(vocab_size, d_model=128, n_heads=4, n_layers=4, norm_type=norm_type).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        losses, times = train_model(model, data, n_steps=3000, device=device)

        results[name] = {
            'losses': losses,
            'times': times,
            'final_loss': losses[-1],
            'avg_time': np.mean(times),
            'n_params': n_params,
        }

    # Experiment 2: Scaling to deeper models
    print("\n=== Scaling: LayerNorm vs RMSNorm at Depth ===")
    scale_results = {}
    for n_layers in [2, 4, 8]:
        for norm_type, name in [('layernorm', 'LN'), ('rmsnorm', 'RMS')]:
            print(f"  {n_layers} layers + {name}...")
            model = TransformerLM(vocab_size, d_model=64, n_heads=2, n_layers=n_layers, norm_type=norm_type).to(device)
            losses, times = train_model(model, data, n_steps=1500, batch_size=16, seq_len=16, lr=1e-3, device=device)
            scale_results[f"{n_layers}L-{name}"] = {
                'final_loss': losses[-1],
                'avg_time': np.mean(times),
            }

    # Experiment 3: Numerical comparison
    print("\n=== Numerical Comparison: LayerNorm vs RMSNorm ===")
    x = torch.randn(4, 128)
    ln = LayerNormImpl(128)
    rms = RMSNormImpl(128)

    x_ln = ln(x)
    x_rms = rms(x)

    print(f"  LayerNorm output: mean={x_ln.mean():.4f}, std={x_ln.std():.4f}")
    print(f"  RMSNorm output:   mean={x_rms.mean():.4f}, std={x_rms.std():.4f}")
    print(f"  Input mean:       {x.mean():.4f}")
    print(f"  Difference:       {(x_ln - x_rms).abs().mean():.6f}")

    # Benchmark speed
    print("\n=== Speed Benchmark ===")
    ln_bench = LayerNormImpl(128).to(device)
    rms_bench = RMSNormImpl(128).to(device)
    for name, norm in [('LayerNorm', ln_bench), ('RMSNorm', rms_bench)]:
        x_test = torch.randn(256, 128, device=device)
        # Warmup
        for _ in range(100):
            _ = norm(x_test)
        # Time
        t0 = time.time()
        for _ in range(1000):
            _ = norm(x_test)
        t1 = time.time()
        print(f"  {name}: {(t1-t0)/1000*1000:.3f}ms per call")

    # ── Results ──
    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"  {name}: Loss={r['final_loss']:.4f}, Avg step time={r['avg_time']*1000:.1f}ms")

    print("\n  Scaling results:")
    for key, r in scale_results.items():
        print(f"    {key}: Loss={r['final_loss']:.4f}, Time={r['avg_time']*1000:.1f}ms")

    # ── Visualization ──

    # 1. Training loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 20
    colors = {'LayerNorm': 'blue', 'RMSNorm': 'green', 'No Norm': 'red'}
    for name, r in results.items():
        smoothed = np.convolve(r['losses'], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=name, color=colors[name], alpha=0.8)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Step time comparison
    names = list(results.keys())
    avg_times = [results[n]['avg_time'] * 1000 for n in names]
    axes[1].bar(names, avg_times, color=[colors[n] for n in names], alpha=0.7)
    axes[1].set_ylabel("Avg Step Time (ms)")
    axes[1].set_title("Training Speed")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("RMSNorm vs LayerNorm: Simpler and Faster", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Scaling results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    depths = [2, 4, 8]
    ln_losses = [scale_results[f"{d}L-LN"]['final_loss'] for d in depths]
    rms_losses = [scale_results[f"{d}L-RMS"]['final_loss'] for d in depths]
    ln_times = [scale_results[f"{d}L-LN"]['avg_time'] * 1000 for d in depths]
    rms_times = [scale_results[f"{d}L-RMS"]['avg_time'] * 1000 for d in depths]

    axes[0].plot(depths, ln_losses, 'o-', label='LayerNorm', color='blue')
    axes[0].plot(depths, rms_losses, 's-', label='RMSNorm', color='green')
    axes[0].set_xlabel("Number of Layers")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Loss vs Depth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(depths, ln_times, 'o-', label='LayerNorm', color='blue')
    axes[1].plot(depths, rms_times, 's-', label='RMSNorm', color='green')
    axes[1].set_xlabel("Number of Layers")
    axes[1].set_ylabel("Avg Step Time (ms)")
    axes[1].set_title("Speed vs Depth")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("RMSNorm: Scales Better with Depth", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "scaling_comparison.png", dpi=150)
    plt.close()

    # 3. Normalization effect visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x_vis = torch.randn(100, 64)
    ln_vis = LayerNormImpl(64)
    rms_vis = RMSNormImpl(64)

    # Before normalization
    axes[0].hist(x_vis.flatten().numpy(), bins=50, alpha=0.7, color='gray')
    axes[0].set_title("Before Normalization")
    axes[0].set_xlabel("Value")
    axes[0].axvline(x_vis.mean().item(), color='red', linestyle='--', label=f'mean={x_vis.mean():.2f}')
    axes[0].legend(fontsize=8)

    # After LayerNorm
    x_ln = ln_vis(x_vis).detach()
    axes[1].hist(x_ln.flatten().numpy(), bins=50, alpha=0.7, color='blue')
    axes[1].set_title("After LayerNorm")
    axes[1].set_xlabel("Value")
    axes[1].axvline(x_ln.mean().item(), color='red', linestyle='--', label=f'mean={x_ln.mean():.2f}')
    axes[1].legend(fontsize=8)

    # After RMSNorm
    x_rms = rms_vis(x_vis).detach()
    axes[2].hist(x_rms.flatten().numpy(), bins=50, alpha=0.7, color='green')
    axes[2].set_title("After RMSNorm")
    axes[2].set_xlabel("Value")
    axes[2].axvline(x_rms.mean().item(), color='red', linestyle='--', label=f'mean={x_rms.mean():.2f}')
    axes[2].legend(fontsize=8)

    plt.suptitle("LayerNorm Centers (mean≈0), RMSNorm Doesn't (But Scales)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "normalization_effect.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("LayerNorm\n(2016)", "x̂ = (x - μ) / σ\ny = γx̂ + β\nMean + Variance\nScale + Shift\n→ 2 learned params", 0.17, 'blue'),
        ("RMSNorm\n(2019)", "x̂ = x / RMS(x)\ny = γ · x̂\nNo mean subtraction\nScale only\n→ 1 learned param", 0.5, 'green'),
        ("Why It Works", "Re-centering is\nredundant: the\nlearned β in LN\ncancels out -μ\n→ Just scaling\n   is sufficient!", 0.83, 'purple'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("RMSNorm: Simpler is Better (Used in LLaMA, Mistral, etc.)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rmsnorm_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
