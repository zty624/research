"""
Minimal SwiGLU / GLU Activation Reproduction
=============================================
Reproduces core ideas from GLU Variants (2002.05202, Shazeer) and GLU (1705.03122, Dauphin et al.):
1. Gated Linear Unit: σ(Wx+b) ⊗ (Vx+c) — gate controls information flow
2. SwiGLU: Swish(xW) ⊗ (xV) — used in LLaMA, PaLM, etc.
3. Compare: ReLU, GELU, Swish, GLU, SwiGLU in Transformer FFN
4. Key insight: gating improves over plain activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Activation Functions ──

class Swish(nn.Module):
    """Swish/SiLU: x * σ(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GELUExact(nn.Module):
    """GELU exact: x * Φ(x)"""
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ── FFN Variants ──

class ReLUFFN(nn.Module):
    """Standard FFN: W2(ReLU(W1(x)))"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class GELUFFN(nn.Module):
    """GELU FFN: W2(GELU(W1(x))) — GPT-2/3 style"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class SwishFFN(nn.Module):
    """Swish/SiLU FFN: W2(Swish(W1(x)))"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = Swish()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class GLUFFN(nn.Module):
    """GLU FFN: W2(σ(W1(x)) ⊗ W3(x)) — Gated Linear Unit"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff)  # gate
        self.w3 = nn.Linear(d_model, d_ff)  # value
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate = torch.sigmoid(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: W2(Swish(W1(x)) ⊗ W3(x)) — LLaMA style"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff)  # gate
        self.w3 = nn.Linear(d_model, d_ff)  # value
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = Swish()

    def forward(self, x):
        gate = self.act(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class GeGLUFFN(nn.Module):
    """GeGLU FFN: W2(GELU(W1(x)) ⊗ W3(x))"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate = F.gelu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


# ── Transformer ──

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_type='swiglu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        ffn_map = {
            'relu': ReLUFFN,
            'gelu': GELUFFN,
            'swish': SwishFFN,
            'glu': GLUFFN,
            'swiglu': SwiGLUFFN,
            'geglu': GeGLUFFN,
        }
        self.ffn = ffn_map[ffn_type](d_model)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32, d_model=128, n_heads=4, n_layers=4, ffn_type='swiglu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(128, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_type) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            # Use causal mask via attention
            residual = h
            h_norm = layer.norm1(h)
            attn_out, _ = layer.attn(h_norm, h_norm, h_norm, attn_mask=mask)
            h = residual + attn_out
            h = h + layer.ffn(layer.norm2(h))
        return self.head(self.norm(h))


# ── Data ──

def generate_data(vocab_size=32, length=8000):
    data = []
    for i in range(length):
        base = (i % 10)
        noise = np.random.randint(0, 3)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Training ──

def train_model(model, data, n_steps=3000, batch_size=32, seq_len=32, lr=3e-4, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    for step in range(n_steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "40-swiglu"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    data = generate_data(vocab_size, length=8000)

    # Experiment 1: FFN activation comparison
    print("=== FFN Activation Comparison ===")
    ffn_types = ['relu', 'gelu', 'swish', 'glu', 'swiglu', 'geglu']
    results = {}

    for ffn_type in ffn_types:
        print(f"\n  {ffn_type.upper()}:")
        model = TransformerLM(vocab_size, d_model=128, n_heads=4, n_layers=4, ffn_type=ffn_type).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")
        losses = train_model(model, data, n_steps=3000, device=device)
        results[ffn_type] = {
            'losses': losses,
            'final_loss': losses[-1],
            'params': n_params,
        }

    # Experiment 2: Activation function visualization
    print("\n=== Activation Function Properties ===")
    x = torch.linspace(-4, 4, 200)
    activations = {
        'ReLU': F.relu(x),
        'GELU': F.gelu(x),
        'Swish': x * torch.sigmoid(x),
        'Sigmoid': torch.sigmoid(x),
    }
    for name, act in activations.items():
        print(f"  {name}: range=[{act.min():.2f}, {act.max():.2f}], "
              f"smooth={'Yes' if name != 'ReLU' else 'No (not differentiable at 0)'}")

    # ── Results ──
    print("\n=== Summary ===")
    for ffn_type, r in results.items():
        print(f"  {ffn_type:8s}: Loss={r['final_loss']:.4f}, Params={r['params']:,}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'relu': 'gray', 'gelu': 'blue', 'swish': 'cyan',
              'glu': 'orange', 'swiglu': 'green', 'geglu': 'purple'}
    window = 20
    for ffn_type, r in results.items():
        smoothed = np.convolve(r['losses'], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=ffn_type.upper(), color=colors[ffn_type], alpha=0.8)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Final loss bar chart
    names = list(results.keys())
    final_losses = [results[n]['final_loss'] for n in names]
    bars = axes[1].bar([n.upper() for n in names], final_losses,
                       color=[colors[n] for n in names], alpha=0.7)
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Final Training Loss")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, final_losses):
        axes[1].text(bar.get_x() + bar.get_width()/2, loss + 0.005,
                    f'{loss:.3f}', ha='center', fontsize=7)

    plt.suptitle("SwiGLU: Gated Activations Outperform Plain Ones", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Activation function shapes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = torch.linspace(-4, 4, 200).numpy()

    # Plain activations
    axes[0].plot(x, np.maximum(0, x), label='ReLU', color='gray', linewidth=2)
    axes[0].plot(x, F.gelu(torch.tensor(x)).numpy(), label='GELU', color='blue')
    axes[0].plot(x, x / (1 + np.exp(-x)), label='Swish', color='cyan')
    axes[0].set_title("Plain Activations")
    axes[0].set_xlabel("x")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gating: sigmoid and swish as gates
    axes[1].plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid (GLU gate)', color='orange')
    axes[1].plot(x, x / (1 + np.exp(-x)), label='Swish (SwiGLU gate)', color='green')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title("Gate Functions")
    axes[1].set_xlabel("x")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gated output: gate * value
    value = np.sin(x * 2)  # arbitrary value function
    axes[2].plot(x, value, label='Value (Vx)', color='gray', alpha=0.5)
    axes[2].plot(x, 1 / (1 + np.exp(-x)) * value, label='GLU: σ(x)·Vx', color='orange')
    axes[2].plot(x, x / (1 + np.exp(-x)) * value, label='SwiGLU: Swish(x)·Vx', color='green')
    axes[2].set_title("Gated Output: gate ⊗ value")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Activation Functions: Plain vs Gated", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "activation_comparison.png", dpi=150)
    plt.close()

    # 3. Parameter comparison (GLU variants have 3 weight matrices vs 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    param_counts = {n: results[n]['params'] for n in names}
    ax.bar([n.upper() for n in names], list(param_counts.values()),
           color=[colors[n] for n in names], alpha=0.7)
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count (GLU variants have 3 weight matrices)")
    ax.grid(True, alpha=0.3, axis='y')
    for i, (n, p) in enumerate(param_counts.items()):
        ax.text(i, p + 1000, f'{p:,}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "param_comparison.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("ReLU\nFFN", "y = W₂(ReLU(W₁x))\n2 weight matrices\nSimple gating:\non/off only", 0.14, 'gray'),
        ("GLU\n(2017)", "y = W₂(σ(W₁x) ⊗ W₃x)\n3 weight matrices\nSmooth gating\nvia sigmoid", 0.42, 'orange'),
        ("SwiGLU\n(2020)", "y = W₂(Swish(W₁x) ⊗ W₃x)\n3 weight matrices\nNon-monotonic gate\n→ LLaMA standard", 0.71, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("SwiGLU: Gating Improves Transformers (LLaMA, PaLM, etc.)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "swiglu_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
