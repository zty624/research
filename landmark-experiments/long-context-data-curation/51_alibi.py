"""
Minimal ALiBi Reproduction
============================
Reproduces core ideas from ALiBi (1910.03193, Press et al.):
1. No positional embeddings needed — just add linear bias to attention scores
2. Attention(bi) = qi·kj + m·(i-j) where m is a head-specific slope
3. Slopes form a geometric sequence: m_h = 2^(-8h/H)
4. Enables length extrapolation: train on short, test on long sequences
5. At 1.3B params: matches learned positional embeddings, extrapolates to 2x length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── ALiBi Attention ──

class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi (Attention with Linear Biases)."""
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        # ALiBi slopes: geometric sequence 2^(-8h/H) for h = 1, ..., H
        # These are NOT learned — fixed at initialization
        slopes = 2.0 ** (-8.0 * torch.arange(1, n_heads + 1) / n_heads)
        self.register_buffer('slopes', slopes)

        # Precompute causal mask and position differences
        self._precompute(max_len)

    def _precompute(self, max_len):
        # Position differences: i - j (positive = looking backward)
        pos = torch.arange(max_len)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
        # Causal mask: can only attend to j <= i
        causal = diff >= 0
        self.register_buffer('causal_mask', causal)
        self.register_buffer('pos_diff', diff.float())

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Add ALiBi bias: m_h * (i - j) for each head
        # slopes: (H,), pos_diff: (T, T) → bias: (H, T, T)
        # For extrapolation: T can be > max_len, compute on the fly
        if T <= self.pos_diff.shape[0]:
            alibi_bias = self.slopes.unsqueeze(1).unsqueeze(2) * self.pos_diff[:T, :T].unsqueeze(0)
            causal = self.causal_mask[:T, :T]
        else:
            pos = torch.arange(T, device=x.device)
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            alibi_bias = self.slopes.unsqueeze(1).unsqueeze(2) * diff.float().unsqueeze(0)
            causal = diff >= 0

        attn = attn + alibi_bias
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Standard Attention with Learned Position Embedding ──

class LearnedPosAttention(nn.Module):
    """Standard multi-head attention with learned positional embeddings."""
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        # Add positional embedding to input
        pos = self.pos_emb(torch.arange(T, device=x.device))
        x = x + pos

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        attn = attn.masked_fill(causal, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Sinusoidal Position Attention ──

class SinusoidalPosAttention(nn.Module):
    """Standard attention with sinusoidal positional embeddings."""
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        # Sinusoidal embeddings (fixed, not learned)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pe[:T].unsqueeze(0)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        attn = attn.masked_fill(causal, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Transformer Blocks ──

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_cls, max_len=512, **attn_kwargs):
        super().__init__()
        self.attn = attn_cls(d_model, n_heads, max_len, **attn_kwargs)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        h = h + self.ff(self.norm2(h))
        return h


class LM(nn.Module):
    """Simple language model with configurable attention type."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 attn_type='alibi', max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)

        attn_map = {
            'alibi': ALiBiAttention,
            'learned': LearnedPosAttention,
            'sinusoidal': SinusoidalPosAttention,
        }
        attn_cls = attn_map[attn_type]

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, attn_cls, max_len)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.attn_type = attn_type

    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)


# ── Data ──

def generate_sequence(batch_size, seq_len, vocab_size=20):
    """Simple sequence modeling: next-token prediction on periodic sequences.
    Pattern: [a, b, c, a, b, c, ...] with random period 2-5.
    Requires understanding relative position to predict next token.
    """
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(batch_size):
        period = np.random.randint(2, 6)
        pattern = torch.randint(1, vocab_size, (period,))
        for j in range(seq_len):
            sequences[i, j] = pattern[j % period]
    # Input and target (shift by 1)
    x = sequences[:, :-1]
    y = sequences[:, 1:]
    return x, y


# ── Training ──

def train_lm(model, vocab_size, seq_len, n_steps=3000, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    for step in range(n_steps):
        x, y = generate_sequence(64, seq_len + 1, vocab_size=vocab_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")

    return losses


def evaluate_extrapolation(model, vocab_size, train_len, test_lens, device='cpu'):
    """Test on sequences longer than training length."""
    results = {}
    model.eval()
    with torch.no_grad():
        for test_len in test_lens:
            x, y = generate_sequence(128, test_len + 1, vocab_size=vocab_size)
            x, y = x.to(device), y.to(device)

            try:
                logits = model(x)
                pred = logits.argmax(-1)
                acc = (pred == y).float().mean().item()
                loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                                       y.reshape(-1)).item()
                results[test_len] = {'acc': acc, 'loss': loss, 'success': True}
            except Exception as e:
                results[test_len] = {'acc': 0, 'loss': float('inf'), 'success': False}
    model.train()
    return results


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "51-alibi"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 20
    train_len = 64

    # Experiment 1: Train different position encodings, compare training
    print("=== Training Comparison ===")
    attn_types = ['alibi', 'learned', 'sinusoidal']
    train_results = {}

    for attn_type in attn_types:
        print(f"\n  {attn_type}:")
        model = LM(vocab_size, d_model=64, n_heads=4, n_layers=2,
                    attn_type=attn_type, max_len=256).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")
        losses = train_lm(model, vocab_size, train_len, n_steps=3000, device=device)
        train_results[attn_type] = {'losses': losses, 'final_loss': losses[-1]}

        # Extrapolation test
        extrap = evaluate_extrapolation(model, vocab_size, train_len,
                                         [64, 96, 128, 160], device)
        train_results[attn_type]['extrap'] = extrap
        for sl, r in extrap.items():
            print(f"      Len={sl}: acc={r['acc']:.4f}" if r['success'] else f"      Len={sl}: FAILED")

    # Experiment 2: ALiBi slope visualization
    print("\n=== ALiBi Slope Analysis ===")
    n_heads = 8
    slopes = 2.0 ** (-8.0 * torch.arange(1, n_heads + 1) / n_heads)
    print(f"  Slopes: {slopes.numpy()}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'alibi': 'blue', 'learned': 'red', 'sinusoidal': 'green'}
    for attn_type, r in train_results.items():
        axes[0].plot(r['losses'], label=attn_type.capitalize(), color=colors[attn_type])

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss (train length=64)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Length extrapolation
    test_lens = [64, 96, 128, 160]
    for attn_type, r in train_results.items():
        accs = [r['extrap'].get(sl, {}).get('acc', 0) for sl in test_lens]
        axes[1].plot(test_lens, accs, 'o-', label=attn_type.capitalize(), color=colors[attn_type])

    axes[1].axvline(x=train_len, color='gray', linestyle='--', alpha=0.5, label='Train length')
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Copy Accuracy")
    axes[1].set_title("Length Extrapolation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ALiBi: Attention with Linear Biases for Length Extrapolation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "alibi_comparison.png", dpi=150)
    plt.close()

    # 3. ALiBi bias pattern
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    T = 32

    for idx in range(min(n_heads, 8)):
        ax = axes[idx // 4, idx % 4]
        slope = slopes[idx].item()

        # Create ALiBi bias for this head
        pos = torch.arange(T)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        bias = slope * diff.float()

        # Apply causal mask
        causal = diff >= 0
        bias = bias.masked_fill(~causal, float('-inf'))

        im = ax.imshow(bias.numpy(), cmap='RdBu_r', aspect='auto')
        ax.set_title(f"Head {idx+1}: m={slope:.4f}")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("ALiBi: Per-Head Linear Bias Patterns", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "alibi_bias_patterns.png", dpi=150)
    plt.close()

    # 4. Slope distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    head_indices = np.arange(1, n_heads + 1)
    ax.bar(head_indices, slopes.numpy(), color='blue', alpha=0.7)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Slope (m)")
    ax.set_title("ALiBi: Slope per Head (Geometric Sequence)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "alibi_slopes.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Learned\nPosition", "Add position embedding\nto input representation\nFixed max length\nCan't extrapolate\n→ Train len = test len", 0.14, 'red'),
        ("ALiBi\nBias", "No position embedding\nAdd m·(i-j) to attention\nSlopes: 2^(-8h/H)\nFixed, not learned\n→ Extrapolates!", 0.5, 'blue'),
        ("Why It\nWorks", "Linear bias penalizes\ndistant positions\nproportionally\nSloper decay gracefully\n→ Smooth length gen.", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("ALiBi: No Positional Embeddings, Just Linear Biases", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "alibi_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
