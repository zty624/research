"""
Minimal RoPE (Rotary Position Embedding) Reproduction
=====================================================
Reproduces core ideas from RoPE (2104.09864, Su et al.):
1. Encode position through rotation in 2D subspaces
2. q_m · k_n depends only on relative position (m - n)
3. No learned parameters for position encoding
4. Compare: sinusoidal, learned, RoPE position encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── RoPE Implementation ──

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        # Compute inverse frequencies: θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

        # Precompute for max_seq_len
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        self.register_buffer('cos_cached', freqs.cos())  # (max_seq_len, dim/2)
        self.register_buffer('sin_cached', freqs.sin())  # (max_seq_len, dim/2)

    def forward(self, x, seq_len=None):
        """Return cos and sin for positions 0..seq_len-1."""
        seq_len = seq_len or x.shape[1]
        return (
            self.cos_cached[:seq_len].unsqueeze(0),  # (1, T, D/2)
            self.sin_cached[:seq_len].unsqueeze(0),   # (1, T, D/2)
        )


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding to input tensor x.
    x: (B, T, D) where D is even
    cos, sin: (1, T, D/2)
    """
    d = x.shape[-1]
    x1 = x[..., :d//2]  # first half
    x2 = x[..., d//2:]  # second half

    # Rotation: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    return rotated


# ── Attention with Different Position Encodings ──

class AttentionSinusoidal(nn.Module):
    """Multi-head attention with sinusoidal position encoding."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        # Sinusoidal position encoding
        pe = torch.zeros(512, d_model)
        position = torch.arange(0, 512).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pe[:, :T, :]  # Add position encoding

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class AttentionLearned(nn.Module):
    """Multi-head attention with learned position embedding."""
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class AttentionRoPE(nn.Module):
    """Multi-head attention with Rotary Position Embedding."""
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.d_head, max_len)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to q and k
        cos, sin = self.rope(x, T)
        # cos, sin: (1, T, D_head/2) → expand to (B, H, T, D_head/2)
        cos = cos.unsqueeze(1).expand(B, self.n_heads, -1, -1)
        sin = sin.unsqueeze(1).expand(B, self.n_heads, -1, -1)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Transformer Models ──

class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2, pos_type='rope'):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)

        if pos_type == 'sinusoidal':
            attn_cls = lambda: AttentionSinusoidal(d_model, n_heads)
        elif pos_type == 'learned':
            attn_cls = lambda: AttentionLearned(d_model, n_heads)
        elif pos_type == 'rope':
            attn_cls = lambda: AttentionRoPE(d_model, n_heads)
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': attn_cls(),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model)),
                'norm2': nn.LayerNorm(d_model),
            }))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x)
        for layer in self.layers:
            h = h + layer['attn'](layer['norm1'](h))
            h = h + layer['ff'](layer['norm2'](h))
        return self.head(self.final_norm(h))


# ── Data ──

def generate_copy_task_data(vocab_size=8, seq_len=32, n_samples=5000):
    """Copy task: repeat a sequence. Tests position awareness."""
    sequences = torch.randint(1, vocab_size, (n_samples, seq_len))
    # Input: [seq] [0-pad]
    # Output: [0-pad] [seq]  (copy after seeing full sequence)
    sep_token = vocab_size  # separator
    input_seq = torch.cat([sequences, torch.full((n_samples, 4), sep_token, dtype=torch.long)], dim=1)
    target_seq = torch.cat([torch.full((n_samples, seq_len + 4), 0, dtype=torch.long), sequences], dim=1)
    return input_seq, target_seq


def generate_structured_data(vocab_size=32, length=8000):
    """Structured sequence data for language modeling."""
    data = []
    for i in range(length):
        base = (i % 10)
        noise = np.random.randint(0, 3)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Training ──

def train_lm(model, data, n_steps=3000, batch_size=32, seq_len=32, lr=3e-4, device='cpu'):
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
    results_dir = Path(__file__).parent / "results" / "39-rope"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    data = generate_structured_data(vocab_size, length=8000)

    # Experiment 1: Compare position encodings
    print("=== Position Encoding Comparison ===")
    pos_types = ['sinusoidal', 'learned', 'rope']
    results = {}

    for pos_type in pos_types:
        print(f"\n  {pos_type}:")
        model = TransformerLM(vocab_size, d_model=64, n_heads=2, n_layers=2, pos_type=pos_type).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")
        losses = train_lm(model, data, n_steps=3000, device=device)
        results[pos_type] = {'losses': losses, 'final_loss': losses[-1], 'params': n_params}

    # Experiment 2: Length extrapolation
    print("\n=== Length Extrapolation ===")
    # Train on short sequences, test on longer ones
    seq_len_train = 16

    extrap_results = {}
    for pos_type in pos_types:
        print(f"\n  {pos_type}:")
        model = TransformerLM(vocab_size, d_model=64, n_heads=2, n_layers=2, pos_type=pos_type).to(device)
        # Train on short sequences
        _ = train_lm(model, data, n_steps=2000, seq_len=seq_len_train, device=device)

        # Test on various lengths
        length_accs = {}
        model.eval()
        for test_len in [16, 24, 32, 48, 64]:
            correct = total = 0
            with torch.no_grad():
                for _ in range(50):
                    start = torch.randint(0, len(data) - test_len - 1, (1,)).item()
                    x = data[start:start+test_len].unsqueeze(0).to(device)
                    y = data[start+1:start+test_len+1].unsqueeze(0).to(device)
                    if test_len > 512:
                        continue
                    try:
                        logits = model(x)
                        pred = logits.argmax(-1)
                        correct += (pred[0, seq_len_train:] == y[0, seq_len_train:]).sum().item()
                        total += (test_len - seq_len_train)
                    except:
                        break
            acc = correct / max(total, 1)
            length_accs[test_len] = acc
            print(f"    Len {test_len}: {acc:.4f}")

        extrap_results[pos_type] = length_accs

    # Experiment 3: Verify relative position property
    print("\n=== RoPE Relative Position Property ===")
    d_head = 64
    rope = RotaryEmbedding(d_head, max_seq_len=100).to(device)

    # Compute q·k^T for different positions
    q = torch.randn(1, 1, d_head, device=device)  # single query
    k = torch.randn(1, 1, d_head, device=device)  # single key

    cos_q, sin_q = rope(q, 100)
    cos_k, sin_k = rope(q, 100)

    # Apply RoPE at different positions
    cos_q_h = cos_q.unsqueeze(1).expand(1, 1, -1, -1)[:, 0, :, :]  # (1, T, D/2)
    sin_q_h = sin_q.unsqueeze(1).expand(1, 1, -1, -1)[:, 0, :, :]
    cos_k_h = cos_k.unsqueeze(1).expand(1, 1, -1, -1)[:, 0, :, :]
    sin_k_h = sin_k.unsqueeze(1).expand(1, 1, -1, -1)[:, 0, :, :]

    # Compute dot products at different relative positions
    rel_dots = []
    for m in range(0, 10):
        for n in range(m, m + 20):
            q_m = apply_rotary_emb(q.expand(1, 1, -1), cos_q_h[:, m:m+1, :], sin_q_h[:, m:m+1, :])
            k_n = apply_rotary_emb(k.expand(1, 1, -1), cos_k_h[:, n:n+1, :], sin_k_h[:, n:n+1, :])
            dot = (q_m * k_n).sum().item()
            rel_dots.append((n - m, dot))

    # Group by relative distance
    from collections import defaultdict
    dist_dots = defaultdict(list)
    for dist, dot in rel_dots:
        dist_dots[dist].append(dot)

    dists = sorted(dist_dots.keys())
    avg_dots = [np.mean(dist_dots[d]) for d in dists]
    std_dots = [np.std(dist_dots[d]) for d in dists]

    print(f"  q·k at relative distance 0: {avg_dots[0]:.4f} ± {std_dots[0]:.4f}")
    print(f"  q·k at relative distance 5: {avg_dots[5]:.4f} ± {std_dots[5]:.4f}")
    print(f"  q·k at relative distance 10: {avg_dots[10]:.4f} ± {std_dots[10]:.4f}")
    print(f"  Variance of q·k at same distance: {np.mean(std_dots):.4f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'sinusoidal': 'orange', 'learned': 'red', 'rope': 'blue'}
    window = 20
    for pos_type, r in results.items():
        smoothed = np.convolve(r['losses'], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=pos_type.capitalize(), color=colors[pos_type])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final loss comparison
    names = list(results.keys())
    final_losses = [results[n]['final_loss'] for n in names]
    axes[1].bar([n.capitalize() for n in names], final_losses,
                color=[colors[n] for n in names], alpha=0.7)
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Final Training Loss")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("RoPE: Position Encoding Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Length extrapolation
    fig, ax = plt.subplots(figsize=(10, 5))
    test_lengths = [16, 24, 32, 48, 64]
    for pos_type in pos_types:
        accs = [extrap_results[pos_type].get(l, 0) for l in test_lengths]
        ax.plot(test_lengths, accs, 'o-', label=pos_type.capitalize(), color=colors[pos_type])
    ax.axvline(x=seq_len_train, color='gray', linestyle='--', label='Train length')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Token Accuracy")
    ax.set_title("Length Extrapolation (trained on len=16)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "extrapolation.png", dpi=150)
    plt.close()

    # 3. Relative position property
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].errorbar(dists, avg_dots, yerr=std_dots, fmt='o-', capsize=3, color='blue')
    axes[0].set_xlabel("Relative Position (n - m)")
    axes[0].set_ylabel("q_m · k_n (dot product)")
    axes[0].set_title("RoPE: q·k Depends Only on Relative Position")
    axes[0].grid(True, alpha=0.3)

    # Rotation visualization
    angles = np.linspace(0, 2 * np.pi, 100)
    r = 1.0
    axes[1].plot(r * np.cos(angles), r * np.sin(angles), 'gray', alpha=0.3)

    # Show rotations at different positions
    for pos in [0, 1, 5, 10]:
        theta = pos * 0.3  # simplified rotation angle
        x_rot = r * np.cos(theta)
        y_rot = r * np.sin(theta)
        axes[1].plot(x_rot, y_rot, 'o', markersize=8, label=f'pos={pos}')
        axes[1].annotate(f'pos={pos}', (x_rot, y_rot), textcoords="offset points",
                        xytext=(5, 5))

    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_aspect('equal')
    axes[1].set_title("RoPE: Rotation in 2D Subspace")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("RoPE: Relative Position Encoding via Rotation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "relative_position.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Absolute\nPosition", "Add position\nto input: x + PE\nSinusoidal or learned\n→ Position mixed\n   with content", 0.14, 'orange'),
        ("RoPE\n(Relative)", "Rotate q and k:\nq_m → R(θm)q_m\nk_n → R(θn)k_n\n→ q·k = f(m-n)\n   Relative by design!", 0.5, 'blue'),
        ("Key Property", "Inner product\ndepends only on\nrelative position\n→ Better length\n   extrapolation", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("RoPE: Position Encoding via Rotation (LLaMA, Mistral, etc.)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rope_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
