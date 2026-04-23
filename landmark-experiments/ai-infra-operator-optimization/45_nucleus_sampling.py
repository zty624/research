"""
Minimal Nucleus Sampling / Temperature Reproduction
=====================================================
Reproduces core ideas from Nucleus Sampling (1904.09751, Holtzman et al.):
1. Temperature: control sharpness of output distribution
2. Top-k: restrict to k most likely tokens
3. Top-p (nucleus): restrict to smallest set with cumulative prob ≥ p
4. Compare: greedy vs temperature sampling vs top-k vs top-p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Simple LM ──

class SimpleLM(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask)
        return self.head(self.norm(h))


# ── Sampling Methods ──

def sample_greedy(logits):
    """Greedy: always pick the most likely token."""
    return logits.argmax(dim=-1)


def sample_temperature(logits, temperature=1.0):
    """Temperature scaling: sharp (T<1) or flat (T>1)."""
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def sample_top_k(logits, k=10):
    """Top-k: restrict to k most likely tokens."""
    top_k = min(k, logits.size(-1))
    values, indices = torch.topk(logits, top_k, dim=-1)
    probs = F.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, 1).squeeze(-1)
    return indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def sample_top_p(logits, p=0.9):
    """Nucleus (top-p): smallest set with cumulative prob ≥ p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Keep at least one token
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_logits[sorted_mask] = float('-inf')

    # Sample from filtered distribution
    probs = F.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1).squeeze(-1)
    return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def sample_top_p_with_temp(logits, p=0.9, temperature=1.0):
    """Nucleus sampling with temperature."""
    logits = logits / temperature
    return sample_top_p(logits, p)


# ── Data ──

def generate_data(vocab_size=32, length=8000):
    data = []
    for i in range(length):
        base = (i % 10)
        noise = np.random.randint(0, 3)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Training ──

def train_model(model, data, n_steps=3000, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    seq_len = 32
    batch_size = 32

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

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f}")


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "45-nucleus-sampling"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    data = generate_data(vocab_size)

    # Train model
    print("=== Training Model ===")
    model = SimpleLM(vocab_size).to(device)
    train_model(model, data, device=device)

    # ── Sampling Analysis ──
    print("\n=== Sampling Analysis ===")
    model.eval()
    prompt = data[:16].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(prompt)
        next_logits = logits[0, -1, :].cpu()

    probs = F.softmax(next_logits, dim=-1).numpy()

    # Experiment 1: Temperature effect
    print("\n  Temperature effect on distribution:")
    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        temp_probs = F.softmax(next_logits / temp, dim=-1).numpy()
        entropy = -np.sum(temp_probs * np.log(temp_probs + 1e-10))
        top1_prob = temp_probs.max()
        print(f"    T={temp}: entropy={entropy:.2f}, top-1 prob={top1_prob:.3f}")

    # Experiment 2: Top-k vs Top-p comparison
    print("\n  Top-k effective vocabulary:")
    for k in [1, 5, 10, 20, 32]:
        top_k_probs = probs.copy()
        top_k_idx = np.argsort(top_k_probs)[::-1]
        mask = np.zeros_like(top_k_probs, dtype=bool)
        mask[top_k_idx[:k]] = True
        top_k_probs[~mask] = 0
        top_k_probs /= top_k_probs.sum()
        print(f"    k={k}: covers {top_k_probs.sum():.4f} probability mass")

    print("\n  Top-p effective vocabulary:")
    for p in [0.5, 0.8, 0.9, 0.95, 1.0]:
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        n_tokens = np.searchsorted(cumsum, p) + 1
        print(f"    p={p}: {n_tokens} tokens needed")

    # Experiment 3: Generation quality
    print("\n=== Generation Quality ===")
    n_gen = 30
    n_runs = 50

    sampling_methods = {
        'Greedy': lambda l: sample_greedy(l.unsqueeze(0)),
        'T=0.5': lambda l: sample_temperature(l.unsqueeze(0), 0.5),
        'T=1.0': lambda l: sample_temperature(l.unsqueeze(0), 1.0),
        'T=2.0': lambda l: sample_temperature(l.unsqueeze(0), 2.0),
        'Top-k=5': lambda l: sample_top_k(l.unsqueeze(0), 5),
        'Top-p=0.9': lambda l: sample_top_p(l.unsqueeze(0), 0.9),
        'Nucleus+T=0.7': lambda l: sample_top_p_with_temp(l.unsqueeze(0), 0.9, 0.7),
    }

    diversity_results = {}
    for method_name, sample_fn in sampling_methods.items():
        all_tokens = []
        for _ in range(n_runs):
            x = prompt.clone()
            gen_tokens = []
            for _ in range(n_gen):
                with torch.no_grad():
                    logits = model(x)
                    next_logits = logits[0, -1, :]
                    next_token = sample_fn(next_logits).item()
                gen_tokens.append(next_token)
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

            all_tokens.append(gen_tokens)

        # Compute diversity metrics
        unique_ratio = len(set(t for ts in all_tokens for t in ts)) / vocab_size
        avg_unique_per_gen = np.mean([len(set(ts)) for ts in all_tokens]) / n_gen

        diversity_results[method_name] = {
            'unique_ratio': unique_ratio,
            'avg_unique_per_gen': avg_unique_per_gen,
            'all_tokens': all_tokens,
        }
        print(f"  {method_name:15s}: vocab coverage={unique_ratio:.2%}, avg unique/gen={avg_unique_per_gen:.2%}")

    # ── Visualization ──

    # 1. Temperature effect on distribution
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    for idx, temp in enumerate(temps):
        temp_probs = F.softmax(next_logits / temp, dim=-1).cpu().numpy()
        axes[idx].bar(range(vocab_size), temp_probs, alpha=0.7,
                     color=['red', 'orange', 'green', 'blue', 'purple'][idx])
        axes[idx].set_title(f"T = {temp}")
        axes[idx].set_xlabel("Token")
        axes[idx].set_ylim(0, max(temp_probs) * 1.2)
        if idx == 0:
            axes[idx].set_ylabel("Probability")

    plt.suptitle("Temperature: Control Distribution Sharpness", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "temperature_effect.png", dpi=150)
    plt.close()

    # 2. Top-p (nucleus) visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Show cumulative probability
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    axes[0].plot(range(len(sorted_probs)), cumsum, color='blue', linewidth=2)
    axes[0].axhline(y=0.9, color='red', linestyle='--', label='p=0.9')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', label='p=0.5')
    axes[0].set_xlabel("Token Rank")
    axes[0].set_ylabel("Cumulative Probability")
    axes[0].set_title("Nucleus: Cumulative Probability of Sorted Tokens")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Effective vocabulary size for different p
    p_values = np.linspace(0.1, 1.0, 50)
    n_tokens_list = []
    for p in p_values:
        n_t = np.searchsorted(cumsum, p) + 1
        n_tokens_list.append(n_t)

    axes[1].plot(p_values, n_tokens_list, color='green', linewidth=2)
    axes[1].set_xlabel("Threshold p")
    axes[1].set_ylabel("Number of Tokens in Nucleus")
    axes[1].set_title("Nucleus Size vs p Threshold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Nucleus Sampling: Adaptive Vocabulary Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "nucleus_visualization.png", dpi=150)
    plt.close()

    # 3. Diversity comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = list(diversity_results.keys())
    coverage = [diversity_results[m]['unique_ratio'] for m in methods]
    unique_gen = [diversity_results[m]['avg_unique_per_gen'] for m in methods]
    colors = ['gray', 'red', 'green', 'orange', 'cyan', 'blue', 'purple']

    axes[0].barh(methods, coverage, color=colors[:len(methods)], alpha=0.7)
    axes[0].set_xlabel("Vocabulary Coverage")
    axes[0].set_title("Token Diversity")

    axes[1].barh(methods, unique_gen, color=colors[:len(methods)], alpha=0.7)
    axes[1].set_xlabel("Avg Unique Tokens / Generation")
    axes[1].set_title("Per-Generation Diversity")

    plt.suptitle("Sampling Strategy: Quality vs Diversity Trade-off", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "diversity_comparison.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Greedy", "Always pick\nmost likely\n→ Repetitive\n→ Boring", 0.1, 'gray'),
        ("Temperature", "Scale logits\nT<1: sharper\nT>1: flatter\n→ Control randomness", 0.35, 'orange'),
        ("Top-k", "Keep top k\ntokens only\nFixed cutoff\n→ May be too\n   narrow or wide", 0.6, 'cyan'),
        ("Top-p\n(Nucleus)", "Keep smallest set\nwith prob ≥ p\nAdaptive cutoff!\n→ Best trade-off", 0.87, 'blue'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Sampling Strategies: From Greedy to Nucleus", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "sampling_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
