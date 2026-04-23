"""
Minimal Attention Sinks / StreamingLLM Reproduction
=====================================================
Reproduces core ideas from Attention Sinks (2309.17453, Xiao et al.):
1. LLMs have "attention sinks" — initial tokens receive disproportionate attention
2. Simply evicting KV cache for old tokens causes catastrophic collapse
3. StreamingLLM: keep attention sink tokens + recent window
4. Compare: full KV cache vs naive eviction vs StreamingLLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Simple Transformer ──

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv_cache = (k, v)

        # Attention
        S = k.shape[2]  # total sequence length including cache
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask
        if mask is None:
            # Default causal: query at position i can attend to keys at positions <= i+T-S
            # With cache: current q starts at position (S-T), attends to all prior
            causal_mask = torch.tril(torch.ones(T, S, device=x.device))
            # Shift for cached positions
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)

        return self.out(out), new_kv_cache, attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, kv_cache=None):
        h = self.norm1(x)
        attn_out, new_cache, attn_weights = self.attn(h, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_cache, attn_weights


class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_caches=None):
        B, T = x.shape
        start_pos = 0
        if kv_caches is not None and kv_caches[0] is not None:
            # Get position offset from cache
            start_pos = kv_caches[0][0].shape[2]  # cached key length

        positions = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos_emb(positions[:, :T])

        new_caches = []
        all_attn_weights = []

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache, attn_weights = layer(h, kv_cache=cache)
            new_caches.append(new_cache)
            all_attn_weights.append(attn_weights)

        return self.head(self.norm(h)), new_caches, all_attn_weights


# ── Generation with KV Cache ──

def generate_full(model, prompt, n_tokens=50, temperature=0.8, device='cpu'):
    """Full KV cache generation (baseline)."""
    model.eval()
    x = prompt.clone()
    all_probs = []
    all_attn = []

    # Initial forward
    with torch.no_grad():
        logits, caches, attn_weights = model(x)
        all_attn.append(attn_weights[-1].detach())  # last layer attention

        # Sample next token
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
        all_probs.append(probs[0].cpu().numpy())
        x = torch.cat([x, next_token], dim=1)

    # Autoregressive with cache
    for _ in range(n_tokens - 1):
        with torch.no_grad():
            logits, caches, attn_weights = model(next_token, kv_caches=caches)
            all_attn.append(attn_weights[-1].detach())

            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            all_probs.append(probs[0].cpu().numpy())
            x = torch.cat([x, next_token], dim=1)

    return x, all_probs, all_attn


def generate_streaming(model, prompt, n_tokens=50, temperature=0.8,
                       window_size=20, n_sinks=4, device='cpu'):
    """StreamingLLM: keep attention sinks + sliding window."""
    model.eval()
    x = prompt.clone()
    all_probs = []

    # Initial forward
    with torch.no_grad():
        logits, caches, attn_weights = model(x)

        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
        all_probs.append(probs[0].cpu().numpy())
        x = torch.cat([x, next_token], dim=1)

    # Autoregressive with streaming cache management
    for step in range(n_tokens - 1):
        # Check if cache exceeds window
        if caches[0][0].shape[2] > window_size:
            # Evict middle tokens, keep sinks + recent
            new_caches = []
            for layer_cache in caches:
                k, v = layer_cache
                # Keep first n_sinks tokens (attention sinks) + recent window
                k_sink = k[:, :, :n_sinks, :]
                v_sink = v[:, :, :n_sinks, :]
                k_recent = k[:, :, -(window_size - n_sinks):, :]
                v_recent = v[:, :, -(window_size - n_sinks):, :]
                new_k = torch.cat([k_sink, k_recent], dim=2)
                new_v = torch.cat([v_sink, v_recent], dim=2)
                new_caches.append((new_k, new_v))
            caches = new_caches

        with torch.no_grad():
            logits, caches, attn_weights = model(next_token, kv_caches=caches)

            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            all_probs.append(probs[0].cpu().numpy())
            x = torch.cat([x, next_token], dim=1)

    return x, all_probs


def generate_naive_evict(model, prompt, n_tokens=50, temperature=0.8,
                         window_size=20, device='cpu'):
    """Naive eviction: just keep recent window (no sink preservation)."""
    model.eval()
    x = prompt.clone()
    all_probs = []

    with torch.no_grad():
        logits, caches, _ = model(x)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
        all_probs.append(probs[0].cpu().numpy())
        x = torch.cat([x, next_token], dim=1)

    for step in range(n_tokens - 1):
        # Naive eviction: just keep last window_size tokens
        if caches[0][0].shape[2] > window_size:
            new_caches = []
            for layer_cache in caches:
                k, v = layer_cache
                k = k[:, :, -window_size:, :]
                v = v[:, :, -window_size:, :]
                new_caches.append((k, v))
            caches = new_caches

        with torch.no_grad():
            logits, caches, _ = model(next_token, kv_caches=caches)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            all_probs.append(probs[0].cpu().numpy())
            x = torch.cat([x, next_token], dim=1)

    return x, all_probs


# ── Data ──

def generate_data(vocab_size=32, length=8000):
    data = []
    for i in range(length):
        base = (i % 10)
        noise = np.random.randint(0, 3)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "42-attention-sinks"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    data = generate_data(vocab_size, length=8000)

    # Train model
    print("=== Training Model ===")
    model = TransformerLM(vocab_size, d_model=64, n_heads=2, n_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = 32

    for step in range(2000):
        starts = torch.randint(0, len(data) - seq_len - 1, (32,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f}")

    # Experiment 1: Attention pattern analysis
    print("\n=== Attention Pattern Analysis ===")
    model.eval()
    test_seq = data[:40].unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, attn_weights = model(test_seq)

    # Analyze attention to first tokens (attention sinks)
    for layer_idx, attn in enumerate(attn_weights):
        # attn: (B, H, T, T) — last layer
        avg_attn = attn[0].mean(dim=0).cpu().numpy()  # (T, T)
        # How much attention does position 0 (sink) get?
        sink_attn = avg_attn[:, 0].mean()
        print(f"  Layer {layer_idx}: Avg attention to position 0 = {sink_attn:.4f}")
        print(f"  Layer {layer_idx}: Max attention to position 0 = {avg_attn[:, 0].max():.4f}")

    # Experiment 2: Generation quality comparison
    print("\n=== Generation with Different Cache Strategies ===")
    prompt = data[:10].unsqueeze(0).to(device)
    n_gen = 40

    # Full KV cache (baseline)
    print("  Full KV cache...")
    _, full_probs, full_attn = generate_full(model, prompt, n_gen, device=device)

    # Naive eviction
    print("  Naive eviction (window=20)...")
    _, naive_probs = generate_naive_evict(model, prompt, n_gen, window_size=20, device=device)

    # StreamingLLM
    print("  StreamingLLM (sinks=4, window=20)...")
    _, streaming_probs = generate_streaming(model, prompt, n_gen, window_size=20, n_sinks=4, device=device)

    # Compute perplexity divergence from full cache
    full_entropy = [-np.sum(p * np.log(p + 1e-10)) for p in full_probs]
    naive_kl = [np.sum(p * np.log((p + 1e-10) / (q + 1e-10))) for p, q in zip(full_probs, naive_probs)]
    stream_kl = [np.sum(p * np.log((p + 1e-10) / (q + 1e-10))) for p, q in zip(full_probs, streaming_probs)]

    print(f"\n  Avg KL divergence from Full KV cache:")
    print(f"    Naive eviction:  {np.mean(naive_kl):.4f}")
    print(f"    StreamingLLM:    {np.mean(stream_kl):.4f}")

    # ── Visualization ──

    # 1. Attention heatmap (attention sinks visualization)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for layer_idx in range(min(2, len(attn_weights))):
        attn = attn_weights[layer_idx][0].mean(dim=0).cpu().numpy()
        im = axes[layer_idx].imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=attn.max() * 0.5)
        axes[layer_idx].set_title(f"Layer {layer_idx} Attention (avg over heads)")
        axes[layer_idx].set_xlabel("Key Position")
        axes[layer_idx].set_ylabel("Query Position")
        plt.colorbar(im, ax=axes[layer_idx], shrink=0.8)

    plt.suptitle("Attention Sinks: First Tokens Receive Disproportionate Attention", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "attention_sinks.png", dpi=150)
    plt.close()

    # 2. Attention to position 0 over sequence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for layer_idx in range(min(2, len(attn_weights))):
        attn = attn_weights[layer_idx][0].mean(dim=0).cpu().numpy()
        sink_attn_over_pos = attn[:, 0]
        axes[0].plot(sink_attn_over_pos, label=f'Layer {layer_idx}', alpha=0.8)
        # Also plot attention to position 1
        if attn.shape[1] > 1:
            axes[1].plot(attn[:, 1], label=f'Layer {layer_idx} pos=1', alpha=0.8, linestyle='--')
            axes[1].plot(attn[:, -1], label=f'Layer {layer_idx} pos=-1', alpha=0.8)

    axes[0].set_title("Attention to Position 0 (Sink)")
    axes[0].set_xlabel("Query Position")
    axes[0].set_ylabel("Attention Weight")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Attention to Position 1 and Last")
    axes[1].set_xlabel("Query Position")
    axes[1].set_ylabel("Attention Weight")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Attention Sinks: Why We Can't Evict First Tokens", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "sink_analysis.png", dpi=150)
    plt.close()

    # 3. KL divergence over generation steps
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = range(len(naive_kl))
    ax.plot(steps, naive_kl, label='Naive Eviction', color='red', alpha=0.7)
    ax.plot(steps, stream_kl, label='StreamingLLM', color='blue', alpha=0.7)
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("KL Divergence from Full Cache")
    ax.set_title("Cache Eviction Impact on Output Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "kl_divergence.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Attention\nSinks", "First tokens get\nhigh attention\n(they serve as\n'rest tokens')\n→ Can't evict them!", 0.14, 'red'),
        ("Naive\nEviction", "Just keep recent\nwindow in KV cache\n→ Model collapses!\nSink tokens lost\n→ High KL divergence", 0.42, 'orange'),
        ("StreamingLLM\n(2023)", "Keep sink tokens\n+ sliding window\n→ Stable generation!\nInfinite-length\nstreaming", 0.71, 'blue'),
        ("Key Insight", "LLMs use first\ntokens as 'overflow'\nfor softmax\n→ Must preserve\n   them in cache", 0.93, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Attention Sinks & StreamingLLM: Why First Tokens Matter", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "attention_sinks_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
