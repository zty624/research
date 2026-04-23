"""
Minimal Sparse Attention / Longformer Reproduction
====================================================
Reproduces core ideas from Longformer (2004.05150, Beltagy et al.):
1. Full attention is O(N²) — too expensive for long sequences
2. Sliding window attention: each token attends to local window
3. Global attention: select tokens attend to all (for [CLS], task tokens)
4. Combine: local + global = efficient long-document understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time


# ── Attention Implementations ──

class FullAttention(nn.Module):
    """Standard O(N²) self-attention."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            # Causal
            causal = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
            attn = attn.masked_fill(causal, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class SlidingWindowAttention(nn.Module):
    """Local sliding window attention: O(N × w) instead of O(N²)."""
    def __init__(self, d_model, n_heads, window_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Create sliding window mask
        w = self.window_size
        mask = torch.zeros(T, T, device=x.device)
        for i in range(T):
            left = max(0, i - w // 2)
            right = min(T, i + w // 2 + 1)
            mask[i, left:right] = 1.0

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class LongformerAttention(nn.Module):
    """Sliding window + global attention (Longformer)."""
    def __init__(self, d_model, n_heads, window_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, global_token_indices=None):
        """global_token_indices: list of positions with global attention."""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build combined mask: sliding window + global tokens
        w = self.window_size
        mask = torch.zeros(T, T, device=x.device)
        for i in range(T):
            left = max(0, i - w // 2)
            right = min(T, i + w // 2 + 1)
            mask[i, left:right] = 1.0

        # Global tokens: attend to and are attended by all positions
        if global_token_indices is not None:
            for gi in global_token_indices:
                if gi < T:
                    mask[gi, :] = 1.0  # global token attends to all
                    mask[:, gi] = 1.0  # all attend to global token

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Memory/Time Benchmarking ──

def benchmark_attention(attn_cls, d_model, n_heads, seq_lengths, device, **kwargs):
    """Benchmark attention memory and time."""
    results = {}
    for T in seq_lengths:
        try:
            attn = attn_cls(d_model, n_heads, **kwargs).to(device)
            x = torch.randn(1, T, d_model, device=device)

            # Warmup
            for _ in range(3):
                _ = attn(x)

            # Time
            t0 = time.time()
            for _ in range(10):
                _ = attn(x)
            t1 = time.time()
            avg_time = (t1 - t0) / 10

            results[T] = {'time': avg_time, 'success': True}
        except RuntimeError as e:
            if 'out of memory' in str(e):
                results[T] = {'time': float('inf'), 'success': False}
            else:
                raise

    return results


# ── Document Classification Task ──

class DocumentClassifier(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2,
                 n_classes=10, attn_type='full', window_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)

        if attn_type == 'full':
            attn_cls = lambda: FullAttention(d_model, n_heads)
        elif attn_type == 'sliding':
            attn_cls = lambda: SlidingWindowAttention(d_model, n_heads, window_size)
        elif attn_type == 'longformer':
            attn_cls = lambda: LongformerAttention(d_model, n_heads, window_size)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.attn_layers = nn.ModuleList([attn_cls() for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, n_classes)
        self.attn_type = attn_type

    def forward(self, x, global_token_indices=None):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))

        for attn, norm in zip(self.attn_layers, self.norms):
            if self.attn_type == 'longformer':
                h = h + attn(norm(h), global_token_indices=global_token_indices)
            else:
                h = h + attn(norm(h))

        # Use first token for classification (like [CLS])
        return self.classifier(h[:, 0, :])


# ── Data ──

def generate_long_documents(vocab_size=32, n_docs=2000, doc_len=128, n_classes=10):
    """Generate long documents: first few tokens determine class."""
    docs = torch.randint(1, vocab_size, (n_docs, doc_len))
    # Class determined by first 3 tokens
    labels = (docs[:, :3].sum(dim=1) % n_classes).long()
    # Prepend [CLS] token (token 0)
    docs = torch.cat([torch.zeros(n_docs, 1, dtype=torch.long), docs], dim=1)
    return docs, labels


# ── Training ──

def train_classifier(model, docs, labels, n_epochs=15, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_train = int(0.8 * len(docs))
    train_docs, test_docs = docs[:n_train].to(device), docs[n_train:].to(device)
    train_labels, test_labels = labels[:n_train].to(device), labels[n_train:].to(device)

    train_losses = []
    test_accs = []

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n_train)[:128]
        logits = model(train_docs[idx], global_token_indices=[0] if model.attn_type == 'longformer' else None)
        loss = F.cross_entropy(logits, train_labels[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            # Evaluate in chunks to avoid OOM
            correct = total = 0
            for i in range(0, len(test_docs), 64):
                batch_docs = test_docs[i:i+64]
                batch_labels = test_labels[i:i+64]
                logits = model(batch_docs, global_token_indices=[0] if model.attn_type == 'longformer' else None)
                correct += (logits.argmax(1) == batch_labels).sum().item()
                total += len(batch_labels)
            test_accs.append(correct / total)

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {test_accs[-1]:.4f}")

    return train_losses, test_accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "47-sparse-attention"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Compare attention types on document classification
    print("=== Document Classification ===")
    docs, labels = generate_long_documents(vocab_size=32, n_docs=2000, doc_len=128)

    attn_types = ['full', 'sliding', 'longformer']
    doc_results = {}

    for attn_type in attn_types:
        print(f"\n  {attn_type}:")
        model = DocumentClassifier(vocab_size=32, d_model=64, n_heads=2, n_layers=2,
                                    attn_type=attn_type, window_size=32).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")
        losses, accs = train_classifier(model, docs, labels, n_epochs=15, device=device)
        doc_results[attn_type] = {'losses': losses, 'accs': accs, 'final_acc': accs[-1]}

    # Experiment 2: Attention pattern visualization
    print("\n=== Attention Pattern Visualization ===")
    d_model = 64
    T = 64

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, attn_type in enumerate(attn_types):
        if attn_type == 'full':
            attn = FullAttention(d_model, 2).to(device)
        elif attn_type == 'sliding':
            attn = SlidingWindowAttention(d_model, 2, window_size=16).to(device)
        else:
            attn = LongformerAttention(d_model, 2, window_size=16).to(device)

        x = torch.randn(1, T, d_model, device=device)
        with torch.no_grad():
            qkv = attn.qkv(x).reshape(1, T, 3, 2, 32).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(32)

            if attn_type == 'sliding':
                w = 16
                mask = torch.zeros(T, T, device=device)
                for i in range(T):
                    left = max(0, i - w // 2)
                    right = min(T, i + w // 2 + 1)
                    mask[i, left:right] = 1.0
                scores = scores.masked_fill(mask == 0, float('-inf'))
            elif attn_type == 'longformer':
                w = 16
                mask = torch.zeros(T, T, device=device)
                for i in range(T):
                    left = max(0, i - w // 2)
                    right = min(T, i + w // 2 + 1)
                    mask[i, left:right] = 1.0
                mask[0, :] = 1.0  # global token
                mask[:, 0] = 1.0
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            avg_attn = attn_weights[0].mean(0).cpu().numpy()

        im = axes[idx].imshow(avg_attn, cmap='Blues', aspect='auto')
        axes[idx].set_title(f"{attn_type.capitalize()} Attention")
        plt.colorbar(im, ax=axes[idx], shrink=0.8)

    plt.suptitle("Attention Patterns: Full vs Sliding Window vs Longformer", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "attention_patterns.png", dpi=150)
    plt.close()

    # Experiment 3: Memory scaling
    print("\n=== Memory/Time Scaling ===")
    seq_lengths = [64, 128, 256, 512]

    # Full attention benchmark
    print("  Full attention:")
    full_bench = benchmark_attention(FullAttention, 64, 2, seq_lengths, device)
    for T, r in full_bench.items():
        print(f"    T={T}: {r['time']*1000:.2f}ms" if r['success'] else f"    T={T}: OOM")

    # Sliding window benchmark
    print("  Sliding window (w=32):")
    sw_bench = benchmark_attention(SlidingWindowAttention, 64, 2, seq_lengths, device, window_size=32)
    for T, r in sw_bench.items():
        print(f"    T={T}: {r['time']*1000:.2f}ms" if r['success'] else f"    T={T}: OOM")

    # ── Visualization ──

    # 1. Training comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'full': 'red', 'sliding': 'blue', 'longformer': 'green'}
    for attn_type, r in doc_results.items():
        axes[0].plot(r['losses'], label=attn_type.capitalize(), color=colors[attn_type])
        axes[1].plot(r['accs'], label=attn_type.capitalize(), color=colors[attn_type])

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Sparse Attention: Full vs Sliding Window vs Longformer", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Time scaling
    fig, ax = plt.subplots(figsize=(8, 5))

    for bench, name, color in [(full_bench, 'Full O(N²)', 'red'),
                                (sw_bench, 'Sliding O(N×w)', 'blue')]:
        ts = [T for T, r in bench.items() if r['success']]
        times = [bench[T]['time'] * 1000 for T in ts]
        if ts:
            ax.plot(ts, times, 'o-', label=name, color=color)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Attention Computation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "time_scaling.png", dpi=150)
    plt.close()

    # 3. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Full\nAttention", "Every token attends\nto every other token\nO(N²) memory & time\n→ Too expensive\n   for long docs", 0.14, 'red'),
        ("Sliding\nWindow", "Local attention only\nEach token attends\nto w neighbors\nO(N × w) → Fast!\n→ Miss global info", 0.42, 'blue'),
        ("Longformer\n(Local+Global)", "Sliding window +\nglobal tokens\n[CLS] attends to all\nAll attend to [CLS]\n→ Best of both!", 0.75, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Sparse Attention: From O(N²) to O(N) for Long Sequences", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "sparse_attention_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
