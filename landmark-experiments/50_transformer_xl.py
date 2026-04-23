"""
Minimal Transformer-XL Reproduction
=====================================
Reproduces core ideas from Transformer-XL (1802.04799, Dai et al.):
1. Fixed-context Transformers can't model long-range dependencies
2. Segment-level recurrence: cache hidden states from previous segment
3. Relative positional encoding: positions are relative, not absolute
4. Enables learning dependencies 80% longer than RNNs, 1800% longer than vanilla Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Relative Positional Encoding ──

class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Transformer-XL."""
    def __init__(self, d_model, max_len=512, clamp_len=None):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.clamp_len = clamp_len

        # Learnable relative position embeddings
        self.rel_emb = nn.Parameter(torch.randn(2 * max_len, d_model) * 0.02)

    def forward(self, qlen, klen):
        """Get relative position embeddings.
        qlen: query length, klen: key length (including memory)
        """
        # Relative positions: klen - 1 - i + j for query i, key j
        # i in [0, qlen), j in [0, klen)
        rp = torch.arange(klen - 1, -qlen, -1, device=self.rel_emb.device)  # (klen + qlen - 1,)
        if self.clamp_len is not None:
            rp = rp.clamp(-self.clamp_len, self.clamp_len)

        # Map to indices
        rp = rp + self.max_len  # shift to non-negative
        rp = rp.clamp(0, 2 * self.max_len - 1)

        return self.rel_emb[rp]  # (qlen + klen - 1, d_model)


# ── Transformer-XL Attention ──

class TransformerXLAttention(nn.Module):
    """Multi-head attention with relative positional encoding and memory."""
    def __init__(self, d_model, n_heads, dropatt=0.0, clamp_len=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        # For relative position
        self.r_w_bias = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.r_r_bias = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)

        # Learnable relative position embeddings (simplified)
        self.rel_pos_enc = nn.Parameter(torch.randn(2 * 256, self.d_head) * 0.02)
        self.clamp_len = clamp_len
        self.dropatt = nn.Dropout(dropatt)

    def _get_rel_pos_bias(self, qlen, klen, device):
        """Compute relative position bias matrix.
        Returns (H, qlen, klen) bias from relative position embeddings.
        """
        # Position differences: i - j for query i, key j
        # i in [0, qlen), j in [0, klen) where klen = T + M
        # Offset: memory keys have positions [-M, 0), query has [0, T)
        # So i - j ranges from -(klen-1) to (qlen-1)
        q_pos = torch.arange(qlen, device=device)
        k_pos = torch.arange(klen, device=device)
        diff = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # (qlen, klen)

        if self.clamp_len is not None:
            diff = diff.clamp(-self.clamp_len, self.clamp_len)

        # Map to embedding indices (shift by max_len to make non-negative)
        max_len = self.rel_pos_enc.shape[0] // 2
        idx = diff + max_len
        idx = idx.clamp(0, 2 * max_len - 1)

        # Get embeddings: (qlen, klen, d_head)
        r = self.rel_pos_enc[idx]

        # Compute bias via (q + r_r_bias) · r
        # We'll add this in forward
        return r

    def forward(self, x, mem=None):
        """x: (B, T, D), mem: (B, M, D) from previous segment."""
        B, T, D = x.shape
        M = mem.shape[1] if mem is not None else 0
        klen = T + M

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Prepend memory to keys and values
        if mem is not None:
            mem_qkv = self.qkv(mem).reshape(B, M, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
            k = torch.cat([mem_qkv[1], k], dim=2)
            v = torch.cat([mem_qkv[2], v], dim=2)

        # Content-based attention: (q + r_w_bias) · k^T
        q_with_bias = q + self.r_w_bias.unsqueeze(0).unsqueeze(2)
        attn = (q_with_bias @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Relative position attention: (q + r_r_bias) · r
        r = self._get_rel_pos_bias(T, klen, x.device)  # (T, klen, d_head)
        q_with_rbias = q + self.r_r_bias.unsqueeze(0).unsqueeze(2)  # (B, H, T, d_head)
        # einsum: (B,H,T,d_head) · (T,klen,d_head) -> (B,H,T,klen)
        rel_attn = torch.einsum('bhti,tji->bhtj', q_with_rbias, r) / math.sqrt(self.d_head)

        attn = attn + rel_attn

        # Causal mask
        causal = torch.tril(torch.ones(T, klen, device=x.device)).bool()
        # For memory: allow attending to all memory positions
        if M > 0:
            causal[:, :M] = True
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = self.dropatt(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Transformer-XL Block ──

class TransformerXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropatt=0.0, clamp_len=None):
        super().__init__()
        self.attn = TransformerXLAttention(d_model, n_heads, dropatt, clamp_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mem=None):
        h = self.norm1(x)
        h = x + self.drop(self.attn(h, mem))
        h = h + self.drop(self.ff(self.norm2(h)))
        return h


# ── Transformer-XL ──

class TransformerXL(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, mem_len=64, clamp_len=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.mem_len = mem_len

        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerXLBlock(d_model, n_heads, d_ff, clamp_len=clamp_len)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._reset_memory()

    def _reset_memory(self):
        self.mems = [None] * self.n_layers

    def forward(self, x, use_memory=True):
        """x: (B, T) token indices."""
        B, T = x.shape
        h = self.emb(x) * math.sqrt(self.d_model)

        new_mems = []
        for i, block in enumerate(self.blocks):
            mem = self.mems[i] if use_memory else None
            new_mems.append(h.detach()[:, -self.mem_len:].clone() if use_memory else None)
            h = block(h, mem)

        if use_memory:
            self.mems = new_mems

        h = self.norm(h)
        return self.head(h)


# ── Vanilla Transformer (for comparison) ──

class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) * math.sqrt(self.d_model) + self.pos_emb(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            h = block(h, src_mask=mask)
        h = self.norm(h)
        return self.head(h)


# ── Data: Copy task with long-range dependency ──

def generate_copy_data(batch_size, seq_len, vocab_size=10, delay=5):
    """Copy task: memorize first `delay` tokens, repeat them at end.
    Input:  [token_1..token_delay, fill, ..., fill, 0, 0, ..., 0]
    Target: [fill, ..., fill, token_1, ..., token_delay]
    """
    tokens = torch.randint(1, vocab_size, (batch_size, delay))
    fill = torch.full((batch_size, seq_len - 2 * delay), vocab_size)  # fill token
    zeros = torch.zeros(batch_size, delay, dtype=torch.long)  # delimiter
    inputs = torch.cat([tokens, fill, zeros], dim=1)
    targets = torch.cat([fill[:, :1], tokens, fill[:, 1:]], dim=1)  # shift right
    # Actually: target is just the shifted input
    targets = torch.cat([inputs[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long)], dim=1)
    return inputs, targets


# ── Training ──

def train_model(model, vocab_size, seq_len, delay, n_steps=3000, lr=1e-3,
                device='cpu', use_memory=True, seg_len=None):
    """Train on copy task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    accs = []

    for step in range(n_steps):
        x, y = generate_copy_data(64, seq_len, vocab_size=vocab_size, delay=delay)
        x, y = x.to(device), y.to(device)

        if use_memory and seg_len is not None:
            # Process in segments with memory
            model._reset_memory() if hasattr(model, '_reset_memory') else None
            total_loss = 0
            n_segs = (seq_len + seg_len - 1) // seg_len
            for seg_idx in range(n_segs):
                start = seg_idx * seg_len
                end = min(start + seg_len, seq_len)
                seg_x = x[:, start:end]
                seg_y = y[:, start:end]
                logits = model(seg_x, use_memory=True)
                total_loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size + 1),
                    seg_y.reshape(-1),
                    ignore_index=0
                )
            loss = total_loss
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size + 1),
                y.reshape(-1),
                ignore_index=0
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Evaluate accuracy on copy tokens
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                x_e, y_e = generate_copy_data(128, seq_len, vocab_size=vocab_size, delay=delay)
                x_e, y_e = x_e.to(device), y_e.to(device)

                if use_memory and seg_len is not None:
                    model._reset_memory() if hasattr(model, '_reset_memory') else None
                    all_logits = []
                    n_segs = (seq_len + seg_len - 1) // seg_len
                    for seg_idx in range(n_segs):
                        start = seg_idx * seg_len
                        end = min(start + seg_len, seq_len)
                        seg_logits = model(x_e[:, start:end], use_memory=True)
                        all_logits.append(seg_logits)
                    logits_e = torch.cat(all_logits, dim=1)
                else:
                    logits_e = model(x_e)

                pred = logits_e.argmax(-1)
                # Check accuracy on the copy portion (last `delay` tokens)
                copy_pred = pred[:, -delay:]
                copy_target = x_e[:, :delay]
                acc = (copy_pred == copy_target).float().mean().item()
                accs.append(acc)

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f} | Copy Acc: {accs[-1]:.4f}")

    return losses, accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "50-transformer-xl"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 10
    seq_len = 80
    delay = 10  # must memorize first 10 tokens and reproduce at end

    # Experiment 1: Transformer-XL with memory vs Vanilla Transformer
    print("=== Transformer-XL vs Vanilla Transformer ===")

    # Vanilla Transformer (full context)
    print("\n  Vanilla Transformer:")
    vanilla = VanillaTransformer(vocab_size + 1, d_model=64, n_heads=2, n_layers=2, d_ff=256).to(device)
    v_losses, v_accs = train_model(vanilla, vocab_size, seq_len, delay,
                                    n_steps=3000, device=device, use_memory=False)

    # Transformer-XL with segment-level recurrence
    print("\n  Transformer-XL (seg_len=40, mem_len=40):")
    txl = TransformerXL(vocab_size + 1, d_model=64, n_heads=2, n_layers=2,
                        d_ff=256, mem_len=40, clamp_len=80).to(device)
    xl_losses, xl_accs = train_model(txl, vocab_size, seq_len, delay,
                                      n_steps=3000, device=device,
                                      use_memory=True, seg_len=40)

    # Transformer-XL without memory (ablation)
    print("\n  Transformer-XL (no memory, seg_len=40):")
    txl_nomem = TransformerXL(vocab_size + 1, d_model=64, n_heads=2, n_layers=2,
                               d_ff=256, mem_len=0, clamp_len=80).to(device)
    nm_losses, nm_accs = train_model(txl_nomem, vocab_size, seq_len, delay,
                                      n_steps=3000, device=device,
                                      use_memory=True, seg_len=40)

    # Experiment 2: Longer sequences
    print("\n=== Scaling to Longer Sequences ===")
    long_results = {}
    for sl in [60, 100, 140]:
        print(f"\n  Seq len={sl}:")
        # Transformer-XL
        m1 = TransformerXL(vocab_size + 1, d_model=64, n_heads=2, n_layers=2,
                           d_ff=256, mem_len=40, clamp_len=sl).to(device)
        _, a1 = train_model(m1, vocab_size, sl, delay, n_steps=2000,
                            device=device, use_memory=True, seg_len=40)

        # Vanilla
        m2 = VanillaTransformer(vocab_size + 1, d_model=64, n_heads=2, n_layers=2, d_ff=256).to(device)
        _, a2 = train_model(m2, vocab_size, sl, delay, n_steps=2000,
                            device=device, use_memory=False)

        long_results[sl] = {'txl_acc': a1[-1], 'vanilla_acc': a2[-1]}
        print(f"    TXL: {a1[-1]:.4f}, Vanilla: {a2[-1]:.4f}")

    # ── Visualization ──

    # 1. Training comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.arange(len(v_losses))
    axes[0].plot(steps, v_losses, label='Vanilla Transformer', color='red', alpha=0.7)
    axes[0].plot(steps, xl_losses, label='Transformer-XL (with memory)', color='blue')
    axes[0].plot(steps, nm_losses, label='Transformer-XL (no memory)', color='blue', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    eval_steps = np.arange(100, 100 * len(v_accs) + 1, 100)
    axes[1].plot(eval_steps, v_accs, label='Vanilla Transformer', color='red', alpha=0.7)
    axes[1].plot(eval_steps, xl_accs, label='Transformer-XL (with memory)', color='blue')
    axes[1].plot(eval_steps, nm_accs, label='Transformer-XL (no memory)', color='blue', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Copy Accuracy")
    axes[1].set_title("Copy Task Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Transformer-XL: Segment-Level Recurrence for Long Sequences", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "txl_comparison.png", dpi=150)
    plt.close()

    # 2. Sequence length scaling
    fig, ax = plt.subplots(figsize=(8, 5))

    seq_lens = sorted(long_results.keys())
    txl_accs_l = [long_results[s]['txl_acc'] for s in seq_lens]
    vanilla_accs_l = [long_results[s]['vanilla_acc'] for s in seq_lens]

    ax.plot(seq_lens, txl_accs_l, 'o-', label='Transformer-XL', color='blue')
    ax.plot(seq_lens, vanilla_accs_l, 's--', label='Vanilla Transformer', color='red')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Copy Accuracy")
    ax.set_title("Performance vs Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "seq_length_scaling.png", dpi=150)
    plt.close()

    # 3. Relative vs Absolute Position Encoding Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute positional encoding attention pattern
    T = 16
    abs_pos = torch.zeros(T, T)
    for i in range(T):
        for j in range(T):
            abs_pos[i, j] = abs(i - j)  # distance-based

    im0 = axes[0].imshow(abs_pos, cmap='Blues', aspect='auto')
    axes[0].set_title("Absolute Position\n(fixed distance pattern)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Relative positional encoding attention pattern
    # In Transformer-XL, positions are relative regardless of segment
    M = 8  # memory length
    rel_pos = torch.zeros(T, T + M)
    for i in range(T):
        for j in range(T + M):
            rel_pos[i, j] = abs(i - (j - M))  # relative distance

    im1 = axes[1].imshow(rel_pos, cmap='Blues', aspect='auto')
    axes[1].set_title("Relative Position (with memory)\n(can attend to cached segment)")
    axes[1].set_xlabel("Key position (incl. memory)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.suptitle("Position Encoding: Absolute vs Relative", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "position_encoding.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Vanilla\nTransformer", "Fixed context window\nCan't use info\nfrom previous segments\nO(L²) per segment\n→ Forgetful", 0.14, 'red'),
        ("Segment-Level\nRecurrence", "Cache hidden states\nfrom previous segment\nFeed as memory to\nnext segment\n→ Remember!", 0.5, 'blue'),
        ("Relative\nPosition", "Positions are relative\nnot absolute\nSame pattern across\nsegments → generalize\n→ Longer deps!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Transformer-XL: Beyond Fixed-Length Context", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "transformer_xl_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
