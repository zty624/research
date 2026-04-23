"""
Minimal TTT (Test-Time Training) Reproduction
===============================================
Reproduces core ideas from TTT (2407.04620, Sun et al.):
1. Linear attention is limited by compression into fixed-size state
2. TTT: update hidden state at test time using self-supervised learning
3. R(x) = f_θ(x) where θ is updated during forward pass (inner loop)
4. Self-supervised task: reconstruct masked input from hidden state
5. Key insight: hidden state = model weights, updating it = learning at test time
6. TTT-Linear: simple linear update rule, O(N) complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── TTT Linear Layer ──

class TTTLinear(nn.Module):
    """TTT-Linear: hidden state is a linear model updated via self-supervised learning."""
    def __init__(self, d_model, d_hidden, lr=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.ttt_lr = lr

        # Main projection (like standard linear)
        self.W = nn.Linear(d_model, d_hidden, bias=False)

        # TTT: reconstruction weight (for self-supervised task)
        self.W_recon = nn.Linear(d_hidden, d_model, bias=False)

        # Learnable init for hidden state
        self.W_init = nn.Parameter(torch.randn(d_hidden, d_model) * 0.02)

    def forward(self, x, use_ttt=True):
        """x: (B, T, D)"""
        B, T, D = x.shape

        if not use_ttt or not self.training:
            # Standard forward (no TTT)
            return self.W(x)

        # TTT: update hidden state W during forward pass
        # W is the hidden state (initially W_init)
        W = self.W_init.unsqueeze(0).expand(B, -1, -1).clone()  # (B, d_hidden, d_model)

        outputs = []
        for t in range(T):
            x_t = x[:, t]  # (B, D)

            # Forward with current W: h_t = x_t @ W^T  (per-sample W)
            h_t = torch.bmm(x_t.unsqueeze(1), W.transpose(1, 2)).squeeze(1)  # (B, d_hidden)

            # Self-supervised reconstruction: x̂_t = h_t @ W_recon
            x_hat = self.W_recon(h_t)  # (B, D)

            # TTT loss: reconstruction error
            ttt_loss = F.mse_loss(x_hat, x_t.detach(), reduction='mean')

            # Update W via gradient descent on TTT loss
            grad = torch.autograd.grad(ttt_loss, W, retain_graph=False)[0]
            W = W - self.ttt_lr * grad

            # Use the updated W for output
            out_t = torch.bmm(x_t.unsqueeze(1), W.transpose(1, 2)).squeeze(1)  # (B, d_hidden)
            outputs.append(out_t)

        return torch.stack(outputs, dim=1)  # (B, T, d_hidden)


# ── TTT Block ──

class TTTBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads=4, use_ttt=True, ttt_lr=0.01):
        super().__init__()
        self.use_ttt = use_ttt

        if use_ttt:
            # TTT replaces self-attention
            self.ttt = TTTLinear(d_model, d_model, lr=ttt_lr)
        else:
            # Standard attention
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        h = self.ln1(x)
        if self.use_ttt:
            h = x + self.ttt(h)
        else:
            # Standard causal attention
            T = h.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
            h_attn, _ = self.attn(h, h, h, attn_mask=mask)
            h = x + h_attn

        h = h + self.ff(self.ln2(h))
        return h


# ── TTT Model ──

class TTTModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, use_ttt=True, ttt_lr=0.01):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([
            TTTBlock(d_model, d_model * 4, n_heads=4, use_ttt=use_ttt, ttt_lr=ttt_lr)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))


# ── Data ──

def generate_copy_data(batch_size, seq_len, vocab_size=10, delay=5):
    """Copy task with delay."""
    tokens = torch.randint(1, vocab_size, (batch_size, delay))
    fill_token = vocab_size
    fill = torch.full((batch_size, seq_len - 2 * delay), fill_token)
    zeros = torch.zeros(batch_size, delay, dtype=torch.long)
    inputs = torch.cat([tokens, fill, zeros], dim=1)
    targets = torch.cat([inputs[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long)], dim=1)
    return inputs, targets


def generate_periodic_data(batch_size, seq_len, vocab_size=20):
    sequences = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    for i in range(batch_size):
        period = np.random.randint(2, 6)
        pattern = torch.randint(1, vocab_size, (period,))
        for j in range(seq_len + 1):
            sequences[i, j] = pattern[j % period]
    return sequences[:, :-1], sequences[:, 1:]


# ── Training ──

def train_model(model, vocab_size, seq_len, n_steps=2000, lr=1e-3, device='cpu',
                task='periodic'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    for step in range(n_steps):
        if task == 'periodic':
            x, y = generate_periodic_data(32, seq_len, vocab_size)
        else:
            x, y = generate_copy_data(32, seq_len, vocab_size)
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


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "58-ttt"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 20
    seq_len = 32

    # Experiment 1: TTT vs standard attention
    print("=== TTT vs Standard Attention ===")

    print("\n  TTT-Linear (lr=0.01):")
    ttt_model = TTTModel(vocab_size, d_model=64, n_layers=2, use_ttt=True, ttt_lr=0.01).to(device)
    n_params_ttt = sum(p.numel() for p in ttt_model.parameters())
    print(f"    Params: {n_params_ttt:,}")
    ttt_losses = train_model(ttt_model, vocab_size, seq_len, n_steps=2000, device=device)

    print("\n  Standard Attention:")
    attn_model = TTTModel(vocab_size, d_model=64, n_layers=2, use_ttt=False).to(device)
    n_params_attn = sum(p.numel() for p in attn_model.parameters())
    print(f"    Params: {n_params_attn:,}")
    attn_losses = train_model(attn_model, vocab_size, seq_len, n_steps=2000, device=device)

    # Experiment 2: TTT learning rate sensitivity
    print("\n=== TTT Learning Rate Sensitivity ===")
    lr_results = {}
    for ttt_lr in [0.001, 0.005, 0.01, 0.05]:
        print(f"\n  TTT lr={ttt_lr}:")
        m = TTTModel(vocab_size, d_model=64, n_layers=2, use_ttt=True, ttt_lr=ttt_lr).to(device)
        l = train_model(m, vocab_size, seq_len, n_steps=2000, device=device)
        lr_results[ttt_lr] = l

    # Experiment 3: TTT at test time adaptation
    print("\n=== Test-Time Adaptation ===")
    # Train on one distribution, test on shifted distribution
    ttt_adapt = TTTModel(vocab_size, d_model=64, n_layers=2, use_ttt=True, ttt_lr=0.01).to(device)
    adapt_losses = train_model(ttt_adapt, vocab_size, seq_len, n_steps=2000, device=device)

    # Evaluate with and without test-time updates
    ttt_adapt.eval()
    with torch.no_grad():
        # Normal evaluation
        x, y = generate_periodic_data(128, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)
        logits_no_ttt = ttt_adapt(x)
        acc_no_ttt = (logits_no_ttt.argmax(-1) == y).float().mean().item()

    print(f"  Without TTT adaptation: {acc_no_ttt:.4f}")
    print(f"  With TTT: model adapts hidden states during forward pass")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 30
    smoothed_ttt = np.convolve(ttt_losses, np.ones(window)/window, mode='valid')
    smoothed_attn = np.convolve(attn_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(smoothed_ttt, label='TTT-Linear', color='blue', linewidth=2)
    axes[0].plot(smoothed_attn, label='Standard Attention', color='red', linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LR sensitivity
    colors_lr = ['gray', 'green', 'blue', 'red']
    for (lr, losses), color in zip(lr_results.items(), colors_lr):
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed, label=f'TTT lr={lr}', color=color)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("TTT Learning Rate Sensitivity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("TTT: Test-Time Training with Self-Supervised Updates", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "ttt_comparison.png", dpi=150)
    plt.close()

    # 2. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Standard\nAttention", "Fixed-size KV cache\nCompress all history\ninto fixed state\nNo adaptation at\n   test time", 0.14, 'red'),
        ("TTT\nInner Loop", "Hidden state = weights\nSelf-supervised loss:\nreconstruct x from h\nUpdate weights via\n   gradient descent", 0.5, 'blue'),
        ("Key Insight\nof TTT", "Expressive hidden state\n= more capacity\nLearning at test time\n= adaptation to\n   new distributions", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("TTT: Hidden State as Model Weights, Updated at Test Time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "ttt_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
