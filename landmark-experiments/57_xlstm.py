"""
Minimal xLSTM Reproduction
============================
Reproduces core ideas from xLSTM (2405.04517, Beck et al.):
1. Extended LSTM with exponential gating and matrix memory
2. sLSTM: scalar memory with exponential gating (like standard LSTM but better)
3. mLSTM: matrix memory with covariance update (stores key-value pairs)
4. Enables LSTMs to compete with Transformers on language tasks
5. Key: exponential gating enables forget/remember decisions, matrix memory stores structured info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── sLSTM: Scalar Memory with Exponential Gating ──

class sLSTMBlock(nn.Module):
    """sLSTM with exponential gating."""
    def __init__(self, d_model, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        hidden = d_model * expand_factor

        # Input projections
        self.W_i = nn.Linear(d_model, hidden, bias=False)
        self.W_f = nn.Linear(d_model, hidden, bias=False)
        self.W_o = nn.Linear(d_model, hidden, bias=False)
        self.W_z = nn.Linear(d_model, hidden, bias=False)

        # Bias initialization
        self.b_i = nn.Parameter(torch.zeros(hidden))
        self.b_f = nn.Parameter(torch.ones(hidden))  # forget bias = 1
        self.b_o = nn.Parameter(torch.zeros(hidden))
        self.b_z = nn.Parameter(torch.zeros(hidden))

        self.out_proj = nn.Linear(hidden, d_model)

    def forward(self, x):
        """x: (B, T, D) — process full sequence."""
        B, T, D = x.shape
        H = self.W_i.out_features

        # Compute all gates at once
        i = self.W_i(x) + self.b_i  # input gate (logits)
        f = self.W_f(x) + self.b_f  # forget gate (logits)
        o = self.W_o(x) + self.b_o  # output gate (logits)
        z = self.W_z(x) + self.b_z  # cell candidate

        # Exponential gating (clamped for stability)
        i = torch.exp(i.clamp(max=5))  # exp for input gate
        f = torch.exp(f.clamp(max=5))  # exp for forget gate

        # Running LSTM
        c = torch.zeros(B, H, device=x.device)
        n = torch.zeros(B, H, device=x.device)  # normalizer
        outputs = []

        for t in range(T):
            # Cell state update
            c = f[:, t] * c + i[:, t] * torch.tanh(z[:, t])
            # Normalizer update (stabilization)
            n = f[:, t] * n + i[:, t]
            # Normalized cell state
            c_norm = c / (n + 1e-8)
            c_norm = c_norm.clamp(-10, 10)
            # Output gate
            h = torch.sigmoid(o[:, t]) * c_norm
            outputs.append(h)

        h_out = torch.stack(outputs, dim=1)  # (B, T, H)
        return self.out_proj(h_out)


# ── mLSTM: Matrix Memory with Covariance Update ──

class mLSTMBlock(nn.Module):
    """mLSTM with matrix memory (key-value store)."""
    def __init__(self, d_model, head_dim=32, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden = n_heads * head_dim

        # Key, Value, Query projections
        self.W_k = nn.Linear(d_model, self.hidden, bias=False)
        self.W_v = nn.Linear(d_model, self.hidden, bias=False)
        self.W_q = nn.Linear(d_model, self.hidden, bias=False)

        # Gates
        self.W_i = nn.Linear(d_model, self.hidden, bias=False)
        self.W_f = nn.Linear(d_model, self.hidden, bias=False)
        self.W_o = nn.Linear(d_model, self.hidden, bias=False)

        self.b_i = nn.Parameter(torch.zeros(self.hidden))
        self.b_f = nn.Parameter(torch.ones(self.hidden))

        self.out_proj = nn.Linear(self.hidden, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, T, D) — parallel scan for efficiency."""
        B, T, D = x.shape
        H = self.hidden

        q = self.W_q(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.W_k(x).reshape(B, T, self.n_heads, self.head_dim)
        v = self.W_v(x).reshape(B, T, self.n_heads, self.head_dim)

        # Gates (clamped for stability)
        i = torch.exp((self.W_i(x) + self.b_i).clamp(max=5))  # (B, T, H)
        f = torch.exp((self.W_f(x) + self.b_f).clamp(max=5))  # (B, T, H)
        o = torch.sigmoid(self.W_o(x))  # (B, T, H)

        # Reshape for per-head processing
        i = i.reshape(B, T, self.n_heads, self.head_dim)
        f = f.reshape(B, T, self.n_heads, self.head_dim)
        o = o.reshape(B, T, self.n_heads, self.head_dim)

        # Matrix memory recurrence: C_t = f_t * C_{t-1} + i_t * v_t ⊗ k_t
        # Output: h_t = o_t * C_t * q_t (query the matrix memory)
        C = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device)
        n = torch.zeros(B, self.n_heads, 1, 1, device=x.device)
        outputs = []

        for t in range(T):
            # Update matrix memory: C = f*C + i*v*k^T
            f_t = f[:, t].unsqueeze(-1)  # (B, H, d, 1)
            i_t = i[:, t].unsqueeze(-1)  # (B, H, d, 1)
            v_t = v[:, t]  # (B, H, d)
            k_t = k[:, t].unsqueeze(-1)  # (B, H, d, 1)

            C = f_t * C + i_t * (v_t.unsqueeze(-1) @ k_t.transpose(-2, -1))  # (B, H, d, d)
            n = f_t.squeeze(-1).mean(dim=-1, keepdim=True).unsqueeze(-1) * n + i_t.squeeze(-1).mean(dim=-1, keepdim=True).unsqueeze(-1)

            # Clip matrix memory for stability
            C = C.clamp(-10, 10)

            # Query: h = o * C * q
            q_t = q[:, t].unsqueeze(-1)  # (B, H, d, 1)
            h_t = o[:, t] * (C @ q_t).squeeze(-1)  # (B, H, d)
            h_t = h_t.clamp(-10, 10)
            outputs.append(h_t)

        h_out = torch.stack(outputs, dim=1).reshape(B, T, self.hidden)
        return self.out_proj(h_out)


# ── xLSTM Block ──

class xLSTMBlock(nn.Module):
    def __init__(self, d_model, block_type='s', expand_factor=2, n_heads=4, head_dim=16):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        if block_type == 's':
            self.block = sLSTMBlock(d_model, expand_factor)
        else:
            self.block = mLSTMBlock(d_model, head_dim, n_heads)

    def forward(self, x):
        return x + self.block(self.ln(x))


# ── xLSTM Model ──

class xLSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=4, expand_factor=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        # Alternate sLSTM and mLSTM blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            block_type = 's' if i % 2 == 0 else 'm'
            self.blocks.append(xLSTMBlock(d_model, block_type, expand_factor,
                                          n_heads=4, head_dim=d_model // 4))
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_out(h))


# ── Baseline: Standard LSTM ──

class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.lstm(h)
        return self.head(h)


# ── Baseline: Transformer ──

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            h = block(h, src_mask=mask)
        return self.head(self.ln(h))


# ── Data ──

def generate_periodic_data(batch_size, seq_len, vocab_size=20):
    sequences = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    for i in range(batch_size):
        period = np.random.randint(2, 6)
        pattern = torch.randint(1, vocab_size, (period,))
        for j in range(seq_len + 1):
            sequences[i, j] = pattern[j % period]
    return sequences[:, :-1], sequences[:, 1:]


# ── Training ──

def train_model(model, vocab_size, seq_len, n_steps=3000, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    for step in range(n_steps):
        x, y = generate_periodic_data(64, seq_len, vocab_size)
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
    results_dir = Path(__file__).parent / "results" / "57-xlstm"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 20
    seq_len = 32

    # Experiment 1: xLSTM vs LSTM vs Transformer
    print("=== xLSTM vs LSTM vs Transformer ===")

    print("\n  xLSTM:")
    xlstm = xLSTMModel(vocab_size, d_model=32, n_layers=2, expand_factor=2).to(device)
    n_params_x = sum(p.numel() for p in xlstm.parameters())
    print(f"    Params: {n_params_x:,}")
    xlstm_losses = train_model(xlstm, vocab_size, seq_len, n_steps=2000, lr=1e-4, device=device)

    print("\n  LSTM:")
    lstm = LSTMBaseline(vocab_size, d_model=32, n_layers=2).to(device)
    n_params_l = sum(p.numel() for p in lstm.parameters())
    print(f"    Params: {n_params_l:,}")
    lstm_losses = train_model(lstm, vocab_size, seq_len, n_steps=2000, device=device)

    print("\n  Transformer:")
    tf = TransformerBaseline(vocab_size, d_model=32, n_heads=4, n_layers=2).to(device)
    n_params_t = sum(p.numel() for p in tf.parameters())
    print(f"    Params: {n_params_t:,}")
    tf_losses = train_model(tf, vocab_size, seq_len, n_steps=2000, device=device)

    # Experiment 2: Different sequence lengths
    print("\n=== Sequence Length Scaling ===")
    len_results = {}
    for sl in [16, 32, 64]:
        print(f"\n  Len={sl}:")
        x = xLSTMModel(vocab_size, d_model=32, n_layers=2, expand_factor=2).to(device)
        l = LSTMBaseline(vocab_size, d_model=32, n_layers=2).to(device)
        xl = train_model(x, vocab_size, sl, n_steps=1500, device=device)
        ll = train_model(l, vocab_size, sl, n_steps=1500, device=device)
        len_results[sl] = {'xlstm': xl[-1], 'lstm': ll[-1]}
        print(f"    xLSTM: {xl[-1]:.4f}, LSTM: {ll[-1]:.4f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 50
    for losses, label, color in [(xlstm_losses, 'xLSTM', 'blue'),
                                  (lstm_losses, 'LSTM', 'green'),
                                  (tf_losses, 'Transformer', 'red')]:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=label, color=color, linewidth=2)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sequence length scaling
    lens = sorted(len_results.keys())
    axes[1].plot(lens, [len_results[s]['xlstm'] for s in lens], 'o-', label='xLSTM', color='blue')
    axes[1].plot(lens, [len_results[s]['lstm'] for s in lens], 's--', label='LSTM', color='green')
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Loss vs Sequence Length")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("xLSTM: Extended LSTM with Exponential Gating", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "xlstm_comparison.png", dpi=150)
    plt.close()

    # 2. sLSTM vs mLSTM comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train sLSTM-only and mLSTM-only models
    print("\n=== sLSTM vs mLSTM Ablation ===")

    class sLSTMOnly(nn.Module):
        def __init__(self, vocab_size, d_model=64, n_layers=4):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([xLSTMBlock(d_model, 's', 2) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
        def forward(self, x):
            h = self.emb(x)
            for b in self.blocks:
                h = b(h)
            return self.head(self.ln(h))

    class mLSTMOnly(nn.Module):
        def __init__(self, vocab_size, d_model=64, n_layers=4):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([xLSTMBlock(d_model, 'm', n_heads=4, head_dim=d_model//4) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
        def forward(self, x):
            h = self.emb(x)
            for b in self.blocks:
                h = b(h)
            return self.head(self.ln(h))

    s_model = sLSTMOnly(vocab_size, d_model=32, n_layers=2).to(device)
    m_model = mLSTMOnly(vocab_size, d_model=32, n_layers=2).to(device)

    s_losses = train_model(s_model, vocab_size, seq_len, n_steps=1500, device=device)
    m_losses = train_model(m_model, vocab_size, seq_len, n_steps=1500, device=device)

    smoothed_s = np.convolve(s_losses, np.ones(window)/window, mode='valid')
    smoothed_m = np.convolve(m_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(smoothed_s, label='sLSTM (scalar memory)', color='blue')
    axes[0].plot(smoothed_m, label='mLSTM (matrix memory)', color='purple')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("sLSTM vs mLSTM")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Model size comparison
    models = ['xLSTM', 'LSTM', 'Transformer']
    params = [n_params_x, n_params_l, n_params_t]
    final_losses = [xlstm_losses[-1], lstm_losses[-1], tf_losses[-1]]

    x_pos = np.arange(len(models))
    width = 0.35
    axes[1].bar(x_pos - width/2, [p/1000 for p in params], width, label='Params (K)', color='blue', alpha=0.7)
    axes[1].bar(x_pos + width/2, final_losses, width, label='Final Loss', color='red', alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    axes[1].set_title("Model Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("xLSTM: Architecture Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "xlstm_analysis.png", dpi=150)
    plt.close()

    # 3. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Standard\nLSTM", "Scalar cell state c\nSigmoid gates\nLimited memory\ncapacity\n→ Forgets easily", 0.14, 'gray'),
        ("sLSTM\nExponential", "Exponential gating\nexp(i), exp(f)\nStabilization via\nnormalizer n\n→ Better control", 0.5, 'blue'),
        ("mLSTM\nMatrix Memory", "Matrix cell state C\nC = f*C + i*v⊗k^T\nKey-value store\nh = o*C*q\n→ Structured memory!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("xLSTM: From Scalar to Matrix Memory", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "xlstm_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
