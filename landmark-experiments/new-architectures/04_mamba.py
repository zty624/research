"""
Minimal Mamba / Selective SSM Reproduction
============================================
Reproduces the core ideas from "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces" (2312.00752):
1. SSM recurrence: h_t = A*h_{t-1} + B*x_t, y_t = C*h_t
2. Selection mechanism: B, C, Δ are input-dependent (vs fixed in S4)
3. Compare: Time-invariant SSM vs Selective SSM on synthetic tasks
4. Demonstrates selective copying and induction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ── S4-style Time-Invariant SSM ──

class S4Block(nn.Module):
    """Time-invariant SSM (like S4): B, C, Δ are fixed parameters."""
    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Fixed parameters
        self.log_A = nn.Parameter(torch.log(torch.ones(d_model, d_state) * 0.9))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.log_dt = nn.Parameter(torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min))

        self.D = nn.Parameter(torch.ones(d_model))  # skip connection
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        B, L, D = x.shape
        A = -torch.exp(self.log_A)  # Ensure A is negative (stable)
        dt = torch.exp(self.log_dt)  # (D,)

        # Discretize: A_bar = exp(A*dt), B_bar = (A^{-1}(A_bar - I)) @ B
        # Simplified: A_bar ≈ I + A*dt (Euler), B_bar ≈ B*dt
        A_bar = torch.exp(A * dt.unsqueeze(-1))  # (D, N)
        B_bar = self.B * dt.unsqueeze(-1)  # (D, N)

        # Recurrence
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)  # (B, D, N)
            y = (h * self.C).sum(-1) + self.D * x[:, t]  # (B, D)
            outputs.append(y)

        out = torch.stack(outputs, dim=1)  # (B, L, D)
        return self.proj(F.silu(out))


# ── Selective SSM (Mamba-style) ──

class SelectiveSSMBlock(nn.Module):
    """Selective SSM (Mamba): B, C, Δ are input-dependent."""
    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is still a fixed parameter (diagonal)
        self.log_A = nn.Parameter(torch.log(torch.ones(d_model, d_state) * 0.9))
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections for B, C, Δ
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_dt = nn.Linear(d_model, 1)

        self.dt_min = dt_min
        self.dt_max = dt_max

        # Gating (like Mamba: split input, one branch through SSM, other as gate)
        self.proj_in = nn.Linear(d_model, d_model * 2)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        B, L, D = x.shape
        A = -torch.exp(self.log_A)  # (D, N), negative for stability

        # Gating: split input
        xz = self.proj_in(x)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, D)

        # Input-dependent parameters
        Bt = self.proj_B(x_branch)  # (B, L, N)
        Ct = self.proj_C(x_branch)  # (B, L, N)
        dt = F.softplus(self.proj_dt(x_branch).squeeze(-1))  # (B, L)
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(dt)
        # dt: (B, L) → (B, L, 1) for broadcasting

        # Discretize per step
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1).unsqueeze(-1))
        # A_bar: (B, L, D, N)
        # B_bar: (B, L, D, N) — project input through B for each position
        # We need B_bar to be (B, L, D, N) to multiply with x[:, t] of shape (B, D)
        B_bar = Bt.unsqueeze(2).expand(-1, -1, D, -1) * dt.unsqueeze(-1).unsqueeze(-1)
        # B_bar: (B, L, D, N)

        # Recurrence
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t_idx in range(L):
            h = A_bar[:, t_idx] * h + B_bar[:, t_idx] * x_branch[:, t_idx].unsqueeze(-1)
            # h: (B, D, N), x_branch[:, t_idx].unsqueeze(-1): (B, D, 1)
            y = (h * Ct[:, t_idx].unsqueeze(1)).sum(-1) + self.D * x_branch[:, t_idx]
            # y: (B, D)
            outputs.append(y)

        out = torch.stack(outputs, dim=1)  # (B, L, D)
        out = out * F.silu(z)  # Gating
        return self.proj_out(out)


# ── Synthetic Tasks ──

def generate_selective_copy(seq_len=64, vocab_size=10, n_special=2, device='cpu'):
    """
    Selective Copy: special tokens appear at random positions in a sequence.
    Model must output the special tokens in order at the end.
    Tokens 0..n_special-1 are "memorize" tokens, rest are noise.
    """
    B = 32
    x = torch.randint(n_special, vocab_size, (B, seq_len), device=device)
    # Place 3-5 special tokens at random positions
    special = torch.randint(0, n_special, (B, 5), device=device)
    positions = torch.randint(0, seq_len - 5, (B, 5), device=device)
    for i in range(5):
        x[torch.arange(B), positions[:, i]] = special[:, i]

    # Target: the special tokens in order (padded)
    y = special
    return x, y


def generate_induction_head(seq_len=64, vocab_size=20, device='cpu'):
    """
    Induction Head: sequence has pattern A...A, model must predict
    what comes after the second A.
    """
    B = 32
    x = torch.randint(2, vocab_size, (B, seq_len), device=device)
    # Place a pattern: token X at position i, token Y at position i+1,
    # then token X again later, expecting Y
    for b in range(B):
        pos1 = np.random.randint(0, seq_len // 2)
        pos2 = np.random.randint(seq_len // 2, seq_len - 1)
        x[b, pos2] = x[b, pos1]  # Same token appears twice
        # Target: predict x[b, pos1+1] when seeing x[b, pos2]
    return x, x  # Auto-regressive: predict next token


# ── Training ──

def train_model(model, task_fn, n_steps=3000, lr=1e-3, device='cpu', task_name=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    accs = []

    for step in range(n_steps):
        x, y = task_fn(device=device)
        out = model(x)  # (B, L, vocab)

        # For selective copy: predict the special tokens from the last few positions
        if "selective" in task_name:
            pred = out[:, -5:, :]  # Last 5 positions
            target = y  # (B, 5)
            loss = F.cross_entropy(pred.reshape(-1, out.size(-1)), target.reshape(-1))
            acc = (pred.argmax(-1) == target).float().mean().item()
        else:
            # Next token prediction
            loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1))
            acc = (out[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "04-mamba"
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = 32
    n_steps = 3000

    # ── Selective Copy Task ──
    print("=== Selective Copy Task ===\n")

    print("Training S4 (time-invariant SSM)...")
    s4_model = nn.Sequential(
        nn.Embedding(10, d_model),
        S4Block(d_model),
        nn.Linear(d_model, 10)
    ).to(device)
    s4_losses, s4_accs = train_model(s4_model, generate_selective_copy, n_steps, device=device, task_name="selective")

    print("\nTraining Selective SSM (Mamba-style)...")
    mamba_model = nn.Sequential(
        nn.Embedding(10, d_model),
        SelectiveSSMBlock(d_model),
        nn.Linear(d_model, 10)
    ).to(device)
    mamba_losses, mamba_accs = train_model(mamba_model, generate_selective_copy, n_steps, device=device, task_name="selective")

    # ── Visualization ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    window = 30
    s4_loss_smooth = np.convolve(s4_losses, np.ones(window)/window, mode='valid')
    mamba_loss_smooth = np.convolve(mamba_losses, np.ones(window)/window, mode='valid')
    s4_acc_smooth = np.convolve(s4_accs, np.ones(window)/window, mode='valid')
    mamba_acc_smooth = np.convolve(mamba_accs, np.ones(window)/window, mode='valid')

    ax1.plot(s4_loss_smooth, label='S4 (time-invariant)', color='red')
    ax1.plot(mamba_loss_smooth, label='Selective SSM (Mamba)', color='blue')
    ax1.set_title("Selective Copy: Training Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss (smoothed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(s4_acc_smooth, label='S4 (time-invariant)', color='red')
    ax2.plot(mamba_acc_smooth, label='Selective SSM (Mamba)', color='blue')
    ax2.set_title("Selective Copy: Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (smoothed)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "selective_copy_comparison.png", dpi=150)
    plt.close()

    # ── Δ visualization: show input-dependent step sizes ──
    mamba_block = None
    for module in mamba_model.modules():
        if isinstance(module, SelectiveSSMBlock):
            mamba_block = module
            break

    if mamba_block is not None:
        mamba_block.eval()
        with torch.no_grad():
            x, _ = generate_selective_copy(device=device)
            x_emb = mamba_model[0](x)  # Embedding
            x_branch = mamba_block.proj_in(x_emb)
            x_b, _ = x_branch.chunk(2, dim=-1)
            dt = F.softplus(mamba_block.proj_dt(x_b).squeeze(-1))
            dt = mamba_block.dt_min + (mamba_block.dt_max - mamba_block.dt_min) * torch.sigmoid(dt)

        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(dt[:5].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.set_title("Input-Dependent Step Size (Δ) — Brighter = Larger Δ (more attention to current input)")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Batch Sample")
        plt.colorbar(im, label='Δ value')
        plt.tight_layout()
        plt.savefig(results_dir / "delta_visualization.png", dpi=150)
        plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
