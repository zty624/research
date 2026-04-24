"""
Minimal SEDD (Discrete Diffusion) Reproduction
===============================================
Reproduces core ideas from "Score Entropy Discrete Diffusion" (2310.16834,
Lou et al.):
1. Discrete state space: tokens (not continuous)
2. Forward process: uniform transition noise (replace token with random)
3. Reverse process: learn to denoise by predicting the clean token
4. Score entropy: alternative training objective for discrete diffusion
5. Compare: continuous diffusion (DDPM) vs discrete diffusion (SEDD)
6. Show: sampling quality vs number of denoising steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Discrete Diffusion Model ──

class DiscreteDenoiser(nn.Module):
    """Transformer-based denoiser for discrete diffusion.
    Input: (B, T) noisy token sequence + timestep t
    Output: (B, T, V) logits over vocabulary for each position
    """
    def __init__(self, vocab_size=32, d_model=128, n_heads=4, n_layers=4,
                 max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 for [MASK]
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       dropout=0.1, activation='gelu', batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, t):
        """x: (B, T) token indices, t: (B,) timesteps in [0, 1]."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos) + self.time_emb(t.unsqueeze(-1)).unsqueeze(1)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


# ── Discrete Forward Process ──

def discrete_forward_noise(x0, t, vocab_size, mask_token_id=None):
    """Apply discrete forward noise: replace tokens with [MASK] or random.

    At time t, each token is independently replaced with probability t:
    - 80% → [MASK] token
    - 10% → random token
    - 10% → keep original

    This follows the BERT-style masking schedule adapted for diffusion.
    """
    B, T = x0.shape
    # Noise probability increases with t
    keep_mask = torch.rand(B, T, device=x0.device) >= t.unsqueeze(-1)

    # Create noisy version
    x_noisy = x0.clone()

    # For positions to noise:
    noise_mask = ~keep_mask
    n_noise = noise_mask.sum().item()

    if n_noise > 0:
        rand_vals = torch.rand(B, T, device=x0.device)
        # 80% mask, 10% random, 10% keep
        mask_token = vocab_size  # [MASK] token ID = vocab_size
        mask_positions = noise_mask & (rand_vals < 0.8)
        random_positions = noise_mask & (rand_vals >= 0.8) & (rand_vals < 0.9)

        x_noisy[mask_positions] = mask_token
        x_noisy[random_positions] = torch.randint(0, vocab_size,
                                                   (random_positions.sum().item(),),
                                                   device=x0.device)

    return x_noisy, keep_mask


# ── Training: Score Entropy ──

def train_sedd(model, data_fn, n_steps=3000, batch_size=64, lr=1e-3,
               vocab_size=32, device='cpu'):
    """Train with score entropy discrete diffusion objective."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []
    accs = []

    for step in range(n_steps):
        x0 = data_fn(batch_size).to(device)
        B, T = x0.shape

        # Sample timestep
        t = torch.rand(B, device=device) * 0.98 + 0.01  # avoid t=0,1

        # Apply forward noise
        x_noisy, keep_mask = discrete_forward_noise(x0, t, vocab_size)

        # Predict clean tokens
        logits = model(x_noisy, t)  # (B, T, V)

        # Score entropy loss: cross-entropy on noised positions
        # Weight by 1/t to emphasize early denoising (high noise)
        ce = F.cross_entropy(logits.reshape(-1, vocab_size),
                             x0.reshape(-1), reduction='none')
        ce = ce.reshape(B, T)

        # Only compute loss on noised positions
        noise_mask = ~keep_mask
        if noise_mask.any():
            # Time weighting: 1/(1-t) to upweight early steps
            weight = 1.0 / (1.0 - t.unsqueeze(-1) + 1e-6)
            loss = (ce * noise_mask.float() * weight).sum() / noise_mask.float().sum()
        else:
            loss = ce.mean() * 0.0  # no noised tokens

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Accuracy on noised positions
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            if noise_mask.any():
                acc = (pred[noise_mask] == x0[noise_mask]).float().mean().item()
            else:
                acc = 1.0

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    return losses, accs


def train_x0_prediction(model, data_fn, n_steps=3000, batch_size=64, lr=1e-3,
                         vocab_size=32, device='cpu'):
    """Train with simple x0-prediction objective (no score entropy weighting)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []
    accs = []

    for step in range(n_steps):
        x0 = data_fn(batch_size).to(device)
        B, T = x0.shape

        t = torch.rand(B, device=device) * 0.98 + 0.01
        x_noisy, keep_mask = discrete_forward_noise(x0, t, vocab_size)

        logits = model(x_noisy, t)

        # Simple CE on noised positions (no time weighting)
        ce = F.cross_entropy(logits.reshape(-1, vocab_size),
                             x0.reshape(-1), reduction='none')
        ce = ce.reshape(B, T)
        noise_mask = ~keep_mask
        if noise_mask.any():
            loss = ce[noise_mask].mean()
        else:
            loss = ce.mean() * 0.0

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            if noise_mask.any():
                acc = (pred[noise_mask] == x0[noise_mask]).float().mean().item()
            else:
                acc = 1.0

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    return losses, accs


# ── Sampling ──

def sample_discrete_diffusion(model, n_samples, seq_len, n_steps, vocab_size,
                               device='cpu', schedule='linear'):
    """Sample from discrete diffusion via iterative denoising."""
    model.eval()
    mask_token = vocab_size

    # Start from all [MASK]
    x = torch.full((n_samples, seq_len), mask_token, dtype=torch.long, device=device)

    trajectories = [x.cpu().clone()]

    for step in range(n_steps):
        # Current noise level: goes from 1.0 → 0.0
        if schedule == 'linear':
            t_cur = 1.0 - step / n_steps
            t_next = 1.0 - (step + 1) / n_steps
        elif schedule == 'cosine':
            t_cur = np.cos(step / n_steps * np.pi / 2) ** 2
            t_next = np.cos((step + 1) / n_steps * np.pi / 2) ** 2

        t_tensor = torch.full((n_samples,), t_cur, device=device)

        with torch.no_grad():
            logits = model(x, t_tensor)  # (B, T, V)

        # For positions that are [MASK] or should be re-noised:
        # Predict clean tokens
        probs = F.softmax((logits / 0.5).clamp(-20, 20), dim=-1)  # temperature

        # Determine which positions to update
        # Unmask probability: p_unmask = (t_cur - t_next) / t_cur
        if t_cur > 0:
            p_unmask = (t_cur - t_next) / t_cur
        else:
            p_unmask = 1.0

        # Update: sample new tokens for a fraction of positions
        update_mask = (torch.rand(n_samples, seq_len, device=device) < p_unmask) & (x == mask_token)

        # Sample from predicted distribution
        new_tokens = torch.multinomial(probs.reshape(-1, vocab_size), 1).reshape(n_samples, seq_len)
        x = torch.where(update_mask, new_tokens, x)

        trajectories.append(x.cpu().clone())

    return x, trajectories


# ── Synthetic Data ──

def generate_structured_sequences(batch_size, vocab_size=32, seq_len=16, device='cpu'):
    """Generate synthetic structured token sequences.

    Patterns: [A B A B ...], [A A B B ...], etc.
    """
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    patterns = torch.randint(0, 4, (batch_size,))  # 4 pattern types

    for i in range(batch_size):
        p = patterns[i].item()
        if p == 0:  # Alternating AB
            a, b = torch.randint(0, vocab_size, (2,))
            for j in range(seq_len):
                sequences[i, j] = a if j % 2 == 0 else b
        elif p == 1:  # Repeating AAA
            a = torch.randint(0, vocab_size, (1,))
            sequences[i] = a
        elif p == 2:  # Block AABBAABB
            a, b = torch.randint(0, vocab_size, (2,))
            for j in range(seq_len):
                sequences[i, j] = a if (j // 2) % 2 == 0 else b
        else:  # Incremental
            start = torch.randint(0, vocab_size // 2, (1,))
            for j in range(seq_len):
                sequences[i, j] = (start + j) % vocab_size

    # Add small noise
    noise_mask = torch.rand(batch_size, seq_len) < 0.05
    noise_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    sequences = torch.where(noise_mask, noise_tokens, sequences)

    return sequences


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "110-sedd-discrete-diffusion"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    seq_len = 16

    data_fn = lambda bs: generate_structured_sequences(bs, vocab_size, seq_len, device='cpu')

    # ── Train SEDD (score entropy) ──
    print("=== Training SEDD (Score Entropy) ===")
    sedd_model = DiscreteDenoiser(vocab_size, d_model=128, n_heads=4,
                                   n_layers=4, max_len=seq_len).to(device)
    sedd_params = sum(p.numel() for p in sedd_model.parameters())
    print(f"  Params: {sedd_params:,}")
    sedd_losses, sedd_accs = train_sedd(sedd_model, data_fn, n_steps=3000,
                                         batch_size=64, vocab_size=vocab_size, device=device)

    # ── Train x0-prediction baseline ──
    print("\n=== Training x0-Prediction (No Score Entropy) ===")
    x0_model = DiscreteDenoiser(vocab_size, d_model=128, n_heads=4,
                                 n_layers=4, max_len=seq_len).to(device)
    x0_losses, x0_accs = train_x0_prediction(x0_model, data_fn, n_steps=3000,
                                               batch_size=64, vocab_size=vocab_size, device=device)

    # ── Sampling comparison ──
    print("\n=== Sampling Comparison ===")
    step_counts = [2, 4, 8, 16, 32, 64]

    sedd_qualities = []
    x0_qualities = []

    # Reference data for comparison
    ref_data = data_fn(500)

    for n_steps in step_counts:
        # SEDD sampling
        sedd_samples, _ = sample_discrete_diffusion(
            sedd_model, 100, seq_len, n_steps, vocab_size, device=device)

        # x0-prediction sampling
        x0_samples, _ = sample_discrete_diffusion(
            x0_model, 100, seq_len, n_steps, vocab_size, device=device)

        # Quality: fraction of samples matching a known pattern
        def pattern_match_rate(samples):
            matches = 0
            for s in samples:
                s = s.cpu().numpy()
                # Check if alternating
                if len(set(s[::2])) <= 2 and len(set(s[1::2])) <= 2:
                    matches += 1
                # Check if uniform
                elif len(set(s)) <= 2:
                    matches += 1
            return matches / len(samples)

        sq = pattern_match_rate(sedd_samples)
        xq = pattern_match_rate(x0_samples)
        sedd_qualities.append(sq)
        x0_qualities.append(xq)

        print(f"  Steps={n_steps}: SEDD pattern rate={sq:.3f}, x0-pred pattern rate={xq:.3f}")

    # ── Denoising trajectory visualization ──
    print("\n=== Denoising Trajectory ===")
    samples_8, traj_8 = sample_discrete_diffusion(
        sedd_model, 5, seq_len, 16, vocab_size, device=device)

    # ── Visualization ──

    # 1. Training loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 30

    sedd_s = np.convolve(sedd_losses, np.ones(w)/w, mode='valid')
    x0_s = np.convolve(x0_losses, np.ones(w)/w, mode='valid')
    axes[0].plot(sedd_s, label='SEDD (Score Entropy)', color='blue')
    axes[0].plot(x0_s, label='x0-Prediction', color='red')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sedd_a = np.convolve(sedd_accs, np.ones(w)/w, mode='valid')
    x0_a = np.convolve(x0_accs, np.ones(w)/w, mode='valid')
    axes[1].plot(sedd_a, label='SEDD (Score Entropy)', color='blue')
    axes[1].plot(x0_a, label='x0-Prediction', color='red')
    axes[1].set_title("Denoising Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('SEDD vs x0-Prediction: Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training_comparison.png', dpi=150)
    plt.close()

    # 2. Sampling quality vs steps
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(step_counts, sedd_qualities, marker='o', label='SEDD', color='blue', linewidth=2)
    ax.plot(step_counts, x0_qualities, marker='s', label='x0-Prediction', color='red', linewidth=2)
    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("Pattern Match Rate")
    ax.set_title("Sampling Quality vs Denoising Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'sampling_quality.png', dpi=150)
    plt.close()

    # 3. Denoising trajectory
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    sample_idx = 0
    for step_idx, ax in enumerate(axes):
        t_idx = step_idx * (len(traj_8) - 1) // 4
        tokens = traj_8[t_idx][sample_idx].cpu().numpy()
        # Show as heatmap of token values
        ax.imshow(tokens.reshape(1, -1), cmap='tab20', aspect='auto', vmin=0, vmax=vocab_size)
        t_val = 1.0 - t_idx / (len(traj_8) - 1) if len(traj_8) > 1 else 0
        ax.set_title(f"t={t_val:.2f} ({t_idx}/{len(traj_8)-1} steps)", fontsize=10)
        ax.set_xlabel("Position")
        ax.set_yticks([])
    axes[0].set_ylabel("Sample")
    plt.suptitle("Discrete Diffusion: Denoising Trajectory", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'denoising_trajectory.png', dpi=150)
    plt.close()

    # 4. Continuous vs Discrete diffusion concept
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Continuous diffusion
    ax = axes[0]
    ax.axis('off')
    ax.set_title("Continuous Diffusion (DDPM)", fontsize=13, fontweight='bold')
    cont_text = (
        "State space: R^n (continuous)\n"
        "Forward: x_t = sqrt(α_t) x_0 + sqrt(1-α_t) ε\n"
        "Reverse: predict noise ε or x_0\n"
        "Loss: MSE on noise prediction\n\n"
        "x_0 → x_0.1 → x_0.5 → x_0.9 → x_1.0\n"
        "clean → slightly noisy → noisy → very noisy → pure noise"
    )
    ax.text(0.05, 0.95, cont_text, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0f0ff', alpha=0.9))

    # Discrete diffusion
    ax = axes[1]
    ax.axis('off')
    ax.set_title("Discrete Diffusion (SEDD)", fontsize=13, fontweight='bold')
    disc_text = (
        "State space: {0, 1, ..., V-1}^T (discrete)\n"
        "Forward: replace token → [MASK] w.p. t\n"
        "Reverse: predict clean token from context\n"
        "Loss: Score entropy weighted CE\n\n"
        "[A B C D] → [A _ C _] → [_ _ C _] → [_ _ _ _]\n"
        "clean   → partially masked → mostly masked → all [MASK]"
    )
    ax.text(0.05, 0.95, disc_text, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe0e0', alpha=0.9))

    plt.suptitle('Continuous vs Discrete Diffusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'continuous_vs_discrete.png', dpi=150)
    plt.close()

    # 5. Noise schedule visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    t_range = np.linspace(0, 1, 100)
    # Linear schedule
    mask_prob_linear = t_range
    # Cosine schedule
    mask_prob_cosine = 1 - np.cos(t_range * np.pi / 2) ** 2
    # Sqrt schedule
    mask_prob_sqrt = np.sqrt(t_range)

    ax.plot(t_range, mask_prob_linear, label='Linear', color='blue')
    ax.plot(t_range, mask_prob_cosine, label='Cosine', color='red')
    ax.plot(t_range, mask_prob_sqrt, label='Sqrt', color='green')
    ax.set_xlabel("Diffusion Time t")
    ax.set_ylabel("Mask Probability")
    ax.set_title("Discrete Diffusion Noise Schedules")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'noise_schedules.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
