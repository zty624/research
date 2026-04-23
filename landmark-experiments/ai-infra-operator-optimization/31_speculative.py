"""
Minimal Speculative Decoding Reproduction
=========================================
Reproduces core ideas from speculative decoding (2211.17192, Leviathan et al.):
1. Draft model proposes tokens, target model verifies in parallel
2. Accept/reject based on modified rejection sampling
3. Exact output distribution preserved (no quality loss!)
4. Compare: autoregressive vs speculative decoding speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── Small Language Model ──

class SmallLM(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate_autoregressive(self, prompt, n_tokens=20, temperature=1.0):
        """Standard autoregressive generation."""
        x = prompt.clone()
        tokens = []
        for _ in range(n_tokens):
            if x.shape[1] >= self.max_len:
                break
            logits = self.forward(x)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            token = torch.multinomial(probs, 1)
            x = torch.cat([x, token], dim=1)
            tokens.append(token.item())
        return x, tokens

    @torch.no_grad()
    def get_probs(self, x, temperature=1.0):
        """Get probability distribution for next token."""
        logits = self.forward(x)
        return F.softmax(logits[:, -1, :] / temperature, dim=-1)


# ── Speculative Decoding ──

@torch.no_grad()
def speculative_decode(target_model, draft_model, prompt, n_tokens=20,
                       gamma=4, temperature=1.0):
    """Speculative decoding: draft proposes, target verifies.

    gamma: number of tokens draft proposes per step
    """
    x = prompt.clone()
    all_tokens = []
    n_draft_tokens = 0
    n_accepted_tokens = 0
    n_target_calls = 0

    while len(all_tokens) < n_tokens:
        # Step 1: Draft model proposes gamma tokens
        draft_tokens = []
        draft_probs = []
        draft_x = x.clone()

        for _ in range(gamma):
            if draft_x.shape[1] >= target_model.max_len:
                break
            probs = draft_model.get_probs(draft_x, temperature)
            token = torch.multinomial(probs, 1)
            draft_tokens.append(token)
            draft_probs.append(probs)
            draft_x = torch.cat([draft_x, token], dim=1)

        n_draft_tokens += len(draft_tokens)

        # Step 2: Target model verifies all draft tokens in ONE forward pass
        # We need target probs at each position
        draft_tensor = torch.cat(draft_tokens, dim=1)  # (1, gamma)
        target_x = torch.cat([x, draft_tensor], dim=1)

        # Get target logits for all positions at once
        target_logits = target_model(target_x)
        n_target_calls += 1

        # Step 3: Accept/reject each draft token
        accepted = 0
        for i in range(len(draft_tokens)):
            # Target probability for this position
            target_p = F.softmax(target_logits[:, x.shape[1] + i - 1, :] / temperature, dim=-1)
            draft_p = draft_probs[i]

            token_id = draft_tokens[i].item()

            # Acceptance criterion: r < min(1, p_target(x) / p_draft(x))
            ratio = (target_p[0, token_id] / (draft_p[0, token_id] + 1e-10)).item()
            acceptance_prob = min(1.0, ratio)

            if np.random.random() < acceptance_prob:
                accepted += 1
                all_tokens.append(token_id)
            else:
                # Reject: sample from adjusted distribution
                # q(x) = max(0, p_target(x) - p_draft(x)) / Z
                adjusted = torch.clamp(target_p - draft_p, min=0)
                adjusted = adjusted / (adjusted.sum() + 1e-10)
                corrected_token = torch.multinomial(adjusted, 1)
                all_tokens.append(corrected_token.item())
                break

        n_accepted_tokens += accepted

        # Also sample one more token from target (the bonus token after all accepted)
        if accepted == len(draft_tokens) and len(all_tokens) < n_tokens:
            # All draft tokens accepted, sample bonus from target
            bonus_p = F.softmax(target_logits[:, -1, :] / temperature, dim=-1)
            bonus_token = torch.multinomial(bonus_p, 1)
            all_tokens.append(bonus_token.item())

        # Update prompt
        new_tokens = torch.tensor([all_tokens[-min(len(all_tokens), gamma+1):]],
                                   dtype=torch.long, device=x.device)
        # Actually just rebuild x from prompt + all generated tokens
        gen_tokens = torch.tensor([all_tokens], dtype=torch.long, device=x.device)
        x = torch.cat([prompt, gen_tokens], dim=1)

    return x, all_tokens[:n_tokens], n_draft_tokens, n_accepted_tokens, n_target_calls


# ── Training ──

def train_lm(model, data_seq, n_steps=3000, batch_size=32, seq_len=16, lr=1e-3, device='cpu'):
    """Train language model on a data sequence."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        # Sample random subsequences
        starts = torch.randint(0, len(data_seq) - seq_len - 1, (batch_size,))
        x = torch.stack([data_seq[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data_seq[s+1:s+seq_len+1] for s in starts]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f}")

    return losses


def generate_data(vocab_size=32, length=5000):
    """Generate structured sequence data: repeating patterns with some randomness."""
    data = []
    for i in range(length):
        # Pattern: digits follow a structured sequence
        base = (i % 10)
        noise = np.random.randint(0, 3)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "31-speculative"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    data = generate_data(vocab_size, length=5000)

    # Train target model (larger)
    print("=== Training Target Model ===")
    target = SmallLM(vocab_size, d_model=64, n_heads=2, n_layers=3).to(device)
    target_params = sum(p.numel() for p in target.parameters())
    print(f"  Target params: {target_params:,}")
    target_losses = train_lm(target, data, n_steps=3000, device=device)

    # Train draft model (smaller, faster)
    print("\n=== Training Draft Model ===")
    draft = SmallLM(vocab_size, d_model=32, n_heads=2, n_layers=1, max_len=64).to(device)
    draft_params = sum(p.numel() for p in draft.parameters())
    print(f"  Draft params: {draft_params:,}")
    draft_losses = train_lm(draft, data, n_steps=3000, lr=3e-3, device=device)

    # ── Benchmark ──
    print("\n=== Benchmarking Generation ===")
    n_gen = 30
    n_runs = 20

    # Autoregressive (target)
    ar_times = []
    ar_tokens_per_sec = []
    for _ in range(n_runs):
        prompt = torch.tensor([[data[0].item()]], dtype=torch.long, device=device)
        t0 = time.time()
        _, tokens = target.generate_autoregressive(prompt, n_gen, temperature=0.8)
        t1 = time.time()
        ar_times.append(t1 - t0)
        ar_tokens_per_sec.append(n_gen / (t1 - t0))

    # Speculative decoding
    spec_times = []
    spec_tokens_per_sec = []
    spec_accept_rates = []
    spec_target_calls = []

    for gamma in [2, 4, 6]:
        times_g = []
        tps_g = []
        acc_g = []
        calls_g = []
        for _ in range(n_runs):
            prompt = torch.tensor([[data[0].item()]], dtype=torch.long, device=device)
            t0 = time.time()
            _, tokens, n_draft, n_accepted, n_calls = speculative_decode(
                target, draft, prompt, n_gen, gamma=gamma, temperature=0.8
            )
            t1 = time.time()
            times_g.append(t1 - t0)
            tps_g.append(len(tokens) / (t1 - t0))
            acc_g.append(n_accepted / max(n_draft, 1))
            calls_g.append(n_calls)

        spec_times.append(times_g)
        spec_tokens_per_sec.append(tps_g)
        spec_accept_rates.append(acc_g)
        spec_target_calls.append(calls_g)

        avg_ar = np.mean(ar_tokens_per_sec)
        avg_spec = np.mean(tps_g)
        speedup = avg_spec / avg_ar
        print(f"  γ={gamma}: Accept rate={np.mean(acc_g):.2%}, "
              f"Target calls={np.mean(calls_g):.1f}, "
              f"Speedup={speedup:.2f}x")

    # ── Quality Check ──
    print("\n=== Quality Check ===")
    # Verify speculative decoding produces same distribution
    prompt = torch.tensor([[data[0].item()]], dtype=torch.long, device=device)

    # Generate many samples autoregressive
    ar_samples = []
    for _ in range(100):
        _, tokens = target.generate_autoregressive(prompt, 10, temperature=1.0)
        ar_samples.append(tokens[0] if tokens else 0)

    # Generate many samples speculative
    spec_samples = []
    for _ in range(100):
        _, tokens, _, _, _ = speculative_decode(target, draft, prompt, 10, gamma=4, temperature=1.0)
        spec_samples.append(tokens[0] if tokens else 0)

    # Compare distributions
    ar_dist = np.bincount(ar_samples, minlength=vocab_size) / len(ar_samples)
    spec_dist = np.bincount(spec_samples, minlength=vocab_size) / len(spec_samples)
    kl_div = np.sum(ar_dist * np.log((ar_dist + 1e-10) / (spec_dist + 1e-10)))
    print(f"  KL divergence (AR || Spec): {kl_div:.4f}")
    print(f"  (Should be near 0 if distributions match)")

    # ── Visualization ──

    # 1. Training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 30
    target_s = np.convolve(target_losses, np.ones(window)/window, mode='valid')
    draft_s = np.convolve(draft_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(target_s, label=f'Target ({target_params:,} params)', color='blue')
    axes[0].plot(draft_s, label=f'Draft ({draft_params:,} params)', color='red')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Generation speedup
    gammas = [2, 4, 6]
    ar_mean = np.mean(ar_tokens_per_sec)
    spec_means = [np.mean(tps) for tps in spec_tokens_per_sec]
    speedups = [s / ar_mean for s in spec_means]

    bars = axes[1].bar(['Auto-\nregressive'] + [f'Speculative\nγ={g}' for g in gammas],
                [ar_mean] + spec_means,
                color=['gray', 'blue', 'green', 'orange'], alpha=0.7)
    axes[1].set_ylabel("Tokens/second")
    axes[1].set_title("Generation Speed")
    axes[1].grid(True, alpha=0.3, axis='y')
    # Add speedup labels above bars
    heights = [ar_mean] + spec_means
    splayers = [1.0] + speedups
    for i, (h, sp) in enumerate(zip(heights, splayers)):
        axes[1].text(i, h + 1, f'{sp:.1f}x', ha='center', fontweight='bold')

    plt.suptitle("Speculative Decoding: Draft-then-Verify", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "speedup_comparison.png", dpi=150)
    plt.close()

    # 3. Accept rate vs gamma
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    avg_accepts = [np.mean(a) for a in spec_accept_rates]
    axes[0].bar([f'γ={g}' for g in gammas], avg_accepts, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[0].set_ylabel("Accept Rate")
    axes[0].set_title("Draft Accept Rate")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(avg_accepts):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center')

    # Target calls
    avg_calls = [np.mean(c) for c in spec_target_calls]
    ar_calls = n_gen  # AR needs n_gen forward passes
    axes[1].bar(['AR'] + [f'γ={g}' for g in gammas],
                [ar_calls] + avg_calls,
                color=['gray', 'blue', 'green', 'orange'], alpha=0.7)
    axes[1].set_ylabel("Target Model Forward Passes")
    axes[1].set_title("Target Model Calls (lower = better)")
    axes[1].grid(True, alpha=0.3, axis='y')

    # Distribution comparison
    x_range = range(min(15, vocab_size))
    axes[2].bar(np.array(x_range) - 0.15, ar_dist[x_range], 0.3, label='AR', color='gray', alpha=0.7)
    axes[2].bar(np.array(x_range) + 0.15, spec_dist[x_range], 0.3, label='Speculative', color='blue', alpha=0.7)
    axes[2].set_xlabel("Token")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Output Distribution (should match)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Speculative Decoding Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "speculative_analysis.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    steps = [
        ("1. Draft\nModel", "Fast small model\nproposes γ tokens\nsequentially", 0.14, 'red'),
        ("2. Target\nVerify", "Large model scores\nall γ tokens in\nONE forward pass", 0.43, 'blue'),
        ("3. Accept\n/ Reject", "Modified rejection\nsampling preserves\nexact distribution", 0.71, 'green'),
        ("4. Bonus\nToken", "If all accepted,\nsample one more\nfrom target", 0.93, 'purple'),
    ]

    for name, desc, x_pos, color in steps:
        ax.text(x_pos, 0.75, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.28, 0.57, 0.82]:
        ax.annotate('→', xy=(x, 0.55), fontsize=24, ha='center', va='center', color='gray')

    ax.set_title("Speculative Decoding: No Quality Loss, Faster Generation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "speculative_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
