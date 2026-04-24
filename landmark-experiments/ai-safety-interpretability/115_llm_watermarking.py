"""
Minimal LLM Watermarking Reproduction
======================================
Reproduces core ideas from "A Watermark for Large Language Models"
(2301.10226, Kirchenbauer et al.):
1. Soft watermarking: bias token probabilities during generation
2. Green/red token lists: partition vocabulary, boost green tokens
3. Detection: z-test on green token frequency in generated text
4. Compare: watermarked vs unwatermarked text statistics
5. Show: detection power vs watermark strength vs text quality
6. Demonstrate: robustness to paraphrase attack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Watermarking ──

class WatermarkProcessor:
    """Kirchenbauer-style watermarking for autoregressive generation.

    Key idea: hash the previous token to seed a random partition of the
    vocabulary into "green" and "red" lists. Boost green token logits
    by a fixed delta. During detection, count green tokens and run a
    z-test against the null hypothesis (no watermark).
    """
    def __init__(self, vocab_size=100, green_fraction=0.5, delta=2.0,
                 hashing_key=42, device='cpu'):
        self.vocab_size = vocab_size
        self.green_fraction = green_fraction
        self.delta = delta
        self.hashing_key = hashing_key
        self.device = device
        # Pre-compute green lists for each previous token
        self.green_lists = {}
        rng = np.random.RandomState(hashing_key)
        for prev_tok in range(vocab_size + 1):  # +1 for start token
            perm = rng.permutation(vocab_size)
            n_green = int(vocab_size * green_fraction)
            self.green_lists[prev_tok] = set(perm[:n_green].tolist())

    def get_green_list(self, prev_token):
        return self.green_lists.get(prev_token, set())

    def modify_logits(self, logits, prev_token):
        """Boost green token logits by delta."""
        green = self.get_green_list(prev_token)
        if not green:
            return logits
        mask = torch.zeros_like(logits)
        for g in green:
            mask[g] = self.delta
        return logits + mask

    def detect(self, tokens, z_threshold=4.0):
        """Detect watermark in a token sequence using z-test.

        Returns: (is_watermarked, z_score, green_count, total_count)
        """
        n = len(tokens) - 1  # exclude first token
        if n == 0:
            return False, 0.0, 0, 0

        green_count = 0
        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            green_list = self.get_green_list(prev)
            if curr in green_list:
                green_count += 1

        # Under null (no watermark): green count ~ Binomial(n, green_fraction)
        # z = (observed - expected) / std
        expected = n * self.green_fraction
        std = np.sqrt(n * self.green_fraction * (1 - self.green_fraction))
        z = (green_count - expected) / (std + 1e-10)

        return z > z_threshold, z, green_count, n


# ── Tiny LM ──

class TinyLM(nn.Module):
    """Small language model for watermarking experiments."""
    def __init__(self, vocab_size=100, d_model=64, n_heads=2, n_layers=2, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=20, temperature=1.0,
                 watermark_processor=None):
        """Generate with optional watermarking."""
        B = prompts.shape[0]
        current = prompts.clone()
        for _ in range(max_new_tokens):
            if current.shape[1] >= self.max_len:
                break
            logits = self.forward(current)[:, -1, :] / temperature
            logits = logits.clamp(-10, 10)

            # Apply watermark if provided
            if watermark_processor is not None:
                for b in range(B):
                    prev_tok = current[b, -1].item()
                    logits[b] = watermark_processor.modify_logits(logits[b], prev_tok)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            current = torch.cat([current, next_tok], dim=1)
        return current


# ── Training ──

def train_lm(model, n_steps=2000, batch_size=32, lr=1e-3, vocab_size=100,
              max_len=32, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []

    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, max_len), device=device)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f}")
    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "115-llm-watermarking"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 100
    max_len = 32

    # Train model
    print("=== Training Tiny LM ===")
    model = TinyLM(vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=max_len).to(device)
    losses = train_lm(model, n_steps=2000, device=device)

    # ── Experiment 1: Detection power vs watermark strength ──
    print("\n=== Experiment 1: Detection Power vs Delta ===")
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    n_gen = 100
    prompt_len = 4

    detection_results = {}
    for delta in deltas:
        wm = WatermarkProcessor(vocab_size, green_fraction=0.5, delta=delta, device=device)
        prompts = torch.randint(0, vocab_size, (n_gen, prompt_len), device=device)
        generated = model.generate(prompts, max_new_tokens=20, watermark_processor=wm)

        z_scores = []
        green_fracs = []
        for i in range(n_gen):
            tokens = generated[i].cpu().tolist()
            _, z, gc, n = wm.detect(tokens)
            z_scores.append(z)
            green_fracs.append(gc / max(n, 1))

        detection_rate = sum(1 for z in z_scores if z > 4.0) / len(z_scores)
        detection_results[delta] = {
            'z_scores': z_scores,
            'green_fracs': green_fracs,
            'detection_rate': detection_rate,
            'mean_z': np.mean(z_scores),
        }
        print(f"  delta={delta:.1f}: detection_rate={detection_rate:.2f}, mean_z={np.mean(z_scores):.2f}")

    # ── Experiment 2: Text quality vs watermark strength ──
    print("\n=== Experiment 2: Perplexity vs Delta ===")
    perplexities = {}
    for delta in deltas:
        wm = WatermarkProcessor(vocab_size, green_fraction=0.5, delta=delta, device=device)
        prompts = torch.randint(0, vocab_size, (50, prompt_len), device=device)
        generated = model.generate(prompts, max_new_tokens=20, watermark_processor=wm)

        # Compute perplexity
        with torch.no_grad():
            logits = model(generated[:, :-1])
            targets = generated[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='none')
            ce = ce.reshape(targets.shape).mean(dim=-1)
            ppls = torch.exp(ce)
        perplexities[delta] = ppls.mean().item()
        print(f"  delta={delta:.1f}: PPL={ppls.mean().item():.2f}")

    # ── Experiment 3: Green fraction analysis ──
    print("\n=== Experiment 3: Green Token Fraction ===")
    gf_deltas = [0.0, 1.0, 2.0, 4.0]
    gf_data = {}
    for delta in gf_deltas:
        wm = WatermarkProcessor(vocab_size, green_fraction=0.5, delta=delta, device=device)
        prompts = torch.randint(0, vocab_size, (200, prompt_len), device=device)
        generated = model.generate(prompts, max_new_tokens=20, watermark_processor=wm)

        all_gf = []
        for i in range(200):
            tokens = generated[i].cpu().tolist()
            _, _, gc, n = wm.detect(tokens)
            all_gf.append(gc / max(n, 1))
        gf_data[delta] = all_gf

    # ── Experiment 4: Robustness to token substitution ──
    print("\n=== Experiment 4: Robustness to Substitution Attack ===")
    wm = WatermarkProcessor(vocab_size, green_fraction=0.5, delta=2.0, device=device)
    prompts = torch.randint(0, vocab_size, (100, prompt_len), device=device)
    generated = model.generate(prompts, max_new_tokens=20, watermark_processor=wm)

    # Original detection
    orig_z = []
    for i in range(100):
        tokens = generated[i].cpu().tolist()
        _, z, _, _ = wm.detect(tokens)
        orig_z.append(z)

    # Substitution attack: replace random fraction of tokens
    sub_fracs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    robust_data = {}
    for sf in sub_fracs:
        attacked_z = []
        for i in range(100):
            tokens = generated[i].cpu().clone()
            n_replace = int(len(tokens) * sf)
            replace_idx = np.random.choice(len(tokens), n_replace, replace=False)
            tokens[replace_idx] = torch.randint(0, vocab_size, (n_replace,))
            _, z, _, _ = wm.detect(tokens)
            attacked_z.append(z)
        detection_rate = sum(1 for z in attacked_z if z > 4.0) / len(attacked_z)
        robust_data[sf] = {'z_scores': attacked_z, 'detection_rate': detection_rate}
        print(f"  sub_frac={sf:.1f}: detection_rate={detection_rate:.2f}, mean_z={np.mean(attacked_z):.2f}")

    # ── Visualization ──

    # 1. Detection power vs delta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    d_vals = [d for d in deltas]
    det_rates = [detection_results[d]['detection_rate'] for d in deltas]
    mean_zs = [detection_results[d]['mean_z'] for d in deltas]

    axes[0].plot(d_vals, det_rates, marker='o', color='blue', linewidth=2)
    axes[0].set_xlabel("Watermark Strength (delta)")
    axes[0].set_ylabel("Detection Rate")
    axes[0].set_title("Detection Power vs Watermark Strength")
    axes[0].axhline(0.05, color='gray', linestyle='--', alpha=0.5, label='False positive rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(d_vals, mean_zs, marker='o', color='red', linewidth=2)
    axes[1].axhline(4.0, color='green', linestyle='--', alpha=0.7, label='Detection threshold')
    axes[1].set_xlabel("Watermark Strength (delta)")
    axes[1].set_ylabel("Mean Z-Score")
    axes[1].set_title("Z-Score vs Watermark Strength")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LLM Watermarking: Detection Power (2301.10226)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'detection_power.png', dpi=150)
    plt.close()

    # 2. Quality vs watermark
    fig, ax = plt.subplots(figsize=(8, 5))
    ppl_vals = [perplexities[d] for d in deltas]
    ax.plot(d_vals, ppl_vals, marker='s', color='red', linewidth=2)
    ax.set_xlabel("Watermark Strength (delta)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Text Quality vs Watermark Strength")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'quality_vs_watermark.png', dpi=150)
    plt.close()

    # 3. Green fraction histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, delta in zip(axes.flat, gf_deltas):
        ax.hist(gf_data[delta], bins=30, color='steelblue', alpha=0.7)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Expected (null)')
        ax.set_title(f"delta={delta:.1f}")
        ax.set_xlabel("Green Token Fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Green Token Fraction Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'green_fraction.png', dpi=150)
    plt.close()

    # 4. Robustness
    fig, ax = plt.subplots(figsize=(8, 5))
    sfs = list(robust_data.keys())
    drs = [robust_data[sf]['detection_rate'] for sf in sfs]
    ax.plot(sfs, drs, marker='o', color='blue', linewidth=2)
    ax.set_xlabel("Token Substitution Fraction")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Robustness to Substitution Attack")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'robustness.png', dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    concept = (
        "LLM Watermarking (2301.10226)\n"
        "=" * 50 + "\n\n"
        "Generation (Watermarking):\n"
        "  1. Hash previous token → seed RNG\n"
        "  2. Partition vocab into green/red lists\n"
        "  3. Boost green token logits by delta\n"
        "  4. Sample from modified distribution\n\n"
        "Detection:\n"
        "  1. For each token, check if it's in the green list\n"
        "     (green list determined by previous token)\n"
        "  2. Count green tokens: G\n"
        "  3. Under null (no watermark): G ~ Binom(n, 0.5)\n"
        "  4. Z-test: z = (G - n/2) / sqrt(n/4)\n"
        "  5. If z > threshold → watermarked\n\n"
        "Tradeoffs:\n"
        "  delta ↑ → detection easier, text quality ↓\n"
        "  green_fraction ↓ → detection easier, quality ↓\n"
        "  Robustness: survives token substitution\n"
        "  Weakness: paraphrasing can remove watermark"
    )
    ax.text(0.05, 0.95, concept, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
