"""
Minimal VALL-E Codec Language Model Reproduction
=================================================
Reproduces the core ideas from "Neural Codec Language Models are Zero-Shot
Text to Speech Synthesizers" (Wang et al., 2023, 2301.02111):
1. Treat audio codec tokens as a language model vocabulary
2. Given a prompt (codec tokens from a speaker), autoregressively generate
   continuation tokens in the same "voice"
3. Hierarchical generation: first RVQ level autoregressive, then
   subsequent levels conditioned on prior levels
4. Temperature sampling effect on generation diversity
5. Synthetic: small vocabulary of codec-like tokens with learnable patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Positional Encoding ──

class RotaryPositionalEncoding(nn.Module):
    """Simplified rotary positional encoding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self._max_len = max_len

    def forward(self, x):
        """x: (B, T, D)"""
        T = x.size(1)
        t = torch.arange(T, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (T, D//2)
        cos = freqs.cos().unsqueeze(0)  # (1, T, D//2)
        sin = freqs.sin().unsqueeze(0)

        x1, x2 = x.chunk(2, dim=-1)
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1)


# ── Transformer Blocks ──

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, D)"""
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(h)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


# ── VALL-E AR Model (first RVQ level) ──

class ValleARModel(nn.Module):
    """
    Autoregressive model for the first RVQ level.
    Given prompt tokens + context, generates continuation tokens.
    """
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 2, d_model)  # +2 for BOS/EOS
        self.pos_enc = RotaryPositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Special tokens
        self.bos_token = vocab_size
        self.eos_token = vocab_size + 1

    def forward(self, tokens):
        """tokens: (B, T) -> logits: (B, T, vocab_size)"""
        x = self.token_emb(tokens)
        x = self.pos_enc(x)

        # Causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.masked_fill(mask, float('-inf'))

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output_proj(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=0):
        """
        Autoregressively generate continuation after prompt.
        prompt: (B, prompt_len) token indices
        """
        self.eval()
        B = prompt.size(0)
        tokens = prompt

        for _ in range(max_new_tokens):
            logits = self.forward(tokens)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == self.eos_token).all():
                break

        return tokens


# ── VALL-E NAR Model (subsequent RVQ levels) ──

class ValleNARModel(nn.Module):
    """
    Non-autoregressive model for subsequent RVQ levels.
    Given tokens from previous levels, predicts tokens at the current level.
    Uses level embedding to distinguish which level is being predicted.
    """
    def __init__(self, vocab_size=256, n_levels=4, d_model=128, n_heads=4,
                 n_layers=4, d_ff=512, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_levels = n_levels
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.level_emb = nn.Embedding(n_levels, d_model)
        self.pos_enc = RotaryPositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, level):
        """
        tokens: (B, T) — tokens from previous levels (or sum of embeddings)
        level: int — which RVQ level to predict
        """
        # Clamp tokens to valid range (AR model may produce BOS/EOS beyond vocab_size)
        tokens = tokens.clamp(0, self.vocab_size - 1)
        x = self.token_emb(tokens) + self.level_emb(
            torch.tensor(level, device=tokens.device))
        x = self.pos_enc(x)

        # No causal mask for NAR — bidirectional
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output_proj(x)  # (B, T, vocab_size)


# ── Synthetic Codec Token Patterns ──

class SyntheticCodecData:
    """
    Generate synthetic codec-like token sequences with learnable patterns.
    Each "speaker" has a characteristic transition matrix, so the model
    must learn to continue in the same "voice" given a prompt.
    """
    def __init__(self, vocab_size=64, n_speakers=4, n_levels=4, seq_len=32,
                 device='cpu'):
        self.vocab_size = vocab_size
        self.n_speakers = n_speakers
        self.n_levels = n_levels
        self.seq_len = seq_len
        self.device = device

        # Each speaker has a unique transition pattern
        # Build Markov chain transition matrices per speaker
        self.transition_matrices = []
        for s in range(n_speakers):
            # Create a sparse, peaked transition matrix
            mat = np.random.dirichlet(np.ones(vocab_size) * 0.3, size=vocab_size)
            # Add speaker-specific structure: each speaker "prefers" a subset
            preferred = np.random.choice(vocab_size, size=vocab_size // 4, replace=False)
            for p in preferred:
                mat[:, p] += 2.0
            mat = mat / mat.sum(axis=1, keepdims=True)
            self.transition_matrices.append(mat)

    def generate_sequence(self, speaker_id, length=None):
        """Generate a codec token sequence for a given speaker."""
        if length is None:
            length = self.seq_len

        mat = self.transition_matrices[speaker_id]
        tokens = np.zeros(length, dtype=np.int64)
        tokens[0] = np.random.randint(self.vocab_size)

        for t in range(1, length):
            tokens[t] = np.random.choice(self.vocab_size, p=mat[tokens[t-1]])

        return tokens

    def generate_batch(self, batch_size=16, prompt_len=8):
        """
        Generate batch with prompt + continuation pairs.
        Returns: prompt (B, prompt_len), target (B, seq_len),
                 speaker_ids (B,)
        """
        prompts = []
        targets = []
        speaker_ids = []

        for _ in range(batch_size):
            speaker = np.random.randint(self.n_speakers)
            seq = self.generate_sequence(speaker)
            prompts.append(seq[:prompt_len])
            targets.append(seq)
            speaker_ids.append(speaker)

        prompt_t = torch.tensor(np.array(prompts), dtype=torch.long, device=self.device)
        target_t = torch.tensor(np.array(targets), dtype=torch.long, device=self.device)
        speaker_t = torch.tensor(speaker_ids, dtype=torch.long, device=self.device)

        return prompt_t, target_t, speaker_t


# ── Training ──

def train_valle_ar(model, dataset, n_steps=3000, lr=1e-3, device='cpu'):
    """Train the autoregressive model (first RVQ level)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    accs = []

    for step in range(n_steps):
        _, target, _ = dataset.generate_batch(batch_size=32)

        # Teacher forcing: predict next token
        input_tokens = target[:, :-1]
        output_targets = target[:, 1:]

        logits = model(input_tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               output_targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = logits.argmax(-1)
        acc = (pred == output_targets).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs


def train_valle_nar(model, dataset, n_steps=1500, lr=1e-3, device='cpu'):
    """Train the non-autoregressive model (subsequent RVQ levels)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    accs = []

    for step in range(n_steps):
        _, target, _ = dataset.generate_batch(batch_size=32)

        # Random level to predict (1 to n_levels-1)
        level = np.random.randint(1, dataset.n_levels)

        # For synthetic data: use previous-level tokens as input
        # Simulate multi-level by adding noise to create "previous level" tokens
        prev_level_tokens = (target + torch.randint_like(target, 5) - 2) % dataset.vocab_size

        logits = model(prev_level_tokens, level)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = (logits.argmax(-1) == target).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Level {level} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.4f}")

    return losses, accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "74-valle"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ──
    print("=== VALL-E: Codec Language Model ===\n")

    vocab_size = 64
    n_levels = 4
    dataset = SyntheticCodecData(vocab_size=vocab_size, n_speakers=4,
                                  n_levels=n_levels, seq_len=32, device=device)

    # AR model (first level)
    ar_model = ValleARModel(vocab_size=vocab_size, d_model=128, n_heads=4,
                             n_layers=4, d_ff=512).to(device)
    # NAR model (subsequent levels)
    nar_model = ValleNARModel(vocab_size=vocab_size, n_levels=n_levels,
                               d_model=128, n_heads=4, n_layers=4,
                               d_ff=512).to(device)

    ar_params = sum(p.numel() for p in ar_model.parameters())
    nar_params = sum(p.numel() for p in nar_model.parameters())
    print(f"AR model parameters: {ar_params:,}")
    print(f"NAR model parameters: {nar_params:,}")
    print(f"Vocab size: {vocab_size}, RVQ levels: {n_levels}, Speakers: 4")

    # ── Train AR Model ──
    print("\n--- Training AR Model (Level 0) ---")
    ar_losses, ar_accs = train_valle_ar(ar_model, dataset, n_steps=3000,
                                         device=device)

    # ── Train NAR Model ──
    print("\n--- Training NAR Model (Levels 1-3) ---")
    nar_losses, nar_accs = train_valle_nar(nar_model, dataset, n_steps=1500,
                                            device=device)

    # ── Visualization ──

    # 1. Training convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    window = 30

    ar_loss_s = np.convolve(ar_losses, np.ones(window)/window, mode='valid')
    ar_acc_s = np.convolve(ar_accs, np.ones(window)/window, mode='valid')
    nar_loss_s = np.convolve(nar_losses, np.ones(window)/window, mode='valid')
    nar_acc_s = np.convolve(nar_accs, np.ones(window)/window, mode='valid')

    ax1.plot(ar_loss_s, label='AR Model', color='blue')
    ax1.plot(nar_loss_s, label='NAR Model', color='orange')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss (smoothed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ar_acc_s, label='AR Model', color='blue')
    ax2.plot(nar_acc_s, label='NAR Model', color='orange')
    ax2.set_title("Token Prediction Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (smoothed)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("VALL-E: AR and NAR Model Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "training_convergence.png", dpi=150)
    plt.close()

    # 2. Prompt -> Continuation Generation
    print("\nGenerating prompt-continuation pairs...")
    ar_model.eval()
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))

    speaker_names = [f"Speaker {i}" for i in range(4)]
    prompt_len = 8

    for speaker_id in range(4):
        for col, temp in enumerate([0.5, 1.0, 1.5]):
            # Generate a sequence for this speaker
            seq = dataset.generate_sequence(speaker_id, length=32)
            prompt = torch.tensor(seq[:prompt_len], dtype=torch.long,
                                   device=device).unsqueeze(0)

            generated = ar_model.generate(prompt, max_new_tokens=24,
                                           temperature=temp, top_k=10)

            gen_tokens = generated[0].cpu().numpy()

            ax = axes[speaker_id, col]
            ax.imshow(gen_tokens.reshape(1, -1), aspect='auto', cmap='tab20',
                     vmin=0, vmax=vocab_size)
            ax.axvline(x=prompt_len - 0.5, color='red', linewidth=2, linestyle='--')
            ax.set_yticks([])
            if speaker_id == 0:
                ax.set_title(f"Temperature = {temp}")
            if col == 0:
                ax.set_ylabel(speaker_names[speaker_id], fontsize=10)
            ax.set_xlabel("Position")

    plt.suptitle("VALL-E AR: Prompt (left of red line) -> Generated Continuation",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "prompt_continuation.png", dpi=150)
    plt.close()

    # 3. Temperature Sampling Effect
    print("Analyzing temperature sampling effect...")
    temperatures = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    n_samples = 20

    diversity_scores = []
    repetition_rates = []

    for temp in temperatures:
        all_generated = []
        for _ in range(n_samples):
            speaker = np.random.randint(4)
            seq = dataset.generate_sequence(speaker, length=8)
            prompt = torch.tensor(seq, dtype=torch.long,
                                   device=device).unsqueeze(0)
            gen = ar_model.generate(prompt, max_new_tokens=24,
                                     temperature=temp, top_k=0)
            all_generated.append(gen[0, 8:].cpu().numpy())  # Only continuation

        # Diversity: average number of unique tokens
        all_flat = np.concatenate(all_generated)
        diversity = len(np.unique(all_flat)) / vocab_size

        # Repetition: fraction of consecutive identical tokens
        reps = 0
        total = 0
        for gen in all_generated:
            for j in range(1, len(gen)):
                total += 1
                if gen[j] == gen[j-1]:
                    reps += 1
        rep_rate = reps / max(total, 1)

        diversity_scores.append(diversity)
        repetition_rates.append(rep_rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(temperatures, diversity_scores, 'o-', color='blue')
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Vocabulary Diversity (unique tokens / vocab size)")
    ax1.set_title("Temperature vs Token Diversity")
    ax1.grid(True, alpha=0.3)

    ax2.plot(temperatures, repetition_rates, 's-', color='red')
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Repetition Rate (consecutive identical tokens)")
    ax2.set_title("Temperature vs Repetition")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("VALL-E: Effect of Temperature on Generation", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "temperature_effect.png", dpi=150)
    plt.close()

    # 4. Speaker Consistency: continuation should match prompt speaker
    print("Evaluating speaker consistency...")
    ar_model.eval()
    consistency_data = {}

    for speaker_id in range(4):
        # Transition matrix for this speaker
        mat = dataset.transition_matrices[speaker_id]

        correct_transitions = 0
        total_transitions = 0

        for _ in range(50):
            seq = dataset.generate_sequence(speaker_id, length=8)
            prompt = torch.tensor(seq, dtype=torch.long,
                                   device=device).unsqueeze(0)
            gen = ar_model.generate(prompt, max_new_tokens=24,
                                     temperature=0.8, top_k=5)
            gen_tokens = gen[0, 8:].cpu().numpy()

            # Check if transitions follow this speaker's pattern
            for j in range(1, len(gen_tokens)):
                prev_t = gen_tokens[j-1]
                curr_t = gen_tokens[j]
                if prev_t < vocab_size:
                    expected_dist = mat[prev_t]
                    # Top-5 most likely next tokens for this speaker
                    top5 = np.argsort(expected_dist)[-5:]
                    if curr_t in top5:
                        correct_transitions += 1
                    total_transitions += 1

        rate = correct_transitions / max(total_transitions, 1)
        consistency_data[speaker_id] = rate
        print(f"  Speaker {speaker_id}: transition consistency = {rate:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    speakers = list(consistency_data.keys())
    rates = list(consistency_data.values())
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    bars = ax.bar(speakers, rates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Top-5 Transition Consistency")
    ax.set_title("VALL-E: Speaker Consistency in Generated Continuations")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{rate:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / "speaker_consistency.png", dpi=150)
    plt.close()

    # 5. Multi-level generation demo
    print("Demo: hierarchical multi-level generation...")
    nar_model.eval()
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))

    # Generate level 0 with AR model
    speaker = np.random.randint(4)
    seq = dataset.generate_sequence(speaker, length=32)
    prompt = torch.tensor(seq[:8], dtype=torch.long, device=device).unsqueeze(0)
    level0_gen = ar_model.generate(prompt, max_new_tokens=24,
                                    temperature=0.8, top_k=5)

    axes[0].imshow(level0_gen[0].cpu().numpy().reshape(1, -1), aspect='auto',
                  cmap='tab20', vmin=0, vmax=vocab_size)
    axes[0].axvline(x=7.5, color='red', linewidth=2, linestyle='--')
    axes[0].set_title("Level 0 (AR): Prompt -> Generated")
    axes[0].set_ylabel("L0")
    axes[0].set_yticks([])

    # Generate levels 1-3 with NAR model
    colors_map = ['Blues', 'Greens', 'Purples']
    for level in range(1, 4):
        with torch.no_grad():
            nar_logits = nar_model(level0_gen[0].unsqueeze(0), level)
            level_tokens = nar_logits.argmax(-1).squeeze(0)

        axes[level].imshow(level_tokens.cpu().numpy().reshape(1, -1), aspect='auto',
                          cmap='tab20', vmin=0, vmax=vocab_size)
        axes[level].set_title(f"Level {level} (NAR): Conditioned on Level 0")
        axes[level].set_ylabel(f"L{level}")
        axes[level].set_yticks([])
        axes[level].set_xlabel("Position")

    plt.suptitle("VALL-E: Hierarchical Codec Token Generation (AR + NAR)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "hierarchical_generation.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
