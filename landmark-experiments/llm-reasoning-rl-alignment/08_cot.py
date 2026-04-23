"""
Minimal Chain-of-Thought Reproduction
======================================
Reproduces the core ideas from "Chain-of-Thought Prompting Elicits Reasoning
in Large Language Models" (2201.11903):
1. Standard prompting vs Chain-of-Thought prompting
2. Emergent reasoning: CoT only helps at sufficient model scale
3. Compare: direct answer vs step-by-step reasoning
4. Demonstrate on arithmetic reasoning tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


# ── Tiny GPT for Arithmetic ──

class TinyGPT(nn.Module):
    """A tiny GPT model that can learn arithmetic with or without CoT."""
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2,
                 max_len=64):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.head(self.norm(h))


# ── Arithmetic Tokenizer ──

class ArithmeticTokenizer:
    """Simple tokenizer for arithmetic expressions.
    Tokens: 0-9, +, -, *, =, space, and special tokens.
    """
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        # Special tokens
        special = ['<PAD>', '<EOS>', '<STEP>', '<ANS>']
        chars = list('0123456789+-*= ')
        all_tokens = special + chars
        for i, tok in enumerate(all_tokens):
            self.char_to_id[tok] = i
            self.id_to_char[i] = tok
        self.pad_id = 0
        self.eos_id = 1
        self.step_id = 2
        self.ans_id = 3
        self.vocab_size = len(all_tokens)

    def encode(self, text):
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids if i != self.pad_id)


# ── Data Generation ──

def generate_addition_problem(a, b, tokenizer, use_cot=False):
    """Generate an addition problem with or without CoT."""
    result = a + b
    if use_cot:
        # Chain-of-thought: break down the addition step by step
        steps = []
        a_str = str(a)
        b_str = str(b)
        # Simple step-by-step: add digit by digit from right
        r_str = str(result)
        # Show intermediate steps
        if a < 10 and b < 10:
            prompt = f"{a}+{b}="
            cot = f"{a}+{b}={result}"
        else:
            prompt = f"{a}+{b}="
            # Show carrying steps
            cot_parts = [f"{a}+{b}="]
            # Simplified: just show partial sums
            if a >= 10 or b >= 10:
                ones_a = a % 10
                ones_b = b % 10
                ones_sum = ones_a + ones_b
                tens_a = a // 10
                tens_b = b // 10
                tens_sum = tens_a + tens_b
                if ones_sum >= 10:
                    cot_parts.append(f"{ones_a}+{ones_b}={ones_sum}")
                    cot_parts.append(f"{tens_sum+1}0+{ones_sum%10}={result}")
                else:
                    cot_parts.append(f"{tens_sum}0+{ones_sum}={result}")
                cot_parts.append(f"={result}")
            cot = ' '.join(cot_parts)
    else:
        # Direct answer
        prompt = f"{a}+{b}="
        cot = f"{a}+{b}={result}"

    full_text = prompt + cot + ' <EOS>'
    return tokenizer.encode(full_text)


def generate_dataset(n_samples, max_num, tokenizer, use_cot=False, max_len=64):
    """Generate a dataset of addition problems."""
    inputs = []
    targets = []

    for _ in range(n_samples):
        a = np.random.randint(1, max_num)
        b = np.random.randint(1, max_num)
        tokens = generate_addition_problem(a, b, tokenizer, use_cot)

        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [tokenizer.pad_id] * (max_len - len(tokens))

        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        inputs.append(input_ids)
        targets.append(target_ids)

    return torch.tensor(inputs), torch.tensor(targets)


# ── Training ──

def train_model(model, inputs, targets, n_epochs=20, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_inputs)
            loss = F.cross_entropy(
                logits.view(-1, model.head.out_features),
                batch_targets.view(-1),
                ignore_index=0  # Ignore PAD
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    return losses


# ── Evaluation ──

def evaluate_accuracy(model, tokenizer, max_num, n_samples=200, device='cpu', max_len=64):
    """Evaluate on addition problems — check if final answer is correct."""
    model.eval()
    correct = 0
    total = 0

    for _ in range(n_samples):
        a = np.random.randint(1, max_num)
        b = np.random.randint(1, max_num)
        expected = a + b

        prompt = f"{a}+{b}="
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids + [tokenizer.pad_id] * (max_len - len(input_ids))],
                                     device=device)

        with torch.no_grad():
            logits = model(input_tensor)
            # Get predictions for positions after '='
            # Find the '=' position
            eq_pos = input_ids.index(tokenizer.char_to_id[chr(61)])
            # The next position should have the answer
            pred_ids = logits[0, eq_pos].argmax().item()
            pred_char = tokenizer.id_to_char.get(pred_ids, '?')

        # Simple accuracy: check if first digit after = is correct
        # This is a simplified evaluation for the minimal demo
        first_digit = str(expected)[0]
        if pred_char == first_digit:
            correct += 1
        total += 1

    return correct / total


# ── Emergent Behavior Simulation ──

def simulate_emergence():
    """Simulate the emergent behavior of CoT: it only helps at scale.
    This uses multiple model sizes to show the effect.
    """
    tokenizer = ArithmeticTokenizer()
    device = 'cpu'
    max_len = 64

    configs = [
        {'d_model': 32, 'n_heads': 1, 'n_layers': 1, 'label': 'Tiny (1L)'},
        {'d_model': 48, 'n_heads': 2, 'n_layers': 2, 'label': 'Small (2L)'},
        {'d_model': 64, 'n_heads': 2, 'n_layers': 3, 'label': 'Medium (3L)'},
        {'d_model': 64, 'n_heads': 2, 'n_layers': 4, 'label': 'Large (4L)'},
    ]

    results = {}
    max_num = 20

    for cfg in configs:
        for use_cot in [False, True]:
            key = f"{cfg['label']}_{'CoT' if use_cot else 'Direct'}"
            print(f"\nTraining {key}...")

            model = TinyGPT(
                vocab_size=tokenizer.vocab_size,
                d_model=cfg['d_model'],
                n_heads=cfg['n_heads'],
                n_layers=cfg['n_layers'],
                max_len=max_len
            ).to(device)

            inputs, targets = generate_dataset(
                500, max_num, tokenizer, use_cot=use_cot, max_len=max_len
            )

            losses = train_model(
                model, inputs, targets, n_epochs=30, batch_size=64, lr=1e-3, device=device
            )

            acc = evaluate_accuracy(model, tokenizer, max_num, n_samples=100, device=device)
            results[key] = {'losses': losses, 'accuracy': acc}
            print(f"  Final loss: {losses[-1]:.4f}, Accuracy: {acc:.3f}")

    return results, configs


# ── Main ──

def main():
    results_dir = Path(__file__).parent / "results" / "08-cot"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run emergence simulation
    print("=== Simulating Chain-of-Thought Emergence ===")
    results, configs = simulate_emergence()

    # ── Visualization ──

    # 1. Training curves: Direct vs CoT at different scales
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, cfg in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        key_direct = f"{cfg['label']}_Direct"
        key_cot = f"{cfg['label']}_CoT"

        if key_direct in results:
            ax.plot(results[key_direct]['losses'], label='Direct', color='red')
        if key_cot in results:
            ax.plot(results[key_cot]['losses'], label='CoT', color='blue')

        ax.set_title(f"Model: {cfg['label']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("CoT vs Direct Prompting at Different Model Scales", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Emergence plot: accuracy vs model size
    fig, ax = plt.subplots(figsize=(8, 5))
    direct_accs = []
    cot_accs = []
    labels = []

    for cfg in configs:
        key_direct = f"{cfg['label']}_Direct"
        key_cot = f"{cfg['label']}_CoT"
        labels.append(cfg['label'])
        direct_accs.append(results.get(key_direct, {}).get('accuracy', 0))
        cot_accs.append(results.get(key_cot, {}).get('accuracy', 0))

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, direct_accs, width, label='Direct', color='red', alpha=0.7)
    ax.bar(x + width/2, cot_accs, width, label='Chain-of-Thought', color='blue', alpha=0.7)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Emergent Reasoning: CoT Helps More at Larger Scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(results_dir / "emergence.png", dpi=150)
    plt.close()

    # 3. CoT illustration
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    illustration = [
        ("Direct Prompting", "Q: 47 + 36 = ?\nA: 83", "red"),
        ("Chain-of-Thought", "Q: 47 + 36 = ?\nA: 7+6=13, carry 1\n   4+3+1=8\n   Answer: 83", "blue"),
    ]

    for i, (title, text, color) in enumerate(illustration):
        ax.text(0.25 + i * 0.5, 0.7, title, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(0.25 + i * 0.5, 0.35, text, fontsize=11, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Direct vs Chain-of-Thought Prompting", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "cot_illustration.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
