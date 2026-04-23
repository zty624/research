"""
Minimal Grokking Reproduction
================================
Reproduces core ideas from Grokking (2201.02177, Power et al.):
1. Grokking: sudden generalization long after overfitting
2. Model memorizes training data, then suddenly learns the underlying algorithm
3. Happens with small datasets and weight decay
4. Key insight: phase transition from memorization to generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Modular Arithmetic Task ──

class ModularArithmetic:
    """Task: predict (a + b) % p given a and b."""
    def __init__(self, p=59):
        self.p = p
        self.vocab_size = p  # tokens 0..p-1

    def generate(self, frac_train=0.3):
        """Generate all pairs and split into train/test."""
        data = []
        for a in range(self.p):
            for b in range(self.p):
                c = (a + b) % self.p
                data.append((a, b, c))

        np.random.shuffle(data)
        n_train = int(len(data) * frac_train)
        train = data[:n_train]
        test = data[n_train:]

        return train, test


# ── Model ──

class GrokkingModel(nn.Module):
    """Simple transformer-like model for modular arithmetic."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(3, d_model)  # a, b, = positions

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.0, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, 3) — [a, b, equals_sign]
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask)
        # Predict from last position
        return self.head(self.norm(h[:, -1, :]))


# ── Training ──

def train_grokking(model, train_data, test_data, n_steps=10000, lr=1e-3,
                    weight_decay=0.1, device='cpu', eval_every=100):
    """Train with weight decay to observe grokking."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    # Prepare data
    train_a = torch.tensor([d[0] for d in train_data], dtype=torch.long, device=device)
    train_b = torch.tensor([d[1] for d in train_data], dtype=torch.long, device=device)
    train_c = torch.tensor([d[2] for d in train_data], dtype=torch.long, device=device)

    test_a = torch.tensor([d[0] for d in test_data], dtype=torch.long, device=device)
    test_b = torch.tensor([d[1] for d in test_data], dtype=torch.long, device=device)
    test_c = torch.tensor([d[2] for d in test_data], dtype=torch.long, device=device)

    equals_token = model.vocab_size - 1  # use last token as equals sign

    train_losses = []
    test_accs = []
    train_accs = []
    steps = []

    batch_size = 256

    for step in range(n_steps):
        # Sample batch
        idx = torch.randint(0, len(train_data), (min(batch_size, len(train_data)),))

        x = torch.stack([train_a[idx], train_b[idx],
                        torch.full_like(train_a[idx], equals_token)], dim=1)
        y = train_c[idx]

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0:
            # Train accuracy
            with torch.no_grad():
                train_pred = model(torch.stack([train_a, train_b,
                    torch.full_like(train_a, equals_token)], dim=1))
                train_acc = (train_pred.argmax(1) == train_c).float().mean().item()

                test_pred = model(torch.stack([test_a, test_b,
                    torch.full_like(test_a, equals_token)], dim=1))
                test_acc = (test_pred.argmax(1) == test_c).float().mean().item()

            train_losses.append(loss.item())
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            steps.append(step + 1)

            if train_acc > 0.99 and test_acc < 0.5:
                print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                      f"Train: {train_acc:.4f} | Test: {test_acc:.4f} — MEMORIZED")
            elif test_acc > 0.9:
                print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                      f"Train: {train_acc:.4f} | Test: {test_acc:.4f} — GROKKED!")
            elif (step + 1) % 1000 == 0:
                print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                      f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

    return steps, train_losses, train_accs, test_accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "46-grokking"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Observe grokking with modular arithmetic
    print("=== Grokking: Modular Arithmetic (a+b)%p ===")
    task = ModularArithmetic(p=59)
    train_data, test_data = task.generate(frac_train=0.3)
    print(f"  Train: {len(train_data)} examples, Test: {len(test_data)} examples")

    print("\n  Training with weight decay=0.1 (needed for grokking)...")
    model = GrokkingModel(task.vocab_size, d_model=128, n_heads=4, n_layers=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    steps, losses, train_accs, test_accs = train_grokking(
        model, train_data, test_data, n_steps=20000, weight_decay=0.1, device=device
    )

    # Experiment 2: Without weight decay (should overfit and stay memorized)
    print("\n=== Without Weight Decay (should not grok) ===")
    model_no_wd = GrokkingModel(task.vocab_size, d_model=128, n_heads=4, n_layers=2).to(device)
    steps2, losses2, train_accs2, test_accs2 = train_grokking(
        model_no_wd, train_data, test_data, n_steps=20000, weight_decay=0.0, device=device
    )

    # Experiment 3: Different training fractions
    print("\n=== Training Fraction Effect ===")
    frac_results = {}
    for frac in [0.2, 0.3, 0.5]:
        print(f"\n  Fraction={frac}:")
        train_d, test_d = task.generate(frac_train=frac)
        m = GrokkingModel(task.vocab_size, d_model=64, n_heads=2, n_layers=1).to(device)
        s, l, ta, tea = train_grokking(
            m, train_d, test_d, n_steps=15000, weight_decay=0.1, device=device
        )
        frac_results[frac] = {'steps': s, 'train_accs': ta, 'test_accs': tea}

    # ── Visualization ──

    # 1. Grokking phenomenon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, train_accs, label='Train Acc', color='blue')
    axes[0].plot(steps, test_accs, label='Test Acc', color='red')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Grokking: Sudden Generalization")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, losses, color='green')
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Training Loss (with weight decay)")
    axes[1].grid(True, alpha=0.3)

    # Annotate phases
    for ax in [axes[0]]:
        ax.annotate('Memorization\nphase', xy=(steps[len(steps)//4], 0.5),
                   fontsize=10, color='blue', ha='center')
        ax.annotate('Grokking!', xy=(steps[3*len(steps)//4], 0.7),
                   fontsize=10, color='red', ha='center', fontweight='bold')

    plt.suptitle("Grokking: Models Can Suddenly Generalize Long After Overfitting", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "grokking_phenomenon.png", dpi=150)
    plt.close()

    # 2. With vs without weight decay
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, train_accs, label='Train (WD=0.1)', color='blue')
    axes[0].plot(steps, test_accs, label='Test (WD=0.1)', color='red')
    axes[0].plot(steps2, train_accs2, label='Train (WD=0)', color='blue', linestyle='--', alpha=0.5)
    axes[0].plot(steps2, test_accs2, label='Test (WD=0)', color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Weight Decay is Necessary for Grokking")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Training fraction effect
    for frac, r in frac_results.items():
        axes[1].plot(r['steps'], r['test_accs'], label=f'{frac:.0%} train data')
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Effect of Training Data Fraction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Grokking: Conditions for Sudden Generalization", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "grokking_conditions.png", dpi=150)
    plt.close()

    # 3. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Phase 1\nMemorization", "Overfit training data\nLow train loss\nRandom test acc\n→ No real\n   understanding", 0.14, 'blue'),
        ("Phase 2\nGrokking!", "Sudden generalization\nTest accuracy jumps\nfrom ~0% to ~100%\n→ Algorithm\n   discovered!", 0.5, 'red'),
        ("Why It\nHappens", "Weight decay favors\nsimple solutions\nMemorization is complex\nGeneralization is simple\n→ Phase transition!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Grokking: Delayed Generalization in Neural Networks", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "grokking_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
