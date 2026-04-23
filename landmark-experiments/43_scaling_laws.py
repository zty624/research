"""
Minimal Scaling Laws / Chinchilla Reproduction
================================================
Reproduces core ideas from Scaling Laws (2001.08361, Kaplan et al.) and Chinchilla (2203.15556, Hoffmann et al.):
1. Neural scaling laws: loss = f(parameters, data, compute)
2. Chinchilla: optimal model size and data size for given compute budget
3. Key finding: most LLMs are undertrained (too big, too little data)
4. Compare: scaling behavior of model size vs data size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Small Transformer ──

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.0, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask)
        return self.head(h)


# ── Data ──

def generate_data(vocab_size=32, length=20000):
    data = []
    for i in range(length):
        base = (i % 10)
        noise = np.random.randint(0, 3)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Training ──

def train_and_evaluate(model, data, n_steps, seq_len=32, batch_size=32, lr=3e-4, device='cpu'):
    """Train model and return final loss and parameter count."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    for step in range(n_steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    n_params = sum(p.numel() for p in model.parameters())
    return losses, n_params


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "43-scaling-laws"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    full_data = generate_data(vocab_size, length=20000)

    # Experiment 1: Scale model size (vary d_model and n_layers)
    print("=== Experiment 1: Scaling Model Size ===")
    model_configs = [
        {'d_model': 32, 'n_heads': 2, 'n_layers': 1},
        {'d_model': 48, 'n_heads': 2, 'n_layers': 2},
        {'d_model': 64, 'n_heads': 2, 'n_layers': 2},
        {'d_model': 64, 'n_heads': 2, 'n_layers': 3},
        {'d_model': 96, 'n_heads': 3, 'n_layers': 3},
        {'d_model': 128, 'n_heads': 4, 'n_layers': 4},
    ]

    scale_results = {}
    for config in model_configs:
        model = TinyTransformer(vocab_size, **config).to(device)
        losses, n_params = train_and_evaluate(model, full_data, n_steps=2000, device=device)
        final_loss = np.mean(losses[-100:])
        scale_results[n_params] = {
            'losses': losses,
            'final_loss': final_loss,
            'n_params': n_params,
            'config': config,
        }
        print(f"  Params={n_params:>7,} | d={config['d_model']:>3d} L={config['n_layers']} | Loss={final_loss:.4f}")

    # Experiment 2: Scale data size (fixed model, varying data)
    print("\n=== Experiment 2: Scaling Data Size ===")
    data_sizes = [500, 1000, 2000, 5000, 10000, 20000]
    data_results = {}

    for n_data in data_sizes:
        data = full_data[:n_data]
        model = TinyTransformer(vocab_size, d_model=64, n_heads=2, n_layers=2).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        losses, _ = train_and_evaluate(model, data, n_steps=2000, device=device)
        final_loss = np.mean(losses[-100:])
        data_results[n_data] = {
            'losses': losses,
            'final_loss': final_loss,
        }
        print(f"  Data={n_data:>6d} tokens | Loss={final_loss:.4f}")

    # Experiment 3: Chinchilla analysis — compute-optimal frontier
    print("\n=== Experiment 3: Compute-Optimal Frontier ===")
    # For each compute budget (params * steps), find the best model size
    compute_budgets = []
    for n_params, r in scale_results.items():
        # Compute ≈ n_params * n_steps (FLOPs proportional)
        compute = n_params * 2000  # all trained for 2000 steps
        compute_budgets.append((compute, n_params, r['final_loss']))

    # Also train smaller models for more steps
    print("  Training small model for more steps...")
    small_model = TinyTransformer(vocab_size, d_model=32, n_heads=2, n_layers=1).to(device)
    small_losses, small_params = train_and_evaluate(small_model, full_data, n_steps=8000, device=device)
    small_final = np.mean(small_losses[-100:])
    print(f"    Small model: {small_params:,} params, 8000 steps → Loss={small_final:.4f}")

    print("  Training medium model for more steps...")
    med_model = TinyTransformer(vocab_size, d_model=64, n_heads=2, n_layers=2).to(device)
    med_losses, med_params = train_and_evaluate(med_model, full_data, n_steps=4000, device=device)
    med_final = np.mean(med_losses[-100:])
    print(f"    Medium model: {med_params:,} params, 4000 steps → Loss={med_final:.4f}")

    # ── Visualization ──

    # 1. Scaling law: loss vs parameters
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    params_list = sorted(scale_results.keys())
    losses_list = [scale_results[p]['final_loss'] for p in params_list]

    axes[0].loglog(params_list, losses_list, 'o-', color='blue', linewidth=2)
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Scaling Law: Loss vs Model Size")
    axes[0].grid(True, alpha=0.3, which='both')

    # Fit power law: L(N) = a * N^(-b)
    log_params = np.log(params_list)
    log_losses = np.log(losses_list)
    coeffs = np.polyfit(log_params, log_losses, 1)
    a, b = np.exp(coeffs[1]), coeffs[0]
    fit_x = np.logspace(np.log10(min(params_list)), np.log10(max(params_list)), 100)
    fit_y = a * fit_x ** b
    axes[0].loglog(fit_x, fit_y, '--', color='red', alpha=0.5, label=f'L ∝ N^{{{b:.2f}}}')
    axes[0].legend()

    # 2. Scaling law: loss vs data
    data_list = sorted(data_results.keys())
    data_loss_list = [data_results[d]['final_loss'] for d in data_list]

    axes[1].loglog(data_list, data_loss_list, 'o-', color='green', linewidth=2)
    axes[1].set_xlabel("Training Data (tokens)")
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Scaling Law: Loss vs Data Size")
    axes[1].grid(True, alpha=0.3, which='both')

    # Fit power law for data
    log_data = np.log(data_list)
    log_dlosses = np.log(data_loss_list)
    d_coeffs = np.polyfit(log_data, log_dlosses, 1)
    da, db = np.exp(d_coeffs[1]), d_coeffs[0]
    fit_dx = np.logspace(np.log10(min(data_list)), np.log10(max(data_list)), 100)
    fit_dy = da * fit_dx ** db
    axes[1].loglog(fit_dx, fit_dy, '--', color='red', alpha=0.5, label=f'L ∝ D^{{{db:.2f}}}')
    axes[1].legend()

    plt.suptitle("Neural Scaling Laws: Power-Law Behavior", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "scaling_laws.png", dpi=150)
    plt.close()

    # 3. Chinchilla: compute-optimal frontier
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compare: same compute, different model sizes
    compute_1 = small_params * 8000  # small model, many steps
    compute_2 = med_params * 4000    # medium model, medium steps
    compute_3 = list(scale_results.keys())[-1] * 2000  # large model, few steps

    compute_vals = [compute_1, compute_2, compute_3]
    model_sizes = [small_params, med_params, list(scale_results.keys())[-1]]
    final_losses = [small_final, med_final, list(scale_results.values())[-1]['final_loss']]
    labels = ['Small\n(many steps)', 'Medium\n(med steps)', 'Large\n(few steps)']

    axes[0].scatter(model_sizes, final_losses, s=200, c=['green', 'blue', 'red'], zorder=5)
    for i, (ms, fl, label) in enumerate(zip(model_sizes, final_losses, labels)):
        axes[0].annotate(label, (ms, fl), textcoords="offset points",
                        xytext=(15, -10), fontsize=8,
                        color=['green', 'blue', 'red'][i])
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Chinchilla: Same Compute, Different Trade-offs")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Loss vs compute
    axes[1].loglog(compute_vals, final_losses, 'o-', color='purple', linewidth=2)
    axes[1].set_xlabel("Compute (params × steps)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss vs Compute Budget")
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle("Chinchilla: Optimal Model Size for Given Compute", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "chinchilla_analysis.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Scaling Laws\n(2020)", "Loss ∝ N^(-α)\nLoss ∝ D^(-β)\nPower-law behavior\nPredictable from\nsmall experiments", 0.14, 'blue'),
        ("Chinchilla\n(2022)", "For fixed compute C:\nN* ∝ C^a, D* ∝ C^b\nMost models are\nOVERSIZED and\nUNDERTRAINED!", 0.5, 'green'),
        ("Key Finding", "Gopher (280B) was\nbeaten by Chinchilla\n(70B, 4× more data)\n→ Train smaller models\n   on more data!", 0.86, 'red'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Scaling Laws & Chinchilla: Compute-Optimal LLM Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "scaling_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
