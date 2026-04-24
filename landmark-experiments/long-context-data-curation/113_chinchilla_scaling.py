"""
Minimal Chinchilla Scaling Laws Reproduction
=============================================
Reproduces core ideas from "Training Compute-Optimal Large Language Models"
(2203.15556, Hoffmann et al.):
1. Scaling laws: model performance follows power law in parameters (N) and data (D)
2. Compute budget C ≈ 6ND: fixed compute, how to split between N and D?
3. Chinchilla finding: previous models were undertrained (too large, too little data)
4. Optimal allocation: N ∝ C^0.5, D ∝ C^0.5 (not N ∝ C^0.7 as Kaplan et al.)
5. Compare: Kaplan scaling vs Chinchilla scaling
6. Show: compute-optimal frontier and model/data tradeoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


# ── Scaling Law Models ──

class KaplanScalingLaw:
    """Kaplan et al. (2020) scaling law: L(C) = (C_c / C)^α.

    Key finding: N ∝ C^0.7, D ∝ C^0.3
    Implies: bigger models are better, data is secondary.
    """
    def __init__(self, alpha=0.076, c0=8.8e8):
        self.alpha = alpha
        self.c0 = c0

    def loss(self, compute):
        """Predict loss given compute budget (FLOPs)."""
        return (self.c0 / compute) ** self.alpha

    def optimal_allocation(self, compute):
        """Kaplan allocation: N ∝ C^0.7."""
        N = compute ** 0.73 / 1e3
        D = compute / (6 * N + 1e-10)
        return N, D


class ChinchillaScalingLaw:
    """Hoffmann et al. (2022) scaling law.

    Key finding: N ∝ C^0.5, D ∝ C^0.5
    Implies: model size and data should scale equally.
    Previous models (GPT-3, etc.) were undertrained.
    """
    def __init__(self, a=6.007, b=16.518, alpha=0.34, beta=0.28,
                 E=1.69, A=406.4, B=410.7):
        # Loss decomposition: L(N,D) = E + A/N^α + B/D^β
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.E = E
        self.A = A
        self.B = B

    def loss_from_nd(self, N, D):
        """Predict loss from model size N and data size D."""
        return self.E + self.A / (N ** self.alpha) + self.B / (D ** self.beta)

    def loss(self, compute):
        """Predict loss at compute-optimal allocation."""
        N, D = self.optimal_allocation(compute)
        return self.loss_from_nd(N, D)

    def optimal_allocation(self, compute):
        """Chinchilla allocation: N ∝ C^0.5, D ∝ C^0.5."""
        # From: C = 6ND, minimize L = E + A/N^α + B/D^β
        # Lagrangian gives: N* = (αA / (6βB))^(1/(α+β)) * C^(β/(α+β)) / 6^(β/(α+β))
        r = self.alpha / self.beta
        # Simplified: N ∝ C^(β/(α+β))
        N_exp = self.beta / (self.alpha + self.beta)
        D_exp = self.alpha / (self.alpha + self.beta)

        # Scale factor calibrated to match paper
        N = 0.6 * compute ** N_exp
        D = compute / (6 * N + 1e-10)
        return N, D


# ── Empirical Scaling Experiment ──

class TinyTransformer(nn.Module):
    """Minimal transformer for measuring actual scaling behavior."""
    def __init__(self, vocab_size=32, d_model=32, n_heads=2, n_layers=1, max_len=16):
        super().__init__()
        self.vocab_size = vocab_size
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


def measure_scaling_point(d_model, n_layers, n_data_tokens, vocab_size=32,
                           max_len=16, n_steps=500, device='cpu'):
    """Train a model with given size and data, measure final loss."""
    n_heads = max(1, d_model // 16)
    model = TinyTransformer(vocab_size, d_model, n_heads, n_layers, max_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Generate synthetic data
    data = torch.randint(0, vocab_size, (n_data_tokens // max_len + 1, max_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    final_loss = float('inf')

    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (32,))
        batch = data[idx]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return n_params, final_loss


# ── Compute-Optimal Frontier ──

def compute_optimal_frontier(chinchilla, computes):
    """Compute the optimal (N, D) and resulting loss for each compute budget."""
    results = []
    for C in computes:
        N_opt, D_opt = chinchilla.optimal_allocation(C)
        L_opt = chinchilla.loss_from_nd(N_opt, D_opt)
        results.append({
            'compute': C,
            'N_optimal': N_opt,
            'D_optimal': D_opt,
            'loss_optimal': L_opt,
        })
    return results


def suboptimal_frontier(chinchilla, computes, oversize_factor=4):
    """Compute loss when model is oversized (like GPT-3 vs Chinchilla)."""
    results = []
    for C in computes:
        N_opt, D_opt = chinchilla.optimal_allocation(C)
        # Oversized model: more params, less data
        N_sub = N_opt * oversize_factor
        D_sub = C / (6 * N_sub + 1e-10)
        L_sub = chinchilla.loss_from_nd(N_sub, D_sub)
        results.append({
            'compute': C,
            'N_suboptimal': N_sub,
            'D_suboptimal': D_sub,
            'loss_suboptimal': L_sub,
        })
    return results


# ── Iso-FLOP Analysis ──

def iso_flop_analysis(chinchilla, compute_budget, N_range):
    """For fixed compute, vary N and compute loss to find optimal."""
    results = []
    for N in N_range:
        D = compute_budget / (6 * N + 1e-10)
        if D < 1:
            continue
        L = chinchilla.loss_from_nd(N, D)
        results.append({'N': N, 'D': D, 'loss': L})
    return results


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "113-chinchilla-scaling"
    results_dir.mkdir(parents=True, exist_ok=True)

    kaplan = KaplanScalingLaw()
    chinchilla = ChinchillaScalingLaw()

    # ── Experiment 1: Scaling law comparison ──
    print("=== Experiment 1: Kaplan vs Chinchilla Scaling Laws ===")
    computes = np.logspace(16, 22, 50)  # FLOPs range

    kaplan_losses = [kaplan.loss(C) for C in computes]
    chinchilla_losses = [chinchilla.loss(C) for C in computes]

    # ── Experiment 2: Optimal allocation ──
    print("\n=== Experiment 2: Optimal N/D Allocation ===")
    frontier = compute_optimal_frontier(chinchilla, computes[::5])
    for r in frontier[:3]:
        print(f"  C={r['compute']:.1e}: N_opt={r['N_optimal']:.1e}, "
              f"D_opt={r['D_optimal']:.1e}, L={r['loss_optimal']:.3f}")

    # ── Experiment 3: Suboptimal (oversized) models ──
    print("\n=== Experiment 3: Oversized Models (GPT-3 style) ===")
    subopt = suboptimal_frontier(chinchilla, computes[::5], oversize_factor=4)
    for opt, sub in zip(frontier[:3], subopt[:3]):
        delta = sub['loss_suboptimal'] - opt['loss_optimal']
        print(f"  C={opt['compute']:.1e}: optimal L={opt['loss_optimal']:.3f}, "
              f"oversized L={sub['loss_suboptimal']:.3f}, gap={delta:.3f}")

    # ── Experiment 4: Iso-FLOP analysis ──
    print("\n=== Experiment 4: Iso-FLOP Analysis ===")
    budget = 1e19
    N_range = np.logspace(4, 8, 30)
    iso_results = iso_flop_analysis(chinchilla, budget, N_range)
    best = min(iso_results, key=lambda x: x['loss'])
    print(f"  Budget={budget:.0e}: optimal N={best['N']:.1e}, D={best['D']:.1e}, "
          f"loss={best['loss']:.3f}")

    # ── Experiment 5: Empirical scaling (small scale) ──
    print("\n=== Experiment 5: Empirical Scaling Measurement ===")
    configs = [
        (32, 1, 5000),    # tiny
        (64, 2, 20000),   # small
        (128, 2, 80000),  # medium
        (64, 4, 20000),   # deep
        (128, 4, 80000),  # large (if memory allows)
    ]

    empirical = []
    for d_model, n_layers, n_data in configs:
        try:
            n_params, loss = measure_scaling_point(
                d_model, n_layers, n_data, n_steps=300, device=device)
            compute = 6 * n_params * n_data
            empirical.append({'N': n_params, 'D': n_data, 'compute': compute, 'loss': loss})
            print(f"  d={d_model}, L={n_layers}, D={n_data}: "
                  f"N={n_params:,}, loss={loss:.3f}, C={compute:.1e}")
        except RuntimeError as e:
            print(f"  d={d_model}, L={n_layers}: OOM, skipping")

    # ── Visualization ──

    # 1. Scaling law comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(computes, kaplan_losses, label='Kaplan et al. (2020)', color='red', linewidth=2)
    ax.loglog(computes, chinchilla_losses, label='Chinchilla (2022)', color='blue', linewidth=2)
    if empirical:
        emp_c = [e['compute'] for e in empirical]
        emp_l = [e['loss'] for e in empirical]
        ax.scatter(emp_c, emp_l, s=100, color='green', zorder=5, label='Empirical (this work)')
    ax.set_xlabel("Compute (FLOPs)")
    ax.set_ylabel("Loss")
    ax.set_title("Scaling Laws: Kaplan vs Chinchilla")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'scaling_law_comparison.png', dpi=150)
    plt.close()

    # 2. Optimal N/D allocation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    c_vals = [r['compute'] for r in frontier]
    n_vals = [r['N_optimal'] for r in frontier]
    d_vals = [r['D_optimal'] for r in frontier]

    axes[0].loglog(c_vals, n_vals, label='Chinchilla: N_opt ∝ C^0.50', color='blue', linewidth=2)
    # Kaplan: N ∝ C^0.73
    kaplan_n = [c ** 0.73 / 1e3 for c in c_vals]
    axes[0].loglog(c_vals, kaplan_n, label='Kaplan: N_opt ∝ C^0.73', color='red', linewidth=2, linestyle='--')
    axes[0].set_xlabel("Compute (FLOPs)")
    axes[0].set_ylabel("Optimal Model Size (N)")
    axes[0].set_title("Optimal Model Size vs Compute")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(c_vals, d_vals, label='Chinchilla: D_opt ∝ C^0.50', color='blue', linewidth=2)
    kaplan_d = [c / (6 * n + 1e-10) for c, n in zip(c_vals, kaplan_n)]
    axes[1].loglog(c_vals, kaplan_d, label='Kaplan: D_opt ∝ C^0.27', color='red', linewidth=2, linestyle='--')
    axes[1].set_xlabel("Compute (FLOPs)")
    axes[1].set_ylabel("Optimal Data Size (D)")
    axes[1].set_title("Optimal Data Size vs Compute")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Chinchilla: Compute-Optimal Allocation (2203.15556)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'optimal_allocation.png', dpi=150)
    plt.close()

    # 3. Suboptimal comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    c_sub = [r['compute'] for r in subopt]
    l_opt = [r['loss_optimal'] for r in frontier]
    l_sub = [r['loss_suboptimal'] for r in subopt]
    ax.semilogx(c_sub, l_opt, label='Compute-Optimal (Chinchilla)', color='blue', linewidth=2)
    ax.semilogx(c_sub, l_sub, label='Oversized Model (4x params)', color='red', linewidth=2, linestyle='--')
    ax.set_xlabel("Compute (FLOPs)")
    ax.set_ylabel("Loss")
    ax.set_title("Compute-Optimal vs Oversized Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'optimal_vs_oversized.png', dpi=150)
    plt.close()

    # 4. Iso-FLOP curve
    fig, ax = plt.subplots(figsize=(10, 6))
    iso_N = [r['N'] for r in iso_results]
    iso_L = [r['loss'] for r in iso_results]
    ax.semilogx(iso_N, iso_L, color='steelblue', linewidth=2)
    ax.axvline(best['N'], color='red', linestyle='--', alpha=0.7,
               label=f'Optimal N={best["N"]:.1e}')
    ax.scatter([best['N']], [best['loss']], color='red', s=100, zorder=5)
    ax.set_xlabel("Model Size (N)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Iso-FLOP Analysis (C={budget:.0e} FLOPs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'iso_flop.png', dpi=150)
    plt.close()

    # 5. Famous model comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    famous = (
        "Chinchilla Scaling Laws: Key Takeaways (2203.15556)\n"
        "=" * 60 + "\n\n"
        "The Compute Budget Equation:\n"
        "  C = 6 × N × D  (FLOPs ≈ 6 × params × data tokens)\n\n"
        "Loss Decomposition:\n"
        "  L(N, D) = E + A/N^α + B/D^β\n"
        "  E = irreducible loss (data entropy)\n"
        "  A/N^α = model capacity term\n"
        "  B/D^β = data term\n\n"
        "Optimal Allocation:\n"
        "  ┌────────────────────────────────────────┐\n"
        "  │ Kaplan (2020):  N ∝ C^0.73, D ∝ C^0.27 │\n"
        "  │ Chinchilla (2022): N ∝ C^0.50, D ∝ C^0.50 │\n"
        "  └────────────────────────────────────────┘\n\n"
        "Implication: GPT-3 (175B, 300B tokens) was undertrained!\n"
        "Chinchilla (70B, 1.4T tokens) achieves better performance\n"
        "with same compute but smaller model + more data.\n\n"
        "Rule of thumb: token-to-parameter ratio ≈ 20:1\n"
        "  (20 tokens of training data per model parameter)"
    )
    ax.text(0.05, 0.95, famous, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'summary.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
