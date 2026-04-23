"""
Minimal Sparse Autoencoder for Mechanistic Interpretability Reproduction
========================================================================
Reproduces core ideas from:
1. Sparse Autoencoders (Cunningham et al. 2309.08600, Bricken et al. 2309.08600):
   - Decompose polysemantic neuron activations into sparse monosemantic features
   - L1 sparsity penalty encourages sparse activation patterns
   - Each SAE feature ideally responds to one interpretable concept
2. Causal Abstraction / Interchange Intervention (Geiger et al. 2301.04709):
   - Features should be causally meaningful, not just correlated
3. Key insight: SAE transforms dense, polysemantic representations into
   sparse, interpretable features with minimal reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Small Transformer for generating MLP activations ──

class TinyTransformer(nn.Module):
    """Minimal transformer to produce MLP activations for SAE training."""
    def __init__(self, vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(32, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'attn_norm': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                ),
                'ff_norm': nn.LayerNorm(d_model),
            }))

        self.head = nn.Linear(d_model, vocab_size)

        # Storage for MLP activations
        self.mlp_acts = []

    def forward(self, x, store_acts=False):
        """x: (B, T) token indices. Returns logits and optionally stores MLP acts."""
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))

        if store_acts:
            self.mlp_acts = []

        for layer in self.layers:
            # Self-attention
            h_norm = layer['attn_norm'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)
            h = h + attn_out

            # Feed-forward (MLP)
            h_norm = layer['ff_norm'](h)
            ff_out = layer['ff'](h_norm)
            if store_acts:
                # Store pre-residual MLP output (before adding back)
                self.mlp_acts.append(ff_out.detach())
            h = h + ff_out

        return self.head(h)


def pretrain_transformer(model, n_steps=3000, lr=1e-3, device='cpu', vocab_size=100, seq_len=16):
    """Pre-train transformer on synthetic next-token prediction task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(n_steps):
        # Random sequences with some structure: periodic patterns
        batch_size = 64
        x = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
        # Add simple structure: every 4th token repeats
        x[:, 4::4] = x[:, :1]

        input_ids = x[:, :-1]
        target_ids = x[:, 1:]

        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")


def collect_activations(model, n_batches=100, batch_size=64, seq_len=16,
                        vocab_size=100, device='cpu'):
    """Collect MLP activations from the transformer."""
    model.eval()
    all_acts = []

    with torch.no_grad():
        for _ in range(n_batches):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            x[:, 4::4] = x[:, :1]  # Same structure as training
            model(x, store_acts=True)
            # Collect from last layer, all positions
            for act in model.mlp_acts:
                all_acts.append(act.cpu())  # (B, T, d_model)

    # Concatenate: (N, d_model)
    acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in all_acts], dim=0)
    return acts


# ── Sparse Autoencoder ──

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder: encodes dense activations into sparse features.

    encoder: x → z = ReLU(W_enc @ x + b_enc)   (sparse features)
    decoder: z → x̂ = W_dec @ z + b_dec          (reconstruction)

    L1 penalty on z encourages sparsity.
    """
    def __init__(self, n_input, n_features, tied_decoder=False):
        super().__init__()
        self.n_input = n_input
        self.n_features = n_features
        self.tied_decoder = tied_decoder

        self.W_enc = nn.Parameter(torch.randn(n_features, n_input) * (1.0 / n_input))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        if not tied_decoder:
            self.W_dec = nn.Parameter(torch.randn(n_input, n_features) * (1.0 / n_features))
        self.b_dec = nn.Parameter(torch.zeros(n_input))

    def encode(self, x):
        """Encode to sparse features: z = ReLU(W_enc @ x + b_enc)"""
        z = F.relu(x @ self.W_enc.T + self.b_enc)
        return z

    def decode(self, z):
        """Decode from sparse features: x̂ = W_dec @ z + b_dec"""
        if self.tied_decoder:
            x_hat = z @ self.W_enc.T + self.b_dec
        else:
            x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x, l1_coef=1e-3):
        """Reconstruction loss + L1 sparsity penalty."""
        x_hat, z = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = z.abs().mean()  # L1 per element, averaged
        total_loss = recon_loss + l1_coef * sparsity_loss
        return total_loss, recon_loss, sparsity_loss


# ── Training SAE ──

def train_sae(sae, activations, n_epochs=50, lr=1e-3, l1_coef=1e-3,
              batch_size=512, device='cpu'):
    """Train sparse autoencoder on collected activations."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    N = activations.shape[0]
    losses = {'total': [], 'recon': [], 'sparsity': []}
    dead_ratios = []
    l0_values = []

    for epoch in range(n_epochs):
        sae.train()
        perm = torch.randperm(N)
        epoch_total = epoch_recon = epoch_spars = 0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx = activations[idx].to(device)

            total_loss, recon_loss, sparsity_loss = sae.loss(bx, l1_coef=l1_coef)

            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            # Normalize decoder columns to unit norm (standard practice)
            with torch.no_grad():
                if hasattr(sae, 'W_dec'):
                    W_dec = sae.W_dec.data
                    norms = W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
                    sae.W_dec.data = W_dec / norms

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_spars += sparsity_loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate metrics
        sae.eval()
        with torch.no_grad():
            sample = activations[:2000].to(device)
            _, z = sae(sample)

            # Dead features: features that never activate (>0) on any sample
            active = (z > 1e-6).any(dim=0)
            dead_ratio = 1.0 - active.float().mean().item()

            # L0: average number of active features per sample
            l0 = (z > 1e-6).float().sum(dim=1).mean().item()

        losses['total'].append(epoch_total / n_batches)
        losses['recon'].append(epoch_recon / n_batches)
        losses['sparsity'].append(epoch_spars / n_batches)
        dead_ratios.append(dead_ratio)
        l0_values.append(l0)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1} | Total: {losses['total'][-1]:.4f} | "
                  f"Recon: {losses['recon'][-1]:.4f} | L1: {losses['sparsity'][-1]:.6f} | "
                  f"L0: {l0:.1f} | Dead: {dead_ratio:.1%}")

    return losses, dead_ratios, l0_values


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "90-sae"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Pre-train a small Transformer ──
    print("=== Step 1: Pre-train Tiny Transformer ===")
    vocab_size = 100
    seq_len = 16
    d_model = 64
    d_ff = 128

    transformer = TinyTransformer(vocab_size=vocab_size, d_model=d_model,
                                  n_heads=4, n_layers=2, d_ff=d_ff).to(device)
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"  Transformer params: {n_params:,}")

    pretrain_transformer(transformer, n_steps=3000, lr=1e-3, device=device,
                         vocab_size=vocab_size, seq_len=seq_len)

    # ── Step 2: Collect MLP activations ──
    print("\n=== Step 2: Collect MLP Activations ===")
    activations = collect_activations(transformer, n_batches=100, batch_size=64,
                                       seq_len=seq_len, vocab_size=vocab_size, device=device)
    print(f"  Collected activations: {activations.shape}")
    print(f"  Activation mean: {activations.mean():.4f}, std: {activations.std():.4f}")

    # ── Step 3: Train SAE with different sparsity levels ──
    print("\n=== Step 3: Train Sparse Autoencoders ===")

    n_features = 256  # Overcomplete: more features than input dim
    l1_coefs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    sae_results = {}

    for l1_coef in l1_coefs:
        print(f"\n  --- L1 coefficient = {l1_coef} ---")
        sae = SparseAutoencoder(n_input=d_model, n_features=n_features).to(device)
        losses, dead_ratios, l0_values = train_sae(
            sae, activations, n_epochs=50, lr=1e-3,
            l1_coef=l1_coef, batch_size=512, device=device)

        # Final evaluation
        sae.eval()
        with torch.no_grad():
            sample = activations[:2000].to(device)
            x_hat, z = sae(sample)

            recon_mse = F.mse_loss(x_hat, sample).item()
            l0 = (z > 1e-6).float().sum(dim=1).mean().item()
            dead = 1.0 - (z > 1e-6).any(dim=0).float().mean().item()

            # Fraction of variance explained
            var_total = (sample - sample.mean(0)).pow(2).sum()
            var_residual = (sample - x_hat).pow(2).sum()
            fve = 1.0 - var_residual.item() / var_total.item()

        sae_results[l1_coef] = {
            'losses': losses, 'dead_ratios': dead_ratios, 'l0_values': l0_values,
            'final_recon': recon_mse, 'final_l0': l0, 'final_dead': dead,
            'fve': fve, 'sae': sae, 'z': z.cpu(), 'x_hat': x_hat.cpu(),
        }

        print(f"  Final: Recon={recon_mse:.6f}, L0={l0:.1f}, Dead={dead:.1%}, FVE={fve:.2%}")

    # ── Visualization ──

    # 1. Sparsity vs Reconstruction trade-off
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    l0_vals = [sae_results[c]['final_l0'] for c in l1_coefs]
    recon_vals = [sae_results[c]['final_recon'] for c in l1_coefs]
    dead_vals = [sae_results[c]['final_dead'] * 100 for c in l1_coefs]
    fve_vals = [sae_results[c]['fve'] * 100 for c in l1_coefs]

    axes[0].plot(l0_vals, recon_vals, 'o-', color='blue')
    for i, c in enumerate(l1_coefs):
        axes[0].annotate(f'L1={c}', (l0_vals[i], recon_vals[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[0].set_xlabel("L0 (avg active features)")
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_title("Sparsity vs Reconstruction Quality")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(l1_coefs)), dead_vals, color='salmon', edgecolor='red')
    axes[1].set_xticks(range(len(l1_coefs)))
    axes[1].set_xticklabels([f'{c:.0e}' for c in l1_coefs], fontsize=9)
    axes[1].set_xlabel("L1 Coefficient")
    axes[1].set_ylabel("Dead Feature Ratio (%)")
    axes[1].set_title("Dead Features vs Sparsity Penalty")
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].plot(l0_vals, fve_vals, 'o-', color='green')
    for i, c in enumerate(l1_coefs):
        axes[2].annotate(f'L1={c}', (l0_vals[i], fve_vals[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel("L0 (avg active features)")
    axes[2].set_ylabel("Fraction of Variance Explained (%)")
    axes[2].set_title("Sparsity vs Explained Variance")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("SAE: Sparsity-Quality Trade-off Across L1 Penalties", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "sparsity_tradeoff.png", dpi=150)
    plt.close()

    # 2. Feature activation heatmap (best L1 coef)
    best_coef = l1_coefs[2]  # 1e-3, middle ground
    z_best = sae_results[best_coef]['z']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Show activation heatmap: samples x features (top 50 active features)
    z_np = np.nan_to_num(z_best.numpy(), nan=0.0, posinf=1e6, neginf=0.0)
    feature_activity = (z_np > 1e-6).sum(axis=0)
    active_features = np.argsort(feature_activity)[-50:]  # Top 50 most active

    heatmap = z_np[:100, active_features]
    im = axes[0].imshow(heatmap, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[0].set_xlabel("Feature Index (top-50 active)")
    axes[0].set_ylabel("Sample")
    axes[0].set_title(f"Feature Activations (L1={best_coef:.0e})")
    plt.colorbar(im, ax=axes[0], shrink=0.6)

    # Feature activation frequency histogram
    all_freq = (z_np > 1e-6).mean(axis=0)
    axes[1].hist(all_freq, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].axvline(x=0.0, color='red', linestyle='--', label='Never active (dead)')
    n_dead = (all_freq == 0).sum()
    axes[1].set_xlabel("Feature Activation Frequency")
    axes[1].set_ylabel("Number of Features")
    axes[1].set_title(f"Feature Activation Distribution\n({n_dead} dead / {n_features} total)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("SAE: Feature Activation Patterns", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "feature_heatmap.png", dpi=150)
    plt.close()

    # 3. Training curves for different L1 values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(l1_coefs)))

    for i, coef in enumerate(l1_coefs):
        r = sae_results[coef]
        epochs = range(1, len(r['losses']['recon']) + 1)

        axes[0, 0].plot(epochs, r['losses']['recon'], color=colors[i],
                        label=f'L1={coef:.0e}', alpha=0.8)
        axes[0, 1].plot(epochs, r['losses']['sparsity'], color=colors[i],
                        label=f'L1={coef:.0e}', alpha=0.8)
        axes[1, 0].plot(epochs, r['l0_values'], color=colors[i],
                        label=f'L1={coef:.0e}', alpha=0.8)
        axes[1, 1].plot(epochs, [d * 100 for d in r['dead_ratios']], color=colors[i],
                        label=f'L1={coef:.0e}', alpha=0.8)

    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Reconstruction Loss")
    axes[0, 0].set_title("Reconstruction Loss")
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("L1 Sparsity Loss")
    axes[0, 1].set_title("Sparsity Penalty")
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("L0 (avg active features)")
    axes[1, 0].set_title("Feature Sparsity (L0)")
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Dead Feature Ratio (%)")
    axes[1, 1].set_title("Dead Features Over Training")
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("SAE: Training Dynamics at Different Sparsity Levels", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 4. Comparison: SAE features vs raw neurons (polysemantic vs monosemantic)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw neuron activations
    raw_acts = activations[:200].numpy()
    # Compute selectivity: how many samples each neuron activates on
    raw_selectivity = (raw_acts > raw_acts.mean(axis=0, keepdims=True)).mean(axis=0)

    # SAE feature activations
    z_np_full = np.nan_to_num(sae_results[best_coef]['z'][:200].numpy(), nan=0.0, posinf=1e6, neginf=0.0)
    sae_selectivity = (z_np_full > 1e-6).mean(axis=0)

    axes[0].hist(raw_selectivity, bins=30, color='coral', edgecolor='black', alpha=0.7,
                 label='Raw neurons')
    axes[0].hist(sae_selectivity[sae_selectivity > 0], bins=30, color='steelblue',
                 edgecolor='black', alpha=0.7, label='SAE features (active)')
    axes[0].set_xlabel("Fraction of samples activating")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Activation Selectivity: Raw vs SAE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Max activation correlation: how polysemantic are raw neurons vs SAE features?
    # Compute pairwise correlation for raw neurons
    raw_corr = np.corrcoef(raw_acts.T)
    np.fill_diagonal(raw_corr, 0)
    raw_max_corr = np.nan_to_num(np.abs(raw_corr).max(axis=1))

    sae_corr = np.corrcoef(z_np_full.T)
    np.fill_diagonal(sae_corr, 0)
    sae_max_corr = np.nan_to_num(np.abs(sae_corr).max(axis=1))

    axes[1].hist(raw_max_corr, bins=30, color='coral', edgecolor='black', alpha=0.7,
                 label='Raw neurons')
    axes[1].hist(sae_max_corr[sae_selectivity > 0], bins=30, color='steelblue',
                 edgecolor='black', alpha=0.7, label='SAE features (active)')
    axes[1].set_xlabel("Max absolute correlation with other feature/neuron")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Feature Correlation: Polysemantic vs Monosemantic")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("SAE: From Polysemantic Neurons to Monosemantic Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "polysemantic_vs_monosemantic.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Polysemantic\nNeurons", "Dense activations\nEach neuron responds\nto multiple concepts\nHard to interpret\n→ Superposition", 0.14, 'coral'),
        ("Sparse\nAutoencoder", "Overcomplete dict\nL1 sparsity penalty\nEncodes x→z (sparse)\nDecodes z→x̂ (recon)\n→ Monosemantic", 0.5, 'steelblue'),
        ("Monosemantic\nFeatures", "Each feature =\none interpretable\nconcept\nL0 << n_features\n→ Interpretability!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Sparse Autoencoders: Decomposing Polysemantic Representations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "sae_concept.png", dpi=150)
    plt.close()

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"  Input dim: {d_model}, SAE features: {n_features} (overcomplete)")
    print(f"  {'L1 Coef':>10s} | {'L0':>6s} | {'Recon MSE':>10s} | {'Dead%':>6s} | {'FVE':>6s}")
    print("  " + "-" * 50)
    for coef in l1_coefs:
        r = sae_results[coef]
        print(f"  {coef:>10.0e} | {r['final_l0']:>6.1f} | {r['final_recon']:>10.6f} | "
              f"{r['final_dead']:>5.1%} | {r['fve']:>5.1%}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
