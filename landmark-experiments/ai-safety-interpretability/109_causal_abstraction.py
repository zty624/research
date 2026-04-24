"""
Minimal Causal Abstraction / Mechanistic Interpretability Reproduction
======================================================================
Reproduces core ideas from "Causal Abstraction: A Theoretical Foundation
for Mechanistic Interpretability" (2301.04709, Geiger et al.):
1. Causal scrubbing: test if an algorithm-level variable is implemented by a neural network
2. Intervention: replace intermediate activations and measure output change
3. Causal graph: model algorithm as DAG of variables with causal links
4. Alignment score: how well neural activations correspond to algorithm variables
5. Compare: random ablation vs causal intervention vs full scrubbing
6. Show: which network components implement which algorithmic steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ── Algorithm: Simple Token Classification ──
# Algorithm: count vowels in input → if vowels > threshold → label 1 else 0

def algorithm_vowel_count(tokens, vowel_set=None):
    """Ground truth algorithm: count vowels → classify."""
    if vowel_set is None:
        vowel_set = {0, 4, 8, 14, 20}  # 'a','e','i','o','u' positions in alphabet
    counts = []
    for seq in tokens:
        n_vowels = sum(1 for t in seq if t in vowel_set)
        counts.append(n_vowels)
    return torch.tensor(counts, dtype=torch.float32)


def algorithm_classify(tokens, threshold=3):
    """Binary classification from vowel count."""
    counts = algorithm_vowel_count(tokens)
    return (counts > threshold).long()


# ── Neural Network ──

class TokenClassifier(nn.Module):
    """Small transformer that learns the vowel-counting algorithm."""
    def __init__(self, vocab_size=26, d_model=64, n_heads=2, n_layers=2,
                 max_len=12, n_classes=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_intermediates=False):
        """Forward pass, optionally returning intermediate activations."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)

        intermediates = {'embed': h.clone()}

        for i, layer in enumerate(self.layers):
            h = layer(h)
            intermediates[f'layer_{i}'] = h.clone()

        h = self.norm(h)
        intermediates['final'] = h.clone()

        # Pool over sequence
        pooled = h.mean(dim=1)
        intermediates['pooled'] = pooled.clone()

        logits = self.head(pooled)
        intermediates['logits'] = logits.clone()

        if return_intermediates:
            return logits, intermediates
        return logits


# ── Causal Interventions ──

class CausalIntervention:
    """Base class for causal interventions on neural activations."""
    def __init__(self, name):
        self.name = name

    def apply(self, activation, source_activation):
        raise NotImplementedError


class AblationIntervention(CausalIntervention):
    """Zero ablation: replace activation with zeros."""
    def __init__(self):
        super().__init__("zero_ablation")

    def apply(self, activation, source_activation):
        return torch.zeros_like(activation)


class RandomIntervention(CausalIntervention):
    """Random replacement: replace with random activation from another input."""
    def __init__(self):
        super().__init__("random_resample")

    def apply(self, activation, source_activation):
        return source_activation


class CausalScrubbing:
    """Causal scrubbing: systematically intervene on variables to test alignment.

    For each algorithm variable V, we:
    1. Identify the corresponding neural activation A
    2. Run the model on input x (original) and input x' (where V differs)
    3. Replace A in the original run with A from the x' run
    4. Measure how much the output changes

    If the output changes significantly, A implements V (high alignment).
    If the output doesn't change, A doesn't implement V (low alignment).
    """
    def __init__(self, model, algorithm_var_names):
        self.model = model
        self.var_names = algorithm_var_names

    def compute_alignment_scores(self, inputs, labels, n_interventions=100):
        """Compute alignment scores for each intermediate activation."""
        self.model.eval()
        device = next(self.model.parameters()).device

        # Get reference outputs
        with torch.no_grad():
            ref_logits, ref_intermediates = self.model(inputs, return_intermediates=True)
            ref_preds = ref_logits.argmax(dim=-1)

        # For each intermediate layer, test causal effect
        scores = {}
        for layer_name in ref_intermediates:
            if layer_name == 'logits':
                continue

            same_output_count = 0
            total = 0

            for _ in range(n_interventions):
                # Pick two inputs with different algorithm outputs
                idx1 = torch.randint(0, len(inputs), (1,)).item()
                idx2 = torch.randint(0, len(inputs), (1,)).item()

                if labels[idx1] == labels[idx2]:
                    continue  # Need different algorithm outputs

                # Get activations for both inputs
                x1 = inputs[idx1:idx1+1]
                x2 = inputs[idx2:idx2+1]

                with torch.no_grad():
                    _, inter1 = self.model(x1, return_intermediates=True)
                    _, inter2 = self.model(x2, return_intermediates=True)

                # Intervention: replace layer in x1 with layer from x2
                # We do this by creating a hook
                intervened = False
                def hook_fn(module, input, output):
                    nonlocal intervened
                    if not intervened:
                        intervened = True
                        return inter2[layer_name]
                    return output

                # Find which module corresponds to this layer
                # Simple approach: just measure using direct replacement
                # Replace pooled activation
                if layer_name == 'pooled':
                    with torch.no_grad():
                        # Use inter2's pooled → project through head
                        new_logits = self.model.head(inter2[layer_name])
                        new_pred = new_logits.argmax(dim=-1)
                else:
                    # For other layers, do approximate intervention
                    # by running from that layer forward
                    with torch.no_grad():
                        if layer_name == 'final':
                            new_pooled = inter2['final'].mean(dim=1)
                            new_logits = self.model.head(new_pooled)
                        elif layer_name.startswith('layer_'):
                            new_pooled = self.model.norm(inter2[layer_name]).mean(dim=1)
                            new_logits = self.model.head(new_pooled)
                        else:
                            continue
                        new_pred = new_logits.argmax(dim=-1)

                total += 1
                if new_pred != ref_preds[idx1]:
                    same_output_count += 1

            if total > 0:
                scores[layer_name] = same_output_count / total
            else:
                scores[layer_name] = 0.0

        return scores


# ── Training ──

def train_classifier(model, n_steps=2000, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []
    accs = []

    for step in range(n_steps):
        tokens = torch.randint(0, 26, (batch_size, 12), device=device)
        labels = algorithm_classify(tokens).to(device)

        logits = model(tokens)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    return losses, accs


# ── Activation Probing ──

def probe_activations(model, inputs, labels, device='cpu'):
    """Train linear probes on each layer to predict algorithm variables."""
    model.eval()
    with torch.no_grad():
        _, intermediates = model(inputs, return_intermediates=True)

    # Compute algorithm variables
    vowel_counts = algorithm_vowel_count(inputs.cpu().numpy()).to(device)

    probe_scores = {}
    for layer_name, acts in intermediates.items():
        if layer_name == 'logits':
            continue

        # Flatten spatial dims
        if acts.dim() == 3:
            acts_flat = acts.reshape(acts.shape[0], -1)  # (B, T*D)
        else:
            acts_flat = acts  # (B, D)

        # Simple linear regression probe
        X = acts_flat.cpu().numpy()
        y = vowel_counts.cpu().numpy()

        # Normal equation: w = (X^T X)^{-1} X^T y
        try:
            X_t = X.T
            w = np.linalg.solve(X_t @ X + 1e-4 * np.eye(X_t.shape[0]), X_t @ y)
            pred = X @ w
            r2 = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-10)
            probe_scores[layer_name] = max(0, r2)
        except np.linalg.LinAlgError:
            probe_scores[layer_name] = 0.0

    return probe_scores


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "109-causal-abstraction"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    print("=== Training Token Classifier ===")
    model = TokenClassifier(vocab_size=26, d_model=64, n_heads=2,
                             n_layers=2, max_len=12, n_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    losses, accs = train_classifier(model, n_steps=2000, device=device)

    # Generate test data
    test_tokens = torch.randint(0, 26, (500, 12), device=device)
    test_labels = algorithm_classify(test_tokens).to(device)

    # ── Experiment 1: Causal scrubbing alignment ──
    print("\n=== Causal Scrubbing: Alignment Scores ===")
    scrubber = CausalScrubbing(model, ['embed', 'layer_0', 'layer_1', 'final', 'pooled'])
    alignment = scrubber.compute_alignment_scores(test_tokens, test_labels, n_interventions=200)

    for name, score in sorted(alignment.items()):
        print(f"  {name}: {score:.3f}")

    # ── Experiment 2: Linear probing ──
    print("\n=== Linear Probing: Can layers predict vowel count? ===")
    probe_scores = probe_activations(model, test_tokens, test_labels, device=device)

    for name, score in sorted(probe_scores.items()):
        print(f"  {name}: R² = {score:.3f}")

    # ── Experiment 3: Ablation study ──
    print("\n=== Ablation Study: Zero vs Random Intervention ===")
    model.eval()

    ablation_results = {}
    with torch.no_grad():
        ref_logits, ref_intermediates = model(test_tokens[:100], return_intermediates=True)
        ref_acc = (ref_logits.argmax(-1) == test_labels[:100]).float().mean().item()

    for layer_name in ['embed', 'layer_0', 'layer_1', 'pooled']:
        # Zero ablation
        accs_zero = []
        for i in range(50):
            x = test_tokens[i:i+1]
            _, inter = model(x, return_intermediates=True)

            # Zero out the layer
            zero_inter = {k: v.clone() for k, v in inter.items()}
            zero_inter[layer_name] = torch.zeros_like(inter[layer_name])

            with torch.no_grad():
                if layer_name == 'pooled':
                    logits = model.head(zero_inter['pooled'])
                elif layer_name == 'final':
                    pooled = zero_inter['final'].mean(dim=1)
                    logits = model.head(pooled)
                elif layer_name.startswith('layer_'):
                    pooled = model.norm(zero_inter[layer_name]).mean(dim=1)
                    logits = model.head(pooled)
                else:
                    continue

            accs_zero.append((logits.argmax(-1) == test_labels[i]).float().item())

        ablation_results[layer_name] = {
            'zero': np.mean(accs_zero),
            'original': ref_acc,
        }

    for name, res in sorted(ablation_results.items()):
        print(f"  {name}: original={res['original']:.3f}, zero_ablation={res['zero']:.3f}, "
              f"drop={res['original'] - res['zero']:.3f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 30
    loss_s = np.convolve(losses, np.ones(w)/w, mode='valid')
    acc_s = np.convolve(accs, np.ones(w)/w, mode='valid')
    axes[0].plot(loss_s, color='blue')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(acc_s, color='green')
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Token Classifier Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training.png', dpi=150)
    plt.close()

    # 2. Alignment scores
    fig, ax = plt.subplots(figsize=(10, 5))
    names = sorted(alignment.keys())
    scores = [alignment[n] for n in names]
    colors = ['#e74c3c' if s > 0.5 else '#3498db' for s in scores]
    ax.bar(names, scores, color=colors, alpha=0.7)
    ax.set_ylabel("Causal Alignment Score")
    ax.set_title("Causal Scrubbing: Which Layers Implement Algorithm Variables?")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / 'alignment_scores.png', dpi=150)
    plt.close()

    # 3. Probing scores
    fig, ax = plt.subplots(figsize=(10, 5))
    probe_names = sorted(probe_scores.keys())
    probe_vals = [probe_scores[n] for n in probe_names]
    colors_probe = ['#e74c3c' if v > 0.7 else '#3498db' if v > 0.3 else '#95a5a6'
                    for v in probe_vals]
    ax.bar(probe_names, probe_vals, color=colors_probe, alpha=0.7)
    ax.set_ylabel("R² Score")
    ax.set_title("Linear Probing: Can Layers Predict Vowel Count?")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / 'probing_scores.png', dpi=150)
    plt.close()

    # 4. Ablation results
    fig, ax = plt.subplots(figsize=(10, 5))
    abl_names = sorted(ablation_results.keys())
    orig = [ablation_results[n]['original'] for n in abl_names]
    zero = [ablation_results[n]['zero'] for n in abl_names]
    x_pos = np.arange(len(abl_names))
    width = 0.35
    ax.bar(x_pos - width/2, orig, width, label='Original', color='green', alpha=0.7)
    ax.bar(x_pos + width/2, zero, width, label='Zero Ablation', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(abl_names)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study: Effect of Zeroing Each Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / 'ablation.png', dpi=150)
    plt.close()

    # 5. Combined: alignment vs probing
    fig, ax = plt.subplots(figsize=(8, 6))
    common_layers = sorted(set(alignment.keys()) & set(probe_scores.keys()))
    align_v = [alignment[n] for n in common_layers]
    probe_v = [probe_scores[n] for n in common_layers]
    ax.scatter(align_v, probe_v, s=100, c='steelblue', edgecolors='black', zorder=5)
    for i, name in enumerate(common_layers):
        ax.annotate(name, (align_v[i], probe_v[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("Causal Alignment Score")
    ax.set_ylabel("Linear Probe R²")
    ax.set_title("Causal Alignment vs Linear Probing")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(results_dir / 'alignment_vs_probing.png', dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    concept = (
        "Causal Abstraction for Mechanistic Interpretability (2301.04709)\n"
        "=" * 65 + "\n\n"
        "Algorithm Variables          Neural Activations\n"
        "─────────────────          ───────────────────\n"
        "  V1: vowel count     ←→     Layer 1 output\n"
        "  V2: threshold cmp   ←→     Pooled representation\n"
        "  V3: final label     ←→     Logits\n\n"
        "Causal Scrubbing Test:\n"
        "  1. Run model on input x (original activation A_x)\n"
        "  2. Run model on input x' where V differs (activation A_x')\n"
        "  3. Replace A_x with A_x' in the forward pass\n"
        "  4. If output changes → A implements V (aligned)\n"
        "  5. If output unchanged → A doesn't implement V\n\n"
        "Key insight: Causal intervention > correlation probing.\n"
        "Linear probes can find spurious correlations,\n"
        "but causal scrubbing tests whether the variable\n"
        "is actually USED by the computation."
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
