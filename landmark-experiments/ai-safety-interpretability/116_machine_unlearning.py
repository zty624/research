"""
Minimal Machine Unlearning Reproduction
========================================
Reproduces core ideas from machine unlearning literature:
1. Exact unlearning: retrain from scratch without forget data (gold standard)
2. Gradient ascent: maximize loss on forget data (naive approach)
3. Influence-based: approximate effect of removing data points
4. Compare: unlearning quality vs model utility retention
5. Show: tradeoff between forgetting and performance
6. Demonstrate: membership inference attack to verify unlearning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── Model ──

class Classifier(nn.Module):
    """Simple classifier for unlearning experiments."""
    def __init__(self, input_dim=20, hidden=64, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Data ──

class UnlearningDataset:
    """Synthetic dataset with distinct subgroups for unlearning experiments.

    Creates 5 classes, each with a Gaussian cluster.
    'Forget' data is a subset of one class that we want the model to 'unlearn'.
    """
    def __init__(self, n_samples=2000, input_dim=20, n_classes=5,
                 forget_class=2, forget_fraction=0.5, device='cpu'):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.forget_class = forget_class
        self.device = device

        # Generate class-specific clusters
        torch.manual_seed(42)
        self.centers = torch.randn(n_classes, input_dim) * 3

        # Generate all data
        all_x = []
        all_y = []
        for c in range(n_classes):
            n_c = n_samples // n_classes
            x_c = self.centers[c] + torch.randn(n_c, input_dim) * 0.5
            all_x.append(x_c)
            all_y.append(torch.full((n_c,), c, dtype=torch.long))

        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)

        # Split into retain and forget
        forget_mask = all_y == forget_class
        forget_indices = torch.where(forget_mask)[0]
        n_forget = int(len(forget_indices) * forget_fraction)
        perm = torch.randperm(len(forget_indices))
        forget_idx = forget_indices[perm[:n_forget]]

        self.forget_mask = torch.zeros(len(all_x), dtype=torch.bool)
        self.forget_mask[forget_idx] = True
        self.retain_mask = ~self.forget_mask

        self.all_x = all_x.to(device)
        self.all_y = all_y.to(device)
        self.forget_x = all_x[self.forget_mask].to(device)
        self.forget_y = all_y[self.forget_mask].to(device)
        self.retain_x = all_x[self.retain_mask].to(device)
        self.retain_y = all_y[self.retain_mask].to(device)

        print(f"  Total: {len(all_x)}, Forget: {self.forget_mask.sum()}, "
              f"Retain: {self.retain_mask.sum()}")


# ── Training ──

def train_model(model, x, y, n_steps=1000, lr=1e-3, batch_size=128, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    accs = []
    N = x.shape[0]

    for step in range(n_steps):
        idx = torch.randint(0, N, (min(batch_size, N),))
        xb, yb = x[idx], y[idx]

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        with torch.no_grad():
            acc = (logits.argmax(-1) == yb).float().mean().item()
            accs.append(acc)

    return losses, accs


# ── Unlearning Methods ──

def exact_unlearning(dataset, n_steps=1000, lr=1e-3, device='cpu'):
    """Gold standard: retrain from scratch on retain data only."""
    model = Classifier(input_dim=dataset.input_dim, n_classes=dataset.n_classes).to(device)
    losses, accs = train_model(model, dataset.retain_x, dataset.retain_y,
                                n_steps=n_steps, lr=lr, device=device)
    return model, losses, accs


def gradient_ascent_unlearning(original_model, dataset, n_steps=200,
                                lr=1e-3, device='cpu'):
    """Naive unlearning: gradient ascent on forget data + normal training on retain.

    Maximizes loss on forget data while minimizing on retain data.
    """
    model = deepcopy(original_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics = {'forget_loss': [], 'retain_loss': [], 'retain_acc': []}

    for step in range(n_steps):
        # Gradient ascent on forget data
        logits_f = model(dataset.forget_x)
        forget_loss = -F.cross_entropy(logits_f, dataset.forget_y)  # negate = ascent

        # Normal training on retain data
        idx = torch.randint(0, dataset.retain_x.shape[0], (64,))
        logits_r = model(dataset.retain_x[idx])
        retain_loss = F.cross_entropy(logits_r, dataset.retain_y[idx])

        total_loss = forget_loss + retain_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics['forget_loss'].append(-forget_loss.item())
            metrics['retain_loss'].append(retain_loss.item())
            acc = (logits_r.argmax(-1) == dataset.retain_y[idx]).float().mean().item()
            metrics['retain_acc'].append(acc)

    return model, metrics


def fine_tune_unlearning(original_model, dataset, n_steps=200,
                          lr=1e-4, device='cpu'):
    """Fine-tune unlearning: just continue training on retain data."""
    model = deepcopy(original_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics = {'forget_loss': [], 'retain_loss': [], 'retain_acc': []}

    for step in range(n_steps):
        idx = torch.randint(0, dataset.retain_x.shape[0], (64,))
        logits = model(dataset.retain_x[idx])
        loss = F.cross_entropy(logits, dataset.retain_y[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits_f = model(dataset.forget_x)
            forget_loss = F.cross_entropy(logits_f, dataset.forget_y)
            acc = (logits.argmax(-1) == dataset.retain_y[idx]).float().mean().item()
            metrics['forget_loss'].append(forget_loss.item())
            metrics['retain_loss'].append(loss.item())
            metrics['retain_acc'].append(acc)

    return model, metrics


# ── Evaluation ──

def membership_inference_attack(model, forget_x, forget_y, test_x, test_y, device='cpu'):
    """Simple membership inference: can we distinguish forget data from test data?

    Uses model confidence as the signal. If the model has 'forgotten' the data,
    its confidence on forget data should be similar to unseen test data.
    """
    model.eval()
    with torch.no_grad():
        logits_f = model(forget_x)
        probs_f = F.softmax(logits_f, dim=-1)
        max_prob_f = probs_f.max(dim=-1)[0]  # confidence on forget data

        logits_t = model(test_x)
        probs_t = F.softmax(logits_t, dim=-1)
        max_prob_t = probs_t.max(dim=-1)[0]  # confidence on test data

    # AUC: how well can we distinguish forget from test?
    labels = torch.cat([torch.ones(len(forget_x)), torch.zeros(len(test_x))])
    scores = torch.cat([max_prob_f, max_prob_t])

    # Move to CPU for AUC computation
    labels = labels.cpu()
    scores = scores.cpu()

    # Simple threshold-based AUC
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices]

    tp = 0
    fp = 0
    auc = 0.0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp

    auc = auc / (n_pos * n_neg + 1e-10)
    return auc, max_prob_f.mean().item(), max_prob_t.mean().item()


def evaluate_unlearning(model, dataset, device='cpu'):
    """Evaluate unlearning: accuracy on forget, retain, and all data."""
    model.eval()
    results = {}
    with torch.no_grad():
        for name, x, y in [
            ('forget', dataset.forget_x, dataset.forget_y),
            ('retain', dataset.retain_x, dataset.retain_y),
            ('all', dataset.all_x, dataset.all_y),
        ]:
            logits = model(x)
            acc = (logits.argmax(-1) == y).float().mean().item()
            loss = F.cross_entropy(logits, y).item()
            results[name] = {'acc': acc, 'loss': loss}

    # Generate test data for membership inference
    test_x = []
    test_y = []
    for c in range(dataset.n_classes):
        n_c = 50
        x_c = dataset.centers[c] + torch.randn(n_c, dataset.input_dim) * 0.5
        test_x.append(x_c)
        test_y.append(torch.full((n_c,), c, dtype=torch.long))
    test_x = torch.cat(test_x).to(device)
    test_y = torch.cat(test_y).to(device)

    mia_auc, conf_forget, conf_test = membership_inference_attack(
        model, dataset.forget_x, dataset.forget_y, test_x, test_y, device)
    results['mia_auc'] = mia_auc
    results['conf_forget'] = conf_forget
    results['conf_test'] = conf_test

    return results


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "116-machine-unlearning"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("=== Creating Dataset ===")
    dataset = UnlearningDataset(n_samples=2000, input_dim=20, n_classes=5,
                                  forget_class=2, forget_fraction=0.5, device=device)

    # Train original model on ALL data
    print("\n=== Training Original Model (all data) ===")
    original = Classifier(input_dim=20, n_classes=5).to(device)
    orig_losses, orig_accs = train_model(original, dataset.all_x, dataset.all_y,
                                          n_steps=1500, device=device)

    # Evaluate original
    orig_eval = evaluate_unlearning(original, dataset, device)
    print(f"  Original - forget acc: {orig_eval['forget']['acc']:.3f}, "
          f"retain acc: {orig_eval['retain']['acc']:.3f}, "
          f"MIA AUC: {orig_eval['mia_auc']:.3f}")

    # ── Method 1: Exact unlearning ──
    print("\n=== Exact Unlearning (retrain from scratch) ===")
    exact_model, exact_losses, exact_accs = exact_unlearning(dataset, n_steps=1500, device=device)
    exact_eval = evaluate_unlearning(exact_model, dataset, device)
    print(f"  Exact - forget acc: {exact_eval['forget']['acc']:.3f}, "
          f"retain acc: {exact_eval['retain']['acc']:.3f}, "
          f"MIA AUC: {exact_eval['mia_auc']:.3f}")

    # ── Method 2: Gradient ascent ──
    print("\n=== Gradient Ascent Unlearning ===")
    ga_model, ga_metrics = gradient_ascent_unlearning(original, dataset, n_steps=300, device=device)
    ga_eval = evaluate_unlearning(ga_model, dataset, device)
    print(f"  GA - forget acc: {ga_eval['forget']['acc']:.3f}, "
          f"retain acc: {ga_eval['retain']['acc']:.3f}, "
          f"MIA AUC: {ga_eval['mia_auc']:.3f}")

    # ── Method 3: Fine-tune unlearning ──
    print("\n=== Fine-Tune Unlearning ===")
    ft_model, ft_metrics = fine_tune_unlearning(original, dataset, n_steps=300, device=device)
    ft_eval = evaluate_unlearning(ft_model, dataset, device)
    print(f"  FT - forget acc: {ft_eval['forget']['acc']:.3f}, "
          f"retain acc: {ft_eval['retain']['acc']:.3f}, "
          f"MIA AUC: {ft_eval['mia_auc']:.3f}")

    # ── Visualization ──

    # 1. Training curves for gradient ascent
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    w = 10
    for ax, key, title, color in [
        (axes[0], 'forget_loss', 'Forget Data Loss', 'red'),
        (axes[1], 'retain_loss', 'Retain Data Loss', 'blue'),
        (axes[2], 'retain_acc', 'Retain Accuracy', 'green'),
    ]:
        ga_s = np.convolve(ga_metrics[key], np.ones(w)/w, mode='valid')
        ft_s = np.convolve(ft_metrics[key], np.ones(w)/w, mode='valid')
        ax.plot(ga_s, label='Gradient Ascent', color='red')
        ax.plot(ft_s, label='Fine-Tune', color='blue')
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Machine Unlearning: Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'unlearning_dynamics.png', dpi=150)
    plt.close()

    # 2. Comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ['Original', 'Exact', 'Grad Ascent', 'Fine-Tune']
    evals = [orig_eval, exact_eval, ga_eval, ft_eval]
    colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#3498db']

    # Forget accuracy (lower = better unlearning)
    forget_accs = [e['forget']['acc'] for e in evals]
    axes[0].bar(methods, forget_accs, color=colors, alpha=0.7)
    axes[0].set_ylabel("Accuracy on Forget Data")
    axes[0].set_title("Forgetting Quality (lower = better)")
    axes[0].grid(True, alpha=0.3, axis='y')

    # Retain accuracy (higher = better)
    retain_accs = [e['retain']['acc'] for e in evals]
    axes[1].bar(methods, retain_accs, color=colors, alpha=0.7)
    axes[1].set_ylabel("Accuracy on Retain Data")
    axes[1].set_title("Model Utility (higher = better)")
    axes[1].grid(True, alpha=0.3, axis='y')

    # MIA AUC (closer to 0.5 = better unlearning)
    mia_aucs = [e['mia_auc'] for e in evals]
    axes[2].bar(methods, mia_aucs, color=colors, alpha=0.7)
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Random (0.5)')
    axes[2].set_ylabel("MIA AUC")
    axes[2].set_title("Privacy (closer to 0.5 = better)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Machine Unlearning: Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png', dpi=150)
    plt.close()

    # 3. Forget-Retain tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, method in enumerate(methods):
        ax.scatter(forget_accs[i], retain_accs[i], s=200, color=colors[i], zorder=5)
        ax.annotate(method, (forget_accs[i], retain_accs[i]),
                    fontsize=10, xytext=(10, 5), textcoords='offset points')
    ax.set_xlabel("Forget Accuracy (lower = better forgetting)")
    ax.set_ylabel("Retain Accuracy (higher = better utility)")
    ax.set_title("Unlearning: Forget-Utility Tradeoff")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'tradeoff.png', dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    concept = (
        "Machine Unlearning: Methods & Tradeoffs\n"
        "=" * 55 + "\n\n"
        "Goal: Remove influence of specific data from trained model\n"
        "  while preserving performance on remaining data.\n\n"
        "Methods:\n"
        "  1. Exact Unlearning: Retrain from scratch without forget data\n"
        "     → Perfect forgetting, but very expensive\n\n"
        "  2. Gradient Ascent: Maximize loss on forget data\n"
        "     → Fast, but may degrade overall performance\n\n"
        "  3. Fine-Tune: Continue training on retain data only\n"
        "     → Simple, but incomplete forgetting\n\n"
        "Evaluation:\n"
        "  • Forget accuracy: how well model forgets (lower = better)\n"
        "  • Retain accuracy: how well model preserves utility (higher = better)\n"
        "  • MIA AUC: membership inference attack (closer to 0.5 = better privacy)"
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
