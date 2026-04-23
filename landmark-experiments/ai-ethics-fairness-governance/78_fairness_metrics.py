"""
Minimal Fairness Metrics & Mitigation Reproduction
====================================================
Reproduces core ideas from fairness ML literature (2403.17333, 2405.06909):
1. Demographic Parity: P(Y_hat=1|A=0) ≈ P(Y_hat=1|A=1)
2. Equalized Odds: P(Y_hat=1|Y=y,A=0) ≈ P(Y_hat=1|Y=y,A=1) for y in {0,1}
3. Calibration: P(Y=1|Y_hat=p,A=0) ≈ P(Y=1|Y_hat=p,A=1)
4. Preprocessing mitigation: Reweighing (adjust sample weights by group)
5. Postprocessing mitigation: Threshold adjustment per group
6. Accuracy vs fairness tradeoff: no free lunch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Biased Data ──

def create_biased_dataset(n_samples=2000, n_features=10, bias_strength=0.5, seed=42):
    """Create synthetic classification data with known demographic bias.

    Sensitive attribute A ∈ {0, 1} (e.g., gender or race).
    Label Y depends on features X, but A also influences both X and Y:
    - A=1 group gets systematically lower scores → lower positive rate
    - The "true" decision boundary is fair, but training data reflects historical bias
    """
    rng = np.random.RandomState(seed)

    # Sensitive attribute: 50/50 split
    A = rng.binomial(1, 0.5, n_samples)

    # Features: biased by group
    X = rng.randn(n_samples, n_features).astype(np.float32)

    # Inject bias: group A=1 gets shifted features
    X[A == 1] -= bias_strength

    # True labels: depend on features + direct bias
    true_scores = X @ rng.randn(n_features).astype(np.float32) + 0.5
    # Add direct discrimination: group A=1 needs higher threshold
    true_scores[A == 1] -= bias_strength * 1.5
    Y = (true_scores > 0).astype(np.float32)

    # Base rates differ by group
    base_rate_0 = Y[A == 0].mean()
    base_rate_1 = Y[A == 1].mean()
    print(f"  Base rates: A=0 → {base_rate_0:.3f}, A=1 → {base_rate_1:.3f}")
    print(f"  Disparity: {abs(base_rate_0 - base_rate_1):.3f}")

    return (torch.tensor(X), torch.tensor(A, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


# ── Classifier ──

class FairClassifier(nn.Module):
    def __init__(self, in_dim=10, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Fairness Metrics ──

def demographic_parity(y_pred, A):
    """P(Y_hat=1 | A=0) vs P(Y_hat=1 | A=1). Returns (rate_0, rate_1, diff)."""
    rate_0 = y_pred[A == 0].float().mean().item()
    rate_1 = y_pred[A == 1].float().mean().item()
    return rate_0, rate_1, abs(rate_0 - rate_1)


def equalized_odds(y_pred, y_true, A):
    """P(Y_hat=1 | Y=y, A=a) for each y and a. Returns TPR/FPR disparity."""
    metrics = {}
    for group in [0, 1]:
        mask = (A == group)
        # TPR: P(Y_hat=1 | Y=1, A=group)
        pos_mask = mask & (y_true == 1)
        if pos_mask.sum() > 0:
            tpr = y_pred[pos_mask].float().mean().item()
        else:
            tpr = 0.0
        # FPR: P(Y_hat=1 | Y=0, A=group)
        neg_mask = mask & (y_true == 0)
        if neg_mask.sum() > 0:
            fpr = y_pred[neg_mask].float().mean().item()
        else:
            fpr = 0.0
        metrics[group] = {'tpr': tpr, 'fpr': fpr}

    tpr_diff = abs(metrics[0]['tpr'] - metrics[1]['tpr'])
    fpr_diff = abs(metrics[0]['fpr'] - metrics[1]['fpr'])
    return metrics, tpr_diff, fpr_diff


def calibration_error(y_scores, y_true, A, n_bins=10):
    """Expected calibration error per group."""
    cal_errors = {}
    for group in [0, 1]:
        mask = (A == group)
        scores = y_scores[mask].numpy()
        labels = y_true[mask].numpy()

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = 0
        for i in range(n_bins):
            in_bin = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i + 1])
            n_in_bin = in_bin.sum()
            if n_in_bin > 0:
                avg_conf = scores[in_bin].mean()
                avg_acc = labels[in_bin].mean()
                ece += n_in_bin * abs(avg_acc - avg_conf)
                total += n_in_bin
        cal_errors[group] = ece / max(total, 1)
    return cal_errors


# ── Training ──

def train_classifier(model, X, Y, A, n_epochs=100, lr=1e-2,
                     sample_weights=None, lambda_fair=0.0):
    """Train classifier with optional reweighing and fairness regularization.

    Args:
        sample_weights: per-sample weights for reweighing mitigation
        lambda_fair: coefficient for adversarial fairness loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        logits = model(X)
        probs = torch.sigmoid(logits)

        # Standard BCE loss
        loss = F.binary_cross_entropy(probs, Y, reduction='none')

        if sample_weights is not None:
            loss = (loss * sample_weights).mean()
        else:
            loss = loss.mean()

        # Optional: fairness regularization (penalize correlation with A)
        if lambda_fair > 0:
            corr = (probs - probs.mean()) * (A - A.mean())
            fair_loss = corr.mean() ** 2
            loss = loss + lambda_fair * fair_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict(model, X, threshold=0.5):
    """Get binary predictions at a given threshold."""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
    return (probs >= threshold).long(), probs


# ── Mitigation Methods ──

def compute_reweighing_weights(Y, A):
    """Compute sample weights for reweighing (preprocessing mitigation).

    Weight = (n * P(Y=y, A=a)) / (n_{y,a}) for each (y,a) group.
    This equalizes the joint distribution P(Y,A) across groups.
    """
    n = len(Y)
    weights = torch.ones(n, dtype=torch.float32)

    for y_val in [0, 1]:
        for a_val in [0, 1]:
            mask = (Y == y_val) & (A == a_val)
            n_ya = mask.sum().item()
            n_y = (Y == y_val).sum().item()
            n_a = (A == a_val).sum().item()

            # Expected count under independence: n_y * n_a / n
            expected = n_y * n_a / n
            if n_ya > 0:
                weights[mask] = expected / n_ya

    return weights


def group_threshold_adjustment(probs, Y, A, target_dp=0.0):
    """Postprocessing: find per-group thresholds to achieve demographic parity.

    For each group, adjust threshold so that positive prediction rates
    are approximately equal across groups.
    """
    # Find threshold for majority group that maintains accuracy
    best_thresholds = {}
    global_rate = probs.float().mean().item()

    for group in [0, 1]:
        mask = (A == group)
        group_probs = probs[mask]
        group_labels = Y[mask]

        # Search for threshold that gives rate closest to global rate
        best_thresh = 0.5
        best_diff = float('inf')
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (group_probs >= thresh).float()
            rate = preds.mean().item()
            diff = abs(rate - global_rate + target_dp / 2 * (1 if group == 0 else -1))
            if diff < best_diff:
                best_diff = diff
                best_thresh = thresh

        best_thresholds[group] = best_thresh

    return best_thresholds


# ── Main ──

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    results_dir = Path(__file__).parent / "results" / "78-fairness-metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("=== Creating Biased Dataset ===")
    X, A, Y = create_biased_dataset(n_samples=2000, bias_strength=0.5)

    # Train/val split
    n_train = 1500
    X_train, X_val = X[:n_train], X[n_train:]
    A_train, A_val = A[:n_train], A[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    # ── Experiment 1: Unmitigated classifier ──

    print("\n=== Unmitigated Classifier ===")
    model_base = FairClassifier(in_dim=X.shape[1])
    train_classifier(model_base, X_train, Y_train, A_train, n_epochs=150)
    preds_base, probs_base = predict(model_base, X_val)

    acc_base = (preds_base == Y_val).float().mean().item()
    dp_base = demographic_parity(preds_base, A_val)
    eo_base = equalized_odds(preds_base, Y_val, A_val)

    print(f"  Accuracy:        {acc_base:.4f}")
    print(f"  DP rates:        A=0: {dp_base[0]:.4f}, A=1: {dp_base[1]:.4f}, diff: {dp_base[2]:.4f}")
    print(f"  EO TPR diff:     {eo_base[1]:.4f}, FPR diff: {eo_base[2]:.4f}")

    cal_base = calibration_error(probs_base, Y_val, A_val)
    print(f"  Calibration ECE: A=0: {cal_base[0]:.4f}, A=1: {cal_base[1]:.4f}")

    # ── Experiment 2: Reweighing (preprocessing) ──

    print("\n=== Reweighing (Preprocessing Mitigation) ===")
    reweigh_weights = compute_reweighing_weights(Y_train, A_train)
    print(f"  Weight range: [{reweigh_weights.min():.3f}, {reweigh_weights.max():.3f}]")

    model_rw = FairClassifier(in_dim=X.shape[1])
    train_classifier(model_rw, X_train, Y_train, A_train, n_epochs=150,
                     sample_weights=reweigh_weights)
    preds_rw, probs_rw = predict(model_rw, X_val)

    acc_rw = (preds_rw == Y_val).float().mean().item()
    dp_rw = demographic_parity(preds_rw, A_val)
    eo_rw = equalized_odds(preds_rw, Y_val, A_val)

    print(f"  Accuracy:        {acc_rw:.4f}")
    print(f"  DP rates:        A=0: {dp_rw[0]:.4f}, A=1: {dp_rw[1]:.4f}, diff: {dp_rw[2]:.4f}")
    print(f"  EO TPR diff:     {eo_rw[1]:.4f}, FPR diff: {eo_rw[2]:.4f}")

    # ── Experiment 3: Fairness regularization ──

    print("\n=== Fairness Regularization ===")
    reg_results = {}
    for lam in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model_reg = FairClassifier(in_dim=X.shape[1])
        train_classifier(model_reg, X_train, Y_train, A_train, n_epochs=150,
                         lambda_fair=lam)
        preds_reg, probs_reg = predict(model_reg, X_val)

        acc = (preds_reg == Y_val).float().mean().item()
        dp = demographic_parity(preds_reg, A_val)
        eo = equalized_odds(preds_reg, Y_val, A_val)

        reg_results[lam] = {'acc': acc, 'dp_diff': dp[2], 'tpr_diff': eo[1],
                            'fpr_diff': eo[2]}
        print(f"  lambda={lam:>5.1f}: acc={acc:.4f}, dp_diff={dp[2]:.4f}, "
              f"tpr_diff={eo[1]:.4f}")

    # ── Experiment 4: Postprocessing threshold adjustment ──

    print("\n=== Postprocessing Threshold Adjustment ===")
    group_thresholds = group_threshold_adjustment(probs_base, Y_val, A_val)
    print(f"  Group thresholds: A=0 → {group_thresholds[0]:.3f}, A=1 → {group_thresholds[1]:.3f}")

    # Apply group-specific thresholds
    preds_post = torch.zeros_like(Y_val, dtype=torch.long)
    for group in [0, 1]:
        mask = (A_val == group)
        preds_post[mask] = (probs_base[mask] >= group_thresholds[group]).long()

    acc_post = (preds_post == Y_val).float().mean().item()
    dp_post = demographic_parity(preds_post, A_val)
    eo_post = equalized_odds(preds_post, Y_val, A_val)

    print(f"  Accuracy:        {acc_post:.4f}")
    print(f"  DP rates:        A=0: {dp_post[0]:.4f}, A=1: {dp_post[1]:.4f}, diff: {dp_post[2]:.4f}")
    print(f"  EO TPR diff:     {eo_post[1]:.4f}, FPR diff: {eo_post[2]:.4f}")

    # ── Experiment 5: Accuracy vs fairness tradeoff curve ──

    print("\n=== Accuracy vs Fairness Tradeoff (threshold sweep) ===")
    tradeoff_data = {'thresh': [], 'acc': [], 'dp_diff': [], 'tpr_diff': []}

    for thresh in np.linspace(0.1, 0.9, 81):
        preds_t, probs_t = predict(model_base, X_val, threshold=thresh)
        acc_t = (preds_t == Y_val).float().mean().item()
        dp_t = demographic_parity(preds_t, A_val)
        eo_t = equalized_odds(preds_t, Y_val, A_val)
        tradeoff_data['thresh'].append(thresh)
        tradeoff_data['acc'].append(acc_t)
        tradeoff_data['dp_diff'].append(dp_t[2])
        tradeoff_data['tpr_diff'].append(eo_t[1])

    # Find Pareto-optimal points
    best_dp_idx = np.argmin(tradeoff_data['dp_diff'])
    best_acc_idx = np.argmax(tradeoff_data['acc'])
    print(f"  Best accuracy: thresh={tradeoff_data['thresh'][best_acc_idx]:.2f}, "
          f"acc={tradeoff_data['acc'][best_acc_idx]:.4f}, dp_diff={tradeoff_data['dp_diff'][best_acc_idx]:.4f}")
    print(f"  Best DP:       thresh={tradeoff_data['thresh'][best_dp_idx]:.2f}, "
          f"acc={tradeoff_data['acc'][best_dp_idx]:.4f}, dp_diff={tradeoff_data['dp_diff'][best_dp_idx]:.4f}")

    # ── Visualization ──

    # 1. Accuracy vs fairness tradeoff curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(tradeoff_data['dp_diff'], tradeoff_data['acc'], 'b-', linewidth=2)
    axes[0].scatter([dp_base[2]], [acc_base], c='red', s=100, zorder=5, label='Unmitigated')
    axes[0].scatter([dp_rw[2]], [acc_rw], c='green', s=100, zorder=5, marker='s', label='Reweighing')
    axes[0].scatter([dp_post[2]], [acc_post], c='orange', s=100, zorder=5, marker='^', label='Postprocessing')
    axes[0].set_xlabel("Demographic Parity Difference (lower = fairer)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Fairness Tradeoff")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tradeoff_data['tpr_diff'], tradeoff_data['acc'], 'b-', linewidth=2)
    axes[1].scatter([eo_base[1]], [acc_base], c='red', s=100, zorder=5, label='Unmitigated')
    axes[1].scatter([eo_rw[1]], [acc_rw], c='green', s=100, zorder=5, marker='s', label='Reweighing')
    axes[1].scatter([eo_post[1]], [acc_post], c='orange', s=100, zorder=5, marker='^', label='Postprocessing')
    axes[1].set_xlabel("Equalized Odds TPR Difference (lower = fairer)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Equalized Odds Tradeoff")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_fairness_tradeoff.png", dpi=150)
    plt.close()

    # 2. Before/after mitigation comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ['Unmitigated', 'Reweighing', 'Fair Reg\n(lambda=2)', 'Post-\nprocessing']
    accs = [acc_base, acc_rw, reg_results[2.0]['acc'], acc_post]
    dp_diffs = [dp_base[2], dp_rw[2], reg_results[2.0]['dp_diff'], dp_post[2]]
    tpr_diffs = [eo_base[1], eo_rw[1], reg_results[2.0]['tpr_diff'], eo_post[1]]

    x = np.arange(len(methods))

    axes[0].bar(x, accs, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=9)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Method")
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

    axes[1].bar(x, dp_diffs, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=9)
    axes[1].set_ylabel("DP Difference")
    axes[1].set_title("Demographic Parity Violation (lower = better)")
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(dp_diffs):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=9)

    axes[2].bar(x, tpr_diffs, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'], alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, fontsize=9)
    axes[2].set_ylabel("TPR Difference")
    axes[2].set_title("Equalized Odds Violation (lower = better)")
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(tpr_diffs):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=9)

    plt.suptitle("Fairness Metrics: Before vs After Mitigation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "mitigation_comparison.png", dpi=150)
    plt.close()

    # 3. Fairness regularization tradeoff
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lambdas = sorted(reg_results.keys())
    reg_accs = [reg_results[l]['acc'] for l in lambdas]
    reg_dps = [reg_results[l]['dp_diff'] for l in lambdas]
    reg_tprs = [reg_results[l]['tpr_diff'] for l in lambdas]

    axes[0].plot(lambdas, reg_accs, 'o-', color='#3498db', linewidth=2, label='Accuracy')
    ax2 = axes[0].twinx()
    ax2.plot(lambdas, reg_dps, 's--', color='#e74c3c', linewidth=2, label='DP diff')
    axes[0].set_xlabel("Fairness Regularization Strength (lambda)")
    axes[0].set_ylabel("Accuracy", color='#3498db')
    ax2.set_ylabel("DP Difference", color='#e74c3c')
    axes[0].set_title("Fairness Regularization: Accuracy vs DP")
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, loc='center right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(reg_dps, reg_accs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    for i, lam in enumerate(lambdas):
        axes[1].annotate(f'λ={lam}', (reg_dps[i], reg_accs[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].set_xlabel("Demographic Parity Difference")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Pareto Frontier: Accuracy vs Fairness")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "regularization_tradeoff.png", dpi=150)
    plt.close()

    # 4. Per-group score distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (probs, title) in enumerate([
        (probs_base, "Unmitigated"),
        (probs_rw, "Reweighing"),
        (probs_base, "Postprocessed (same probs)"),
    ]):
        for group in [0, 1]:
            mask = (A_val == group)
            group_probs = probs[mask].numpy()
            axes[idx].hist(group_probs, bins=30, alpha=0.5,
                           label=f'A={group}', density=True,
                           color='#3498db' if group == 0 else '#e74c3c')
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Predicted Probability")
        axes[idx].set_ylabel("Density")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

        # Mark thresholds for postprocessed model
        if idx == 2:
            for group, thresh in group_thresholds.items():
                color = '#3498db' if group == 0 else '#e74c3c'
                axes[idx].axvline(x=thresh, color=color, linestyle='--', alpha=0.8,
                                  label=f'Threshold A={group}: {thresh:.2f}')
            axes[idx].legend(fontsize=8)

    plt.suptitle("Score Distributions by Group: Mitigation Effect", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "score_distributions.png", dpi=150)
    plt.close()

    # 5. Calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_bins = 10
    for idx, (probs, title) in enumerate([
        (probs_base, "Unmitigated"),
        (probs_rw, "Reweighing"),
    ]):
        for group in [0, 1]:
            mask = (A_val == group)
            group_probs = probs[mask].numpy()
            group_labels = Y_val[mask].numpy()
            color = '#3498db' if group == 0 else '#e74c3c'

            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accs = []
            for i in range(n_bins):
                in_bin = (group_probs >= bin_boundaries[i]) & (group_probs < bin_boundaries[i + 1])
                if in_bin.sum() > 0:
                    bin_centers.append(group_probs[in_bin].mean())
                    bin_accs.append(group_labels[in_bin].mean())

            axes[idx].plot(bin_centers, bin_accs, 'o-', color=color,
                           label=f'A={group}', markersize=5)

        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        axes[idx].set_xlabel("Mean Predicted Probability")
        axes[idx].set_ylabel("Fraction of Positives")
        axes[idx].set_title(f"Calibration: {title}")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(results_dir / "calibration.png", dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    texts = [
        ("Preprocessing\nReweighing", "Adjust sample weights:\n  w(y,a) = P(Y=y)P(A=a) / P(Y=y,A=a)\n\nEqualize joint distribution\nbefore training\n→ No model change needed\n→ May reduce accuracy", 0.17, '#2ecc71'),
        ("In-processing\nFair Regularization", "Add fairness penalty:\n  L = L_BCE + λ·L_fair\n\nL_fair = corr(ŷ, A)²\nPenalize correlation\nbetween predictions\nand sensitive attribute\n→ Tunable tradeoff", 0.5, '#3498db'),
        ("Postprocessing\nThreshold Adjust", "Per-group thresholds:\n  ŷ = (p ≥ τ_a) for group a\n\nFind τ_a to equalize\npositive rates\n→ No retraining\n→ Uses same model\n→ May not fix EO", 0.83, '#f39c12'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.78, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Fairness Mitigation: Three Approaches to Reduce Algorithmic Bias",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "fairness_concept.png", dpi=150)
    plt.close()

    # ── Summary ──

    print("\n=== Summary ===")
    print(f"  {'Method':<20s} | {'Accuracy':>9s} | {'DP Diff':>9s} | {'TPR Diff':>9s}")
    print("  " + "-" * 55)
    print(f"  {'Unmitigated':<20s} | {acc_base:>9.4f} | {dp_base[2]:>9.4f} | {eo_base[1]:>9.4f}")
    print(f"  {'Reweighing':<20s} | {acc_rw:>9.4f} | {dp_rw[2]:>9.4f} | {eo_rw[1]:>9.4f}")
    print(f"  {'Fair Reg (λ=2)':<20s} | {reg_results[2.0]['acc']:>9.4f} | "
          f"{reg_results[2.0]['dp_diff']:>9.4f} | {reg_results[2.0]['tpr_diff']:>9.4f}")
    print(f"  {'Postprocessing':<20s} | {acc_post:>9.4f} | {dp_post[2]:>9.4f} | {eo_post[1]:>9.4f}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
