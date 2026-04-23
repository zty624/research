"""
Minimal Red Teaming / Adversarial Attack Reproduction
======================================================
Reproduces core ideas from AI safety red-teaming literature:
1. PGD (Projected Gradient Descent) adversarial attack on input embeddings
   (Madry et al. 1706.06083: adversarial training as saddle-point problem)
2. GCG-style token optimization attack
   (Zou et al. 2307.15043: Greedy Coordinate Gradient for universal jailbreaks)
3. Adversarial training as defense
4. Key insight: small perturbations can flip classifier decisions;
   adversarial training improves robustness but reduces clean accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Data ──

def generate_harmful_benign_data(n_samples=2000, embed_dim=64, seed=42):
    """
    Synthetic harmful/benign input embeddings.

    Harmful embeddings cluster around a "danger" direction,
    benign embeddings are spread in orthogonal directions.
    """
    rng = np.random.RandomState(seed)

    n_harmful = n_samples // 2
    n_benign = n_samples - n_harmful

    # Harmful: tight cluster along danger direction + small noise
    danger_dir = rng.randn(embed_dim)
    danger_dir /= np.linalg.norm(danger_dir)
    harmful = np.outer(np.ones(n_harmful), danger_dir * 2.0) + rng.randn(n_harmful, embed_dim) * 0.3

    # Benign: spread in orthogonal complement + small component along danger dir
    orth_basis = rng.randn(embed_dim, embed_dim)
    orth_basis -= np.outer(orth_basis @ danger_dir, danger_dir)
    benign = orth_basis[:n_benign] * 1.5 + rng.randn(n_benign, embed_dim) * 0.2

    X = np.vstack([harmful, benign]).astype(np.float32)
    y = np.concatenate([np.ones(n_harmful), np.zeros(n_benign)]).astype(np.int64)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Split
    n_train = int(0.8 * len(y))
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


# ── Classifier ──

class HarmfulDetector(nn.Module):
    """Simple MLP classifier: harmful (1) vs benign (0)."""
    def __init__(self, embed_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── PGD Attack ──

def pgd_attack(model, x, y, eps=0.5, alpha=0.05, n_steps=50, targeted=False, target_class=0):
    """
    Projected Gradient Descent adversarial attack.

    Untargeted: maximize loss for true class
    Targeted: minimize loss for target class (make harmful look benign)
    """
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        logits = model(x_adv)
        if targeted:
            # Minimize loss for target class
            loss = -F.cross_entropy(logits, torch.full_like(y, target_class))
        else:
            # Maximize loss for true class
            loss = F.cross_entropy(logits, y)

        loss.backward()
        grad = x_adv.grad.data

        # Update: ascend gradient (untargeted) or descend (targeted)
        if targeted:
            x_adv = x_adv - alpha * grad.sign()
        else:
            x_adv = x_adv + alpha * grad.sign()

        # Project back to eps-ball around original
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, -10.0, 10.0).detach().requires_grad_(True)

    return x_adv.detach()


# ── GCG-style Token Optimization ──

def gcg_attack(model, x_harmful, n_iter=100, top_k=10, replacement_width=5, lr_scale=0.1):
    """
    Simplified GCG-style attack: optimize continuous embeddings via
    greedy coordinate-wise gradient to flip harmful→benign.

    In real GCG, discrete tokens are optimized. Here we approximate by
    optimizing small perturbation vectors with greedy coordinate selection.
    """
    x_adv = x_harmful.clone().detach()
    B, D = x_adv.shape
    perturbation = torch.zeros_like(x_adv, requires_grad=True)

    best_x = x_adv.clone()
    best_loss = float('inf')
    attack_success = torch.zeros(B, dtype=torch.bool)

    for it in range(n_iter):
        perturbation.requires_grad_(True)
        x_perturbed = x_adv + perturbation

        logits = model(x_perturbed)
        # Target: minimize probability of harmful class (class 1)
        target_labels = torch.zeros(B, dtype=torch.long, device=x_adv.device)
        loss = F.cross_entropy(logits, target_labels)

        loss.backward()
        grad = perturbation.grad.data

        with torch.no_grad():
            # Greedy coordinate selection: pick top_k dimensions with largest gradient
            for b in range(B):
                if attack_success[b]:
                    continue

                top_dims = grad[b].abs().topk(top_k).indices
                step = torch.zeros(D, device=x_adv.device)
                step[top_dims] = -lr_scale * grad[b, top_dims].sign()
                perturbation[b] += step

            x_perturbed = x_adv + perturbation
            logits_new = model(x_perturbed)
            preds = logits_new.argmax(dim=1)

            # Track per-sample success
            for b in range(B):
                if preds[b] == 0 and not attack_success[b]:
                    attack_success[b] = True
                    best_x[b] = x_perturbed[b].clone()

            current_loss = F.cross_entropy(logits_new, target_labels).item()
            if current_loss < best_loss:
                best_loss = current_loss

        perturbation = perturbation.detach()

        if (it + 1) % 20 == 0:
            rate = attack_success.float().mean().item()
            print(f"    GCG iter {it+1}/{n_iter} | ASR: {rate:.2%} | Loss: {best_loss:.4f}")

    return best_x, attack_success


# ── Adversarial Training ──

def train_standard(model, X_train, y_train, n_epochs=30, lr=1e-3, batch_size=128, device='cpu'):
    """Standard training (no adversarial defense)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    X_t = torch.tensor(X_train, device=device)
    y_t = torch.tensor(y_train, device=device)
    N = len(y_t)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X_t[idx], y_t[idx]

            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_t).argmax(1)
                acc = (preds == y_t).float().mean().item()
            print(f"    Epoch {epoch+1} | Loss: {epoch_loss/n_batches:.4f} | Acc: {acc:.4f}")


def train_adversarial(model, X_train, y_train, n_epochs=30, lr=1e-3, batch_size=128,
                      device='cpu', eps=0.3, alpha=0.03, pgd_steps=10):
    """Adversarial training: mix clean and adversarial examples."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    X_t = torch.tensor(X_train, device=device)
    y_t = torch.tensor(y_train, device=device)
    N = len(y_t)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X_t[idx], y_t[idx]

            # Generate adversarial examples
            bx_adv = pgd_attack(model, bx, by, eps=eps, alpha=alpha, n_steps=pgd_steps, targeted=True, target_class=0)

            # Train on both clean and adversarial
            x_mix = torch.cat([bx, bx_adv])
            y_mix = torch.cat([by, by])  # Adversarial examples retain true labels

            logits = model(x_mix)
            loss = F.cross_entropy(logits, y_mix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_t).argmax(1)
                acc = (preds == y_t).float().mean().item()
            print(f"    Epoch {epoch+1} | Loss: {epoch_loss/n_batches:.4f} | Acc: {acc:.4f}")


def evaluate(model, X, y, device='cpu'):
    """Evaluate accuracy."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, device=device)
        y_t = torch.tensor(y, device=device)
        preds = model(X_t).argmax(1)
        acc = (preds == y_t).float().mean().item()
    return acc


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "89-red-team"
    results_dir.mkdir(parents=True, exist_ok=True)

    embed_dim = 64
    (X_train, y_train), (X_test, y_test) = generate_harmful_benign_data(
        n_samples=2000, embed_dim=embed_dim)

    # Split harmful test samples for attack evaluation
    harmful_mask = y_test == 1
    X_test_harmful = X_test[harmful_mask]
    y_test_harmful = y_test[harmful_mask]

    # ── Experiment 1: Train standard classifier ──
    print("=== Training Standard Classifier ===")
    model_std = HarmfulDetector(embed_dim=embed_dim, hidden=128).to(device)
    train_standard(model_std, X_train, y_train, n_epochs=30, device=device)

    clean_acc_std = evaluate(model_std, X_test, y_test, device)
    harmful_acc_std = evaluate(model_std, X_test_harmful, y_test_harmful, device)
    print(f"  Standard model — Clean acc: {clean_acc_std:.4f}, Harmful detection: {harmful_acc_std:.4f}")

    # ── Experiment 2: PGD Attack ──
    print("\n=== PGD Attack on Standard Model ===")
    X_h_t = torch.tensor(X_test_harmful, device=device)
    y_h_t = torch.tensor(y_test_harmful, device=device)

    pgd_epsilons = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    pgd_asr = []  # attack success rate: fraction of harmful → benign
    pgd_perturb = []

    for eps in pgd_epsilons:
        X_adv = pgd_attack(model_std, X_h_t, y_h_t, eps=eps, alpha=eps/10, n_steps=50,
                           targeted=True, target_class=0)
        with torch.no_grad():
            preds = model_std(X_adv).argmax(1)
        # Attack success: model now predicts benign (0) for harmful input
        asr = (preds == 0).float().mean().item()
        avg_perturb = (X_adv - X_h_t).norm(dim=1).mean().item()
        pgd_asr.append(asr)
        pgd_perturb.append(avg_perturb)
        print(f"  eps={eps:.1f} | ASR: {asr:.2%} | Avg perturbation: {avg_perturb:.4f}")

    # ── Experiment 3: GCG-style Attack ──
    print("\n=== GCG-style Token Optimization Attack ===")
    X_h_sub = X_h_t[:50]  # Subset for speed
    y_h_sub = y_h_t[:50]
    X_gcg_adv, gcg_success = gcg_attack(model_std, X_h_sub, n_iter=80, top_k=8,
                                         replacement_width=5, lr_scale=0.15, )
    gcg_asr = gcg_success.float().mean().item()
    gcg_avg_perturb = (X_gcg_adv - X_h_sub).norm(dim=1).mean().item()
    print(f"  GCG ASR: {gcg_asr:.2%} | Avg perturbation: {gcg_avg_perturb:.4f}")

    # ── Experiment 4: Adversarial Training as Defense ──
    print("\n=== Adversarial Training Defense ===")
    model_adv = HarmfulDetector(embed_dim=embed_dim, hidden=128).to(device)
    train_adversarial(model_adv, X_train, y_train, n_epochs=30, device=device,
                      eps=0.3, alpha=0.03, pgd_steps=10)

    clean_acc_adv = evaluate(model_adv, X_test, y_test, device)
    harmful_acc_adv = evaluate(model_adv, X_test_harmful, y_test_harmful, device)
    print(f"  Adv-trained model — Clean acc: {clean_acc_adv:.4f}, Harmful detection: {harmful_acc_adv:.4f}")

    # Re-evaluate PGD on adversarially trained model
    print("\n=== PGD Attack on Adv-Trained Model ===")
    pgd_asr_defended = []
    for eps in pgd_epsilons:
        X_adv = pgd_attack(model_adv, X_h_t, y_h_t, eps=eps, alpha=eps/10, n_steps=50,
                           targeted=True, target_class=0)
        with torch.no_grad():
            preds = model_adv(X_adv).argmax(1)
        asr = (preds == 0).float().mean().item()
        pgd_asr_defended.append(asr)
        print(f"  eps={eps:.1f} | ASR: {asr:.2%}")

    # Re-evaluate GCG on adversarially trained model
    print("\n=== GCG Attack on Adv-Trained Model ===")
    X_gcg_adv_def, gcg_success_def = gcg_attack(model_adv, X_h_sub, n_iter=80, top_k=8,
                                                  replacement_width=5, lr_scale=0.15)
    gcg_asr_def = gcg_success_def.float().mean().item()
    print(f"  GCG ASR on adv-trained: {gcg_asr_def:.2%}")

    # ── Visualization ──

    # 1. PGD Attack Success Rate vs Epsilon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(pgd_epsilons, [r * 100 for r in pgd_asr], 'o-', color='red',
                 label='Standard model', linewidth=2)
    axes[0].plot(pgd_epsilons, [r * 100 for r in pgd_asr_defended], 's--', color='green',
                 label='Adv-trained model', linewidth=2)
    axes[0].set_xlabel("Perturbation Budget (epsilon)")
    axes[0].set_ylabel("Attack Success Rate (%)")
    axes[0].set_title("PGD Attack: Harmful → Benign Flip Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 105)

    # 2. Perturbation magnitude vs ASR trade-off
    axes[1].plot(pgd_perturb, [r * 100 for r in pgd_asr], 'o-', color='red',
                 label='Standard model')
    axes[1].axhline(y=gcg_asr * 100, color='blue', linestyle='--',
                    label=f'GCG (std) ASR={gcg_asr:.0%}')
    axes[1].axhline(y=gcg_asr_def * 100, color='green', linestyle='--',
                    label=f'GCG (adv) ASR={gcg_asr_def:.0%}')
    axes[1].set_xlabel("Avg Perturbation Magnitude")
    axes[1].set_ylabel("Attack Success Rate (%)")
    axes[1].set_title("Perturbation vs Attack Success")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-5, 105)

    plt.suptitle("Red Teaming: Adversarial Attacks on Harmful Content Detector", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "attack_success_rate.png", dpi=150)
    plt.close()

    # 2. Defense comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    categories = ['Clean Acc', 'Harmful\nDetection', 'PGD\n(eps=0.3)', 'PGD\n(eps=0.5)', 'GCG']
    std_vals = [clean_acc_std * 100, harmful_acc_std * 100,
                (1 - pgd_asr[2]) * 100, (1 - pgd_asr[4]) * 100,
                (1 - gcg_asr) * 100]
    adv_vals = [clean_acc_adv * 100, harmful_acc_adv * 100,
                (1 - pgd_asr_defended[2]) * 100, (1 - pgd_asr_defended[4]) * 100,
                (1 - gcg_asr_def) * 100]

    x = np.arange(len(categories))
    w = 0.35
    axes[0].bar(x - w/2, std_vals, w, label='Standard', color='salmon', edgecolor='red')
    axes[0].bar(x + w/2, adv_vals, w, label='Adv-Trained', color='lightgreen', edgecolor='green')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=9)
    axes[0].set_ylabel("Robust Accuracy (%)")
    axes[0].set_title("Standard vs Adversarial Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 110)

    # Accuracy drop from adversarial training
    metrics = ['Clean', 'PGD eps=0.3', 'PGD eps=0.5', 'GCG']
    drops = [
        (clean_acc_std - clean_acc_adv) * 100,
        (pgd_asr[2] - pgd_asr_defended[2]) * 100,
        (pgd_asr[4] - pgd_asr_defended[4]) * 100,
        (gcg_asr - gcg_asr_def) * 100,
    ]
    colors = ['red' if d < 0 else 'green' for d in drops]
    axes[1].bar(metrics, drops, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel("Change (percentage points)")
    axes[1].set_title("Effect of Adversarial Training\n(green=improved, red=degraded)")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Red Teaming: Attack & Defense Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "defense_comparison.png", dpi=150)
    plt.close()

    # 3. Embedding space visualization (2D projection)
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Project to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_test)

    harmful_mask_test = y_test == 1
    benign_mask_test = y_test == 0

    # Original
    axes[0].scatter(X_2d[benign_mask_test, 0], X_2d[benign_mask_test, 1],
                    c='blue', alpha=0.3, s=10, label='Benign')
    axes[0].scatter(X_2d[harmful_mask_test, 0], X_2d[harmful_mask_test, 1],
                    c='red', alpha=0.3, s=10, label='Harmful')
    axes[0].set_title("Original Embeddings")
    axes[0].legend(fontsize=8)

    # PGD adversarial (eps=0.5)
    X_adv_05 = pgd_attack(model_std, X_h_t, y_h_t, eps=0.5, alpha=0.05, n_steps=50,
                           targeted=True, target_class=0)
    X_adv_2d = pca.transform(X_adv_05.cpu().numpy())
    axes[1].scatter(X_2d[benign_mask_test, 0], X_2d[benign_mask_test, 1],
                    c='blue', alpha=0.2, s=10, label='Benign')
    axes[1].scatter(X_2d[harmful_mask_test, 0], X_2d[harmful_mask_test, 1],
                    c='red', alpha=0.1, s=10, label='Harmful (orig)')
    axes[1].scatter(X_adv_2d[:, 0], X_adv_2d[:, 1],
                    c='orange', alpha=0.5, s=15, marker='x', label='Harmful (PGD)')
    axes[1].set_title("PGD Adversarial Examples (eps=0.5)")
    axes[1].legend(fontsize=8)

    # GCG adversarial
    X_gcg_2d = pca.transform(X_gcg_adv.cpu().numpy())
    axes[2].scatter(X_2d[benign_mask_test, 0], X_2d[benign_mask_test, 1],
                    c='blue', alpha=0.2, s=10, label='Benign')
    axes[2].scatter(X_2d[harmful_mask_test, 0], X_2d[harmful_mask_test, 1],
                    c='red', alpha=0.1, s=10, label='Harmful (orig)')
    axes[2].scatter(X_gcg_2d[:, 0], X_gcg_2d[:, 1],
                    c='purple', alpha=0.5, s=15, marker='x', label='Harmful (GCG)')
    axes[2].set_title("GCG Adversarial Examples")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Red Teaming: Embedding Space View of Adversarial Attacks", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "embedding_space.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("PGD Attack\n(1706.06083)", "Gradient-based\nIterative perturbation\nProject to eps-ball\nTargeted: flip\nharmful→benign", 0.14, 'red'),
        ("GCG Attack\n(2307.15043)", "Greedy Coordinate\nGradient optimization\nTop-k dimensions\nper iteration\nToken-level search", 0.5, 'purple'),
        ("Adversarial\nTraining", "Train on adversarial\nexamples (mix clean+adv)\nImproves robustness\nTrade-off: lower\nclean accuracy", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Red Teaming: Attack Methods and Adversarial Defense", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "red_team_concept.png", dpi=150)
    plt.close()

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"  Standard model clean acc:        {clean_acc_std:.4f}")
    print(f"  Adv-trained model clean acc:     {clean_acc_adv:.4f}")
    print(f"  Clean accuracy drop:             {(clean_acc_std - clean_acc_adv)*100:.1f}pp")
    print(f"  PGD (eps=0.5) ASR on standard:   {pgd_asr[4]:.2%}")
    print(f"  PGD (eps=0.5) ASR on adv-trained: {pgd_asr_defended[4]:.2%}")
    print(f"  GCG ASR on standard:             {gcg_asr:.2%}")
    print(f"  GCG ASR on adv-trained:          {gcg_asr_def:.2%}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
