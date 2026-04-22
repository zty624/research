"""
Minimal Dropout & Regularization Reproduction
==============================================
Reproduces core ideas from regularization literature:
1. Dropout (1207.0580, Srivastava et al.): random unit dropout as ensemble
2. Weight Decay (L2 regularization): penalize large weights
3. Label Smoothing (1906.02629, Szegedy et al.): soften hard targets
4. Compare: no reg vs dropout vs weight decay vs label smoothing on MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Models ──

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        ]
        if dropout > 0:
            # Insert dropout after each ReLU
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, out_dim)
            )
        else:
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Training ──

def train_model(model, train_loader, test_loader, n_epochs=20, lr=1e-3,
                weight_decay=0.0, label_smoothing=0.0, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    test_accs = []
    train_accs = []

    for epoch in range(n_epochs):
        # Train
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for bx, by in train_loader:
            bx = bx.view(bx.shape[0], -1).to(device)
            by = by.to(device)

            logits = model(bx)
            loss = F.cross_entropy(logits, by, label_smoothing=label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(1) == by).sum().item()
            total += by.shape[0]

        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(correct / total)

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.view(bx.shape[0], -1).to(device)
                by = by.to(device)
                logits = model(bx)
                correct += (logits.argmax(1) == by).sum().item()
                total += by.shape[0]
        test_accs.append(correct / total)

    return train_losses, train_accs, test_accs


# ── Dropout as Ensemble ──

def dropout_as_ensemble(model, x, n_forward=20):
    """Average predictions across multiple dropout masks.
    This approximates model averaging over all possible sub-networks.
    """
    model.train()  # Enable dropout
    predictions = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs)
    predictions = torch.stack(predictions)  # (n_forward, B, C)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    return mean_pred, std_pred


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "28-dropout"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use subset for speed
    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    n_epochs = 25

    # 1. No regularization
    print("=== Training: No Regularization ===")
    mlp_none = MLP(dropout=0.0).to(device)
    none_losses, none_train, none_test = train_model(mlp_none, train_loader, test_loader, n_epochs, device=device)

    # 2. Dropout (0.3)
    print("=== Training: Dropout (p=0.3) ===")
    mlp_drop = MLP(dropout=0.3).to(device)
    drop_losses, drop_train, drop_test = train_model(mlp_drop, train_loader, test_loader, n_epochs, device=device)

    # 3. Weight Decay (L2)
    print("=== Training: Weight Decay (λ=0.01) ===")
    mlp_wd = MLP(dropout=0.0).to(device)
    wd_losses, wd_train, wd_test = train_model(mlp_wd, train_loader, test_loader, n_epochs, weight_decay=0.01, device=device)

    # 4. Label Smoothing (ε=0.1)
    print("=== Training: Label Smoothing (ε=0.1) ===")
    mlp_ls = MLP(dropout=0.0).to(device)
    ls_losses, ls_train, ls_test = train_model(mlp_ls, train_loader, test_loader, n_epochs, label_smoothing=0.1, device=device)

    # 5. Combined (dropout + WD + LS)
    print("=== Training: Combined (dropout + WD + LS) ===")
    mlp_combo = MLP(dropout=0.3).to(device)
    combo_losses, combo_train, combo_test = train_model(
        mlp_combo, train_loader, test_loader, n_epochs,
        weight_decay=0.01, label_smoothing=0.1, device=device
    )

    # ── Dropout as Ensemble Analysis ──
    print("\n=== Dropout as Ensemble Analysis ===")
    test_batch = next(iter(test_loader))
    test_x = test_batch[0][:100].view(100, -1).to(device)
    test_y = test_batch[1][:100].to(device)

    # Without dropout ensemble (deterministic)
    mlp_drop.eval()
    with torch.no_grad():
        logits_det = mlp_drop(test_x)
        acc_det = (logits_det.argmax(1) == test_y).float().mean().item()

    # With dropout ensemble
    mean_pred, std_pred = dropout_as_ensemble(mlp_drop, test_x, n_forward=30)
    acc_ensemble = (mean_pred.argmax(1) == test_y).float().mean().item()

    # Confidence (max probability)
    conf_det = F.softmax(logits_det, dim=-1).max(dim=-1)[0].mean().item()
    conf_ensemble = mean_pred.max(dim=-1)[0].mean().item()

    print(f"  Deterministic accuracy: {acc_det:.3f}, confidence: {conf_det:.3f}")
    print(f"  Ensemble accuracy:      {acc_ensemble:.3f}, confidence: {conf_ensemble:.3f}")
    print(f"  Uncertainty (avg std):  {std_pred.mean().item():.4f}")

    # ── Visualization ──

    # 1. Training loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(none_losses, label='No Reg', color='gray')
    axes[0].plot(drop_losses, label='Dropout', color='blue')
    axes[0].plot(wd_losses, label='Weight Decay', color='orange')
    axes[0].plot(ls_losses, label='Label Smooth', color='green')
    axes[0].plot(combo_losses, label='Combined', color='red')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Train accuracy
    axes[1].plot(none_train, label='No Reg', color='gray')
    axes[1].plot(drop_train, label='Dropout', color='blue')
    axes[1].plot(wd_train, label='Weight Decay', color='orange')
    axes[1].plot(ls_train, label='Label Smooth', color='green')
    axes[1].plot(combo_train, label='Combined', color='red')
    axes[1].set_title("Train Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Test accuracy
    axes[2].plot(none_test, label='No Reg', color='gray')
    axes[2].plot(drop_test, label='Dropout', color='blue')
    axes[2].plot(wd_test, label='Weight Decay', color='orange')
    axes[2].plot(ls_test, label='Label Smooth', color='green')
    axes[2].plot(combo_test, label='Combined', color='red')
    axes[2].set_title("Test Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Regularization Methods: Training Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Final accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ['No Reg', 'Dropout\n(p=0.3)', 'Weight\nDecay', 'Label\nSmooth', 'Combined']
    final_test = [none_test[-1], drop_test[-1], wd_test[-1], ls_test[-1], combo_test[-1]]
    final_train = [none_train[-1], drop_train[-1], wd_train[-1], ls_train[-1], combo_train[-1]]

    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, final_train, width, label='Train', alpha=0.7, color='blue')
    ax.bar(x + width/2, final_test, width, label='Test', alpha=0.7, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy")
    ax.set_title("Regularization: Train vs Test Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.85, 1.0)
    for i, (tr, te) in enumerate(zip(final_train, final_test)):
        ax.text(i - width/2, tr + 0.002, f'{tr:.3f}', ha='center', fontsize=8)
        ax.text(i + width/2, te + 0.002, f'{te:.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "final_accuracy.png", dpi=150)
    plt.close()

    # 3. Dropout ensemble analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confidence distribution
    with torch.no_grad():
        mlp_drop.eval()
        conf_no_drop = F.softmax(mlp_drop(test_x), dim=-1).max(dim=-1)[0].cpu().numpy()
    conf_with_drop = mean_pred.max(dim=-1)[0].cpu().numpy()

    axes[0].hist(conf_no_drop, bins=20, alpha=0.5, label='Deterministic', color='blue')
    axes[0].hist(conf_with_drop, bins=20, alpha=0.5, label='Ensemble (MC Dropout)', color='green')
    axes[0].set_title("Prediction Confidence Distribution")
    axes[0].set_xlabel("Max Probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Uncertainty for correct vs incorrect predictions
    uncertainties = std_pred.mean(dim=-1).cpu().numpy()
    correct_mask = (mean_pred.argmax(1) == test_y).cpu().numpy()

    axes[1].hist(uncertainties[correct_mask], bins=20, alpha=0.5, label='Correct', color='green')
    axes[1].hist(uncertainties[~correct_mask], bins=20, alpha=0.5, label='Incorrect', color='red')
    axes[1].set_title("MC Dropout Uncertainty")
    axes[1].set_xlabel("Avg Prediction Std")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Label smoothing effect
    with torch.no_grad():
        # Hard target distribution
        hard_target = F.one_hot(test_y[:5], 10).float()
        # Soft target
        soft_target = hard_target * (1 - 0.1) + 0.1 / 10

    for i in range(3):
        ax = axes[2] if i == 0 else None
    axes[2].bar(np.arange(10) - 0.15, hard_target[0].cpu().numpy(), 0.3, label='Hard label', color='blue', alpha=0.7)
    axes[2].bar(np.arange(10) + 0.15, soft_target[0].cpu().numpy(), 0.3, label='Smooth label (ε=0.1)', color='green', alpha=0.7)
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Probability")
    axes[2].set_title("Label Smoothing Effect")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Regularization: Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "regularization_analysis.png", dpi=150)
    plt.close()

    # 4. Overfitting gap
    fig, ax = plt.subplots(figsize=(8, 5))
    gap_none = [t - e for t, e in zip(none_train, none_test)]
    gap_drop = [t - e for t, e in zip(drop_train, drop_test)]
    gap_wd = [t - e for t, e in zip(wd_train, wd_test)]
    gap_combo = [t - e for t, e in zip(combo_train, combo_test)]

    ax.plot(gap_none, label='No Reg', color='gray')
    ax.plot(gap_drop, label='Dropout', color='blue')
    ax.plot(gap_wd, label='Weight Decay', color='orange')
    ax.plot(gap_combo, label='Combined', color='red')
    ax.set_title("Generalization Gap (Train - Test Accuracy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "generalization_gap.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Dropout", "Randomly zero units\n→ Implicit ensemble\n→ Reduces co-adaptation", 0.13, 'blue'),
        ("Weight Decay", "Penalize ||W||²\n→ Smoother function\n→ L2 regularization", 0.38, 'orange'),
        ("Label Smooth", "y_k = 1-ε+ε/K\n→ Prevent overconf.\n→ Better calibration", 0.62, 'green'),
        ("Combined", "Stack all three\n→ Best generalization\n→ Industry standard", 0.87, 'red'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Regularization: Preventing Overfitting", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "regularization_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
