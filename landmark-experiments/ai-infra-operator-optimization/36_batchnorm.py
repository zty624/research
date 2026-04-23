"""
Minimal Batch Normalization Reproduction
=========================================
Reproduces core ideas from Batch Normalization (1502.03167, Ioffe & Szegedy):
1. Internal Covariate Shift: distribution of layer inputs changes during training
2. Normalize activations: zero mean, unit variance per mini-batch
3. Learnable scale (γ) and shift (β) to recover representational power
4. Compare: with/without BN on deep networks at various learning rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Custom BatchNorm ──

class CustomBatchNorm(nn.Module):
    """Manual implementation of Batch Normalization."""
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Normalize
            x_norm = (x - mean) / torch.sqrt(var + self.eps)

            # Scale and shift
            out = self.gamma * x_norm + self.beta

            # Update running stats
            self.num_batches_tracked += 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta

        return out


# ── Models ──

class MLPNoBN(nn.Module):
    """Deep MLP without Batch Normalization."""
    def __init__(self, in_dim=784, hidden=256, n_layers=8, n_classes=10):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.classifier(x)


class MLPWithBN(nn.Module):
    """Deep MLP with Batch Normalization."""
    def __init__(self, in_dim=784, hidden=256, n_layers=8, n_classes=10):
        super().__init__()
        layers = []
        bns = []
        layers.append(nn.Linear(in_dim, hidden))
        bns.append(nn.BatchNorm1d(hidden))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            bns.append(nn.BatchNorm1d(hidden))
        self.layers = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        for layer, bn in zip(self.layers, self.bns):
            x = F.relu(bn(layer(x)))
        return self.classifier(x)


class MLPWithCustomBN(nn.Module):
    """Deep MLP with our custom Batch Normalization."""
    def __init__(self, in_dim=784, hidden=256, n_layers=8, n_classes=10):
        super().__init__()
        layers = []
        bns = []
        layers.append(nn.Linear(in_dim, hidden))
        bns.append(CustomBatchNorm(hidden))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            bns.append(CustomBatchNorm(hidden))
        self.layers = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        for layer, bn in zip(self.layers, self.bns):
            x = F.relu(bn(layer(x)))
        return self.classifier(x)


# ── Training ──

def train_model(model, train_loader, test_loader, n_epochs=20, lr=1e-2, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_losses = []
    test_accs = []
    activation_stats = []  # track activation distribution at layer 4

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        test_accs.append(correct / total)

        # Record activation stats at layer 4
        model.eval()
        with torch.no_grad():
            sample = next(iter(test_loader))[0][:64].to(device)
            x = sample.flatten(1)
            for i, (layer, bn) in enumerate(zip(model.layers, model.bns)):
                x = layer(x)
                if i == 3:  # 4th layer
                    activation_stats.append({
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'min': x.min().item(),
                        'max': x.max().item(),
                    })
                x = bn(x) if hasattr(model, 'bns') else x
                x = F.relu(x)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {test_accs[-1]:.4f}")

    return train_losses, test_accs, activation_stats


def train_no_bn(model, train_loader, test_loader, n_epochs=20, lr=1e-2, device='cpu'):
    """Train model without BN (separate function to avoid BN-specific code)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_losses = []
    test_accs = []
    activation_stats = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        test_accs.append(correct / total)

        # Record activation stats at layer 4
        model.eval()
        with torch.no_grad():
            sample = next(iter(test_loader))[0][:64].to(device)
            x = sample.flatten(1)
            for i, layer in enumerate(model.layers):
                x = layer(x)
                if i == 3:
                    activation_stats.append({
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'min': x.min().item(),
                        'max': x.max().item(),
                    })
                x = F.relu(x)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {test_accs[-1]:.4f}")

    return train_losses, test_accs, activation_stats


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "36-batchnorm"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(10000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Experiment 1: BN vs No BN at various learning rates
    print("=== Experiment 1: BN vs No BN at Various Learning Rates ===")
    lrs = [1e-4, 1e-3, 1e-2, 5e-2]
    results_lr = {}

    for lr in lrs:
        print(f"\n  LR = {lr}")

        print("    No BN...")
        no_bn = MLPNoBN(hidden=256, n_layers=8).to(device)
        no_bn_losses, no_bn_accs, no_bn_stats = train_no_bn(
            no_bn, train_loader, test_loader, n_epochs=20, lr=lr, device=device)

        print("    With BN...")
        with_bn = MLPWithBN(hidden=256, n_layers=8).to(device)
        bn_losses, bn_accs, bn_stats = train_model(
            with_bn, train_loader, test_loader, n_epochs=20, lr=lr, device=device)

        results_lr[lr] = {
            'no_bn': {'losses': no_bn_losses, 'accs': no_bn_accs, 'stats': no_bn_stats},
            'bn': {'losses': bn_losses, 'accs': bn_accs, 'stats': bn_stats},
        }

    # Experiment 2: Custom BN validation
    print("\n=== Experiment 2: Custom BN vs PyTorch BN ===")
    custom_bn = MLPWithCustomBN(hidden=256, n_layers=8).to(device)
    custom_losses, custom_accs, _ = train_model(
        custom_bn, train_loader, test_loader, n_epochs=20, lr=1e-2, device=device)

    pytorch_bn = MLPWithBN(hidden=256, n_layers=8).to(device)
    pytorch_losses, pytorch_accs, _ = train_model(
        pytorch_bn, train_loader, test_loader, n_epochs=20, lr=1e-2, device=device)

    print(f"  Custom BN accuracy: {custom_accs[-1]:.4f}")
    print(f"  PyTorch BN accuracy: {pytorch_accs[-1]:.4f}")

    # ── Final results ──
    print("\n=== Final Accuracy by Learning Rate ===")
    for lr in lrs:
        no_bn_acc = results_lr[lr]['no_bn']['accs'][-1]
        bn_acc = results_lr[lr]['bn']['accs'][-1]
        print(f"  LR={lr}: No BN={no_bn_acc:.4f}, BN={bn_acc:.4f}, diff={bn_acc - no_bn_acc:+.4f}")

    # ── Visualization ──

    # 1. Learning rate sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, lr in enumerate(lrs):
        ax = axes[idx // 2][idx % 2]
        r = results_lr[lr]
        ax.plot(r['no_bn']['losses'], label='No BN', color='red')
        ax.plot(r['bn']['losses'], label='With BN', color='blue')
        ax.set_title(f"LR = {lr}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Batch Normalization: Enables Higher Learning Rates", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "lr_sensitivity.png", dpi=150)
    plt.close()

    # 2. Accuracy comparison across LRs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    no_bn_accs = [results_lr[lr]['no_bn']['accs'][-1] for lr in lrs]
    bn_accs = [results_lr[lr]['bn']['accs'][-1] for lr in lrs]

    x = np.arange(len(lrs))
    width = 0.35
    axes[0].bar(x - width/2, no_bn_accs, width, label='No BN', color='red', alpha=0.7)
    axes[0].bar(x + width/2, bn_accs, width, label='With BN', color='blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{lr:.0e}' for lr in lrs])
    axes[0].set_ylabel("Final Test Accuracy")
    axes[0].set_title("Accuracy vs Learning Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 3. Activation distribution (Internal Covariate Shift)
    # Show how activation stats change over training
    lr = 1e-3
    no_bn_means = [s['mean'] for s in results_lr[lr]['no_bn']['stats']]
    bn_means = [s['mean'] for s in results_lr[lr]['bn']['stats']]
    no_bn_stds = [s['std'] for s in results_lr[lr]['no_bn']['stats']]
    bn_stds = [s['std'] for s in results_lr[lr]['bn']['stats']]

    epochs = range(1, len(no_bn_means) + 1)
    axes[1].plot(epochs, no_bn_means, label='No BN mean', color='red', linestyle='--')
    axes[1].plot(epochs, bn_means, label='BN mean', color='blue', linestyle='--')
    axes[1].plot(epochs, no_bn_stds, label='No BN std', color='red')
    axes[1].plot(epochs, bn_stds, label='BN std', color='blue')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Activation Value (Layer 4, pre-BN)")
    axes[1].set_title("Internal Covariate Shift (Layer 4)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Batch Normalization: Stabilizes Activation Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "activation_analysis.png", dpi=150)
    plt.close()

    # 4. Custom vs PyTorch BN
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(custom_losses, label='Custom BN', color='green', linewidth=2)
    ax.plot(pytorch_losses, label='PyTorch BN', color='blue', linewidth=2, linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Custom BN vs PyTorch BN Implementation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "custom_vs_pytorch_bn.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Internal\nCovariate Shift", "Layer input distribution\nchanges as params update\n→ Slow convergence\n→ Careful init needed", 0.14, 'red'),
        ("Batch\nNormalization", "Normalize: x̂ = (x-μ)/σ\nScale: y = γx̂ + β\nPer mini-batch stats\n→ Stable distributions", 0.5, 'blue'),
        ("Benefits", "Higher learning rates\nFaster convergence\nLess sensitive to init\nRegularization effect\n(smooths loss landscape)", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Batch Normalization: Accelerating Deep Network Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "batchnorm_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
