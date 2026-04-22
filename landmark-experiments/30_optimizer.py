"""
Minimal Optimizer Comparison Reproduction
=========================================
Reproduces core ideas from optimizer literature:
1. SGD with momentum (classic)
2. Adam (1412.6980, Kingma & Ba): adaptive learning rates
3. AdamW (1711.05101, Loshchilov): decoupled weight decay
4. Compare: SGD vs Adam vs AdamW with LR schedules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def train_with_optimizer(model, train_loader, test_loader, optimizer_fn,
                         n_epochs=20, device='cpu', scheduler_fn=None):
    optimizer = optimizer_fn(model.parameters())
    scheduler = scheduler_fn(optimizer) if scheduler_fn else None

    train_losses = []
    test_accs = []
    lrs = []

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

        if scheduler:
            scheduler.step()

        train_losses.append(epoch_loss / len(train_loader))
        lrs.append(optimizer.param_groups[0]['lr'])

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(1)
                correct += (preds == by).sum().item()
                total += by.shape[0]
        test_accs.append(correct / total)

    return train_losses, test_accs, lrs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "30-optimizer"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    n_epochs = 25

    # 1. SGD (no momentum)
    print("=== SGD (no momentum) ===")
    sgd_losses, sgd_accs, sgd_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.SGD(p, lr=0.1),
        n_epochs, device
    )

    # 2. SGD + Momentum
    print("=== SGD + Momentum ===")
    sgd_m_losses, sgd_m_accs, sgd_m_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
        n_epochs, device
    )

    # 3. SGD + Momentum + Cosine LR
    print("=== SGD + Momentum + Cosine LR ===")
    sgd_cos_losses, sgd_cos_accs, sgd_cos_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
        n_epochs, device,
        scheduler_fn=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    )

    # 4. Adam
    print("=== Adam ===")
    adam_losses, adam_accs, adam_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.Adam(p, lr=1e-3),
        n_epochs, device
    )

    # 5. AdamW
    print("=== AdamW ===")
    adamw_losses, adamw_accs, adamw_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01),
        n_epochs, device
    )

    # 6. AdamW + Cosine LR
    print("=== AdamW + Cosine LR ===")
    adamw_cos_losses, adamw_cos_accs, adamw_cos_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01),
        n_epochs, device,
        scheduler_fn=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    )

    # 7. AdamW + Warmup + Cosine
    print("=== AdamW + Warmup Cosine ===")
    adamw_warm_losses, adamw_warm_accs, adamw_warm_lrs = train_with_optimizer(
        CNN().to(device), train_loader, test_loader,
        lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01),
        n_epochs, device,
        scheduler_fn=lambda opt: torch.optim.lr_scheduler.SequentialLR(
            opt, [
                torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1e-2, 5),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs - 5)
            ], [5]
        )
    )

    # ── Visualization ──

    optimizers = {
        'SGD': (sgd_losses, sgd_accs, sgd_lrs, 'gray'),
        'SGD+M': (sgd_m_losses, sgd_m_accs, sgd_m_lrs, 'blue'),
        'SGD+M+Cos': (sgd_cos_losses, sgd_cos_accs, sgd_cos_lrs, 'cyan'),
        'Adam': (adam_losses, adam_accs, adam_lrs, 'orange'),
        'AdamW': (adamw_losses, adamw_accs, adamw_lrs, 'red'),
        'AdamW+Cos': (adamw_cos_losses, adamw_cos_accs, adamw_cos_lrs, 'green'),
        'AdamW+WarmCos': (adamw_warm_losses, adamw_warm_accs, adamw_warm_lrs, 'purple'),
    }

    # 1. Training loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, (losses, accs, lrs, color) in optimizers.items():
        axes[0].plot(losses, label=name, color=color)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. Test accuracy
    for name, (losses, accs, lrs, color) in optimizers.items():
        axes[1].plot(accs, label=name, color=color)
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Learning rate schedules
    for name, (losses, accs, lrs, color) in optimizers.items():
        axes[2].plot(lrs, label=name, color=color)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.suptitle("Optimizer Comparison on MNIST", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "optimizer_comparison.png", dpi=150)
    plt.close()

    # 4. Final accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(optimizers.keys())
    final_accs = [optimizers[n][1][-1] for n in names]
    colors = [optimizers[n][3] for n in names]
    bars = ax.bar(names, final_accs, color=colors, alpha=0.7)
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Optimizer: Final Test Accuracy")
    ax.set_ylim(0.9, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f'{v:.4f}',
                ha='center', fontsize=8, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / "final_accuracy.png", dpi=150)
    plt.close()

    # 5. Adam vs AdamW: weight norm comparison
    print("\n=== Weight Norm Comparison ===")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Train with Adam and AdamW, track weight norms
    adam_model = CNN().to(device)
    adamw_model = CNN().to(device)

    adam_opt = torch.optim.Adam(adam_model.parameters(), lr=1e-3)
    adamw_opt = torch.optim.AdamW(adamw_model.parameters(), lr=1e-3, weight_decay=0.01)

    adam_norms = []
    adamw_norms = []

    for epoch in range(n_epochs):
        # Train one epoch each
        for model, opt in [(adam_model, adam_opt), (adamw_model, adamw_opt)]:
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                loss = F.cross_entropy(model(bx), by)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Track weight norms
        adam_norm = sum(p.norm().item()**2 for p in adam_model.parameters())**0.5
        adamw_norm = sum(p.norm().item()**2 for p in adamw_model.parameters())**0.5
        adam_norms.append(adam_norm)
        adamw_norms.append(adamw_norm)

    ax.plot(adam_norms, label='Adam', color='orange')
    ax.plot(adamw_norms, label='AdamW (WD=0.01)', color='red')
    ax.set_title("Weight Norm: Adam vs AdamW")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||W|| (total L2 norm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "weight_norms.png", dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("SGD + Momentum", "v_t = μ·v + ∇L\nw -= lr·v\nSimple but effective\nNeeds LR tuning", 0.17, 'blue'),
        ("Adam", "m = β₁·m + (1-β₁)·g\nv = β₂·v + (1-β₂)·g²\nAdaptive LR per param\nL2 coupled with LR", 0.5, 'orange'),
        ("AdamW", "Same as Adam but\nDecoupled weight decay:\nw -= lr·(adam_step + λ·w)\n→ Better generalization", 0.83, 'red'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Optimizer Evolution: SGD → Adam → AdamW", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "optimizer_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
