"""
Minimal Lottery Ticket Hypothesis Reproduction
================================================
Reproduces core ideas from Lottery Ticket Hypothesis (1803.03635, Frankle & Carlin):
1. Dense networks contain sparse "winning tickets" that can match full performance
2. Iterative Magnitude Pruning (IMP): remove smallest weights, reset remaining to init
3. Winning ticket trains faster and achieves same or better accuracy than dense network
4. Random pruning at same sparsity performs much worse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy


# ── Pruning ──

def magnitude_prune(model, sparsity):
    """Prune smallest magnitude weights to achieve target sparsity."""
    # Collect all weight magnitudes
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name and 'emb' not in name:
            weights.append(param.data.abs().flatten())
    all_weights = torch.cat(weights)
    threshold = torch.quantile(all_weights, sparsity)

    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name and 'emb' not in name:
            masks[name] = (param.data.abs() >= threshold).float()
    return masks


def apply_masks(model, masks):
    """Apply pruning masks to model weights."""
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]


def random_prune(model, sparsity):
    """Random pruning at same sparsity level (baseline)."""
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name and 'emb' not in name:
            masks[name] = (torch.rand_like(param.data) >= sparsity).float()
    return masks


def count_sparsity(model, masks):
    """Count actual sparsity of masked model."""
    total = 0
    pruned = 0
    for name, param in model.named_parameters():
        if name in masks:
            total += masks[name].numel()
            pruned += (masks[name] == 0).sum().item()
    return pruned / total


# ── Model ──

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, n_layers=3, n_classes=10):
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


# ── Training ──

def train_model(model, train_loader, test_loader, n_epochs=15, lr=1e-2, device='cpu',
                masks=None, init_state=None):
    """Train model, optionally with pruning masks and weight reset."""
    if init_state is not None:
        # Reset to initialization (winning ticket)
        model.load_state_dict(init_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_losses = []
    test_accs = []

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

            # Apply masks after each step
            if masks is not None:
                apply_masks(model, masks)

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

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {test_accs[-1]:.4f}")

    return train_losses, test_accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "48-lottery-ticket"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(10000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Step 0: Train dense model and save initialization
    print("=== Step 0: Train Dense Model ===")
    dense_model = MLP(hidden=256, n_layers=3).to(device)
    dense_params = sum(p.numel() for p in dense_model.parameters())
    init_state = copy.deepcopy(dense_model.state_dict())
    print(f"  Dense params: {dense_params:,}")
    dense_losses, dense_accs = train_model(dense_model, train_loader, test_loader,
                                            n_epochs=15, device=device)
    print(f"  Dense final acc: {dense_accs[-1]:.4f}")

    # Step 1: Iterative Magnitude Pruning (IMP)
    print("\n=== Iterative Magnitude Pruning ===")
    sparsities = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]

    # Winning tickets (IMP: prune, reset to init, retrain)
    imp_results = {}
    for sparsity in sparsities:
        print(f"\n  Sparsity={sparsity:.0%}:")
        model = MLP(hidden=256, n_layers=3).to(device)

        if sparsity == 0.0:
            # No pruning — just retrain from init
            masks = None
            losses, accs = train_model(model, train_loader, test_loader,
                                       n_epochs=15, device=device, init_state=init_state)
        else:
            # IMP: prune trained model, get masks
            masks = magnitude_prune(dense_model, sparsity)
            actual_sparsity = count_sparsity(dense_model, masks)
            print(f"    Actual sparsity: {actual_sparsity:.2%}")

            # Reset to init and retrain with masks
            losses, accs = train_model(model, train_loader, test_loader,
                                       n_epochs=15, device=device,
                                       masks=masks, init_state=init_state)

        imp_results[sparsity] = {'losses': losses, 'accs': accs, 'final_acc': accs[-1]}

    # Step 2: Random pruning baseline
    print("\n=== Random Pruning Baseline ===")
    random_results = {}
    for sparsity in [0.5, 0.8, 0.9, 0.95]:
        print(f"\n  Random sparsity={sparsity:.0%}:")
        model = MLP(hidden=256, n_layers=3).to(device)
        masks = random_prune(model, sparsity)
        losses, accs = train_model(model, train_loader, test_loader,
                                   n_epochs=15, device=device,
                                   masks=masks, init_state=init_state)
        random_results[sparsity] = {'losses': losses, 'accs': accs, 'final_acc': accs[-1]}

    # ── Results ──
    print("\n=== Summary ===")
    print(f"  {'Method':>15s} | {'Sparsity':>8s} | {'Final Acc':>9s}")
    print("  " + "-" * 40)
    for sp, r in imp_results.items():
        print(f"  {'IMP (winning)':>15s} | {sp:>8.0%} | {r['final_acc']:>9.4f}")
    for sp, r in random_results.items():
        print(f"  {'Random':>15s} | {sp:>8.0%} | {r['final_acc']:>9.4f}")

    # ── Visualization ──

    # 1. Accuracy vs sparsity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    imp_sparsities = list(imp_results.keys())
    imp_accs = [imp_results[s]['final_acc'] for s in imp_sparsities]
    rand_sparsities = list(random_results.keys())
    rand_accs = [random_results[s]['final_acc'] for s in rand_sparsities]

    axes[0].plot(imp_sparsities, imp_accs, 'o-', label='IMP (Winning Ticket)', color='blue')
    axes[0].plot(rand_sparsities, rand_accs, 's--', label='Random Pruning', color='red')
    axes[0].axhline(y=dense_accs[-1], color='gray', linestyle=':', label='Dense baseline')
    axes[0].set_xlabel("Sparsity")
    axes[0].set_ylabel("Final Test Accuracy")
    axes[0].set_title("Winning Ticket vs Random Pruning")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Training curves at 80% sparsity
    sp = 0.8
    if sp in imp_results and sp in random_results:
        axes[1].plot(imp_results[sp]['accs'], label='IMP (Winning Ticket)', color='blue')
        axes[1].plot(random_results[sp]['accs'], label='Random Pruning', color='red')
        axes[1].plot(dense_accs, label='Dense', color='gray', linestyle=':')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Test Accuracy")
        axes[1].set_title(f"Training Curves at {sp:.0%} Sparsity")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("Lottery Ticket Hypothesis: Winning Tickets Exist", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "winning_ticket.png", dpi=150)
    plt.close()

    # 3. Weight distribution at different sparsities
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, sp in enumerate([0.0, 0.8, 0.95]):
        ax = axes[idx]
        model = MLP(hidden=256, n_layers=3).to(device)
        model.load_state_dict(init_state)
        if sp > 0:
            masks = magnitude_prune(dense_model, sp)
            apply_masks(model, masks)

        # Plot weight distribution of first layer
        w = model.layers[0].weight.data.cpu().numpy().flatten()
        w_nonzero = w[w != 0]
        ax.hist(w_nonzero, bins=50, alpha=0.7, color='blue')
        ax.set_title(f"Sparsity={sp:.0%}\n({len(w_nonzero):,} nonzero weights)")
        ax.set_xlabel("Weight Value")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Pruned Weight Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "weight_distributions.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Dense\nNetwork", "All weights trained\nFull capacity\nMost weights\nare redundant\n→ Over-parameterized", 0.14, 'gray'),
        ("Prune &\nReset", "Remove smallest\nweights by magnitude\nReset remaining to\ninitialization\n→ Winning ticket", 0.5, 'blue'),
        ("Winning\nTicket", "Sparse subnetwork\nthat trains fast\nand matches dense\n→ 'The lottery\n   was won at init!'", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Lottery Ticket Hypothesis: Sparse Subnetworks at Initialization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "lottery_ticket_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
