"""
Minimal ResNet / Skip Connection Reproduction
==============================================
Reproduces core ideas from ResNet (1512.03385, He et al.):
1. Skip connections: F(x) = H(x) - x, learn residual
2. Vanishing gradient solution: gradients flow directly through shortcut
3. Deeper ≠ worse: adding identity shortcuts makes deeper networks trainable
4. Compare: plain network vs residual network at various depths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Building Blocks ──

class PlainBlock(nn.Module):
    """Standard conv block without skip connection."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h)


class ResidualBlock(nn.Module):
    """Conv block with skip (residual) connection: F(x) + x."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + residual  # skip connection
        return F.relu(h)


# ── Networks ──

def make_plain_net(C=16, n_blocks_per_stage=2, n_stages=3, n_classes=10):
    """Plain network (no skip connections)."""
    layers = []
    # Stem
    layers.extend([
        nn.Conv2d(1, C, 3, padding=1, bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(),
    ])
    # Stages
    for stage in range(n_stages):
        for _ in range(n_blocks_per_stage):
            stride = 2 if stage > 0 and _ == 0 else 1
            if stride == 2:
                # Downsample with channel doubling
                layers.append(nn.Conv2d(C, C*2, 3, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(C*2))
                layers.append(nn.ReLU())
                C = C * 2
            layers.append(PlainBlock(C))
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers), C


def make_resnet(C=16, n_blocks_per_stage=2, n_stages=3, n_classes=10):
    """Residual network (with skip connections)."""
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(),
            )
            self.stages = nn.ModuleList()
            self.c = C
            for stage in range(n_stages):
                blocks = []
                for _ in range(n_blocks_per_stage):
                    if stage > 0 and _ == 0:
                        # Downsample: increase channels
                        blocks.append(ResidualBlockWithDownsample(self.c, self.c * 2))
                        self.c = self.c * 2
                    else:
                        blocks.append(ResidualBlock(self.c))
                self.stages.append(nn.Sequential(*blocks))
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            h = self.stem(x)
            for stage in self.stages:
                h = stage(h)
            h = self.pool(h).flatten(1)
            return h

    # Build model and add classifier
    model = ResNet()
    final_c = model.c
    return model, final_c


class ResidualBlockWithDownsample(nn.Module):
    """Residual block that doubles channels and downsamples."""
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C_out)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out)
        # 1x1 conv to match dimensions
        self.shortcut = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, stride=2, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + residual
        return F.relu(h)


class MNISTModel(nn.Module):
    """Wrapper for plain or residual network."""
    def __init__(self, backbone, final_c, n_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(final_c, n_classes)

    def forward(self, x):
        if isinstance(self.backbone, nn.Sequential):
            h = self.backbone(x)
            h = h.flatten(1)
        else:
            h = self.backbone(x)
        return self.classifier(h)


# ── Training ──

def train_model(model, train_loader, test_loader, n_epochs=15, lr=1e-2, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_losses = []
    test_accs = []
    grad_norms = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()

            # Track gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            epoch_grad_norm += total_norm ** 0.5

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)
        grad_norms.append(epoch_grad_norm / n_batches)
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

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Acc: {test_accs[-1]:.4f} | GradNorm: {grad_norms[-1]:.4f}")

    return train_losses, test_accs, grad_norms


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "35-resnet"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(10000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Experiment: vary depth and compare plain vs residual
    depths = [2, 4, 8]  # blocks per stage
    all_results = {}

    for n_blocks in depths:
        print(f"\n=== Depth: {n_blocks} blocks/stage ===")

        # Plain network
        print(f"  Training Plain ({n_blocks} blocks)...")
        backbone, final_c = make_plain_net(C=16, n_blocks_per_stage=n_blocks, n_stages=3)
        plain = MNISTModel(backbone, final_c).to(device)
        plain_params = sum(p.numel() for p in plain.parameters())
        print(f"    Params: {plain_params:,}")
        plain_losses, plain_accs, plain_grads = train_model(plain, train_loader, test_loader, n_epochs=15, device=device)

        # Residual network
        print(f"  Training ResNet ({n_blocks} blocks)...")
        backbone, final_c = make_resnet(C=16, n_blocks_per_stage=n_blocks, n_stages=3)
        resnet = MNISTModel(backbone, final_c).to(device)
        resnet_params = sum(p.numel() for p in resnet.parameters())
        print(f"    Params: {resnet_params:,}")
        resnet_losses, resnet_accs, resnet_grads = train_model(resnet, train_loader, test_loader, n_epochs=15, device=device)

        all_results[n_blocks] = {
            'plain': {'losses': plain_losses, 'accs': plain_accs, 'grads': plain_grads, 'params': plain_params},
            'resnet': {'losses': resnet_losses, 'accs': resnet_accs, 'grads': resnet_grads, 'params': resnet_params},
        }

    # ── Gradient Flow Experiment ──
    print("\n=== Gradient Flow Experiment ===")
    # Measure gradient magnitude at first layer for deep plain vs resnet
    for n_blocks in depths:
        # Plain
        backbone, final_c = make_plain_net(C=16, n_blocks_per_stage=n_blocks, n_stages=3)
        plain = MNISTModel(backbone, final_c).to(device)
        x = torch.randn(4, 1, 28, 28, device=device)
        loss = plain(x).sum()
        loss.backward()
        first_layer_grad = None
        for name, p in plain.named_parameters():
            if 'conv' in name and p.grad is not None:
                first_layer_grad = p.grad.norm().item()
                break

        # ResNet
        backbone, final_c = make_resnet(C=16, n_blocks_per_stage=n_blocks, n_stages=3)
        resnet = MNISTModel(backbone, final_c).to(device)
        x = torch.randn(4, 1, 28, 28, device=device)
        loss = resnet(x).sum()
        loss.backward()
        res_first_grad = None
        for name, p in resnet.named_parameters():
            if 'conv' in name and p.grad is not None:
                res_first_grad = p.grad.norm().item()
                break

        print(f"  Depth {n_blocks}: Plain first-layer grad={first_layer_grad:.6f}, ResNet={res_first_grad:.6f}")

    # ── Final comparison ──
    print("\n=== Final Accuracy ===")
    for n_blocks in depths:
        plain_acc = all_results[n_blocks]['plain']['accs'][-1]
        res_acc = all_results[n_blocks]['resnet']['accs'][-1]
        print(f"  Depth {n_blocks}: Plain={plain_acc:.4f}, ResNet={res_acc:.4f}")

    # ── Visualization ──

    # 1. Training curves at each depth
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, n_blocks in enumerate(depths):
        ax = axes[idx]
        r = all_results[n_blocks]
        ax.plot(r['plain']['losses'], label='Plain', color='red')
        ax.plot(r['resnet']['losses'], label='ResNet', color='blue')
        ax.set_title(f"{n_blocks} blocks/stage")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("ResNet: Skip Connections Enable Training Deeper Networks", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Test accuracy vs depth
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plain_final_accs = [all_results[d]['plain']['accs'][-1] for d in depths]
    res_final_accs = [all_results[d]['resnet']['accs'][-1] for d in depths]

    axes[0].plot(depths, plain_final_accs, 'o-', label='Plain', color='red')
    axes[0].plot(depths, res_final_accs, 's-', label='ResNet', color='blue')
    axes[0].set_xlabel("Blocks per Stage")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Accuracy vs Depth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 3. Gradient norms
    plain_grads = [all_results[d]['plain']['grads'][-1] for d in depths]
    res_grads = [all_results[d]['resnet']['grads'][-1] for d in depths]

    axes[1].plot(depths, plain_grads, 'o-', label='Plain', color='red')
    axes[1].plot(depths, res_grads, 's-', label='ResNet', color='blue')
    axes[1].set_xlabel("Blocks per Stage")
    axes[1].set_ylabel("Gradient Norm (last epoch)")
    axes[1].set_title("Gradient Norm vs Depth")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ResNet: Deeper Plain Networks Degrade, ResNet Does Not", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "depth_comparison.png", dpi=150)
    plt.close()

    # 4. Learning curves (epoch-by-epoch accuracy) for deepest model
    fig, ax = plt.subplots(figsize=(10, 5))
    r = all_results[depths[-1]]
    epochs = range(1, len(r['plain']['accs']) + 1)
    ax.plot(epochs, r['plain']['accs'], label=f'Plain ({depths[-1]} blocks)', color='red', linewidth=2)
    ax.plot(epochs, r['resnet']['accs'], label=f'ResNet ({depths[-1]} blocks)', color='blue', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Learning Curves at Maximum Depth ({depths[-1]} blocks)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "learning_curves_deep.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Plain\nNetwork", "y = F(x)\nEach layer must\nlearn full mapping\n→ Vanishing gradients\n→ Deeper = worse", 0.14, 'red'),
        ("Residual\nLearning", "y = F(x) + x\nLearn residual F\n= H(x) - x\n→ Gradients flow\n   through shortcut", 0.5, 'blue'),
        ("Key Insight", "Identity mapping is\neasy to learn: F=0\nShortcuts add zero\nextra params\n→ Deeper = better", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("ResNet: Skip Connections Solve the Degradation Problem", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "resnet_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
