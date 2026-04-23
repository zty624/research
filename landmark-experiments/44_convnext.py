"""
Minimal ConvNeXt Reproduction
==============================
Reproduces core ideas from ConvNeXt (2201.03545, Liu et al.):
1. Modernize ResNet by adopting Transformer design choices
2. Step-by-step: ResNet → depthwise conv → larger kernel → LayerNorm → GELU → fewer activations → SwiGLU
3. Pure ConvNet can match ViT performance without attention!
4. Compare: ResNet vs ConvNeXt step-by-step modernization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── ResNet Block (baseline) ──

class ResNetBlock(nn.Module):
    """Standard ResNet bottleneck block."""
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.conv3 = nn.Conv2d(C, C, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(C)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + residual)


# ── ConvNeXt Block (modernized) ──

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv + LayerNorm + Linear + GELU + Linear."""
    def __init__(self, C, kernel_size=7):
        super().__init__()
        # Depthwise 7x7 conv (like spatial mixing in ViT)
        self.dwconv = nn.Conv2d(C, C, kernel_size, padding=kernel_size//2, groups=C, bias=False)
        self.norm = nn.LayerNorm(C)  # LayerNorm instead of BatchNorm
        # Inverted bottleneck: narrow → wide → narrow (like SwiGLU FFN)
        self.pw1 = nn.Linear(C, 4 * C)
        self.act = nn.GELU()  # GELU instead of ReLU
        self.pw2 = nn.Linear(4 * C, C)
        # No activation after residual (less activations)
        self.gamma = nn.Parameter(torch.zeros(C))  # LayerScale

    def forward(self, x):
        residual = x
        h = self.dwconv(x)
        # Permute for LayerNorm: (B, C, H, W) → (B, H, W, C)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.pw2(h)
        h = self.gamma * h
        h = h.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        return h + residual


# ── Networks ──

class ResNetTiny(nn.Module):
    """Tiny ResNet for MNIST."""
    def __init__(self, C=32, n_blocks=3, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 4, stride=4, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResNetBlock(C) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, n_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


class ConvNeXtTiny(nn.Module):
    """Tiny ConvNeXt for MNIST."""
    def __init__(self, C=32, n_blocks=3, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 4, stride=4, bias=False),
            nn.LayerNorm([C, 7, 7]),
        )
        self.blocks = nn.Sequential(*[ConvNeXtBlock(C) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(C)
        self.classifier = nn.Linear(C, n_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)
        h = self.norm(h)
        return self.classifier(h)


# ── Incremental Modernization ──

class ModernizedBlock(nn.Module):
    """ResNet block with configurable modernization steps."""
    def __init__(self, C, use_depthwise=False, kernel_size=3,
                 use_layernorm=False, use_gelu=False,
                 fewer_activations=False, use_inverted_bottleneck=False):
        super().__init__()
        self.use_layernorm = use_layernorm

        if use_depthwise:
            self.conv1 = nn.Conv2d(C, C, kernel_size, padding=kernel_size//2, groups=C, bias=False)
            self.bn1 = nn.BatchNorm2d(C) if not use_layernorm else None
            self.conv2 = nn.Conv2d(C, C, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(C) if not use_layernorm else None
        else:
            self.conv1 = nn.Conv2d(C, C, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(C)
            self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(C)

        self.act = nn.GELU() if use_gelu else nn.ReLU()
        self.fewer_activations = fewer_activations
        self.use_inverted_bottleneck = use_inverted_bottleneck

        if use_inverted_bottleneck:
            self.expand = nn.Conv2d(C, 4*C, 1, bias=False)
            self.contract = nn.Conv2d(4*C, C, 1, bias=False)
        else:
            self.expand = None
            self.contract = None

    def forward(self, x):
        residual = x
        h = self.conv1(x)
        if self.bn1 is not None:
            h = self.bn1(h)
        if not self.fewer_activations:
            h = self.act(h)
        h = self.conv2(h) if self.expand is None else self.contract(self.act(self.expand(h)))
        if self.bn2 is not None:
            h = self.bn2(h)
        h = self.act(h + residual)
        return h


# ── Training ──

def train_model(model, train_loader, test_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
    results_dir = Path(__file__).parent / "results" / "44-convnext"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(10000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Experiment 1: ResNet vs ConvNeXt
    print("=== ResNet vs ConvNeXt ===")
    print("  ResNet:")
    resnet = ResNetTiny(C=32, n_blocks=4).to(device)
    resnet_params = sum(p.numel() for p in resnet.parameters())
    print(f"    Params: {resnet_params:,}")
    resnet_losses, resnet_accs = train_model(resnet, train_loader, test_loader, n_epochs=10, device=device)

    print("  ConvNeXt:")
    convnext = ConvNeXtTiny(C=32, n_blocks=4).to(device)
    convnext_params = sum(p.numel() for p in convnext.parameters())
    print(f"    Params: {convnext_params:,}")
    convnext_losses, convnext_accs = train_model(convnext, train_loader, test_loader, n_epochs=10, device=device)

    # Experiment 2: Step-by-step modernization
    print("\n=== Step-by-Step Modernization ===")
    steps = [
        ("ResNet (baseline)", {}),
        ("+ Depthwise Conv", {"use_depthwise": True}),
        ("+ Larger Kernel (7)", {"use_depthwise": True, "kernel_size": 7}),
        ("+ GELU", {"use_depthwise": True, "kernel_size": 7, "use_gelu": True}),
        ("+ Inverted Bottleneck", {"use_depthwise": True, "kernel_size": 7, "use_gelu": True, "use_inverted_bottleneck": True}),
    ]

    step_results = {}
    for name, kwargs in steps:
        print(f"  {name}:")
        # Build model with this configuration
        model = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            *[ModernizedBlock(32, **kwargs) for _ in range(3)],
            nn.AdaptiveAvgPool2d(1),
        )
        class Wrapper(nn.Module):
            def __init__(self, backbone, n_classes=10):
                super().__init__()
                self.backbone = backbone
                self.fc = nn.Linear(32, n_classes)
            def forward(self, x):
                h = self.backbone(x).flatten(1)
                return self.fc(h)

        wrapper = Wrapper(model).to(device)
        n_params = sum(p.numel() for p in wrapper.parameters())
        print(f"    Params: {n_params:,}")
        losses, accs = train_model(wrapper, train_loader, test_loader, n_epochs=10, device=device)
        step_results[name] = {'losses': losses, 'accs': accs, 'final_acc': accs[-1], 'params': n_params}

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"  ResNet:     Acc={resnet_accs[-1]:.4f}, Params={resnet_params:,}")
    print(f"  ConvNeXt:   Acc={convnext_accs[-1]:.4f}, Params={convnext_params:,}")
    print("\n  Modernization steps:")
    for name, r in step_results.items():
        print(f"    {name:35s}: Acc={r['final_acc']:.4f}, Params={r['params']:,}")

    # ── Visualization ──

    # 1. ResNet vs ConvNeXt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(resnet_losses, label='ResNet', color='red')
    axes[0].plot(convnext_losses, label='ConvNeXt', color='blue')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(resnet_accs, label='ResNet', color='red')
    axes[1].plot(convnext_accs, label='ConvNeXt', color='blue')
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ConvNeXt: Modernized ResNet Matches Transformer Design", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "resnet_vs_convnext.png", dpi=150)
    plt.close()

    # 2. Step-by-step modernization
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(step_results.keys())
    final_accs = [step_results[n]['final_acc'] for n in names]
    colors = ['gray', 'orange', 'yellow', 'lightgreen', 'blue']

    bars = ax.bar(range(len(names)), final_accs, color=colors[:len(names)], alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("ConvNeXt: Step-by-Step Modernization of ResNet")
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.002, f'{acc:.3f}',
                ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(results_dir / "modernization_steps.png", dpi=150)
    plt.close()

    # 3. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("ResNet\n(2015)", "3×3 Conv\nBatchNorm\nReLU\nBottleneck\n→ Standard CNN", 0.14, 'red'),
        ("Modernize\nStep-by-Step", "Depthwise 7×7\nLayerNorm\nGELU\nInv. Bottleneck\nFewer activations", 0.5, 'blue'),
        ("ConvNeXt\n(2022)", "Pure ConvNet\nMatches ViT\nNo attention!\nSimpler design\n→ SOTA without Transformer", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("ConvNeXt: A ConvNet for the 2020s", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "convnext_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
