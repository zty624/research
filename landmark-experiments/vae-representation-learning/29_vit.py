"""
Minimal Vision Transformer (ViT) Reproduction
==============================================
Reproduces core ideas from ViT (2010.11929, Dosovitskiy et al.):
1. Patch embedding: split image into patches, linear projection
2. [CLS] token for classification
3. Positional embedding (learned)
4. Compare: ViT vs CNN on MNIST with varying data sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Patch Embedding ──

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, n_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


# ── Vision Transformer ──

class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=64, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout, activation='gelu', batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, D)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, D)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Use [CLS] token for classification
        cls_output = x[:, 0]
        return self.head(cls_output), cls_output


# ── Simple CNN Baseline ──

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h), h


# ── Training ──

def train_model(model, train_loader, test_loader, n_epochs=15, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_losses = []
    test_accs = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            if isinstance(model, VisionTransformer):
                logits, _ = model(bx)
            else:
                logits, _ = model(bx)

            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                if isinstance(model, VisionTransformer):
                    logits, _ = model(bx)
                else:
                    logits, _ = model(bx)
                correct += (logits.argmax(1) == by).sum().item()
                total += by.shape[0]
        test_accs.append(correct / total)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Test Acc: {test_accs[-1]:.4f}")

    return train_losses, test_accs


# ── Data Size Experiment ──

def train_with_data_fraction(model_class, model_kwargs, train_dataset, test_loader,
                             fractions=[0.1, 0.3, 0.5, 1.0], n_epochs=15, device='cpu'):
    """Train models with different data fractions."""
    results = {}
    total_data = len(train_dataset)

    for frac in fractions:
        n = max(int(total_data * frac), 100)
        subset = torch.utils.data.Subset(train_dataset, range(n))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)

        model = model_class(**model_kwargs).to(device)
        _, test_accs = train_model(model, train_loader, test_loader, n_epochs, device=device)
        results[frac] = test_accs
        print(f"  Fraction {frac:.0%}: Final acc = {test_accs[-1]:.4f}")

    return results


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "29-vit"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    n_epochs = 20

    # 1. ViT
    print("=== Training ViT ===")
    vit = VisionTransformer(img_size=28, patch_size=4, embed_dim=64, depth=4, n_heads=4).to(device)
    n_patches = vit.patch_embed.n_patches
    print(f"  Patches: {n_patches} ({28//4}x{28//4}), Params: {sum(p.numel() for p in vit.parameters()):,}")
    vit_losses, vit_accs = train_model(vit, train_loader, test_loader, n_epochs, device=device)

    # 2. CNN baseline
    print("\n=== Training CNN ===")
    cnn = SimpleCNN().to(device)
    print(f"  Params: {sum(p.numel() for p in cnn.parameters()):,}")
    cnn_losses, cnn_accs = train_model(cnn, train_loader, test_loader, n_epochs, device=device)

    # 3. ViT with different patch sizes
    print("\n=== Patch Size Comparison ===")
    patch_results = {}
    for ps in [2, 4, 7]:
        if 28 % ps != 0:
            continue
        print(f"  Patch size {ps}:")
        model = VisionTransformer(img_size=28, patch_size=ps, embed_dim=64, depth=4, n_heads=4).to(device)
        n_p = (28 // ps) ** 2
        n_params = sum(p.numel() for p in model.parameters())
        _, accs = train_model(model, train_loader, test_loader, n_epochs, device=device)
        patch_results[ps] = {'accs': accs, 'n_patches': n_p, 'params': n_params}

    # 4. Data scaling experiment
    print("\n=== Data Scaling: ViT vs CNN ===")
    vit_data = train_with_data_fraction(
        VisionTransformer,
        {'img_size': 28, 'patch_size': 4, 'embed_dim': 64, 'depth': 4, 'n_heads': 4},
        train_dataset, test_loader, fractions=[0.05, 0.1, 0.3, 0.5, 1.0],
        n_epochs=15, device=device
    )

    cnn_data = train_with_data_fraction(
        SimpleCNN, {},
        train_dataset, test_loader, fractions=[0.05, 0.1, 0.3, 0.5, 1.0],
        n_epochs=15, device=device
    )

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(vit_losses, label='ViT', color='blue')
    axes[0].plot(cnn_losses, label='CNN', color='red')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(vit_accs, label='ViT', color='blue')
    axes[1].plot(cnn_accs, label='CNN', color='red')
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Vision Transformer vs CNN on MNIST", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Patch size comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for ps, data in patch_results.items():
        ax.plot(data['accs'], label=f'Patch {ps} ({data["n_patches"]} patches, {data["params"]/1e3:.0f}K params)')
    ax.set_title("ViT: Patch Size Effect")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "patch_size_comparison.png", dpi=150)
    plt.close()

    # 3. Data scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = sorted(vit_data.keys())
    vit_final = [vit_data[f][-1] for f in fracs]
    cnn_final = [cnn_data[f][-1] for f in fracs]

    ax.plot([f*100 for f in fracs], vit_final, 'o-', label='ViT', color='blue')
    ax.plot([f*100 for f in fracs], cnn_final, 's-', label='CNN', color='red')
    ax.set_title("Data Scaling: ViT vs CNN")
    ax.set_xlabel("Training Data (%)")
    ax.set_ylabel("Final Test Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ax.set_ylim(0.8, 1.0)
    plt.savefig(results_dir / "data_scaling.png", dpi=150)
    plt.close()

    # 4. Patch embedding visualization
    print("\n=== Visualizing Patch Embeddings ===")
    vit_vis = VisionTransformer(img_size=28, patch_size=4, embed_dim=64, depth=4, n_heads=4).to(device)
    train_model(vit_vis, train_loader, test_loader, n_epochs=10, device=device)

    with torch.no_grad():
        # Get patch embeddings for test samples
        test_batch = next(iter(test_loader))[0][:64].to(device)
        patches = vit_vis.patch_embed(test_batch)  # (64, 49, 64)
        cls_features = []

        # Get [CLS] features
        B = test_batch.shape[0]
        x = patches
        cls_tokens = vit_vis.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + vit_vis.pos_embed
        for block in vit_vis.blocks:
            x = block(x)
        x = vit_vis.norm(x)
        cls_features = x[:, 0].cpu().numpy()

    # PCA of [CLS] features
    try:
        from sklearn.decomposition import PCA
        labels = next(iter(test_loader))[1][:64].numpy()
        pca = PCA(n_components=2).fit_transform(cls_features)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        scatter = axes[0].scatter(pca[:, 0], pca[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
        axes[0].set_title("ViT [CLS] Features (PCA)")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        plt.colorbar(scatter, ax=axes[0])
        axes[0].grid(True, alpha=0.3)

        # Visualize learned positional embeddings
        pos_emb = vit_vis.pos_embed[0, 1:, :].detach().cpu().numpy()  # (49, 64)
        pos_pca = PCA(n_components=2).fit_transform(pos_emb)
        grid_size = 7
        axes[1].scatter(pos_pca[:, 0], pos_pca[:, 1], c=np.arange(49), cmap='viridis', s=50)
        for i in range(49):
            axes[1].annotate(str(i), (pos_pca[i, 0], pos_pca[i, 1]), fontsize=6)
        axes[1].set_title("Learned Positional Embeddings (PCA)")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("ViT: Feature and Position Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / "feature_analysis.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  PCA visualization failed: {e}")

    # 5. ViT architecture diagram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    steps = [
        ("Input\n28×28", 0.05, 'gray'),
        ("Patches\n7×7×16", 0.2, 'blue'),
        ("+ Pos\nEmbed", 0.35, 'teal'),
        ("Transformer\n×4 layers", 0.55, 'purple'),
        ("[CLS]\ntoken", 0.72, 'orange'),
        ("Class\n10-way", 0.88, 'green'),
    ]

    for name, x_pos, color in steps:
        ax.text(x_pos, 0.5, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.12, 0.27, 0.45, 0.63, 0.80]:
        ax.annotate('→', xy=(x, 0.5), fontsize=20, ha='center', va='center', color='gray')

    ax.set_title("Vision Transformer (ViT) Architecture", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "vit_architecture.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
