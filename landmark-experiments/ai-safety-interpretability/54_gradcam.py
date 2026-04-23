"""
Minimal GradCAM Reproduction
==============================
Reproduces core ideas from GradCAM (2304.08485 / 1610.02391, Selvaraju et al.):
1. Gradient-weighted Class Activation Mapping for visual explanations
2. α_k^c = (1/Z) Σ_i Σ_j ∂y^c / ∂A_k^ij  (global average pooling of gradients)
3. L_GradCAM = ReLU(Σ_k α_k^c · A_k)  (weighted combination + ReLU)
4. Highlight regions most relevant to a specific class prediction
5. Class-agnostic: same image produces different explanations for different classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Simple CNN for MNIST ──

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),  # 28x28
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        self.gradients = {}
        self.activations = {}

    def forward(self, x):
        # Store activations from last conv layer
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.out_channels == 32:
                if i not in self.activations:
                    self.activations[i] = x

        # Register hook on last conv (only if grad tracking enabled)
        if x.requires_grad:
            x.register_hook(self._save_grad)
        self.last_conv_act = x

        return self.classifier(x)

    def _save_grad(self, grad):
        self.gradients['last_conv'] = grad

    def get_gradcam(self, target_class=None):
        """Compute GradCAM for the target class."""
        act = self.last_conv_act  # (B, C, H, W)
        grad = self.gradients['last_conv']  # (B, C, H, W)

        if target_class is not None:
            # Already computed outside; grad is for target class
            pass

        # Global average pool gradients → α_k^c
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination: Σ_k α_k^c · A_k
        cam = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze(1)  # (B, H, W)
        for i in range(cam.shape[0]):
            if cam[i].max() > 0:
                cam[i] = cam[i] / cam[i].max()

        return cam


# ── Training ──

def train_cnn(model, train_loader, n_epochs=5, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == by).sum().item()
            total += by.shape[0]

        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Acc: {correct/total:.4f}")


# ── GradCAM Computation ──

def compute_gradcam(model, image, target_class=None):
    """Compute GradCAM for a single image."""
    model.eval()
    image = image.unsqueeze(0)  # (1, 1, 28, 28)

    # Forward pass
    logits = model(image)

    if target_class is None:
        target_class = logits.argmax(1).item()

    # Backward pass for target class
    model.zero_grad()
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1.0
    logits.backward(gradient=one_hot, retain_graph=True)

    # Get GradCAM
    cam = model.get_gradcam(target_class)
    return cam[0].detach().cpu().numpy(), logits.argmax(1).item()


# ── Vanilla Gradients (for comparison) ──

def compute_vanilla_grad(model, image):
    """Compute vanilla gradient (saliency map)."""
    model.eval()
    image = image.unsqueeze(0).requires_grad_(True)

    logits = model(image)
    target = logits.argmax(1).item()

    model.zero_grad()
    one_hot = torch.zeros_like(logits)
    one_hot[0, target] = 1.0
    logits.backward(gradient=one_hot)

    saliency = image.grad.abs().squeeze().cpu().numpy()
    return saliency, target


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "54-gradcam"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(5000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Train model
    print("=== Training CNN ===")
    model = SimpleCNN(n_classes=10).to(device)
    train_cnn(model, train_loader, n_epochs=5, device=device)

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            correct += (model(bx).argmax(1) == by).sum().item()
            total += by.shape[0]
    print(f"  Test accuracy: {correct/total:.4f}")

    # Experiment 1: GradCAM vs Saliency maps
    print("\n=== GradCAM vs Saliency Maps ===")

    # Get sample images
    test_images = []
    test_labels = []
    for bx, by in test_loader:
        test_images = bx[:8]
        test_labels = by[:8]
        break

    fig, axes = plt.subplots(3, 8, figsize=(20, 8))

    for i in range(8):
        img = test_images[i].to(device)
        label = test_labels[i].item()

        # Original image
        axes[0, i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"Label: {label}")
        axes[0, i].axis('off')

        # Saliency map
        saliency, pred = compute_vanilla_grad(model, img)
        axes[1, i].imshow(saliency, cmap='hot')
        axes[1, i].set_title(f"Saliency (p={pred})")
        axes[1, i].axis('off')

        # GradCAM
        cam, pred = compute_gradcam(model, img)
        # Resize CAM to image size
        cam_resized = np.kron(cam, np.ones((4, 4)))[:28, :28]
        axes[2, i].imshow(img.squeeze().cpu().numpy(), cmap='gray', alpha=0.5)
        axes[2, i].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[2, i].set_title(f"GradCAM (p={pred})")
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Saliency", fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel("GradCAM", fontsize=12, fontweight='bold')

    plt.suptitle("GradCAM: Visual Explanations from Deep Networks", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "gradcam_comparison.png", dpi=150)
    plt.close()

    # Experiment 2: Class-specific explanations
    print("\n=== Class-Specific GradCAM ===")

    # Find a well-classified image
    img = test_images[0].to(device)
    true_label = test_labels[0].item()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for target_class in range(10):
        cam, _ = compute_gradcam(model, img, target_class=target_class)
        cam_resized = np.kron(cam, np.ones((4, 4)))[:28, :28]

        row, col = target_class // 5, target_class % 5
        axes[row, col].imshow(img.squeeze().cpu().numpy(), cmap='gray', alpha=0.5)
        axes[row, col].imshow(cam_resized, cmap='jet', alpha=0.5)
        if target_class == true_label:
            axes[row, col].set_title(f"Class {target_class} ✓", fontweight='bold', color='green')
        else:
            axes[row, col].set_title(f"Class {target_class}")
        axes[row, col].axis('off')

    plt.suptitle(f"GradCAM: Different Class Explanations (True label: {true_label})", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "class_specific_gradcam.png", dpi=150)
    plt.close()

    # Experiment 3: GradCAM on multiple examples
    print("\n=== GradCAM Examples ===")

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    for i in range(4):
        img = test_images[i + 2].to(device)  # Skip first few
        true_label = test_labels[i + 2].item()

        # Show original
        axes[i, 0].imshow(img.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f"True: {true_label}")
        axes[i, 0].axis('off')

        # Show top-5 class GradCAMs
        logits = model(img.unsqueeze(0))
        top5 = logits.argsort(descending=True)[0, :5].tolist()

        for j, cls in enumerate(top5):
            cam, _ = compute_gradcam(model, img, target_class=cls)
            cam_resized = np.kron(cam, np.ones((4, 4)))[:28, :28]

            axes[i, j + 1].imshow(img.squeeze().cpu().numpy(), cmap='gray', alpha=0.5)
            axes[i, j + 1].imshow(cam_resized, cmap='jet', alpha=0.5)
            prob = F.softmax(logits, dim=1)[0, cls].item()
            is_correct = "✓" if cls == true_label else ""
            axes[i, j + 1].set_title(f"Cls {cls} {prob:.2f}{is_correct}", fontsize=9)
            axes[i, j + 1].axis('off')

    plt.suptitle("GradCAM: Top-5 Class Explanations per Image", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "gradcam_top5.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Forward\nPass", "Input image → CNN\nGet feature maps A_k\nfrom last conv layer\nGet class score y^c\n→ Normal prediction", 0.14, 'red'),
        ("Gradient\nComputation", "∂y^c / ∂A_k\nHow much does each\nfeature affect the\nclass prediction?\nα_k = GAP(∂y^c/∂A_k)", 0.5, 'blue'),
        ("GradCAM\nMap", "L = ReLU(Σ α_k·A_k)\nWeighted combination\nof feature maps\n→ Heatmap showing\n   important regions", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("GradCAM: Gradient-Weighted Class Activation Mapping", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "gradcam_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
