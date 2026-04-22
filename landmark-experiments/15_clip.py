"""
Minimal CLIP / Contrastive Learning Reproduction
==================================================
Reproduces core ideas from "Learning Transferable Visual Models From
Natural Language Supervision" (2103.00020, CLIP):
1. Contrastive learning: align image and text embeddings
2. InfoNCE loss (cross-modal negative sampling)
3. Temperature scaling in softmax
4. Zero-shot classification via text prompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Simple Encoders ──

class ImageEncoder(nn.Module):
    """Simple CNN for 28x28 images."""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return F.normalize(self.proj(h), dim=-1)


class TextEncoder(nn.Module):
    """Simple encoder for class label embeddings."""
    def __init__(self, n_classes, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, labels):
        h = self.embed(labels)
        return F.normalize(self.proj(h), dim=-1)


# ── CLIP Model ──

class CLIPModel(nn.Module):
    def __init__(self, n_classes=10, embed_dim=64, init_temp=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(n_classes, embed_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))

    def forward(self, images, labels):
        img_emb = self.image_encoder(images)   # (B, D)
        txt_emb = self.text_encoder(labels)     # (B, D)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_emb @ txt_emb.T  # (B, B)
        return logits, img_emb, txt_emb

    def zero_shot_classify(self, images, all_labels):
        """Classify images by comparing to all class text embeddings."""
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(all_labels)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_emb @ txt_emb.T  # (B, n_classes)
        return logits.argmax(dim=-1)


# ── InfoNCE Loss ──

def contrastive_loss(logits, labels=None):
    """Symmetric InfoNCE loss.
    Given a (B, B) similarity matrix where diagonal = positive pairs,
    maximize similarity of positive pairs while minimizing negatives.
    """
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)      # image → text
    loss_t2i = F.cross_entropy(logits.T, targets)     # text → image
    return (loss_i2t + loss_t2i) / 2


# ── Training ──

def train_clip(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    temps = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits, _, _ = model(bx, by)
            loss = contrastive_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            temps.append(model.logit_scale.exp().item())

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

    return losses, temps


def train_supervised(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    """Standard supervised baseline for comparison."""
    # Use only the image encoder + a classifier head
    classifier = nn.Linear(64, 10).to(device)
    optimizer = torch.optim.AdamW(
        list(model.image_encoder.parameters()) + list(classifier.parameters()), lr=lr
    )
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                img_emb = model.image_encoder(bx)
            logits = classifier(img_emb)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))

    return losses, classifier


# ── Evaluation ──

def evaluate_zero_shot(model, test_loader, n_classes=10, device='cpu'):
    """Zero-shot classification: compare image to all class text embeddings."""
    all_labels = torch.arange(n_classes, device=device)
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model.zero_shot_classify(bx, all_labels)
            correct += (pred == by).sum().item()
            total += by.shape[0]
    return correct / total


def evaluate_linear_probe(model, test_loader, classifier, device='cpu'):
    """Linear probe evaluation."""
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            img_emb = model.image_encoder(bx)
            pred = classifier(img_emb).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += by.shape[0]
    return correct / total


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "15-clip"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Train CLIP
    print("=== Training CLIP (Contrastive) ===")
    clip_model = CLIPModel(n_classes=10, embed_dim=64).to(device)
    clip_losses, temps = train_clip(clip_model, train_loader, n_epochs=10, device=device)
    zero_shot_acc = evaluate_zero_shot(clip_model, test_loader, device=device)
    print(f"  Zero-shot accuracy: {zero_shot_acc:.4f}")

    # Train supervised baseline
    print("\n=== Training Supervised Baseline ===")
    sup_model = CLIPModel(n_classes=10, embed_dim=64).to(device)
    sup_losses, classifier = train_supervised(sup_model, train_loader, n_epochs=10, device=device)
    lin_probe_acc = evaluate_linear_probe(sup_model, test_loader, classifier, device=device)
    print(f"  Linear probe accuracy: {lin_probe_acc:.4f}")

    # Also do linear probe on CLIP embeddings
    print("\n=== Linear Probe on CLIP Embeddings ===")
    clip_classifier = nn.Linear(64, 10).to(device)
    clip_opt = torch.optim.AdamW(clip_classifier.parameters(), lr=1e-3)
    for epoch in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                img_emb = clip_model.image_encoder(bx)
            loss = F.cross_entropy(clip_classifier(img_emb), by)
            clip_opt.zero_grad()
            loss.backward()
            clip_opt.step()

    clip_linprobe_acc = evaluate_linear_probe(clip_model, test_loader, clip_classifier, device=device)
    print(f"  CLIP linear probe accuracy: {clip_linprobe_acc:.4f}")

    # ── Visualization ──

    # 1. Training loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(clip_losses, label='CLIP (Contrastive)', color='blue')
    axes[0].plot(sup_losses, label='Supervised (CE)', color='red')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Temperature evolution
    axes[1].plot(temps, color='green')
    axes[1].set_title("Learned Temperature (1/τ)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Scale (exp of logit_scale)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CLIP: Contrastive Language-Image Pre-training", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 3. Embedding space visualization (t-SNE-like with PCA)
    from sklearn.decomposition import PCA
    clip_model.eval()
    all_img_embs = []
    all_labels = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            embs = clip_model.image_encoder(bx).cpu().numpy()
            all_img_embs.append(embs)
            all_labels.append(by.numpy())

    all_img_embs = np.concatenate(all_img_embs)[:500]
    all_labels = np.concatenate(all_labels)[:500]

    # Also get text embeddings
    with torch.no_grad():
        all_cls = torch.arange(10, device=device)
        txt_embs = clip_model.text_encoder(all_cls).cpu().numpy()

    # PCA
    pca = PCA(n_components=2)
    combined = np.vstack([all_img_embs, txt_embs])
    combined_2d = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(combined_2d[:500, 0], combined_2d[:500, 1],
                         c=all_labels, cmap='tab10', alpha=0.3, s=5, label='Images')
    ax.scatter(combined_2d[500:, 0], combined_2d[500:, 1],
               c=range(10), cmap='tab10', marker='*', s=300, edgecolors='black',
               linewidth=1.5, label='Text embeddings')

    for i in range(10):
        ax.annotate(str(i), (combined_2d[500+i, 0], combined_2d[500+i, 1]),
                    fontsize=12, fontweight='bold', ha='center', va='center')

    ax.set_title("CLIP Embedding Space (PCA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "embedding_space.png", dpi=150)
    plt.close()

    # 4. Accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Zero-Shot\n(CLIP)', 'Linear Probe\n(CLIP emb)', 'Supervised\n(CE baseline)']
    accs = [zero_shot_acc, clip_linprobe_acc, lin_probe_acc]
    colors = ['blue', 'cyan', 'red']
    ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("CLIP: Zero-Shot vs Linear Probe vs Supervised")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.8, 1.0)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    # 5. Similarity matrix visualization
    clip_model.eval()
    with torch.no_grad():
        all_cls = torch.arange(10, device=device)
        txt_embs = clip_model.text_encoder(all_cls)
        # Get mean image embedding per class
        class_embs = []
        for c in range(10):
            mask = torch.tensor(all_labels == c)
            class_imgs = torch.tensor(all_img_embs[mask][:20], dtype=torch.float32).to(device)
            # Re-encode through proj
            mean_emb = torch.tensor(all_img_embs[mask].mean(axis=0), dtype=torch.float32).to(device)
            class_embs.append(F.normalize(mean_emb, dim=-1))
        class_embs = torch.stack(class_embs)

        sim_matrix = (class_embs @ txt_embs.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    ax.set_xlabel("Text Class Embedding")
    ax.set_ylabel("Image Class Mean Embedding")
    ax.set_title("Cross-Modal Similarity Matrix")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(results_dir / "similarity_matrix.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
