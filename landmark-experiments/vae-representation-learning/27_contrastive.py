"""
Minimal Contrastive Learning Reproduction
=========================================
Reproduces core ideas from contrastive learning literature:
1. SimCLR (2002.05709): NT-Xent loss, augmentation invariance
2. MoCo (1911.05722): momentum encoder, queue of negatives
3. BYOL (2006.07733): no negatives, EMA target + prediction head
4. Compare: SimCLR vs MoCo vs BYOL on MNIST representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import random


# ── Augmentations ──

def augment_batch(x, strength=0.3):
    """Simple augmentations for MNIST: noise + crop + brightness."""
    B, C, H, W = x.shape
    aug = x.clone()

    # Gaussian noise
    aug = aug + torch.randn_like(aug) * strength * 0.5

    # Random brightness
    brightness = 1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * strength
    aug = aug * brightness

    # Random crop (shift by up to 2 pixels)
    shift_h = random.randint(-2, 2)
    shift_w = random.randint(-2, 2)
    aug = torch.roll(aug, shifts=(shift_h, shift_w), dims=(2, 3))

    return aug.clamp(0, 1)


# ── Encoder ──

class Encoder(nn.Module):
    """Simple CNN encoder for MNIST."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        h = self.cnn(x).flatten(1)
        return self.proj(h)


# ── NT-Xent Loss (SimCLR) ──

def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross-Entropy Loss.
    For each pair (z1_i, z2_i), treat z2_i as positive,
    all other z2_j (j≠i) as negatives.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=-1)

    # Similarity matrix
    sim = z @ z.T / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Labels: z1_i's positive is z2_i and vice versa
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


# ── SimCLR ──

class SimCLR(nn.Module):
    def __init__(self, hidden_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = Encoder(hidden_dim)
        self.temperature = temperature

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return nt_xent_loss(z1, z2, self.temperature)


# ── MoCo ──

class MoCo(nn.Module):
    def __init__(self, hidden_dim=128, queue_size=256, momentum=0.999, temperature=0.07):
        super().__init__()
        self.encoder_q = Encoder(hidden_dim)
        self.encoder_k = Encoder(hidden_dim)
        self.momentum = momentum
        self.temperature = temperature

        # Initialize k with q's weights
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue
        self.register_buffer('queue', torch.randn(hidden_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    @torch.no_grad()
    def momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def dequeue_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=-1)  # (B, D)
        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=-1)  # (B, D)

        # Positive logits: q @ k.T
        l_pos = (q * k).sum(dim=-1, keepdim=True)  # (B, 1)

        # Negative logits: q @ queue (clone to avoid in-place modify issue)
        l_neg = q @ self.queue.clone()  # (B, queue_size)

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # (B, 1+K)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        self.dequeue_enqueue(k)

        return loss


# ── BYOL ──

class BYOL(nn.Module):
    def __init__(self, hidden_dim=128, projection_dim=64, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        # Online network
        self.online_encoder = Encoder(hidden_dim)
        self.online_projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Target network (EMA)
        self.target_encoder = Encoder(hidden_dim)
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim), nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Initialize target with online
        for p_online, p_target in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            p_target.data.copy_(p_online.data)
            p_target.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for p_online, p_target in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            p_target.data = self.momentum * p_target.data + (1 - self.momentum) * p_online.data

    def forward(self, x1, x2):
        # Online
        z1_online = self.online_projector(self.online_encoder(x1))
        p1 = self.predictor(z1_online)

        z2_online = self.online_projector(self.online_encoder(x2))
        p2 = self.predictor(z2_online)

        # Target (no grad)
        with torch.no_grad():
            self.momentum_update()
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))

        # Symmetric loss
        loss = (F.mse_loss(F.normalize(p1, dim=-1), F.normalize(z2_target, dim=-1).detach()) +
                F.mse_loss(F.normalize(p2, dim=-1), F.normalize(z1_target, dim=-1).detach())) / 2

        return loss


# ── Linear Probe ──

def linear_probe(encoder, train_loader, test_loader, n_epochs=5, lr=1e-3, device='cpu'):
    """Train linear classifier on frozen features."""
    encoder.eval()
    classifier = nn.Linear(128, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                features = encoder(bx)
            logits = classifier(features)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            features = encoder(bx)
            preds = classifier(features).argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.shape[0]

    encoder.train()
    return correct / total


# ── Training ──

def train_model(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, _ in train_loader:
            bx = bx.to(device)
            x1 = augment_batch(bx)
            x2 = augment_batch(bx)

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))
        print(f"  Epoch {epoch+1} | Loss: {losses[-1]:.4f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "27-contrastive"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    n_epochs = 15

    # 1. SimCLR
    print("=== Training SimCLR ===")
    simclr = SimCLR(hidden_dim=128).to(device)
    simclr_losses = train_model(simclr, train_loader, n_epochs, device=device)
    simclr_acc = linear_probe(simclr.encoder, train_loader, test_loader, device=device)

    # 2. MoCo
    print("\n=== Training MoCo ===")
    moco = MoCo(hidden_dim=128, queue_size=256).to(device)
    moco_losses = train_model(moco, train_loader, n_epochs, device=device)
    moco_acc = linear_probe(moco.encoder_q, train_loader, test_loader, device=device)

    # 3. BYOL
    print("\n=== Training BYOL ===")
    byol = BYOL(hidden_dim=128, projection_dim=64).to(device)
    byol_losses = train_model(byol, train_loader, n_epochs, lr=1e-4, device=device)
    byol_acc = linear_probe(byol.online_encoder, train_loader, test_loader, device=device)

    # 4. Supervised baseline
    print("\n=== Supervised Baseline ===")
    sup_encoder = Encoder(128).to(device)
    sup_classifier = nn.Linear(128, 10).to(device)
    optimizer = torch.optim.Adam(
        list(sup_encoder.parameters()) + list(sup_classifier.parameters()), lr=1e-3
    )
    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            features = sup_encoder(bx)
            logits = sup_classifier(features)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    sup_acc = linear_probe(sup_encoder, train_loader, test_loader, device=device)

    # Random baseline
    random_encoder = Encoder(128).to(device)
    random_acc = linear_probe(random_encoder, train_loader, test_loader, n_epochs=20, device=device)

    print(f"\n=== Linear Probe Accuracy ===")
    print(f"  Random:    {random_acc:.3f}")
    print(f"  SimCLR:    {simclr_acc:.3f}")
    print(f"  MoCo:      {moco_acc:.3f}")
    print(f"  BYOL:      {byol_acc:.3f}")
    print(f"  Supervised:{sup_acc:.3f}")

    # ── Visualization ──

    # 1. Training loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(simclr_losses, label='SimCLR', color='blue')
    ax.plot(moco_losses, label='MoCo', color='orange')
    ax.plot(byol_losses, label='BYOL', color='green')
    ax.set_title("Contrastive Learning: Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # 2. Linear probe accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Random', 'SimCLR', 'MoCo', 'BYOL', 'Supervised']
    accs = [random_acc, simclr_acc, moco_acc, byol_acc, sup_acc]
    colors = ['gray', 'blue', 'orange', 'green', 'red']
    bars = ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Linear Probe Accuracy")
    ax.set_title("Contrastive Learning: Representation Quality")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "linear_probe_accuracy.png", dpi=150)
    plt.close()

    # 3. Feature space PCA
    try:
        from sklearn.decomposition import PCA

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        test_batch = next(iter(test_loader))
        imgs, labels = test_batch[0][:500].to(device), test_batch[1][:500]

        encoders = {
            'Random': random_encoder,
            'SimCLR': simclr.encoder,
            'MoCo': moco.encoder_q,
            'BYOL': byol.online_encoder,
            'Supervised': sup_encoder,
        }

        for idx, (name, enc) in enumerate(encoders.items()):
            ax = axes[idx // 3, idx % 3]
            enc.eval()
            with torch.no_grad():
                features = enc(imgs).cpu().numpy()
            pca = PCA(n_components=2).fit_transform(features)
            scatter = ax.scatter(pca[:, 0], pca[:, 1], c=labels, cmap='tab10', alpha=0.5, s=3)
            ax.set_title(f"{name}")
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        axes[1, 2].axis('off')

        plt.suptitle("Feature Space (PCA): Contrastive Learning Methods", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / "feature_space_pca.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  PCA visualization failed: {e}")

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("SimCLR", "Large batch negatives\nNT-Xent loss\nSimple but needs\nbig batch size", 0.14, 'blue'),
        ("MoCo", "Momentum encoder\nQueue of negatives\nWorks with small\nbatch size", 0.5, 'orange'),
        ("BYOL", "No negatives needed!\nEMA target + predictor\nAsymmetric architecture\n→ avoid collapse", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Contrastive Learning: Three Paradigms", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "contrastive_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
