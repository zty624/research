"""
Minimal Self-Supervised Learning Reproduction
===============================================
Reproduces core ideas from contrastive and non-contrastive SSL:
1. SimCLR: contrastive learning with data augmentations + projection head
2. BYOL: bootstrapping without negative pairs
3. MAE: masked autoencoding (mask patches, reconstruct)
4. Compare learned representations via linear probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Encoder ──

class SimpleEncoder(nn.Module):
    """Simple CNN encoder for 28x28 images."""
    def __init__(self, hidden=128, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        h = F.relu(self.fc(h))
        return self.out(h)

    def features(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)


# ── Augmentations ──

def augment_batch(x, strength=0.5):
    """Simple augmentations for MNIST: random crop + noise."""
    B, C, H, W = x.shape
    # Random shift
    pad = 4
    x_pad = F.pad(x, [pad]*4, mode='constant', value=0)
    offsets_h = torch.randint(0, 2*pad, (B,))
    offsets_w = torch.randint(0, 2*pad, (B,))
    x_aug = torch.zeros_like(x)
    for i in range(B):
        x_aug[i] = x_pad[i, :, offsets_h[i]:offsets_h[i]+H, offsets_w[i]:offsets_w[i]+W]

    # Add noise
    x_aug = x_aug + strength * torch.randn_like(x_aug) * 0.1
    return x_aug.clamp(0, 1)


# ── SimCLR ──

class SimCLR(nn.Module):
    def __init__(self, hidden=128, out_dim=64, temperature=0.5):
        super().__init__()
        self.encoder = SimpleEncoder(hidden, out_dim)
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.temperature = temperature

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return z1, z2

    def nt_xent_loss(self, z1, z2):
        """Normalized Temperature-scaled Cross Entropy (NT-Xent)."""
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = z @ z.T / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2*B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ── BYOL ──

class BYOL(nn.Module):
    def __init__(self, hidden=128, out_dim=64, ema_decay=0.99):
        super().__init__()
        self.online_encoder = SimpleEncoder(hidden, out_dim)
        self.online_projector = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

        # Target network (EMA)
        self.target_encoder = SimpleEncoder(hidden, out_dim)
        self.target_projector = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.ema_decay = ema_decay

        # Initialize target = online
        self._init_target()

    def _init_target(self):
        for tp, op in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            tp.data.copy_(op.data)
        for tp, op in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            tp.data.copy_(op.data)

    @torch.no_grad()
    def update_target(self):
        for tp, op in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        for tp, op in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

    def forward(self, x1, x2):
        # Online
        online_z1 = self.predictor(self.online_projector(self.online_encoder(x1)))
        online_z2 = self.predictor(self.online_projector(self.online_encoder(x2)))

        # Target (no grad)
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_encoder(x1))
            target_z2 = self.target_projector(self.target_encoder(x2))

        return online_z1, online_z2, target_z1, target_z2

    def loss(self, online_z1, online_z2, target_z1, target_z2):
        """BYOL loss: cosine similarity between online prediction and target projection."""
        loss1 = 2 - 2 * F.cosine_similarity(online_z1, target_z2.detach(), dim=-1).mean()
        loss2 = 2 - 2 * F.cosine_similarity(online_z2, target_z1.detach(), dim=-1).mean()
        return loss1 + loss2


# ── MAE (Masked Autoencoder) ──

class MAE(nn.Module):
    def __init__(self, hidden=128, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = SimpleEncoder(hidden, hidden)
        # Decoder: reconstruct full image from masked + visible patches
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, 784)
        )
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, hidden))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # Random mask: keep mask_ratio of pixels
        mask = torch.rand(B, 784, device=x.device) < self.mask_ratio
        visible = ~mask

        # Encode only visible pixels (simplified: encode full image, but zero out masked)
        x_masked = x_flat * visible.float()
        encoded = self.encoder(x_masked.view(B, 1, 28, 28))

        # Decode to reconstruct full image
        reconstructed = self.decoder(encoded)
        return reconstructed, mask

    def loss(self, x, reconstructed, mask):
        """MSE loss only on masked pixels."""
        x_flat = x.view(x.shape[0], -1)
        loss = (F.mse_loss(reconstructed, x_flat, reduction='none') * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss


# ── Training Functions ──

def train_simclr(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, _ in train_loader:
            bx = bx.to(device)
            x1 = augment_batch(bx)
            x2 = augment_batch(bx)
            z1, z2 = model(x1, x2)
            loss = model.nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    return losses


def train_byol(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(list(model.online_encoder.parameters()) +
                                   list(model.online_projector.parameters()) +
                                   list(model.predictor.parameters()), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, _ in train_loader:
            bx = bx.to(device)
            x1 = augment_batch(bx)
            x2 = augment_batch(bx)
            oz1, oz2, tz1, tz2 = model(x1, x2)
            loss = model.loss(oz1, oz2, tz1, tz2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    return losses


def train_mae(model, train_loader, n_epochs=10, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, _ in train_loader:
            bx = bx.to(device)
            recon, mask = model(bx)
            loss = model.loss(bx, recon, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    return losses


def linear_probe(encoder, train_loader, test_loader, n_epochs=5, lr=1e-3, device='cpu'):
    """Train a linear classifier on frozen encoder features."""
    classifier = nn.Linear(128, 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                feats = encoder.features(bx)
            loss = F.cross_entropy(classifier(feats), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            feats = encoder.features(bx)
            pred = classifier(feats).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += by.shape[0]
    return correct / total


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "17-ssl"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    n_epochs = 10

    # Train SimCLR
    print("=== Training SimCLR ===")
    simclr = SimCLR(hidden=128, out_dim=64).to(device)
    simclr_losses = train_simclr(simclr, train_loader, n_epochs, device=device)
    simclr_acc = linear_probe(simclr.encoder, train_loader, test_loader, device=device)
    print(f"  SimCLR linear probe: {simclr_acc:.4f}")

    # Train BYOL
    print("\n=== Training BYOL ===")
    byol = BYOL(hidden=128, out_dim=64).to(device)
    byol_losses = train_byol(byol, train_loader, n_epochs, device=device)
    byol_acc = linear_probe(byol.online_encoder, train_loader, test_loader, device=device)
    print(f"  BYOL linear probe: {byol_acc:.4f}")

    # Train MAE
    print("\n=== Training MAE ===")
    mae = MAE(hidden=128).to(device)
    mae_losses = train_mae(mae, train_loader, n_epochs, device=device)
    mae_acc = linear_probe(mae.encoder, train_loader, test_loader, device=device)
    print(f"  MAE linear probe: {mae_acc:.4f}")

    # Random baseline
    print("\n=== Random Encoder Baseline ===")
    random_enc = SimpleEncoder(128, 64).to(device)
    random_acc = linear_probe(random_enc, train_loader, test_loader, device=device)
    print(f"  Random linear probe: {random_acc:.4f}")

    # Supervised baseline
    print("\n=== Supervised Baseline ===")
    sup_enc = SimpleEncoder(128, 64).to(device)
    sup_classifier = nn.Linear(128, 10).to(device)
    optimizer = torch.optim.AdamW(list(sup_enc.parameters()) + list(sup_classifier.parameters()), lr=1e-3)
    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            feats = sup_enc.features(bx)
            loss = F.cross_entropy(sup_classifier(feats), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            feats = sup_enc.features(bx)
            pred = sup_classifier(feats).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += by.shape[0]
    sup_acc = correct / total
    print(f"  Supervised: {sup_acc:.4f}")

    # ── Visualization ──

    # 1. Training losses
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(simclr_losses, label='SimCLR (NT-Xent)', color='blue')
    axes[0].set_title("SimCLR Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(byol_losses, label='BYOL (Cosine)', color='green')
    axes[1].set_title("BYOL Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(mae_losses, label='MAE (Reconstruction)', color='purple')
    axes[2].set_title("MAE Training")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Self-Supervised Learning: Training Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_losses.png", dpi=150)
    plt.close()

    # 2. Linear probe comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ['Random', 'SimCLR', 'BYOL', 'MAE', 'Supervised']
    accs = [random_acc, simclr_acc, byol_acc, mae_acc, sup_acc]
    colors = ['gray', 'blue', 'green', 'purple', 'red']
    ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Linear Probe Accuracy")
    ax.set_title("Self-Supervised Learning: Representation Quality")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "linear_probe_comparison.png", dpi=150)
    plt.close()

    # 3. MAE reconstruction visualization
    mae.eval()
    test_batch = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        recon, mask = mae(test_batch)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(test_batch[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].view(28, 28).cpu(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.suptitle("MAE: Masked Reconstruction", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "mae_reconstruction.png", dpi=150)
    plt.close()

    # 4. Feature space visualization (PCA)
    from sklearn.decomposition import PCA

    encoders = {
        'SimCLR': simclr.encoder,
        'BYOL': byol.online_encoder,
        'MAE': mae.encoder,
        'Random': random_enc,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (name, enc) in enumerate(encoders.items()):
        ax = axes[idx // 2][idx % 2]
        enc.eval()
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for bx, by in test_loader:
                feats = enc.features(bx.to(device)).cpu().numpy()
                all_feats.append(feats)
                all_labels.append(by.numpy())

        feats = np.concatenate(all_feats)[:500]
        labels = np.concatenate(all_labels)[:500]

        pca = PCA(n_components=2)
        feats_2d = pca.fit_transform(feats)

        scatter = ax.scatter(feats_2d[:, 0], feats_2d[:, 1], c=labels,
                           cmap='tab10', alpha=0.3, s=3)
        ax.set_title(f"{name} Features (PCA)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Learned Feature Spaces (colored by digit class)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "feature_spaces.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
