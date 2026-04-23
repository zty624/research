"""
Minimal BYOL (Bootstrapping Your Own Latent) Reproduction
=========================================================
Reproduces core ideas from BYOL (2006.07733, Grill et al., 2020):
1. Two networks: online (encoder + projector + predictor) and target (encoder + projector, EMA updated)
2. No negative pairs needed -- avoids collapse via predictor + EMA asymmetry
3. Online network predicts target network's output from a different augmented view
4. Target encoder updated by exponential moving average of online encoder
5. Key insight: the predictor + EMA creates sufficient asymmetry to prevent collapse
6. Synthetic data with known cluster structure for controlled evaluation
7. Show: representation quality over training, EMA decay effect, collapse analysis
8. Compare: BYOL vs SimCLR (with negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Synthetic Data Generator ──

class SyntheticClusterDataset(torch.utils.data.Dataset):
    """Generate synthetic image-like data with known cluster structure.
    Each class has a distinct pattern (rotated stripes, colored shapes, etc.)
    so we can measure how well representations capture this structure.
    """
    def __init__(self, n_samples=5000, n_classes=10, img_size=32, seed=42):
        super().__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        rng = np.random.RandomState(seed)

        # Generate class-specific template patterns
        self.templates = []
        for c in range(n_classes):
            template = np.zeros((3, img_size, img_size), dtype=np.float32)
            # Each class gets a unique stripe pattern with distinct color + orientation
            angle = c * (180 / n_classes)  # rotation angle in degrees
            # Color: distinct hue per class
            color = np.array([
                (math.sin(c * 2 * math.pi / n_classes) + 1) / 2,
                (math.sin(c * 2 * math.pi / n_classes + 2 * math.pi / 3) + 1) / 2,
                (math.sin(c * 2 * math.pi / n_classes + 4 * math.pi / 3) + 1) / 2,
            ], dtype=np.float32)
            # Draw rotated stripes
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            period = 6 + c % 4  # vary stripe period
            for i in range(img_size):
                for j in range(img_size):
                    # Rotated coordinate
                    x_rot = (i - img_size / 2) * cos_a - (j - img_size / 2) * sin_a
                    stripe = 0.5 + 0.5 * math.sin(x_rot * math.pi / period)
                    template[:, i, j] = color * stripe
            self.templates.append(template)

        # Generate samples: template + noise + random transformations
        self.data = []
        self.labels = []
        for i in range(n_samples):
            c = i % n_classes
            img = self.templates[c].copy()
            # Add Gaussian noise
            img += rng.randn(3, img_size, img_size).astype(np.float32) * 0.15
            # Random brightness shift
            img += rng.uniform(-0.1, 0.1, (1, 1, 1)).astype(np.float32)
            # Random contrast
            img *= rng.uniform(0.8, 1.2, (1, 1, 1)).astype(np.float32)
            img = np.clip(img, 0, 1)
            self.data.append(img)
            self.labels.append(c)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


# ── Augmentations ──

def augment_batch(x, strength=0.3):
    """Simple augmentations: noise + crop + brightness + color jitter."""
    B, C, H, W = x.shape
    aug = x.clone()

    # Gaussian noise
    aug = aug + torch.randn_like(aug) * strength * 0.3

    # Random brightness
    brightness = 1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * strength
    aug = aug * brightness

    # Random contrast
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * strength * 0.5
    mean = aug.mean(dim=(2, 3), keepdim=True)
    aug = (aug - mean) * contrast + mean

    # Random crop (shift by up to 3 pixels)
    shift_h = torch.randint(-3, 4, (1,)).item()
    shift_w = torch.randint(-3, 4, (1,)).item()
    aug = torch.roll(aug, shifts=(shift_h, shift_w), dims=(2, 3))

    # Random horizontal flip
    flip_mask = torch.rand(B, 1, 1, 1, device=x.device) > 0.5
    aug = torch.where(flip_mask, torch.flip(aug, dims=[3]), aug)

    return aug.clamp(0, 1)


# ── Encoder ──

class Encoder(nn.Module):
    """Simple CNN encoder for image-like data."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        h = self.cnn(x).flatten(1)
        return self.fc(h)


# ── BYOL Model ──

class BYOL(nn.Module):
    """BYOL: Bootstrapping Your Own Latent.

    Online network: encoder -> projector -> predictor
    Target network: encoder -> projector (EMA of online, no gradient)

    The predictor creates asymmetry that prevents collapse.
    Without it, the two networks could converge to a constant output.
    """
    def __init__(self, hidden_dim=128, projection_dim=64, prediction_dim=64,
                 ema_decay=0.996):
        super().__init__()
        self.ema_decay = ema_decay

        # Online network
        self.online_encoder = Encoder(hidden_dim)
        self.online_projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(),
            nn.Linear(prediction_dim, projection_dim),
        )

        # Target network (EMA, no gradients)
        self.target_encoder = Encoder(hidden_dim)
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Initialize target = online
        for p_on, p_tgt in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            p_tgt.data.copy_(p_on.data)
            p_tgt.requires_grad = False

    @torch.no_grad()
    def ema_update(self):
        """EMA update: target <- ema_decay * target + (1 - ema_decay) * online."""
        for p_on, p_tgt in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            p_tgt.data = self.ema_decay * p_tgt.data + (1 - self.ema_decay) * p_on.data

    def forward(self, x1, x2):
        """Compute BYOL loss.

        Args:
            x1, x2: two augmented views of the same batch

        Returns:
            BYOL loss (scalar)
        """
        # Online forward
        z1_online = self.online_projector(self.online_encoder(x1))
        p1 = self.online_predictor(z1_online)

        z2_online = self.online_projector(self.online_encoder(x2))
        p2 = self.online_predictor(z2_online)

        # Target forward (no grad)
        with torch.no_grad():
            self.ema_update()
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))

        # Symmetric loss: cosine similarity between prediction and target
        loss = (self._cosine_loss(p1, z2_target) + self._cosine_loss(p2, z1_target)) / 2
        return loss

    def _cosine_loss(self, pred, target):
        """Cosine similarity loss: 2 - 2 * cos_sim(pred, target)."""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()


# ── SimCLR Model ──

class SimCLR(nn.Module):
    """SimCLR: contrastive learning with NT-Xent loss (for comparison)."""
    def __init__(self, hidden_dim=128, projection_dim=64, temperature=0.5):
        super().__init__()
        self.encoder = Encoder(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.temperature = temperature

    def forward(self, x1, x2):
        z1 = F.normalize(self.projector(self.encoder(x1)), dim=-1)
        z2 = F.normalize(self.projector(self.encoder(x2)), dim=-1)

        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = z @ z.T / self.temperature  # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Labels: each z1_i matches z2_i
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)
        loss = F.cross_entropy(sim, labels)
        return loss


# ── Linear Probe ──

def linear_probe(encoder, train_loader, test_loader, n_epochs=10, lr=1e-3,
                 device='cpu', hidden_dim=128):
    """Train linear classifier on frozen features."""
    encoder.eval()
    classifier = nn.Linear(hidden_dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    for _ in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                features = encoder(bx)
            logits = classifier(features)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            features = encoder(bx)
            preds = classifier(features).argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.shape[0]

    encoder.train()
    return correct / max(total, 1)


# ── Collapse Detection ──

@torch.no_grad()
def check_collapse(encoder, dataloader, device='cpu'):
    """Check for representation collapse.
    Returns: std per dimension, effective rank, singular value spectrum.
    """
    encoder.eval()
    features = []
    for bx, _ in dataloader:
        bx = bx.to(device)
        feat = encoder(bx)
        features.append(feat.cpu())
    features = torch.cat(features, dim=0)  # (N, D)

    # Per-dimension standard deviation (collapsed = near 0)
    std_per_dim = features.std(dim=0)

    # Singular value spectrum
    try:
        sv = torch.linalg.svdvals(features)
        sv_norm = sv / sv.sum()
        effective_rank = (-sv_norm * (sv_norm + 1e-10).log()).sum().exp().item()
    except Exception:
        sv = torch.ones(features.shape[1])
        effective_rank = 0.0

    encoder.train()
    return std_per_dim.mean().item(), effective_rank, sv.numpy()


# ── Training Functions ──

def train_byol(model, train_loader, n_epochs=20, lr=1e-3, device='cpu', verbose=True):
    """Train BYOL model."""
    # Collect trainable parameters (predictor may not exist in ablation)
    params = (list(model.online_encoder.parameters()) +
              list(model.online_projector.parameters()))
    if hasattr(model, 'online_predictor'):
        params += list(model.online_predictor.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []
    probe_accs = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for bx, _ in train_loader:
            bx = bx.to(device)
            x1 = augment_batch(bx)
            x2 = augment_batch(bx)

            loss = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

    return losses


def train_simclr(model, train_loader, n_epochs=20, lr=1e-3, device='cpu', verbose=True):
    """Train SimCLR model."""
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    for epoch in range(n_epochs):
        model.train()
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

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "100-byol"
    results_dir.mkdir(parents=True, exist_ok=True)

    hidden_dim = 128
    projection_dim = 64
    n_epochs = 30

    # ── Generate Synthetic Data ──
    print("=== Generating Synthetic Cluster Data ===")
    train_dataset = SyntheticClusterDataset(n_samples=5000, n_classes=10, img_size=32, seed=42)
    test_dataset = SyntheticClusterDataset(n_samples=1000, n_classes=10, img_size=32, seed=123)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=0)
    print(f"  Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples | Classes: 10")

    # Visualize a few samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c in range(10):
        idx = c  # First sample of each class
        img = train_dataset.data[idx].transpose(1, 2, 0)
        axes[c // 5, c % 5].imshow(img)
        axes[c // 5, c % 5].set_title(f"Class {c}", fontsize=9)
        axes[c // 5, c % 5].axis('off')
    plt.suptitle("Synthetic Cluster Data: 10 Classes with Distinct Patterns", fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "synthetic_data_samples.png", dpi=150)
    plt.close()

    # ── Experiment 1: Train BYOL ──
    print("\n=== Training BYOL ===")
    byol = BYOL(hidden_dim=hidden_dim, projection_dim=projection_dim,
                prediction_dim=64, ema_decay=0.996).to(device)
    n_params = sum(p.numel() for p in byol.online_encoder.parameters()) + \
               sum(p.numel() for p in byol.online_projector.parameters()) + \
               sum(p.numel() for p in byol.online_predictor.parameters())
    print(f"  Online network params: {n_params:,}")

    # Track representation quality during training
    byol_probe_accs = []
    byol_collapse_metrics = []
    eval_epochs = list(range(0, n_epochs, 3)) + [n_epochs - 1]

    optimizer = torch.optim.Adam(
        list(byol.online_encoder.parameters()) +
        list(byol.online_projector.parameters()) +
        list(byol.online_predictor.parameters()),
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    byol_losses = []

    for epoch in range(n_epochs):
        byol.train()
        epoch_loss = 0
        n_batches = 0

        for bx, _ in train_loader:
            bx = bx.to(device)
            x1 = augment_batch(bx)
            x2 = augment_batch(bx)
            loss = byol(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(byol.online_encoder.parameters()) +
                list(byol.online_projector.parameters()) +
                list(byol.online_predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        byol_losses.append(avg_loss)

        if epoch in eval_epochs:
            acc = linear_probe(byol.online_encoder, train_loader, test_loader,
                               device=device, hidden_dim=hidden_dim)
            byol_probe_accs.append((epoch, acc))
            mstd, erank, _ = check_collapse(byol.online_encoder, test_loader, device=device)
            byol_collapse_metrics.append((epoch, mstd, erank))
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Probe: {acc:.3f} | Std: {mstd:.4f} | Rank: {erank:.1f}")
        elif (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

    byol_final_acc = byol_probe_accs[-1][1] if byol_probe_accs else 0

    # ── Experiment 2: Train SimCLR ──
    print("\n=== Training SimCLR ===")
    simclr = SimCLR(hidden_dim=hidden_dim, projection_dim=projection_dim,
                    temperature=0.5).to(device)
    simclr_losses = train_simclr(simclr, train_loader, n_epochs=n_epochs, device=device)
    simclr_acc = linear_probe(simclr.encoder, train_loader, test_loader,
                              device=device, hidden_dim=hidden_dim)
    print(f"  SimCLR linear probe accuracy: {simclr_acc:.4f}")

    # ── Experiment 3: Supervised baseline ──
    print("\n=== Training Supervised Baseline ===")
    sup_encoder = Encoder(hidden_dim).to(device)
    sup_classifier = nn.Linear(hidden_dim, 10).to(device)
    optimizer = torch.optim.Adam(
        list(sup_encoder.parameters()) + list(sup_classifier.parameters()), lr=1e-3
    )
    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            feat = sup_encoder(bx)
            loss = F.cross_entropy(sup_classifier(feat), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    sup_acc = linear_probe(sup_encoder, train_loader, test_loader,
                           device=device, hidden_dim=hidden_dim)
    print(f"  Supervised probe accuracy: {sup_acc:.4f}")

    # Random baseline
    random_encoder = Encoder(hidden_dim).to(device)
    random_acc = linear_probe(random_encoder, train_loader, test_loader,
                              n_epochs=20, device=device, hidden_dim=hidden_dim)
    print(f"  Random probe accuracy: {random_acc:.4f}")

    # ── Experiment 4: EMA decay effect ──
    print("\n=== EMA Decay Effect ===")
    ema_decays = [0.9, 0.99, 0.996, 0.999, 1.0]
    ema_decay_accs = {}
    ema_decay_losses = {}
    ema_decay_collapse = {}

    for ema_d in ema_decays:
        print(f"  EMA decay = {ema_d}:")
        model = BYOL(hidden_dim=hidden_dim, projection_dim=projection_dim,
                     prediction_dim=64, ema_decay=ema_d).to(device)
        losses = train_byol(model, train_loader, n_epochs=n_epochs, lr=1e-3,
                            device=device, verbose=False)
        acc = linear_probe(model.online_encoder, train_loader, test_loader,
                           device=device, hidden_dim=hidden_dim)
        mstd, erank, _ = check_collapse(model.online_encoder, test_loader, device=device)

        ema_decay_accs[ema_d] = acc
        ema_decay_losses[ema_d] = losses
        ema_decay_collapse[ema_d] = (mstd, erank)
        print(f"    Probe acc: {acc:.4f} | Std: {mstd:.4f} | Rank: {erank:.1f}")

    # ── Experiment 5: Ablation -- BYOL without predictor (should collapse) ──
    print("\n=== Ablation: BYOL without Predictor (expect collapse) ===")

    class BYOLNoPredictor(nn.Module):
        """BYOL without predictor -- should collapse."""
        def __init__(self, hidden_dim=128, projection_dim=64, ema_decay=0.996):
            super().__init__()
            self.ema_decay = ema_decay
            self.online_encoder = Encoder(hidden_dim)
            self.online_projector = nn.Sequential(
                nn.Linear(hidden_dim, projection_dim), nn.BatchNorm1d(projection_dim),
                nn.ReLU(), nn.Linear(projection_dim, projection_dim),
            )
            self.target_encoder = Encoder(hidden_dim)
            self.target_projector = nn.Sequential(
                nn.Linear(hidden_dim, projection_dim), nn.BatchNorm1d(projection_dim),
                nn.ReLU(), nn.Linear(projection_dim, projection_dim),
            )
            for p_on, p_tgt in zip(
                list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
                list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
            ):
                p_tgt.data.copy_(p_on.data)
                p_tgt.requires_grad = False

        @torch.no_grad()
        def ema_update(self):
            for p_on, p_tgt in zip(
                list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
                list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
            ):
                p_tgt.data = self.ema_decay * p_tgt.data + (1 - self.ema_decay) * p_on.data

        def forward(self, x1, x2):
            z1_online = self.online_projector(self.online_encoder(x1))
            z2_online = self.online_projector(self.online_encoder(x2))
            with torch.no_grad():
                self.ema_update()
                z1_target = self.target_projector(self.target_encoder(x1))
                z2_target = self.target_projector(self.target_encoder(x2))
            loss = (self._cosine_loss(z1_online, z2_target) +
                    self._cosine_loss(z2_online, z1_target)) / 2
            return loss

        def _cosine_loss(self, pred, target):
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)
            return 2 - 2 * (pred * target).sum(dim=-1).mean()

    byol_no_pred = BYOLNoPredictor(hidden_dim, projection_dim, ema_decay=0.996).to(device)
    no_pred_losses = train_byol(byol_no_pred, train_loader, n_epochs=n_epochs, lr=1e-3,
                                device=device, verbose=False)
    no_pred_acc = linear_probe(byol_no_pred.online_encoder, train_loader, test_loader,
                               device=device, hidden_dim=hidden_dim)
    no_pred_mstd, no_pred_erank, no_pred_sv = check_collapse(
        byol_no_pred.online_encoder, test_loader, device=device)
    print(f"  No predictor: Probe acc={no_pred_acc:.4f} | Std={no_pred_mstd:.4f} | Rank={no_pred_erank:.1f}")

    # Also check BYOL's SV spectrum
    _, _, byol_sv = check_collapse(byol.online_encoder, test_loader, device=device)

    # ── Visualization ──

    # 1. Training loss: BYOL vs SimCLR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(byol_losses, color='steelblue', linewidth=2, label='BYOL')
    ax.plot(simclr_losses, color='darkorange', linewidth=2, label='SimCLR')
    ax.set_title("Training Loss: BYOL vs SimCLR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss_comparison.png", dpi=150)
    plt.close()

    # 2. Representation quality over training
    if byol_probe_accs:
        epochs_probe = [e for e, _ in byol_probe_accs]
        accs_probe = [a for _, a in byol_probe_accs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_probe, accs_probe, 'o-', color='steelblue', linewidth=2,
                markersize=5, label='BYOL')
        ax.axhline(y=random_acc, color='gray', linestyle='--', alpha=0.7,
                    label=f'Random ({random_acc:.3f})')
        ax.axhline(y=simclr_acc, color='darkorange', linestyle='--', alpha=0.7,
                    label=f'SimCLR ({simclr_acc:.3f})')
        ax.axhline(y=sup_acc, color='red', linestyle='--', alpha=0.7,
                    label=f'Supervised ({sup_acc:.3f})')
        ax.set_title("BYOL: Representation Quality Over Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Linear Probe Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(sup_acc + 0.1, 1.0))
        plt.tight_layout()
        plt.savefig(results_dir / "representation_quality_over_training.png", dpi=150)
        plt.close()

    # 3. Final accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ['Random', 'BYOL\n(no predictor)', 'SimCLR', 'BYOL', 'Supervised']
    accs = [random_acc, no_pred_acc, simclr_acc, byol_final_acc, sup_acc]
    colors = ['gray', 'salmon', 'darkorange', 'steelblue', 'red']
    bars = ax.bar(methods, accs, color=colors, alpha=0.8)
    ax.set_ylabel("Linear Probe Accuracy")
    ax.set_title("Representation Quality: BYOL vs SimCLR vs Baselines")
    ax.set_ylim(0, max(max(accs) * 1.2, 0.3))
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    # 4. EMA decay effect
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss by EMA decay
    ema_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    for idx, ema_d in enumerate(ema_decays):
        label = f"EMA={ema_d}"
        axes[0].plot(ema_decay_losses[ema_d], label=label, color=ema_colors[idx])
    axes[0].set_title("Training Loss by EMA Decay")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Probe accuracy by EMA decay
    ema_labels = [f"{d}" for d in ema_decays]
    ema_accs = [ema_decay_accs[d] for d in ema_decays]
    bars = axes[1].bar(ema_labels, ema_accs, color=ema_colors, alpha=0.8)
    axes[1].set_title("Linear Probe Accuracy by EMA Decay")
    axes[1].set_xlabel("EMA Decay")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, ema_accs):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Effective rank by EMA decay
    ema_ranks = [ema_decay_collapse[d][1] for d in ema_decays]
    bars = axes[2].bar(ema_labels, ema_ranks, color=ema_colors, alpha=0.8)
    axes[2].set_title("Effective Rank by EMA Decay")
    axes[2].set_xlabel("EMA Decay")
    axes[2].set_ylabel("Effective Rank")
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, ema_ranks):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.5,
                     f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')

    plt.suptitle("BYOL: Effect of EMA Decay on Training and Collapse Prevention",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "ema_decay_effect.png", dpi=150)
    plt.close()

    # 5. Collapse analysis: SV spectrum comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_sv = min(50, len(byol_sv))
    axes[0].plot(range(n_sv), byol_sv[:n_sv] / byol_sv[0], 'o-',
                 color='steelblue', markersize=3, label='BYOL (with predictor)')
    axes[0].plot(range(n_sv), no_pred_sv[:n_sv] / no_pred_sv[0], 'o-',
                 color='salmon', markersize=3, label='BYOL (no predictor)')
    axes[0].set_title("Singular Value Spectrum")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Normalized SV")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Collapse metrics during training
    if byol_collapse_metrics:
        cm = byol_collapse_metrics
        epochs_cm = [x[0] for x in cm]
        mean_stds = [x[1] for x in cm]
        eff_ranks = [x[2] for x in cm]

        axes[1].plot(epochs_cm, mean_stds, 'o-', color='teal', linewidth=2,
                     markersize=5, label='Mean Std')
        ax2 = axes[1].twinx()
        ax2.plot(epochs_cm, eff_ranks, 's-', color='purple', linewidth=2,
                 markersize=5, label='Eff. Rank')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean Std Across Dims", color='teal')
        ax2.set_ylabel("Effective Rank", color='purple')
        axes[1].set_title("BYOL: Collapse Prevention During Training")
        axes[1].grid(True, alpha=0.3)
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.suptitle("BYOL: Collapse Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "collapse_analysis.png", dpi=150)
    plt.close()

    # 6. Feature space PCA
    print("\n=== Feature Space Visualization ===")
    try:
        from sklearn.decomposition import PCA

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        test_imgs = []
        test_labels = []
        for bx, by in test_loader:
            test_imgs.append(bx)
            test_labels.append(by)
        test_imgs = torch.cat(test_imgs)[:500].to(device)
        test_labels = torch.cat(test_labels)[:500]

        encoders = {
            'Random': (random_encoder, 0, 0),
            'SimCLR': (simclr.encoder, 0, 1),
            'BYOL': (byol.online_encoder, 1, 0),
            'Supervised': (sup_encoder, 1, 1),
        }

        for name, (enc, r, c) in encoders.items():
            enc.eval()
            with torch.no_grad():
                features = enc(test_imgs).cpu().numpy()
            pca = PCA(n_components=2).fit_transform(features)
            axes[r, c].scatter(pca[:, 0], pca[:, 1], c=test_labels.numpy(),
                               cmap='tab10', alpha=0.4, s=5)
            axes[r, c].set_title(f"{name}", fontweight='bold')
            axes[r, c].grid(True, alpha=0.3)
            axes[r, c].set_xlabel("PC1")
            axes[r, c].set_ylabel("PC2")

        plt.suptitle("Feature Space (PCA): BYOL vs SimCLR vs Baselines",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / "feature_space_pca.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  PCA visualization failed: {e}")

    # 7. BYOL architecture concept diagram
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')

    # Online path (top)
    online_steps = [
        ("View 1\n(x1)", 0.04, 0.82, 'lightblue'),
        ("Online\nEncoder\nf_\\theta", 0.16, 0.82, 'paleturquoise'),
        ("Online\nProjector\ng_\\theta", 0.28, 0.82, 'paleturquoise'),
        ("Online\nPredictor\nq_\\theta", 0.40, 0.82, 'thistle'),
        ("Prediction\np1 = q(g(f(x1)))", 0.54, 0.82, 'plum'),
    ]

    # Target path (bottom)
    target_steps = [
        ("View 2\n(x2)", 0.04, 0.25, 'lightyellow'),
        ("Target\nEncoder\nf_\\xi", 0.16, 0.25, 'navajowhite'),
        ("Target\nProjector\ng_\\xi", 0.28, 0.25, 'navajowhite'),
        ("Target\nOutput\nz2' = g(f(x2))", 0.40, 0.25, 'moccasin'),
    ]

    # Shared
    shared_steps = [
        ("Cosine\nLoss\n2 - 2*cos(p1, z2')", 0.62, 0.55, 'lightsalmon'),
        ("EMA Update\n\\xi <- \\tau*\\xi + (1-\\tau)*\\theta", 0.80, 0.55, 'lightgreen'),
    ]

    for name, x, y, color in online_steps + target_steps + shared_steps:
        ax.text(x, y, name, fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                          edgecolor='gray', alpha=0.9))

    # Arrows: online path
    for x in [0.09, 0.22, 0.34, 0.47]:
        ax.annotate('', xy=(x + 0.03, 0.82), xytext=(x, 0.82),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    # Arrows: target path
    for x in [0.09, 0.22, 0.34]:
        ax.annotate('', xy=(x + 0.03, 0.25), xytext=(x, 0.25),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    # Prediction + target -> loss
    ax.annotate('', xy=(0.57, 0.65), xytext=(0.60, 0.78),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))
    ax.annotate('', xy=(0.57, 0.42), xytext=(0.45, 0.28),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    # Loss -> EMA
    ax.annotate('', xy=(0.73, 0.55), xytext=(0.69, 0.55),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # EMA -> target (feedback loop)
    ax.annotate('', xy=(0.85, 0.40), xytext=(0.88, 0.48),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='dashed'))

    # Labels
    ax.text(0.30, 0.95, "ONLINE network (trained with gradient)",
            fontsize=11, ha='center', color='steelblue', fontweight='bold')
    ax.text(0.22, 0.10, "TARGET network (EMA of online, NO gradient)",
            fontsize=11, ha='center', color='darkorange', fontweight='bold')

    # Key insight
    ax.text(0.50, 0.02,
            "Key Insight: No negative pairs needed! Predictor + EMA creates asymmetry that prevents collapse\n"
            "Unlike SimCLR, BYOL does NOT need large batches or negative samples",
            fontsize=10, ha='center', va='center', style='italic', color='darkblue',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    ax.set_title("BYOL: Bootstrapping Your Own Latent (2006.07733)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "byol_architecture.png", dpi=150)
    plt.close()

    # 8. Ablation: with vs without predictor
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(byol_losses, color='steelblue', linewidth=2, label='BYOL (with predictor)')
    axes[0].plot(no_pred_losses, color='salmon', linewidth=2, label='BYOL (no predictor)')
    axes[0].set_title("Training Loss: With vs Without Predictor")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy comparison
    abl_methods = ['No Predictor', 'BYOL', 'SimCLR', 'Supervised']
    abl_accs = [no_pred_acc, byol_final_acc, simclr_acc, sup_acc]
    abl_colors = ['salmon', 'steelblue', 'darkorange', 'red']
    bars = axes[1].bar(abl_methods, abl_accs, color=abl_colors, alpha=0.8)
    axes[1].set_ylabel("Linear Probe Accuracy")
    axes[1].set_title("Predictor Ablation: Why BYOL Needs the Predictor")
    axes[1].set_ylim(0, max(max(abl_accs) * 1.2, 0.3))
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, abl_accs):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f'{v:.3f}', ha='center', fontweight='bold')

    plt.suptitle("BYOL: Predictor is Critical for Preventing Collapse",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "predictor_ablation.png", dpi=150)
    plt.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("BYOL Experiment Summary")
    print("=" * 60)
    print(f"Dataset: Synthetic cluster data (5K train, 1K test, 10 classes)")
    print(f"\nRepresentation Quality (Linear Probe):")
    print(f"  Random:           {random_acc:.4f}")
    print(f"  BYOL (no pred):   {no_pred_acc:.4f} (collapsed without predictor)")
    print(f"  SimCLR:           {simclr_acc:.4f} (needs negatives)")
    print(f"  BYOL:             {byol_final_acc:.4f} (no negatives needed!)")
    print(f"  Supervised:       {sup_acc:.4f}")
    print(f"\nEMA Decay Effect:")
    for ema_d in ema_decays:
        mstd, erank = ema_decay_collapse[ema_d]
        print(f"  EMA={ema_d}: acc={ema_decay_accs[ema_d]:.4f}, "
              f"std={mstd:.4f}, rank={erank:.1f}")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
