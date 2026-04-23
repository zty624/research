"""
Minimal DINO (Self-Distillation with No Labels) Reproduction
============================================================
Reproduces core ideas from DINO (2104.14294, Caron et al., 2021):
1. Student-teacher self-distillation: student learns from teacher's output distribution
2. Centering: subtract EMA of batch means from teacher logits to prevent collapse
3. Sharpening: low-temperature softmax on teacher (tau_t=0.04-0.07) for peaked distributions
4. Student temperature tau_s=0.1 for smoother distributions
5. Teacher updated by EMA of student weights (lambda cosine 0.996->1)
6. Stop-gradient on teacher (no backprop through teacher)
7. Multi-crop: 2 global + N local crops; teacher sees only global crops
8. Evaluate: k-NN accuracy on learned features, collapse monitoring, attention maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Augmentations (CIFAR-10) ──

class CIFARAugmentation:
    """Simplified multi-crop augmentation for CIFAR-10.
    Global crops: standard augmentations at full resolution.
    Local crops: smaller random crops for local-to-global matching.
    """
    def __init__(self, n_local_crops=4, local_scale=(0.3, 0.5), global_scale=(0.8, 1.0)):
        self.n_local_crops = n_local_crops
        self.local_scale = local_scale
        self.global_scale = global_scale

    def _random_crop(self, x, scale_range):
        """Random crop + resize to original size."""
        B, C, H, W = x.shape
        scale = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
        crop_h, crop_w = int(H * scale), int(W * scale)
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        cropped = x[:, :, top:top+crop_h, left:left+crop_w]
        # Resize back to original size
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)

    def _augment(self, x, is_global=True):
        """Apply augmentations."""
        scale = self.global_scale if is_global else self.local_scale
        out = self._random_crop(x, scale)
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            out = torch.flip(out, dims=[3])
        # Color jitter (simplified: brightness + contrast)
        brightness = 1 + (torch.rand(1).item() - 0.5) * 0.4
        contrast = 1 + (torch.rand(1).item() - 0.5) * 0.4
        out = out * brightness * contrast
        # Gaussian blur (simplified via small conv)
        if torch.rand(1).item() > 0.5:
            kernel_size = 3
            sigma = 1.0
            # Simple box blur as approximation
            padding = kernel_size // 2
            avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
            out = avg_pool(out)
        # Gaussian noise
        out = out + torch.randn_like(out) * 0.02
        return out.clamp(0, 1)

    def __call__(self, x):
        """Return list of crops: 2 global + N local."""
        global_crops = [self._augment(x, is_global=True) for _ in range(2)]
        local_crops = [self._augment(x, is_global=False) for _ in range(self.n_local_crops)]
        return global_crops, local_crops


# ── Backbone: Small CNN for CIFAR-10 ──

class SmallCNN(nn.Module):
    """Lightweight CNN backbone for CIFAR-10 (32x32).
    Returns feature maps for attention visualization.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, hidden_dim)

        # Store attention weights for visualization
        self.attention_maps = None

    def forward(self, x, return_attention=False):
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 32, 32, 32)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 64, 16, 16)
        x = F.relu(self.bn3(self.conv3(x)))   # (B, 128, 8, 8)
        feat = F.relu(self.bn4(self.conv4(x))) # (B, 256, 4, 4)

        # Attention: channel-wise mean as spatial attention map
        if return_attention:
            self.attention_maps = feat.mean(dim=1, keepdim=True)  # (B, 1, 4, 4)

        x = self.pool(feat).flatten(1)  # (B, 256)
        x = self.fc(x)                  # (B, hidden_dim)
        return x

    def get_attention(self, x):
        """Get spatial attention map for visualization."""
        self.eval()
        with torch.no_grad():
            self.forward(x, return_attention=True)
            attn = self.attention_maps
        self.train()
        return attn


# ── DINO Head ──

class DINOHead(nn.Module):
    """DINO projection head: 3-layer MLP -> L2 normalize -> linear (prototypes).
    This maps backbone features to a probability distribution over prototypes.
    """
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=256, bottleneck_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # Initialize last layer with small weights (paper uses weight_norm but we skip for simplicity)
        nn.init.normal_(self.last_layer.weight, std=0.02)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)  # L2 normalize bottleneck features
        x = self.last_layer(x)       # Project to prototype space
        return x


# ── DINO Model ──

class DINO(nn.Module):
    """DINO: Self-Distillation with No Labels.

    Student-teacher framework where:
    - Student processes all crops (global + local)
    - Teacher processes only global crops
    - Teacher is updated via EMA of student
    - Centering prevents collapse; sharpening gives peaked distributions
    """
    def __init__(self, hidden_dim=256, out_dim=256,
                 teacher_temp=0.07, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        # Student
        self.student_backbone = SmallCNN(hidden_dim)
        self.student_head = DINOHead(hidden_dim, hidden_dim * 2, out_dim)

        # Teacher (same architecture, no gradients)
        self.teacher_backbone = SmallCNN(hidden_dim)
        self.teacher_head = DINOHead(hidden_dim, hidden_dim * 2, out_dim)

        # Initialize teacher = student
        for sp, tp in zip(self.student_backbone.parameters(),
                          self.teacher_backbone.parameters()):
            tp.data.copy_(sp.data)
        for sp, tp in zip(self.student_head.parameters(),
                          self.teacher_head.parameters()):
            tp.data.copy_(sp.data)

        # Freeze teacher
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Centering vector (EMA of batch means)
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.center_momentum = center_momentum

        # Temperatures
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center: EMA of batch means.
        c <- m*c + (1-m)*mean(teacher_output)
        """
        batch_mean = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_mean

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA update: theta_t <- lambda*theta_t + (1-lambda)*theta_s"""
        for sp, tp in zip(self.student_backbone.parameters(),
                          self.teacher_backbone.parameters()):
            tp.data = momentum * tp.data + (1 - momentum) * sp.data
        for sp, tp in zip(self.student_head.parameters(),
                          self.teacher_head.parameters()):
            tp.data = momentum * tp.data + (1 - momentum) * sp.data

    def forward(self, global_crops, local_crops):
        """
        Args:
            global_crops: list of 2 tensors, each (B, C, H, W)
            local_crops: list of N tensors, each (B, C, H, W)
        Returns:
            loss: cross-entropy between teacher and student distributions
        """
        # Teacher forward (only global crops, no grad)
        teacher_outs = []
        with torch.no_grad():
            for crop in global_crops:
                feat = self.teacher_backbone(crop)
                out = self.teacher_head(feat)
                # Centering + sharpening
                out = (out - self.center) / self.teacher_temp
                teacher_out = F.softmax(out, dim=-1)
                teacher_outs.append(teacher_out)

        # Student forward (all crops)
        student_outs = []
        for crop in global_crops + local_crops:
            feat = self.student_backbone(crop)
            out = self.student_head(feat)
            # Student: no centering, uses student temperature
            student_out = F.log_softmax(out / self.student_temp, dim=-1)
            student_outs.append(student_out)

        # Compute loss: cross-entropy between teacher (global) and student (all)
        # Each student crop should match each teacher global crop
        n_global = len(global_crops)
        total_loss = 0
        n_loss_terms = 0
        for t_idx in range(n_global):
            for s_idx in range(len(student_outs)):
                # Skip same-view pairs for global crops
                if s_idx == t_idx:
                    continue
                loss = -torch.sum(teacher_outs[t_idx] * student_outs[s_idx], dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss = total_loss / n_loss_terms

        # Update center after computing loss
        with torch.no_grad():
            all_teacher_logits = []
            for crop in global_crops:
                feat = self.teacher_backbone(crop)
                out = self.teacher_head(feat)
                all_teacher_logits.append(out)
            concat_logits = torch.cat(all_teacher_logits, dim=0)
            self.update_center(concat_logits)

        return total_loss


# ── EMA Schedule ──

def cosine_schedule(base_value, final_value, total_steps, step):
    """Cosine schedule for teacher EMA momentum (0.996 -> 1.0)."""
    if step >= total_steps:
        return final_value
    return final_value + (base_value - final_value) * \
        (1 + math.cos(math.pi * step / total_steps)) / 2


# ── k-NN Evaluation ──

@torch.no_grad()
def knn_accuracy(backbone, train_loader, test_loader, k=10, device='cpu'):
    """Compute k-NN accuracy using learned features."""
    backbone.eval()

    # Build feature bank
    train_features = []
    train_labels = []
    for bx, by in train_loader:
        bx = bx.to(device)
        feat = backbone(bx)
        feat = F.normalize(feat, dim=-1)
        train_features.append(feat.cpu())
        train_labels.append(by)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Evaluate
    correct = 0
    total = 0
    for bx, by in test_loader:
        bx = bx.to(device)
        feat = backbone(bx)
        feat = F.normalize(feat, dim=-1)
        # Compute similarity
        sim = feat.cpu() @ train_features.T  # (B_test, B_train)
        # Get top-k neighbors
        _, topk_idx = sim.topk(k, dim=1)  # (B_test, k)
        topk_labels = train_labels[topk_idx]  # (B_test, k)
        # Vote
        for i in range(topk_labels.shape[0]):
            counts = torch.bincount(topk_labels[i], minlength=10)
            pred = counts.argmax().item()
            if pred == by[i].item():
                correct += 1
            total += 1

    backbone.train()
    return correct / max(total, 1)


# ── Collapse Detection ──

@torch.no_grad()
def check_collapse(model, dataloader, device='cpu'):
    """Check for representation collapse.
    Returns: std of feature dimensions, effective rank, max entropy.
    """
    model.student_backbone.eval()
    features = []
    for bx, _ in dataloader:
        bx = bx.to(device)
        feat = model.student_backbone(bx)
        features.append(feat.cpu())
    features = torch.cat(features, dim=0)  # (N, D)

    # Per-dimension standard deviation (collapsed = near 0 for many dims)
    std_per_dim = features.std(dim=0)

    # Effective rank via singular values
    try:
        sv = torch.linalg.svdvals(features)
        sv_norm = sv / sv.sum()
        effective_rank = (-sv_norm * (sv_norm + 1e-10).log()).sum().exp().item()
    except Exception:
        effective_rank = 0.0

    # Output distribution entropy (how spread are the prototype assignments)
    model.student_head.eval()
    with torch.no_grad():
        sample = features[:256].to(device)
        logits = model.student_head(sample)
        probs = F.softmax(logits, dim=-1)
        avg_prob = probs.mean(dim=0)
        entropy = -(avg_prob * (avg_prob + 1e-10).log()).sum().item()
        max_entropy = math.log(logits.shape[-1])
    model.student_head.train()
    model.student_backbone.train()

    return std_per_dim.mean().item(), effective_rank, entropy, max_entropy


# ── Training ──

def train_dino(model, train_loader, test_loader, augmentor,
               n_epochs=20, lr=1e-3, device='cpu', eval_every=5):
    """Train DINO model."""
    # Separate student parameters (teacher has no grad)
    optimizer = torch.optim.AdamW(
        list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    total_steps = n_epochs * len(train_loader)
    global_step = 0

    losses = []
    knn_accs = []
    collapse_metrics = []  # (step, mean_std, eff_rank, entropy, max_entropy)
    ema_momentums = []
    center_norms = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0

        for bx, _ in train_loader:
            bx = bx.to(device)

            # Multi-crop augmentation
            global_crops, local_crops = augmentor(bx)

            # Forward + loss
            loss = model(global_crops, local_crops)

            # Backward on student only
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
                max_norm=3.0
            )
            optimizer.step()

            # EMA update teacher
            ema_mom = cosine_schedule(0.996, 1.0, total_steps, global_step)
            model.update_teacher(ema_mom)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Track center norm
            center_norms.append(model.center.norm().item())
            ema_momentums.append(ema_mom)

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        # Evaluate k-NN and collapse
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            acc = knn_accuracy(model.student_backbone, train_loader, test_loader, k=10, device=device)
            knn_accs.append((epoch, acc))
            mstd, erank, ent, maxent = check_collapse(model, test_loader, device=device)
            collapse_metrics.append((epoch, mstd, erank, ent, maxent))
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"k-NN: {acc:.3f} | EMA: {ema_mom:.6f} | "
                  f"Std: {mstd:.4f} | Rank: {erank:.1f} | Ent: {ent:.3f}/{maxent:.3f}")
        else:
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | EMA: {ema_mom:.6f}")

    return {
        'losses': losses,
        'knn_accs': knn_accs,
        'collapse_metrics': collapse_metrics,
        'ema_momentums': ema_momentums,
        'center_norms': center_norms,
    }


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "63-dino"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Use CIFAR-10 with a subset for speed
    print("Loading CIFAR-10...")
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # Subset for faster training
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=0)
    # Small subset for k-NN train bank
    knn_train_subset = torch.utils.data.Subset(train_dataset, range(2000))
    knn_train_loader = torch.utils.data.DataLoader(knn_train_subset, batch_size=256, num_workers=0)

    # ── Train DINO ──
    print("\n=== Training DINO ===")
    augmentor = CIFARAugmentation(n_local_crops=4, local_scale=(0.3, 0.5), global_scale=(0.8, 1.0))
    model = DINO(
        hidden_dim=256, out_dim=256,
        teacher_temp=0.07, student_temp=0.1,
        center_momentum=0.9,
    ).to(device)

    n_params = sum(p.numel() for p in model.student_backbone.parameters()) + \
               sum(p.numel() for p in model.student_head.parameters())
    print(f"  Student parameters: {n_params:,}")

    results = train_dino(
        model, knn_train_loader, test_loader, augmentor,
        n_epochs=30, lr=1e-3, device=device, eval_every=5,
    )

    # ── Baseline: supervised CNN ──
    print("\n=== Training Supervised Baseline ===")
    sup_model = SmallCNN(256).to(device)
    sup_classifier = nn.Linear(256, 10).to(device)
    optimizer = torch.optim.AdamW(
        list(sup_model.parameters()) + list(sup_classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    for epoch in range(30):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            feat = sup_model(bx)
            loss = F.cross_entropy(sup_classifier(feat), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f}")

    sup_acc = knn_accuracy(sup_model, knn_train_loader, test_loader, k=10, device=device)
    print(f"  Supervised k-NN accuracy: {sup_acc:.3f}")

    # Random baseline
    random_model = SmallCNN(256).to(device)
    random_acc = knn_accuracy(random_model, knn_train_loader, test_loader, k=10, device=device)
    print(f"  Random k-NN accuracy: {random_acc:.3f}")

    dino_final_acc = results['knn_accs'][-1][1] if results['knn_accs'] else 0
    print(f"  DINO k-NN accuracy: {dino_final_acc:.3f}")

    # ── Visualization ──

    # 1. Training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results['losses'], color='steelblue', linewidth=2)
    ax.set_title("DINO Training Loss", fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # 2. k-NN accuracy over training
    if results['knn_accs']:
        epochs_knn = [e for e, _ in results['knn_accs']]
        accs_knn = [a for _, a in results['knn_accs']]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_knn, accs_knn, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.axhline(y=random_acc, color='gray', linestyle='--', alpha=0.7, label=f'Random ({random_acc:.3f})')
        ax.axhline(y=sup_acc, color='red', linestyle='--', alpha=0.7, label=f'Supervised ({sup_acc:.3f})')
        ax.set_title("DINO: k-NN Accuracy During Training", fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("k-NN Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(sup_acc + 0.1, 1.0))
        plt.tight_layout()
        plt.savefig(results_dir / "knn_accuracy.png", dpi=150)
        plt.close()

    # 3. Collapse analysis (multi-panel)
    if results['collapse_metrics']:
        cm = results['collapse_metrics']
        epochs_cm = [x[0] for x in cm]
        mean_stds = [x[1] for x in cm]
        eff_ranks = [x[2] for x in cm]
        entropies = [x[3] for x in cm]
        max_entropies = [x[4] for x in cm]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Feature std per dimension
        axes[0].plot(epochs_cm, mean_stds, 'o-', color='teal', linewidth=2, markersize=5)
        axes[0].set_title("Feature Dimension Std (higher = less collapse)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean Std Across Dims")
        axes[0].grid(True, alpha=0.3)

        # Effective rank
        axes[1].plot(epochs_cm, eff_ranks, 'o-', color='purple', linewidth=2, markersize=5)
        axes[1].set_title("Effective Rank of Features")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Effective Rank")
        axes[1].grid(True, alpha=0.3)

        # Output entropy vs max entropy
        axes[2].plot(epochs_cm, entropies, 'o-', color='orange', linewidth=2, markersize=5, label='Actual')
        axes[2].plot(epochs_cm, max_entropies, 's--', color='red', linewidth=1, markersize=4, label='Max (uniform)')
        axes[2].set_title("Output Distribution Entropy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Entropy")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("DINO: Collapse Prevention Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / "collapse_analysis.png", dpi=150)
        plt.close()

    # 4. EMA momentum schedule and center norm
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # EMA momentum
    axes[0].plot(results['ema_momentums'], color='steelblue', linewidth=1, alpha=0.8)
    axes[0].set_title("Teacher EMA Momentum (cosine 0.996 -> 1.0)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Lambda")
    axes[0].grid(True, alpha=0.3)

    # Center norm
    axes[1].plot(results['center_norms'], color='teal', linewidth=1, alpha=0.8)
    axes[1].set_title("Centering Vector Norm")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("||c||")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DINO: Training Dynamics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "training_dynamics.png", dpi=150)
    plt.close()

    # 5. Attention maps from student backbone
    print("\n=== Visualizing Attention Maps ===")
    model.student_backbone.eval()
    test_batch = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        attn_maps = model.student_backbone.get_attention(test_batch)  # (8, 1, 4, 4)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(8):
        # Original image
        img = test_batch[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Input", fontsize=11)

        # Attention map (upsample to image size)
        attn = attn_maps[i, 0].cpu().numpy()
        attn_up = F.interpolate(
            attn_maps[i:i+1], size=(32, 32), mode='bilinear', align_corners=False
        )[0, 0].cpu().numpy()
        axes[1, i].imshow(img, alpha=0.3)
        axes[1, i].imshow(attn_up, cmap='jet', alpha=0.7)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel("Attention", fontsize=11)

    plt.suptitle("DINO: Self-Attention Maps (Emergent Object Segmentation)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "attention_maps.png", dpi=150)
    plt.close()

    # 6. Feature space PCA
    print("\n=== Feature Space Visualization ===")
    try:
        from sklearn.decomposition import PCA

        model.student_backbone.eval()
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                feat = model.student_backbone(bx)
                feat = F.normalize(feat, dim=-1)
                all_feats.append(feat.cpu().numpy())
                all_labels.append(by.numpy())

        feats = np.concatenate(all_feats)[:1000]
        labels = np.concatenate(all_labels)[:1000]

        pca = PCA(n_components=2)
        feats_2d = pca.fit_transform(feats)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # DINO features
        scatter = axes[0].scatter(feats_2d[:, 0], feats_2d[:, 1], c=labels,
                                  cmap='tab10', alpha=0.4, s=5)
        axes[0].set_title("DINO Learned Features (PCA)", fontweight='bold')
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Class')

        # Random features
        random_model.eval()
        rand_feats = []
        rand_labels = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                feat = random_model(bx)
                rand_feats.append(feat.cpu().numpy())
                rand_labels.append(by.numpy())
        rand_feats = np.concatenate(rand_feats)[:1000]
        rand_labels = np.concatenate(rand_labels)[:1000]
        rand_2d = PCA(n_components=2).fit_transform(rand_feats)

        scatter2 = axes[1].scatter(rand_2d[:, 0], rand_2d[:, 1], c=rand_labels,
                                   cmap='tab10', alpha=0.4, s=5)
        axes[1].set_title("Random Encoder Features (PCA)", fontweight='bold')
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Class')

        plt.suptitle("DINO vs Random: Feature Space Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / "feature_space_pca.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  PCA visualization failed: {e}")

    # 7. Final accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Random', 'DINO', 'Supervised']
    accs = [random_acc, dino_final_acc, sup_acc]
    colors = ['gray', 'steelblue', 'red']
    bars = ax.bar(methods, accs, color=colors, alpha=0.8)
    ax.set_ylabel("k-NN Accuracy")
    ax.set_title("DINO: Representation Quality (k-NN on CIFAR-10)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(max(accs) + 0.1, 1.0))
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    # 8. DINO concept diagram
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    # Student path
    student_steps = [
        ("Input\nImage", 0.04, 'gray'),
        ("Multi-Crop\n2G + 4L", 0.15, 'teal'),
        ("Student\nBackbone", 0.28, 'steelblue'),
        ("Student\nHead", 0.40, 'steelblue'),
        ("Softmax\n(tau_s=0.1)", 0.52, 'cyan'),
    ]

    # Teacher path
    teacher_steps = [
        ("Input\nImage", 0.04, 'gray'),
        ("Global Crops\nOnly (2)", 0.15, 'teal'),
        ("Teacher\nBackbone", 0.28, 'darkorange'),
        ("Teacher\nHead", 0.40, 'darkorange'),
        ("Center + Sharpen\n(c, tau_t=0.07)", 0.52, 'orange'),
    ]

    # Shared
    shared_steps = [
        ("Cross-Entropy\nLoss", 0.67, 'red'),
        ("EMA Update\n(0.996->1)", 0.82, 'green'),
    ]

    # Draw student path (top)
    for name, x_pos, color in student_steps:
        ax.text(x_pos, 0.75, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Draw teacher path (bottom)
    for name, x_pos, color in teacher_steps:
        ax.text(x_pos, 0.30, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Draw shared steps (middle)
    for name, x_pos, color in shared_steps:
        ax.text(x_pos, 0.52, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Arrows student
    for x in [0.09, 0.21, 0.34, 0.46]:
        ax.annotate('', xy=(x + 0.03, 0.75), xytext=(x, 0.75),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    # Arrows teacher
    for x in [0.09, 0.21, 0.34, 0.46]:
        ax.annotate('', xy=(x + 0.03, 0.30), xytext=(x, 0.30),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    # Student/teacher -> loss
    ax.annotate('', xy=(0.64, 0.63), xytext=(0.57, 0.70),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))
    ax.annotate('', xy=(0.64, 0.42), xytext=(0.57, 0.35),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    # Loss -> EMA
    ax.annotate('', xy=(0.77, 0.52), xytext=(0.72, 0.52),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # EMA -> teacher (feedback)
    ax.annotate('', xy=(0.82, 0.42), xytext=(0.88, 0.42),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5, linestyle='dashed'))

    # Stop-grad label
    ax.text(0.34, 0.15, "Stop Gradient", fontsize=9, ha='center', va='center',
            color='red', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='mistyrose', alpha=0.8))

    # Labels
    ax.text(0.40, 0.90, "STUDENT (learned via gradient)", fontsize=11,
            ha='center', color='steelblue', fontweight='bold')
    ax.text(0.40, 0.10, "TEACHER (EMA of student, no gradient)", fontsize=11,
            ha='center', color='darkorange', fontweight='bold')

    ax.set_title("DINO: Self-Distillation with No Labels (2104.14294)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "dino_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
