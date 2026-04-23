"""
Minimal I-JEPA (Joint-Embedding Predictive Architecture) Reproduction
=====================================================================
Reproduces core ideas from I-JEPA (2301.08243, Assran et al., 2023):
1. Predict representations of masked regions from visible regions (not pixels!)
2. No negative pairs needed (vs contrastive learning like SimCLR)
3. Context encoder processes visible patches, predictor predicts masked representations
4. Target encoder (EMA of context encoder) provides the representation targets
5. Multi-block masking strategy (not random patch masking like MAE)
6. Compare: I-JEPA vs MAE-style pixel reconstruction baseline
7. Show: prediction quality at different mask ratios, representation quality via linear probe
8. Key insight: predicting in representation space avoids low-level pixel bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Patch Embedding ──

class PatchEmbed(nn.Module):
    """Split image into patches and project to embedding dimension."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)            # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)       # (B, n_patches, embed_dim)
        return x


# ── Transformer Blocks ──

class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ── ViT Encoder ──

class ViTEncoder(nn.Module):
    """Vision Transformer encoder that processes patches."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        """Encode patches. If ids_keep given, only process those patches."""
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        B, N, D = x.shape

        pos = self.pos_embed.expand(B, -1, -1)

        if ids_keep is not None:
            ids_exp = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x = torch.gather(x, dim=1, index=ids_exp)
            pos = torch.gather(pos, dim=1, index=ids_exp)

        x = self.pos_drop(x + pos)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# ── I-JEPA Predictor ──

class IJEPAPredictor(nn.Module):
    """Predicts target representations for masked positions.
    Takes context encoder outputs for visible patches + mask tokens for
    masked positions, and predicts what the target encoder would produce.
    """
    def __init__(self, n_patches=64, embed_dim=128, predictor_dim=64,
                 depth=2, n_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n_patches = n_patches
        self.predictor_dim = predictor_dim

        # Project from context encoder dim to predictor dim
        self.context_proj = nn.Linear(embed_dim, predictor_dim)

        # Mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding for all patches
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, predictor_dim) * 0.02)

        # Predictor transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, n_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        # Project predictor output to match target encoder dimension
        self.pred_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, context_output, ids_keep, ids_restore):
        """Predict representations for masked positions.

        Args:
            context_output: (B, N_keep, embed_dim) context encoder output for visible patches
            ids_keep: (B, N_keep) indices of visible patches
            ids_restore: (B, N) indices to restore original ordering

        Returns:
            (B, N, embed_dim) predicted representations for all patches
        """
        B = context_output.shape[0]
        N = self.n_patches
        N_keep = context_output.shape[1]
        N_mask = N - N_keep

        # Project context output to predictor dim
        ctx = self.context_proj(context_output)  # (B, N_keep, pred_dim)

        # Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        # Concatenate visible + mask tokens (in shuffled order)
        x = torch.cat([ctx, mask_tokens], dim=1)  # (B, N, pred_dim)

        # Unshuffle to restore original patch ordering
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        x = torch.gather(x, dim=1, index=ids_restore_exp)

        # Add positional embeddings
        x = x + self.pos_embed

        # Predictor transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Project back to encoder dim
        x = self.pred_proj(x)
        return x


# ── I-JEPA Model ──

class IJEPA(nn.Module):
    """I-JEPA: Joint-Embedding Predictive Architecture.

    Context encoder processes visible patches.
    Predictor predicts target encoder representations for masked patches.
    Target encoder is EMA-updated from context encoder.
    Loss: MSE between predicted and target representations on masked patches.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, encoder_depth=4, predictor_depth=2,
                 n_heads=4, mlp_ratio=4.0, mask_ratio=0.75,
                 ema_decay=0.996, dropout=0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        n_patches = (img_size // patch_size) ** 2

        # Context encoder (trained with gradients)
        self.context_encoder = ViTEncoder(
            img_size, patch_size, in_channels,
            embed_dim, encoder_depth, n_heads, mlp_ratio, dropout
        )

        # Target encoder (EMA of context encoder, no gradients)
        self.target_encoder = ViTEncoder(
            img_size, patch_size, in_channels,
            embed_dim, encoder_depth, n_heads, mlp_ratio, dropout
        )
        # Initialize target = context
        for cp, tp in zip(self.context_encoder.parameters(),
                          self.target_encoder.parameters()):
            tp.data.copy_(cp.data)
            tp.requires_grad = False

        # Predictor
        self.predictor = IJEPAPredictor(
            n_patches, embed_dim, predictor_dim=64,
            depth=predictor_depth, n_heads=n_heads, mlp_ratio=mlp_ratio
        )

        self.ema_decay = ema_decay
        self.n_patches = n_patches

    @torch.no_grad()
    def ema_update(self):
        """EMA update target encoder from context encoder."""
        for cp, tp in zip(self.context_encoder.parameters(),
                          self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * cp.data

    def random_masking(self, x, mask_ratio):
        """Random masking with shuffle/unshuffle trick."""
        B, N, D = x.shape
        N_keep = max(1, int(N * (1 - mask_ratio)))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :N_keep]

        # Binary mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward(self, x):
        """I-JEPA forward pass.

        Returns:
            pred: (B, N, embed_dim) predicted representations for all patches
            target: (B, N, embed_dim) target encoder representations for all patches
            mask: (B, N) binary mask (1=masked, 0=visible)
        """
        patches = self.context_encoder.patch_embed(x)

        # Determine masking
        ids_keep, ids_restore, mask = self.random_masking(patches, self.mask_ratio)

        # Context encoder: only visible patches
        context_out = self.context_encoder(x, ids_keep)  # (B, N_keep, D)

        # Target encoder: all patches (no grad)
        with torch.no_grad():
            target_out = self.target_encoder(x)  # (B, N, D)

        # Predictor: predict representations for all positions
        pred = self.predictor(context_out, ids_keep, ids_restore)  # (B, N, D)

        return pred, target_out, mask

    def loss(self, pred, target, mask):
        """MSE loss on masked patch representations only.

        This is the key difference from MAE: we predict in representation space,
        not pixel space. No decoder needed for pixel reconstruction.
        """
        # Normalize predictions and targets (following I-JEPA paper)
        pred = F.layer_norm(pred, pred.shape[-1:])
        target = F.layer_norm(target, target.shape[-1:])

        loss_per_patch = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # (B, N)
        masked_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)
        return masked_loss

    def get_representations(self, x):
        """Get context encoder representations for linear probing."""
        # Use all patches (no masking) for evaluation
        return self.context_encoder(x).mean(dim=1)  # (B, D)


# ── MAE-style Pixel Reconstruction Baseline ──

class MAEBaseline(nn.Module):
    """Simplified MAE baseline that predicts pixels (not representations).
    Same encoder architecture as I-JEPA for fair comparison.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, encoder_depth=4, decoder_dim=64,
                 decoder_depth=2, n_heads=4, mlp_ratio=4.0,
                 mask_ratio=0.75, dropout=0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        n_patches = (img_size // patch_size) ** 2

        self.encoder = ViTEncoder(
            img_size, patch_size, in_channels,
            embed_dim, encoder_depth, n_heads, mlp_ratio, dropout
        )

        # Lightweight decoder
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, n_patches, decoder_dim) * 0.02
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, n_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        self.patch_dim = patch_size * patch_size * in_channels
        self.pred_head = nn.Linear(decoder_dim, self.patch_dim)
        self.n_patches = n_patches

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        N_keep = max(1, int(N * (1 - mask_ratio)))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :N_keep]
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, ids_restore, mask

    def forward(self, x):
        patches = self.encoder.patch_embed(x)
        ids_keep, ids_restore, mask = self.random_masking(patches, self.mask_ratio)

        encoded = self.encoder(x, ids_keep)  # (B, N_keep, D)

        B = encoded.shape[0]
        N = self.n_patches
        N_mask = N - encoded.shape[1]

        encoded = self.enc_to_dec(encoded)
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x_dec = torch.cat([encoded, mask_tokens], dim=1)
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
        x_dec = torch.gather(x_dec, dim=1, index=ids_restore_exp)
        x_dec = x_dec + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x_dec = block(x_dec)
        x_dec = self.decoder_norm(x_dec)
        pred = self.pred_head(x_dec)
        return pred, mask

    def loss(self, x, pred, mask):
        target = self.patchify(x)
        loss_per_patch = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
        masked_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)
        return masked_loss

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, h * w, C * p * p)
        return x

    def unpatchify(self, patches):
        B = patches.shape[0]
        C = self.in_channels
        p = self.patch_size
        h = w = int(self.n_patches ** 0.5)
        x = patches.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, h * p, w * p)
        return x

    def get_representations(self, x):
        patches = self.encoder.patch_embed(x)
        ids_keep, _, _ = self.random_masking(patches, self.mask_ratio)
        encoded = self.encoder(x, ids_keep)
        return encoded.mean(dim=1)


# ── Multi-block Masking (I-JEPA style) ──

def multiblock_mask(B, N, mask_ratio, device, n_blocks=4):
    """Generate multi-block masks as used in I-JEPA.
    Instead of random patch-level masking (like MAE), I-JEPA uses
    contiguous block-level masking for more semantic prediction.
    """
    h = w = int(math.sqrt(N))
    assert h * w == N, f"N={N} is not a perfect square"

    n_mask = int(N * mask_ratio)
    masks = torch.zeros(B, N, device=device)

    for b in range(B):
        masked = set()
        for _ in range(n_blocks):
            if len(masked) >= n_mask:
                break
            # Random block size
            bh = torch.randint(1, max(2, h // 2 + 1), (1,)).item()
            bw = torch.randint(1, max(2, w // 2 + 1), (1,)).item()
            # Random top-left corner
            top = torch.randint(0, max(1, h - bh + 1), (1,)).item()
            left = torch.randint(0, max(1, w - bw + 1), (1,)).item()

            for i in range(top, min(top + bh, h)):
                for j in range(left, min(left + bw, w)):
                    idx = i * w + j
                    if len(masked) < n_mask:
                        masked.add(idx)

        for idx in masked:
            masks[b, idx] = 1.0

    return masks


# ── Linear Probe ──

def linear_probe(model, train_loader, test_loader, n_epochs=10, lr=1e-3,
                 device='cpu', is_ijepa=True):
    """Train linear classifier on frozen encoder features."""
    embed_dim = model.context_encoder.patch_embed.proj.out_channels if is_ijepa \
        else model.encoder.patch_embed.proj.out_channels

    def get_features(loader, n_samples=3):
        all_feats, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                feats_list = []
                for _ in range(n_samples):
                    feat = model.get_representations(bx)
                    feats_list.append(feat)
                feats = torch.stack(feats_list).mean(dim=0)
                all_feats.append(feats.cpu())
                all_labels.append(by)
        return torch.cat(all_feats), torch.cat(all_labels)

    train_feats, train_labels = get_features(train_loader)
    test_feats, test_labels = get_features(test_loader)

    classifier = nn.Linear(embed_dim, 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    for _ in range(n_epochs):
        n = train_feats.shape[0]
        bs = 256
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(classifier(train_feats[idx]), train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = classifier(test_feats).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()

    return acc


# ── Training ──

def train_ijepa(model, train_loader, n_epochs=20, lr=1e-4, weight_decay=0.05,
                device='cpu', verbose=True):
    """Train I-JEPA with AdamW and cosine LR schedule."""
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []
    pred_qualities = []  # cosine similarity between pred and target

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_pred_q = 0
        n_batches = 0

        for bx, _ in train_loader:
            bx = bx.to(device)
            pred, target, mask = model(bx)
            loss = model.loss(pred, target, mask)

            # Measure prediction quality: cosine sim on masked patches
            with torch.no_grad():
                pred_norm = F.normalize(pred, dim=-1)
                target_norm = F.normalize(target, dim=-1)
                cos_sim = (pred_norm * target_norm).sum(dim=-1)  # (B, N)
                masked_cos = (cos_sim * mask).sum() / mask.sum().clamp(min=1)
                epoch_pred_q += masked_cos.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            # EMA update target encoder
            model.ema_update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pred_q = epoch_pred_q / max(n_batches, 1)
        losses.append(avg_loss)
        pred_qualities.append(avg_pred_q)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | "
                  f"Pred CosSim: {avg_pred_q:.4f}")

    return losses, pred_qualities


def train_mae_baseline(model, train_loader, n_epochs=20, lr=1.5e-4,
                       weight_decay=0.05, device='cpu', verbose=True):
    """Train MAE baseline."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for bx, _ in train_loader:
            bx = bx.to(device)
            pred, mask = model(bx)
            loss = model.loss(bx, pred, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "99-ijepa"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=256, num_workers=0)

    n_epochs = 30

    # ── Experiment 1: Train I-JEPA at 75% mask ratio ──
    print("=== Training I-JEPA (75% mask ratio) ===")
    ijepa = IJEPA(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=128, encoder_depth=4, predictor_depth=2,
        n_heads=4, mlp_ratio=4.0, mask_ratio=0.75,
        ema_decay=0.996
    ).to(device)

    n_ctx = sum(p.numel() for p in ijepa.context_encoder.parameters())
    n_tgt = sum(p.numel() for p in ijepa.target_encoder.parameters())
    n_pred = sum(p.numel() for p in ijepa.predictor.parameters())
    print(f"  Patches: {ijepa.n_patches} ({32//4}x{32//4})")
    print(f"  Context encoder: {n_ctx:,} | Target encoder: {n_tgt:,} | Predictor: {n_pred:,}")
    print(f"  At 75% masking: only {int(ijepa.n_patches * 0.25)}/{ijepa.n_patches} patches visible")

    ijepa_losses, ijepa_pred_q = train_ijepa(ijepa, train_loader, n_epochs=n_epochs, device=device)

    print("  Linear probing I-JEPA...")
    ijepa_acc = linear_probe(ijepa, train_loader, test_loader, device=device, is_ijepa=True)
    print(f"  I-JEPA linear probe accuracy: {ijepa_acc:.4f}")

    # ── Experiment 2: Train MAE baseline at 75% mask ratio ──
    print("\n=== Training MAE Baseline (75% mask ratio, pixel reconstruction) ===")
    mae = MAEBaseline(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=128, encoder_depth=4, decoder_dim=64,
        decoder_depth=2, n_heads=4, mlp_ratio=4.0, mask_ratio=0.75
    ).to(device)
    n_enc = sum(p.numel() for p in mae.encoder.parameters())
    n_dec = sum(p.numel() for p in mae.decoder_blocks.parameters())
    print(f"  Encoder: {n_enc:,} | Decoder: {n_dec:,}")

    mae_losses = train_mae_baseline(mae, train_loader, n_epochs=n_epochs, device=device)

    print("  Linear probing MAE...")
    mae_acc = linear_probe(mae, train_loader, test_loader, device=device, is_ijepa=False)
    print(f"  MAE linear probe accuracy: {mae_acc:.4f}")

    # ── Experiment 3: Compare mask ratios for I-JEPA ──
    print("\n=== Comparing Mask Ratios (I-JEPA) ===")
    mask_ratios = [0.3, 0.5, 0.75, 0.9]
    ratio_losses = {}
    ratio_pred_q = {}
    ratio_accs = {}

    for mr in mask_ratios:
        n_visible = max(1, int(ijepa.n_patches * (1 - mr)))
        print(f"  Mask ratio {mr:.0%} ({n_visible}/{ijepa.n_patches} visible):")

        model = IJEPA(
            img_size=32, patch_size=4, in_channels=3,
            embed_dim=128, encoder_depth=4, predictor_depth=2,
            n_heads=4, mlp_ratio=4.0, mask_ratio=mr, ema_decay=0.996
        ).to(device)

        losses, pred_q = train_ijepa(model, train_loader, n_epochs=n_epochs,
                                     device=device, verbose=False)
        acc = linear_probe(model, train_loader, test_loader, device=device, is_ijepa=True)

        ratio_losses[mr] = losses
        ratio_pred_q[mr] = pred_q
        ratio_accs[mr] = acc
        print(f"    Final loss: {losses[-1]:.4f} | Pred CosSim: {pred_q[-1]:.4f} | Probe acc: {acc:.4f}")

    # ── Experiment 4: I-JEPA vs MAE at each mask ratio ──
    print("\n=== I-JEPA vs MAE at Each Mask Ratio ===")
    compare_ratios = [0.3, 0.5, 0.75, 0.9]
    ijepa_ratio_accs = {}
    mae_ratio_accs = {}

    for mr in compare_ratios:
        # I-JEPA
        ijepa_model = IJEPA(
            img_size=32, patch_size=4, in_channels=3,
            embed_dim=128, encoder_depth=4, predictor_depth=2,
            n_heads=4, mask_ratio=mr, ema_decay=0.996
        ).to(device)
        train_ijepa(ijepa_model, train_loader, n_epochs=n_epochs, device=device, verbose=False)
        ijepa_ratio_accs[mr] = linear_probe(ijepa_model, train_loader, test_loader,
                                             device=device, is_ijepa=True)

        # MAE
        mae_model = MAEBaseline(
            img_size=32, patch_size=4, in_channels=3,
            embed_dim=128, encoder_depth=4, decoder_dim=64,
            decoder_depth=2, n_heads=4, mask_ratio=mr
        ).to(device)
        train_mae_baseline(mae_model, train_loader, n_epochs=n_epochs, device=device, verbose=False)
        mae_ratio_accs[mr] = linear_probe(mae_model, train_loader, test_loader,
                                          device=device, is_ijepa=False)

        print(f"  {mr:.0%} mask: I-JEPA={ijepa_ratio_accs[mr]:.4f}, MAE={mae_ratio_accs[mr]:.4f}")

    # ── Visualization ──

    # 1. Training loss: I-JEPA vs MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ijepa_losses, color='steelblue', linewidth=2, label='I-JEPA')
    axes[0].set_title("I-JEPA Training Loss (75% mask)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (representation space)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(mae_losses, color='darkorange', linewidth=2, label='MAE')
    axes[1].set_title("MAE Training Loss (75% mask)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (pixel space)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("I-JEPA vs MAE: Training Loss Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss_comparison.png", dpi=150)
    plt.close()

    # 2. Prediction quality over training (cosine similarity)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ijepa_pred_q, color='steelblue', linewidth=2)
    ax.set_title("I-JEPA: Prediction Quality (Cosine Similarity on Masked Patches)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "prediction_quality.png", dpi=150)
    plt.close()

    # 3. Mask ratio comparison for I-JEPA
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, mr in enumerate(mask_ratios):
        n_visible = max(1, int(ijepa.n_patches * (1 - mr)))
        label = f"{mr:.0%} ({n_visible} visible)"
        axes[0].plot(ratio_losses[mr], label=label, color=colors[idx])
    axes[0].set_title("I-JEPA Training Loss by Mask Ratio")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (representation space)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for idx, mr in enumerate(mask_ratios):
        n_visible = max(1, int(ijepa.n_patches * (1 - mr)))
        label = f"{mr:.0%} ({n_visible} visible)"
        axes[1].plot(ratio_pred_q[mr], label=label, color=colors[idx])
    axes[1].set_title("Prediction Quality by Mask Ratio")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("I-JEPA: Effect of Mask Ratio", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "mask_ratio_comparison.png", dpi=150)
    plt.close()

    # 4. I-JEPA vs MAE linear probe accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart at 75%
    methods = ['I-JEPA', 'MAE']
    accs_75 = [ijepa_acc, mae_acc]
    bar_colors = ['steelblue', 'darkorange']
    bars = axes[0].bar(methods, accs_75, color=bar_colors, alpha=0.8)
    axes[0].set_ylabel("Linear Probe Accuracy")
    axes[0].set_title("I-JEPA vs MAE (75% mask ratio)")
    axes[0].set_ylim(0, max(max(accs_75) * 1.2, 0.3))
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs_75):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f'{v:.3f}', ha='center', fontweight='bold')

    # Across mask ratios
    x = np.arange(len(compare_ratios))
    width = 0.35
    ijepa_vals = [ijepa_ratio_accs[mr] for mr in compare_ratios]
    mae_vals = [mae_ratio_accs[mr] for mr in compare_ratios]
    bars1 = axes[1].bar(x - width/2, ijepa_vals, width, label='I-JEPA',
                        color='steelblue', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, mae_vals, width, label='MAE',
                        color='darkorange', alpha=0.8)
    axes[1].set_xlabel("Mask Ratio")
    axes[1].set_ylabel("Linear Probe Accuracy")
    axes[1].set_title("I-JEPA vs MAE Across Mask Ratios")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{mr:.0%}" for mr in compare_ratios])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars1, ijepa_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    for bar, v in zip(bars2, mae_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    plt.suptitle("Representation Quality: I-JEPA vs MAE", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "ijepa_vs_mae_accuracy.png", dpi=150)
    plt.close()

    # 5. I-JEPA mask ratio probe accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ratios_str = [f"{mr:.0%}" for mr in mask_ratios]
    accs = [ratio_accs[mr] for mr in mask_ratios]
    bars = ax.bar(ratios_str, accs, color=colors, alpha=0.8)
    ax.set_title("I-JEPA: Linear Probe Accuracy by Mask Ratio")
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(accs) * 1.2)
    plt.tight_layout()
    plt.savefig(results_dir / "ijepa_mask_ratio_accuracy.png", dpi=150)
    plt.close()

    # 6. Feature space PCA
    print("\n=== Feature Space Visualization ===")
    try:
        from sklearn.decomposition import PCA

        ijepa.eval()
        mae.eval()

        all_ijepa_feats, all_mae_feats, all_labels = [], [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                all_ijepa_feats.append(ijepa.get_representations(bx).cpu().numpy())
                # Average over multiple masks for MAE
                feats_list = []
                for _ in range(3):
                    feats_list.append(mae.get_representations(bx))
                all_mae_feats.append(torch.stack(feats_list).mean(dim=0).cpu().numpy())
                all_labels.append(by.numpy())

        ijepa_feats = np.concatenate(all_ijepa_feats)[:500]
        mae_feats = np.concatenate(all_mae_feats)[:500]
        labels = np.concatenate(all_labels)[:500]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        pca_ijepa = PCA(n_components=2).fit_transform(ijepa_feats)
        scatter1 = axes[0].scatter(pca_ijepa[:, 0], pca_ijepa[:, 1], c=labels,
                                   cmap='tab10', alpha=0.4, s=5)
        axes[0].set_title("I-JEPA Features (PCA)", fontweight='bold')
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Class')

        pca_mae = PCA(n_components=2).fit_transform(mae_feats)
        scatter2 = axes[1].scatter(pca_mae[:, 0], pca_mae[:, 1], c=labels,
                                   cmap='tab10', alpha=0.4, s=5)
        axes[1].set_title("MAE Features (PCA)", fontweight='bold')
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Class')

        plt.suptitle("I-JEPA vs MAE: Learned Feature Space", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / "feature_space_pca.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  PCA visualization failed: {e}")

    # 7. I-JEPA architecture concept diagram
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis('off')

    # I-JEPA flow
    boxes = [
        ("Input\nImage", 0.03, 0.7, 'lightblue'),
        ("Patch\nEmbed", 0.12, 0.7, 'lightcyan'),
        ("Random\nMask", 0.21, 0.7, 'lightyellow'),
        ("Visible\nPatches", 0.31, 0.85, 'lightgreen'),
        ("Masked\nPatches", 0.31, 0.3, 'mistyrose'),
        ("Context\nEncoder\n(ViT)", 0.44, 0.85, 'paleturquoise'),
        ("Predictor\n(+ mask tokens)", 0.57, 0.55, 'thistle'),
        ("Predicted\nReps", 0.70, 0.55, 'plum'),
        ("Target\nEncoder\n(EMA, no grad)", 0.44, 0.35, 'navajowhite'),
        ("Target\nReps", 0.57, 0.35, 'moccasin'),
        ("MSE Loss\n(repr space)", 0.75, 0.45, 'lightsalmon'),
        ("EMA\nUpdate", 0.87, 0.6, 'lightgreen'),
    ]

    for name, x, y, color in boxes:
        ax.text(x, y, name, fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                          edgecolor='gray', alpha=0.9))

    # Arrows
    arrow_kw = dict(arrowstyle='->', color='gray', lw=1.5)
    arrows = [
        (0.06, 0.7, 0.09, 0.7),
        (0.15, 0.7, 0.18, 0.7),
        (0.24, 0.78, 0.28, 0.85),
        (0.24, 0.62, 0.28, 0.3),
        (0.34, 0.85, 0.40, 0.85),
        (0.48, 0.80, 0.54, 0.60),
        (0.48, 0.35, 0.54, 0.35),
        (0.60, 0.50, 0.67, 0.55),
        (0.60, 0.35, 0.67, 0.45),
        (0.73, 0.50, 0.75, 0.48),
        (0.73, 0.40, 0.75, 0.42),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_kw)

    # EMA arrow (dashed)
    ax.annotate('', xy=(0.87, 0.70), xytext=(0.87, 0.80),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='dashed'))
    ax.annotate('', xy=(0.84, 0.6), xytext=(0.80, 0.85),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5, linestyle='dashed'))

    # Key difference annotation
    ax.text(0.50, 0.05,
            "Key Difference from MAE: I-JEPA predicts in REPRESENTATION space (not pixel space)\n"
            "No negative pairs needed (vs contrastive) | No pixel-level decoder needed (vs MAE)",
            fontsize=10, ha='center', va='center', style='italic', color='darkblue',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    ax.set_title("I-JEPA: Joint-Embedding Predictive Architecture (2301.08243)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "ijepa_architecture.png", dpi=150)
    plt.close()

    # 8. Prediction quality visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, mr in enumerate(mask_ratios):
        n_visible = max(1, int(ijepa.n_patches * (1 - mr)))
        label = f"{mr:.0%} ({n_visible} visible)"
        ax.plot(ratio_pred_q[mr], label=label, color=colors[idx], linewidth=2)
    ax.set_title("I-JEPA: How Well Can We Predict Masked Representations?")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity (pred vs target)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "prediction_quality_by_mask_ratio.png", dpi=150)
    plt.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("I-JEPA Experiment Summary")
    print("=" * 60)
    print(f"Model: I-JEPA with 4-layer encoder (dim=128), 2-layer predictor")
    print(f"Dataset: CIFAR-10 (5K train subset), patch size 4x4 = 64 patches")
    print(f"\nI-JEPA vs MAE at 75% mask:")
    print(f"  I-JEPA: probe acc = {ijepa_acc:.4f} (predicts representations)")
    print(f"  MAE:    probe acc = {mae_acc:.4f} (predicts pixels)")
    print(f"\nMask Ratio Results (I-JEPA):")
    for mr in mask_ratios:
        n_vis = max(1, int(ijepa.n_patches * (1 - mr)))
        print(f"  {mr:.0%} mask ({n_vis} visible): loss={ratio_losses[mr][-1]:.4f}, "
              f"pred_cos={ratio_pred_q[mr][-1]:.4f}, probe acc={ratio_accs[mr]:.4f}")
    print(f"\nI-JEPA vs MAE across mask ratios:")
    for mr in compare_ratios:
        print(f"  {mr:.0%} mask: I-JEPA={ijepa_ratio_accs[mr]:.4f}, MAE={mae_ratio_accs[mr]:.4f}")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
