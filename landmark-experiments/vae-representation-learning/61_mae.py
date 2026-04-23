"""
Minimal Masked Autoencoder (MAE) Reproduction
==============================================
Reproduces core ideas from MAE (2111.06377, He et al., 2021):
1. Patch embedding: split image into non-overlapping patches, linear projection
2. Random masking at high ratio (75%): only visible patches enter encoder
3. Asymmetric encoder-decoder: deep encoder on visible patches, lightweight decoder on all
4. MSE reconstruction loss on masked patches only
5. Shuffle/unshuffle trick for efficient masking without sparse operations
6. Compare mask ratios (10%, 30%, 50%, 75%, 90%)
7. Evaluate representation quality via linear probing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
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

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        if return_attn:
            attn_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
            x = x + attn_out
        else:
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x


# ── MAE Encoder ──

class MAEEncoder(nn.Module):
    """ViT encoder that only processes visible (unmasked) patches."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Positional embedding for all patch positions
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep):
        """Encode only visible patches.
        Args:
            x: (B, C, H, W) input images
            ids_keep: (B, N_keep) indices of visible patches
        Returns:
            (B, N_keep, embed_dim) encoded visible patches
        """
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        B, N, D = x.shape

        # Gather only visible patch embeddings + their positional embeddings
        pos = self.pos_embed.expand(B, -1, -1)  # (B, n_patches, D)
        # Batched index select: pick visible positions
        ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, D)  # (B, N_keep, D)
        x = torch.gather(x, dim=1, index=ids_keep_exp)
        pos = torch.gather(pos, dim=1, index=ids_keep_exp)

        x = self.pos_drop(x + pos)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_with_attn(self, x, ids_keep):
        """Forward pass that also returns attention weights from last layer."""
        x = self.patch_embed(x)
        B, N, D = x.shape
        pos = self.pos_embed.expand(B, -1, -1)
        ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x = torch.gather(x, dim=1, index=ids_keep_exp)
        pos = torch.gather(pos, dim=1, index=ids_keep_exp)
        x = self.pos_drop(x + pos)

        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x, return_attn=True)
        x = self.norm(x)
        return x, attn_weights


# ── MAE Decoder ──

class MAEDecoder(nn.Module):
    """Lightweight decoder that processes all patches (visible + mask tokens)."""
    def __init__(self, n_patches=64, encoder_dim=128, decoder_dim=64,
                 depth=2, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.n_patches = n_patches
        self.decoder_dim = decoder_dim

        # Project encoder output to decoder dimension
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        # Shared mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding for all patches in decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, n_patches, decoder_dim) * 0.02
        )

        # Decoder transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

    def forward(self, encoded_visible, ids_restore):
        """Decode all patches from encoded visible + mask tokens.
        Args:
            encoded_visible: (B, N_keep, encoder_dim) encoded visible patches
            ids_restore: (B, N) indices to restore original patch ordering
        Returns:
            (B, N, decoder_dim) decoded features for all patches
        """
        B = encoded_visible.shape[0]
        N = self.n_patches
        N_keep = encoded_visible.shape[1]
        N_mask = N - N_keep

        # Project encoder dim -> decoder dim
        encoded_visible = self.enc_to_dec(encoded_visible)  # (B, N_keep, dec_dim)

        # Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N_mask, -1)  # (B, N_mask, dec_dim)

        # Concatenate visible encoded + mask tokens (in shuffled order)
        x = torch.cat([encoded_visible, mask_tokens], dim=1)  # (B, N, dec_dim)

        # Unshuffle: restore original patch ordering using ids_restore
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        x = torch.gather(x, dim=1, index=ids_restore_exp)  # (B, N, dec_dim)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Decoder transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# ── Full MAE Model ──

class MAE(nn.Module):
    """Masked Autoencoder with asymmetric encoder-decoder."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 encoder_dim=128, encoder_depth=4, encoder_heads=4,
                 decoder_dim=64, decoder_depth=2, decoder_heads=4,
                 mask_ratio=0.75, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        n_patches = (img_size // patch_size) ** 2

        self.encoder = MAEEncoder(
            img_size, patch_size, in_channels,
            encoder_dim, encoder_depth, encoder_heads, mlp_ratio, dropout
        )
        self.decoder = MAEDecoder(
            n_patches, encoder_dim, decoder_dim,
            decoder_depth, decoder_heads, mlp_ratio, dropout
        )

        # Prediction head: predict pixel values for each patch
        self.patch_dim = patch_size * patch_size * in_channels
        self.pred_head = nn.Linear(decoder_dim, self.patch_dim)

        self.n_patches = n_patches

    def random_masking(self, x, mask_ratio):
        """Random masking: shuffle patches, keep first (1-mask_ratio), return indices.
        This is the efficient shuffle/unshuffle trick from the paper.

        Args:
            x: (B, N, D) patch embeddings (unused here, just for shape)
            mask_ratio: fraction of patches to mask

        Returns:
            ids_keep: (B, N_keep) indices of visible patches
            ids_restore: (B, N) indices to restore original ordering
            mask: (B, N) binary mask (1=masked, 0=visible)
        """
        B, N, D = x.shape
        N_keep = max(1, int(N * (1 - mask_ratio)))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)  # (B, N)
        ids_shuffle = torch.argsort(noise, dim=1)   # (B, N) ascending
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N)

        # Keep first N_keep patches (after shuffle)
        ids_keep = ids_shuffle[:, :N_keep]  # (B, N_keep)

        # Generate binary mask: 0 = visible, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N_keep] = 0
        # Unshuffle mask to match original patch ordering
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward(self, x):
        """Full MAE forward pass.
        Args:
            x: (B, C, H, W) input images
        Returns:
            pred: (B, N, patch_dim) predicted pixel values for all patches
            mask: (B, N) binary mask (1=masked, 0=visible)
        """
        # Get patch embeddings for masking logic
        patches = self.encoder.patch_embed(x)  # (B, N, D)

        # Determine which patches to mask
        ids_keep, ids_restore, mask = self.random_masking(patches, self.mask_ratio)

        # Encode only visible patches
        encoded = self.encoder(x, ids_keep)  # (B, N_keep, encoder_dim)

        # Decode all patches (visible encoded + mask tokens)
        decoded = self.decoder(encoded, ids_restore)  # (B, N, decoder_dim)

        # Predict pixel values
        pred = self.pred_head(decoded)  # (B, N, patch_dim)

        return pred, mask

    def forward_with_attn(self, x):
        """Forward pass that also returns encoder attention weights."""
        patches = self.encoder.patch_embed(x)
        ids_keep, ids_restore, mask = self.random_masking(patches, self.mask_ratio)
        encoded, attn_weights = self.encoder.forward_with_attn(x, ids_keep)
        decoded = self.decoder(encoded, ids_restore)
        pred = self.pred_head(decoded)
        return pred, mask, attn_weights, ids_keep

    def loss(self, x, pred, mask):
        """MSE loss on masked patches only.
        Args:
            x: (B, C, H, W) original images
            pred: (B, N, patch_dim) predicted patches
            mask: (B, N) binary mask (1=masked, 0=visible)
        Returns:
            scalar MSE loss averaged over masked patches
        """
        target = self.patchify(x)  # (B, N, patch_dim)
        # MSE per patch
        loss_per_patch = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # (B, N)
        # Average only over masked patches
        masked_loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)
        return masked_loss

    def patchify(self, x):
        """Convert image to patches.
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, patch_dim) flattened patches
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h = H // p
        w = W // p
        # (B, C, h, p, w, p) -> (B, h, w, C, p, p) -> (B, N, patch_dim)
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, h, w, C, p, p)
        x = x.reshape(B, h * w, C * p * p)
        return x

    def unpatchify(self, patches):
        """Convert patches back to image.
        Args:
            patches: (B, N, patch_dim)
        Returns:
            (B, C, H, W) reconstructed image
        """
        B = patches.shape[0]
        C = self.in_channels
        p = self.patch_size
        h = w = int(self.n_patches ** 0.5)
        x = patches.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        x = x.reshape(B, C, h * p, w * p)
        return x

    def get_representations(self, x):
        """Get encoder representations for all visible patches, pooled to one vector.
        Used for linear probing.
        """
        patches = self.encoder.patch_embed(x)
        ids_keep, _, _ = self.random_masking(patches, self.mask_ratio)
        encoded = self.encoder(x, ids_keep)  # (B, N_keep, D)
        # Global average pool over visible patches
        return encoded.mean(dim=1)  # (B, D)


# ── Training ──

def train_mae(model, train_loader, n_epochs=20, lr=1.5e-4, weight_decay=0.05,
              device='cpu', verbose=True):
    """Train MAE with AdamW and cosine LR schedule."""
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
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


# ── Linear Probe ──

def linear_probe(model, train_loader, test_loader, n_epochs=10, lr=1e-3, device='cpu'):
    """Train a linear classifier on frozen MAE encoder features.
    Uses average pooling over visible patch representations.
    """
    encoder_dim = model.encoder.patch_embed.proj.out_channels

    # Collect features with multiple masking samples for stability
    def get_features(loader, n_samples=3):
        all_feats = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                # Average representations over multiple random masks
                feats_list = []
                for _ in range(n_samples):
                    feat = model.get_representations(bx)
                    feats_list.append(feat)
                feats = torch.stack(feats_list).mean(dim=0)
                all_feats.append(feats.cpu())
                all_labels.append(by)
        return torch.cat(all_feats), torch.cat(all_labels)

    # Get features
    train_feats, train_labels = get_features(train_loader)
    test_feats, test_labels = get_features(test_loader)

    # Train linear classifier
    classifier = nn.Linear(encoder_dim, 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    for epoch in range(n_epochs):
        # Mini-batch training on features
        n = train_feats.shape[0]
        bs = 256
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(classifier(train_feats[idx]), train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    with torch.no_grad():
        preds = classifier(test_feats).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()

    return acc


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "61-mae"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # Use subset for speed
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=256, num_workers=0)

    n_epochs = 30

    # ── Experiment 1: Train MAE at 75% mask ratio ──
    print("=== Training MAE (75% mask ratio) ===")
    mae_75 = MAE(
        img_size=32, patch_size=4, in_channels=3,
        encoder_dim=128, encoder_depth=4, encoder_heads=4,
        decoder_dim=64, decoder_depth=2, decoder_heads=4,
        mask_ratio=0.75
    ).to(device)
    n_params_enc = sum(p.numel() for p in mae_75.encoder.parameters())
    n_params_dec = sum(p.numel() for p in mae_75.decoder.parameters())
    n_params_total = sum(p.numel() for p in mae_75.parameters())
    print(f"  Patches: {mae_75.n_patches} ({32//4}x{32//4})")
    print(f"  Encoder params: {n_params_enc:,} | Decoder params: {n_params_dec:,} | Total: {n_params_total:,}")
    print(f"  At 75% masking: only {int(mae_75.n_patches * 0.25)}/{mae_75.n_patches} patches visible")

    losses_75 = train_mae(mae_75, train_loader, n_epochs=n_epochs, device=device)

    # Linear probe
    print("  Linear probing...")
    acc_75 = linear_probe(mae_75, train_loader, test_loader, device=device)
    print(f"  Linear probe accuracy: {acc_75:.4f}")

    # ── Experiment 2: Compare mask ratios ──
    print("\n=== Comparing Mask Ratios ===")
    mask_ratios = [0.1, 0.3, 0.5, 0.75, 0.9]
    ratio_losses = {}
    ratio_accs = {}
    ratio_n_visible = {}

    for mr in mask_ratios:
        n_visible = max(1, int(mae_75.n_patches * (1 - mr)))
        ratio_n_visible[mr] = n_visible
        print(f"  Mask ratio {mr:.0%} ({n_visible}/{mae_75.n_patches} visible):")

        model = MAE(
            img_size=32, patch_size=4, in_channels=3,
            encoder_dim=128, encoder_depth=4, encoder_heads=4,
            decoder_dim=64, decoder_depth=2, decoder_heads=4,
            mask_ratio=mr
        ).to(device)

        losses = train_mae(model, train_loader, n_epochs=n_epochs, device=device, verbose=False)
        acc = linear_probe(model, train_loader, test_loader, device=device)

        ratio_losses[mr] = losses
        ratio_accs[mr] = acc
        print(f"    Final loss: {losses[-1]:.6f} | Linear probe acc: {acc:.4f}")

    # ── Experiment 3: Asymmetric design - compare encoder vs decoder depth ──
    print("\n=== Asymmetric Design: Encoder vs Decoder Depth ===")
    depth_configs = [
        {"enc_d": 2, "dec_d": 2, "label": "2 enc / 2 dec"},
        {"enc_d": 4, "dec_d": 2, "label": "4 enc / 2 dec (MAE)"},
        {"enc_d": 2, "dec_d": 4, "label": "2 enc / 4 dec"},
        {"enc_d": 4, "dec_d": 4, "label": "4 enc / 4 dec"},
    ]
    depth_losses = {}
    depth_accs = {}
    depth_params = {}

    for cfg in depth_configs:
        label = cfg["label"]
        print(f"  {label}:")
        model = MAE(
            img_size=32, patch_size=4, in_channels=3,
            encoder_dim=128, encoder_depth=cfg["enc_d"], encoder_heads=4,
            decoder_dim=64, decoder_depth=cfg["dec_d"], decoder_heads=4,
            mask_ratio=0.75
        ).to(device)

        n_p = sum(p.numel() for p in model.parameters())
        depth_params[label] = n_p
        losses = train_mae(model, train_loader, n_epochs=n_epochs, device=device, verbose=False)
        acc = linear_probe(model, train_loader, test_loader, device=device)

        depth_losses[label] = losses
        depth_accs[label] = acc
        print(f"    Params: {n_p:,} | Final loss: {losses[-1]:.6f} | Probe acc: {acc:.4f}")

    # ── Visualization ──

    # 1. Reconstruction quality at different mask ratios
    print("\n=== Generating Reconstructions ===")
    # Use unnormalized images for visualization
    vis_transform = transforms.ToTensor()
    vis_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=vis_transform)
    vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=8, num_workers=0)
    vis_batch = next(iter(vis_loader))[0].to(device)

    fig, axes = plt.subplots(len(mask_ratios) + 1, 8, figsize=(16, 2.2 * (len(mask_ratios) + 1)))

    # Row 0: originals
    for i in range(8):
        axes[0, i].imshow(vis_batch[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=9, fontweight='bold')

    # Each subsequent row: a different mask ratio
    for row, mr in enumerate(mask_ratios, start=1):
        model = MAE(
            img_size=32, patch_size=4, in_channels=3,
            encoder_dim=128, encoder_depth=4, encoder_heads=4,
            decoder_dim=64, decoder_depth=2, decoder_heads=4,
            mask_ratio=mr
        ).to(device)
        # Quick train for visualization
        train_mae(model, train_loader, n_epochs=15, device=device, verbose=False)

        model.eval()
        with torch.no_grad():
            pred, mask = model(vis_batch)

        recon = model.unpatchify(pred)  # (B, C, H, W)
        recon = recon.clamp(0, 1)

        for i in range(8):
            # Show masked input (visible patches only) in top half, reconstruction below
            img = vis_batch[i].permute(1, 2, 0).cpu().numpy()
            mask_i = mask[i].cpu().numpy()
            # Gray out masked patches
            p = 4
            h = w = 8
            display = img.copy()
            for idx in range(model.n_patches):
                if mask_i[idx] == 1:
                    hi, wi = idx // h, idx % h
                    display[hi*p:(hi+1)*p, wi*p:(wi+1)*p] = 0.5
            axes[row, i].imshow(display)
            axes[row, i].axis('off')

        n_vis = ratio_n_visible[mr]
        axes[row, 0].set_ylabel(f"Mask {mr:.0%}\n({n_vis} visible)", fontsize=8, fontweight='bold')

    plt.suptitle("MAE: Reconstruction at Different Mask Ratios\n(gray = masked patches)", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "reconstruction_mask_ratios.png", dpi=150)
    plt.close()

    # 2. Reconstruction comparison: masked input vs reconstruction vs original
    print("=== Generating Detailed Reconstruction Comparison ===")
    mae_75.eval()
    with torch.no_grad():
        pred, mask = mae_75(vis_batch)

    recon = mae_75.unpatchify(pred).clamp(0, 1)

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        # Original
        axes[0, i].imshow(vis_batch[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')

        # Masked input
        img = vis_batch[i].permute(1, 2, 0).cpu().numpy()
        mask_i = mask[i].cpu().numpy()
        display = img.copy()
        p, h, w = 4, 8, 8
        for idx in range(mae_75.n_patches):
            if mask_i[idx] == 1:
                hi, wi = idx // h, idx % h
                display[hi*p:(hi+1)*p, wi*p:(wi+1)*p] = 0.5
        axes[1, i].imshow(display)
        axes[1, i].axis('off')

        # Reconstruction
        axes[2, i].imshow(recon[i].permute(1, 2, 0).cpu().numpy())
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Masked Input\n(75% masked)", fontsize=9, fontweight='bold')
    axes[2, 0].set_ylabel("Reconstruction", fontsize=10, fontweight='bold')
    plt.suptitle("MAE (75% Masking): Original vs Masked vs Reconstructed", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "reconstruction_detail.png", dpi=150)
    plt.close()

    # 3. Training curves for different mask ratios
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, mr in enumerate(mask_ratios):
        label = f"{mr:.0%} ({ratio_n_visible[mr]} visible)"
        axes[0].plot(ratio_losses[mr], label=label, color=colors[idx])
    axes[0].set_title("Training Loss by Mask Ratio")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (masked patches)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Linear probe accuracy bar chart
    ratios_str = [f"{mr:.0%}" for mr in mask_ratios]
    accs = [ratio_accs[mr] for mr in mask_ratios]
    bars = axes[1].bar(ratios_str, accs, color=colors, alpha=0.8)
    axes[1].set_title("Linear Probe Accuracy by Mask Ratio")
    axes[1].set_xlabel("Mask Ratio")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, max(accs) * 1.15)

    plt.suptitle("MAE: Effect of Mask Ratio on Training and Representation Quality", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "mask_ratio_comparison.png", dpi=150)
    plt.close()

    # 4. Asymmetric design comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    depth_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for idx, (label, losses) in enumerate(depth_losses.items()):
        axes[0].plot(losses, label=label, color=depth_colors[idx])
    axes[0].set_title("Training Loss: Encoder vs Decoder Depth")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    labels = list(depth_accs.keys())
    accs = [depth_accs[l] for l in labels]
    params = [depth_params[l] / 1e3 for l in labels]
    bars = axes[1].bar(labels, accs, color=depth_colors, alpha=0.8)
    axes[1].set_title("Linear Probe: Encoder vs Decoder Depth")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, acc, p in zip(bars, accs, params):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{acc:.3f}\n({p:.0f}K params)', ha='center', fontsize=8, fontweight='bold')
    axes[1].set_ylim(0, max(accs) * 1.2)
    axes[1].tick_params(axis='x', labelsize=8)

    plt.suptitle("MAE: Asymmetric Encoder-Decoder Design\n(deeper encoder = better representations)", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "asymmetric_design.png", dpi=150)
    plt.close()

    # 5. Attention patterns in encoder
    print("=== Visualizing Attention Patterns ===")
    mae_75.eval()
    with torch.no_grad():
        sample = vis_batch[:1]
        _, _, attn_weights, ids_keep = mae_75.forward_with_attn(sample)

    # attn_weights: (1, N_keep, N_keep) from last layer
    attn = attn_weights[0].cpu().numpy()  # (N_keep, N_keep)
    n_keep = attn.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Attention heatmap
    im = axes[0].imshow(attn, cmap='Blues', interpolation='nearest')
    axes[0].set_title(f"Encoder Self-Attention\n({n_keep} visible patches)")
    axes[0].set_xlabel("Key patch")
    axes[0].set_ylabel("Query patch")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Average attention per patch position
    avg_attn = attn.mean(axis=0)  # average attention received by each key
    # Map back to spatial grid using ids_keep
    attn_map = np.zeros(mae_75.n_patches)
    ids = ids_keep[0].cpu().numpy()
    for i, idx in enumerate(ids):
        attn_map[idx] = avg_attn[i]

    attn_grid = attn_map.reshape(8, 8)
    im = axes[1].imshow(attn_grid, cmap='Reds', interpolation='nearest')
    axes[1].set_title("Avg Attention per Position\n(visible patches only)")
    axes[1].set_xlabel("Patch column")
    axes[1].set_ylabel("Patch row")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Show which patches are visible vs masked
    vis_mask = np.zeros(mae_75.n_patches)
    for idx in ids:
        vis_mask[idx] = 1
    mask_grid = vis_mask.reshape(8, 8)
    axes[2].imshow(mask_grid, cmap='gray_r', interpolation='nearest')
    axes[2].set_title("Visible vs Masked Patches\n(white=visible, black=masked)")
    axes[2].set_xlabel("Patch column")
    axes[2].set_ylabel("Patch row")

    plt.suptitle("MAE Encoder: Attention Patterns (75% Masking)", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "attention_patterns.png", dpi=150)
    plt.close()

    # 6. Concept diagram: MAE architecture
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    # Architecture flow
    boxes = [
        ("Input\nImage\n32x32x3", 0.03, 0.55, 'lightblue', 3.5),
        ("Patch\nEmbed\n64 patches", 0.14, 0.55, 'lightcyan', 3.5),
        ("Random\nMask\n75%", 0.25, 0.55, 'lightyellow', 3.5),
        ("Visible\nPatches\n(16/64)", 0.36, 0.75, 'lightgreen', 3.2),
        ("Masked\nPatches\n(48/64)", 0.36, 0.3, 'mistyrose', 3.2),
        ("Encoder\n4-layer ViT\ndim=128", 0.50, 0.75, 'paleturquoise', 3.5),
        ("Mask\nToken", 0.50, 0.3, 'navajowhite', 3.0),
        ("Concat +\nUnshuffle\n64 patches", 0.64, 0.55, 'plum', 3.5),
        ("Decoder\n2-layer\ndim=64", 0.77, 0.55, 'thistle', 3.5),
        ("Predict\nPixels", 0.90, 0.55, 'lightsalmon', 3.5),
    ]

    for name, x, y, color, fs in boxes:
        ax.text(x, y, name, fontsize=fs, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='gray', alpha=0.9))

    # Arrows
    arrow_props = dict(arrowstyle='->', color='gray', lw=2)
    arrows = [
        (0.08, 0.55, 0.11, 0.55),   # input -> patch
        (0.19, 0.55, 0.22, 0.55),   # patch -> mask
        (0.28, 0.65, 0.33, 0.75),   # mask -> visible
        (0.28, 0.45, 0.33, 0.3),    # mask -> masked
        (0.39, 0.75, 0.45, 0.75),   # visible -> encoder
        (0.39, 0.3, 0.47, 0.3),     # masked -> mask token
        (0.55, 0.75, 0.60, 0.65),   # encoder -> concat
        (0.55, 0.3, 0.60, 0.45),    # mask token -> concat
        (0.68, 0.55, 0.73, 0.55),   # concat -> decoder
        (0.81, 0.55, 0.86, 0.55),   # decoder -> predict
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)

    # Key insight text
    ax.text(0.50, 0.02,
            "Key Insight: Asymmetric design -- heavy encoder processes only 25% patches, "
            "lightweight decoder reconstructs from all patches",
            fontsize=10, ha='center', va='center', style='italic', color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Masked Autoencoder (MAE) Architecture", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "mae_architecture.png", dpi=150)
    plt.close()

    # 7. Training curve for the main MAE model
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses_75, color='purple', linewidth=2)
    ax.set_title("MAE Training Loss (75% Mask Ratio)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (masked patches only)")
    ax.grid(True, alpha=0.3)
    ax.annotate(f"Final: {losses_75[-1]:.4f}", xy=(len(losses_75)-1, losses_75[-1]),
                xytext=(-50, 20), textcoords='offset points',
                fontsize=10, fontweight='bold', color='purple',
                arrowprops=dict(arrowstyle='->', color='purple'))
    plt.tight_layout()
    plt.savefig(results_dir / "training_curve.png", dpi=150)
    plt.close()

    # 8. Pixel-level reconstruction error heatmap
    print("=== Generating Error Heatmap ===")
    mae_75.eval()
    with torch.no_grad():
        sample = vis_batch[:4]
        pred, mask = mae_75(sample)
        target = mae_75.patchify(sample)
        # Per-patch MSE
        patch_error = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # (B, N)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        # Original image
        axes[0, i].imshow(sample[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Sample {i+1}")

        # Error heatmap (higher = worse reconstruction)
        err = patch_error[i].cpu().numpy().reshape(8, 8)
        mask_i = mask[i].cpu().numpy().reshape(8, 8)
        # Only show error for masked patches (visible patches are not reconstructed)
        display_err = np.where(mask_i == 1, err, np.nan)
        im = axes[1, i].imshow(display_err, cmap='hot', interpolation='nearest')
        axes[1, i].set_title("Reconstruction Error\n(masked patches)")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Error Heatmap", fontsize=10, fontweight='bold')
    plt.suptitle("MAE: Per-Patch Reconstruction Error (75% Masking)", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "error_heatmap.png", dpi=150)
    plt.close()

    # ── Summary ──
    print("\n" + "="*60)
    print("MAE Experiment Summary")
    print("="*60)
    print(f"Model: MAE with 4-layer encoder (dim=128), 2-layer decoder (dim=64)")
    print(f"Dataset: CIFAR-10 (5K train subset), patch size 4x4 = 64 patches")
    print(f"\nMask Ratio Results:")
    for mr in mask_ratios:
        print(f"  {mr:.0%} mask: loss={ratio_losses[mr][-1]:.4f}, probe acc={ratio_accs[mr]:.4f}")
    print(f"\nAsymmetric Design Results:")
    for label in depth_configs:
        l = label["label"]
        print(f"  {l}: probe acc={depth_accs[l]:.4f} ({depth_params[l]:,} params)")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
