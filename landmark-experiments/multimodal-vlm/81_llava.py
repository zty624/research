"""
Minimal LLaVA-style Vision-Language Adapter Reproduction
========================================================
Reproduces core ideas from "Visual Instruction Tuning"
(Liu et al., 2023, 2304.08485, LLaVA):
1. Vision encoder (ViT) extracts visual features from images
2. Projection layer maps visual tokens into language model embedding space
3. Language model (Transformer) processes concatenated visual + text tokens
4. Training: only projection layer is trained (frozen encoders) to align modalities
5. Autoregressive generation conditioned on visual input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Data: Colored Shapes with Text Descriptions ──

SHAPE_TYPES = ["circle", "square", "triangle"]
COLOR_TYPES = ["red", "green", "blue"]
TEMPLATES = [
    "a {color} {shape} on dark background",
    "the image shows a {color} {shape}",
    "this is a {color} {shape}",
]


def generate_shape_image(shape, color, size=32):
    """Generate a 32x32 grayscale-ish image with a colored shape.

    Colors encoded as channels: red=(0.9,0.1,0.1), green=(0.1,0.9,0.1),
    blue=(0.1,0.1,0.9) mapped to single-channel intensity for simplicity.
    We use 3 channels to mimic RGB.
    """
    img = np.zeros((3, size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    r = size // 4

    color_map = {
        "red":   [0.9, 0.1, 0.1],
        "green": [0.1, 0.9, 0.1],
        "blue":  [0.1, 0.1, 0.9],
    }
    c = color_map[color]

    if shape == "circle":
        for i in range(size):
            for j in range(size):
                if (i - cx)**2 + (j - cy)**2 <= r**2:
                    img[:, i, j] = c
    elif shape == "square":
        img[:, cx-r:cx+r, cy-r:cy+r] = np.array(c).reshape(3, 1, 1)
    elif shape == "triangle":
        for i in range(size):
            if i >= cy - r and i <= cy + r:
                half_w = max(1, int(r * (i - (cy - r)) / (2 * r)))
                x_start = max(0, cx - half_w)
                x_end = min(size, cx + half_w)
                for ch in range(3):
                    img[ch, i, x_start:x_end] = c[ch]

    # Add small noise
    img += np.random.randn(*img.shape).astype(np.float32) * 0.02
    img = np.clip(img, 0, 1)
    return img


def generate_dataset(n_samples=500, img_size=32):
    """Generate synthetic image-text pairs."""
    images, texts, labels = [], [], []
    for _ in range(n_samples):
        shape = np.random.choice(SHAPE_TYPES)
        color = np.random.choice(COLOR_TYPES)
        img = generate_shape_image(shape, color, img_size)
        template = np.random.choice(TEMPLATES)
        text = template.format(color=color, shape=shape)
        images.append(img)
        texts.append(text)
        # Label: (color_idx * n_shapes + shape_idx) for 9 classes
        labels.append(COLOR_TYPES.index(color) * len(SHAPE_TYPES) + SHAPE_TYPES.index(shape))

    images = torch.tensor(np.array(images))
    labels = torch.tensor(labels)
    return images, texts, labels


def tokenize(texts, vocab, max_len=12):
    """Simple whitespace tokenization with a fixed vocab."""
    pad_id = vocab.get("<pad>", 0)
    bos_id = vocab.get("<bos>", 1)
    eos_id = vocab.get("<eos>", 2)

    token_ids = []
    for text in texts:
        ids = [bos_id]
        for word in text.lower().split():
            ids.append(vocab.get(word, vocab.get("<unk>", 3)))
        ids.append(eos_id)
        ids = ids[:max_len]
        ids += [pad_id] * (max_len - len(ids))
        token_ids.append(ids)
    return torch.tensor(token_ids, dtype=torch.long)


def build_vocab():
    """Build a minimal vocabulary from the synthetic data."""
    words = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for shape in SHAPE_TYPES:
        words.append(shape)
    for color in COLOR_TYPES:
        words.append(color)
    words.extend(["a", "on", "dark", "background", "the", "image", "shows", "this", "is"])
    return {w: i for i, w in enumerate(words)}


# ── Vision Encoder (Small ViT) ──

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    """Minimal Vision Transformer encoder."""
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, D)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return x  # (B, n_patches+1, D) — all visual tokens including CLS


# ── Language Model (Small Transformer LM) ──

class TransformerLM(nn.Module):
    """Small autoregressive Transformer language model."""
    def __init__(self, vocab_size, embed_dim=64, n_heads=4, n_layers=2, max_len=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, text_tokens, visual_tokens, visual_mask=None):
        """
        text_tokens: (B, T) — token ids for the text
        visual_tokens: (B, V, D) — projected visual features (serves as memory)
        visual_mask: optional attention mask for visual tokens
        Returns: logits (B, T, vocab_size)
        """
        B, T = text_tokens.shape
        text_emb = self.token_embed(text_tokens) + self.pos_embed[:, :T, :]

        # Causal mask for text (autoregressive)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=text_tokens.device)

        out = self.decoder(
            tgt=text_emb,
            memory=visual_tokens,
            tgt_mask=causal_mask,
            memory_mask=visual_mask,
        )
        out = self.norm(out)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate(self, visual_tokens, bos_id, eos_id, pad_id, max_len=12):
        """Autoregressive generation conditioned on visual tokens."""
        B = visual_tokens.shape[0]
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=visual_tokens.device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, visual_tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token.squeeze(-1) == eos_id).all():
                break

        return generated


# ── LLaVA Model ──

class LLaVAModel(nn.Module):
    """LLaVA-style Vision-Language Model with projection adapter."""
    def __init__(self, vocab_size, img_size=32, patch_size=8, embed_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.vision_encoder = ViTEncoder(img_size, patch_size, 3, embed_dim, n_heads, n_layers)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.language_model = TransformerLM(vocab_size, embed_dim, n_heads, n_layers, max_len=20)
        self.embed_dim = embed_dim

    def forward(self, images, text_tokens):
        """Forward pass: images -> visual tokens -> projection -> LM."""
        visual_feats = self.vision_encoder(images)        # (B, V+1, D)
        projected = self.projection(visual_feats)          # (B, V+1, D)
        logits = self.language_model(text_tokens, projected)
        return logits, projected, visual_feats

    @torch.no_grad()
    def generate(self, images, bos_id, eos_id, pad_id, max_len=12):
        """Generate text conditioned on images."""
        visual_feats = self.vision_encoder(images)
        projected = self.projection(visual_feats)
        return self.language_model.generate(projected, bos_id, eos_id, pad_id, max_len)


# ── Training ──

def train_llava(model, images, text_tokens, n_epochs=30, lr=1e-3, batch_size=64, device='cpu'):
    """Train LLaVA: only projection layer is updated (frozen encoders)."""
    # Freeze vision encoder and language model
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    # Only projection is trainable
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(images, text_tokens)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # Teacher forcing: predict next token
            input_ids = by[:, :-1]
            target_ids = by[:, 1:]

            logits, _, _ = model(bx, input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=0  # ignore <pad>
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

    return losses


def train_full(model, images, text_tokens, n_epochs=30, lr=1e-3, batch_size=64, device='cpu'):
    """Train all parameters (ablation: no frozen encoders)."""
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(images, text_tokens)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            input_ids = by[:, :-1]
            target_ids = by[:, 1:]

            logits, _, _ = model(bx, input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {losses[-1]:.4f}")

    return losses


# ── Evaluation ──

def evaluate_generation(model, images, texts, vocab, inv_vocab, n_show=8, device='cpu'):
    """Evaluate generation quality by checking if correct color+shape appear."""
    model.eval()
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    pad_id = vocab["<pad>"]

    images_dev = images[:n_show * 4].to(device)
    generated = model.generate(images_dev, bos_id, eos_id, pad_id, max_len=10)

    results = []
    for i in range(min(n_show * 4, len(images))):
        gen_tokens = [inv_vocab[t.item()] for t in generated[i] if t.item() not in (pad_id, bos_id, eos_id)]
        gen_text = " ".join(gen_tokens)
        real_text = texts[i]
        results.append((real_text, gen_text))

    # Compute word-level accuracy for color+shape keywords
    correct = 0
    total = 0
    for real, gen in results:
        real_words = set(real.lower().split())
        gen_words = set(gen.lower().split())
        keywords = real_words & {"red", "green", "blue", "circle", "square", "triangle"}
        if keywords:
            total += len(keywords)
            correct += len(keywords & gen_words)

    keyword_acc = correct / total if total > 0 else 0
    return results, keyword_acc


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "81-llava"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # Generate data
    print("=== Generating Synthetic Data ===")
    images, texts, labels = generate_dataset(n_samples=500, img_size=32)
    text_tokens = tokenize(texts, vocab, max_len=10)

    # Train LLaVA (projection only)
    print("\n=== Training LLaVA (projection only, frozen encoders) ===")
    model = LLaVAModel(vocab_size, img_size=32, patch_size=8, embed_dim=64, n_heads=4, n_layers=2).to(device)
    losses_proj = train_llava(model, images, text_tokens, n_epochs=40, lr=2e-3, batch_size=64, device=device)

    # Evaluate projection-only model
    results_proj, acc_proj = evaluate_generation(model, images, texts, vocab, inv_vocab, n_show=8, device=device)
    print(f"  Keyword accuracy (projection only): {acc_proj:.3f}")

    # Train full model (ablation)
    print("\n=== Training Full Model (all params, ablation) ===")
    model_full = LLaVAModel(vocab_size, img_size=32, patch_size=8, embed_dim=64, n_heads=4, n_layers=2).to(device)
    losses_full = train_full(model_full, images, text_tokens, n_epochs=40, lr=1e-3, batch_size=64, device=device)

    results_full, acc_full = evaluate_generation(model_full, images, texts, vocab, inv_vocab, n_show=8, device=device)
    print(f"  Keyword accuracy (full model): {acc_full:.3f}")

    # ── Visualization ──

    # 1. Training loss comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses_proj, label='Projection Only (LLaVA-style)', color='blue')
    ax.plot(losses_full, label='Full Model (all params)', color='red')
    ax.set_title("LLaVA Training: Projection-Only vs Full Fine-tuning")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # 2. Projection quality: visual features vs projected features
    model.eval()
    with torch.no_grad():
        sample_imgs = images[:50].to(device)
        visual_feats = model.vision_encoder(sample_imgs)
        projected_feats = model.projection(visual_feats)

        # Visual: t-SNE/PCA of visual features and projected features
        from sklearn.decomposition import PCA

        vis_np = visual_feats.cpu().reshape(50, -1).numpy()
        proj_np = projected_feats.cpu().reshape(50, -1).numpy()
        labels_np = labels[:50].numpy()

        combined = np.vstack([vis_np, proj_np])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        n_cls = len(SHAPE_TYPES) * len(COLOR_TYPES)

        # Visual features (before projection)
        for c in range(n_cls):
            mask = labels_np == c
            axes[0].scatter(combined_2d[:50][mask, 0], combined_2d[:50][mask, 1],
                          alpha=0.7, s=30, label=f"{COLOR_TYPES[c//3]} {SHAPE_TYPES[c%3]}")
        axes[0].set_title("Visual Features (Before Projection)")
        axes[0].legend(fontsize=6, ncol=2)
        axes[0].grid(True, alpha=0.3)

        # Projected features (after projection)
        for c in range(n_cls):
            mask = labels_np == c
            axes[1].scatter(combined_2d[50:][mask, 0], combined_2d[50:][mask, 1],
                          alpha=0.7, s=30, label=f"{COLOR_TYPES[c//3]} {SHAPE_TYPES[c%3]}")
        axes[1].set_title("Projected Features (After Projection)")
        axes[1].legend(fontsize=6, ncol=2)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("LLaVA: Visual Feature Projection Quality", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / "projection_quality.png", dpi=150)
        plt.close()

    # 3. Cross-attention visualization: language tokens attending to visual tokens
    model.eval()
    with torch.no_grad():
        sample_img = images[:1].to(device)
        sample_text = text_tokens[:1].to(device)

        # Use multihead_attn directly from the first decoder layer
        visual_feats = model.vision_encoder(sample_img)
        projected = model.projection(visual_feats)

        # Need to pass through self-attn + norm first (as decoder layer does)
        dec_layer = model.language_model.decoder.layers[0]
        text_emb = model.language_model.token_embed(sample_text) + model.language_model.pos_embed[:, :sample_text.shape[1], :]
        T = sample_text.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Self-attention sublayer
        tgt2 = dec_layer.self_attn(text_emb, text_emb, text_emb, attn_mask=causal_mask)[0]
        tgt = dec_layer.norm1(text_emb + tgt2)

        # Cross-attention: query=text, key=value=visual
        cross_attn_mod = dec_layer.multihead_attn
        tgt2, cross_attn_weights = cross_attn_mod(
            query=tgt, key=projected, value=projected, need_weights=True
        )

        attn_np = cross_attn_weights[0].cpu().numpy()  # (T, V+1)
        token_names = [inv_vocab.get(t.item(), "?") for t in sample_text[0]]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(attn_np, aspect='auto', cmap='Blues')
        ax.set_xlabel("Visual Token Position")
        ax.set_ylabel("Text Token")
        ax.set_yticks(range(len(token_names)))
        ax.set_yticklabels(token_names)
        ax.set_title("Cross-Attention: Language Tokens → Visual Tokens")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        plt.savefig(results_dir / "cross_attention.png", dpi=150)
        plt.close()

    # 4. Sample images and generation results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    model.eval()
    with torch.no_grad():
        sample_imgs = images[:4].to(device)
        gen_tokens = model.generate(sample_imgs, vocab["<bos>"], vocab["<eos>"], vocab["<pad>"], max_len=10)

        for i in range(4):
            # Show image
            img_np = images[i].transpose(0, 2).transpose(0, 1).numpy()  # CHW -> HWC
            axes[0, i].imshow(np.clip(img_np, 0, 1))
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')

            # Show generated text
            gen_words = [inv_vocab[t.item()] for t in gen_tokens[i] if t.item() not in (vocab["<pad>"], vocab["<bos>"], vocab["<eos>"])]
            gen_text = " ".join(gen_words)
            axes[1, i].text(0.5, 0.5, f"GT: {texts[i]}\nGen: {gen_text}",
                          ha='center', va='center', fontsize=9, wrap=True,
                          transform=axes[1, i].transAxes)
            axes[1, i].axis('off')

    plt.suptitle("LLaVA: Image-to-Text Generation Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "generation_samples.png", dpi=150)
    plt.close()

    # 5. Accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Projection Only\n(LLaVA-style)', 'Full Fine-tuning']
    accs = [acc_proj, acc_full]
    colors = ['blue', 'red']
    ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Keyword Accuracy")
    ax.set_title("LLaVA: Generation Quality Comparison")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
