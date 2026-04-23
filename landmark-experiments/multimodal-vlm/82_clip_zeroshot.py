"""
Minimal CLIP Zero-Shot Classification Reproduction
===================================================
Reproduces core ideas from "Learning Transferable Visual Models From
Natural Language Supervision" (Radford et al., 2021, 2103.00020, CLIP):
1. Dual encoder: image encoder + text encoder trained with contrastive loss
2. InfoNCE loss with learnable temperature parameter
3. Symmetric cross-entropy on image-text similarity matrix
4. Zero-shot classification: compare image embeddings to class prompt embeddings
5. Cross-modal retrieval: image-to-text and text-to-image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Data: Simple Image Patterns with Labels ──

CLASSES = ["red_circle", "green_square", "blue_triangle",
           "red_square", "green_triangle", "blue_circle"]
CLASS_PROMPTS = [
    "a photo of a red circle",
    "a photo of a green square",
    "a photo of a blue triangle",
    "a photo of a red square",
    "a photo of a green triangle",
    "a photo of a blue circle",
]

COLOR_MAP = {
    "red":   [0.9, 0.1, 0.1],
    "green": [0.1, 0.9, 0.1],
    "blue":  [0.1, 0.1, 0.9],
}


def generate_image(class_name, size=32):
    """Generate a 32x32 RGB image for a given class."""
    img = np.zeros((3, size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    r = size // 4

    parts = class_name.split("_")
    color, shape = parts[0], parts[1]
    c = COLOR_MAP[color]

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

    img += np.random.randn(*img.shape).astype(np.float32) * 0.03
    img = np.clip(img, 0, 1)
    return img


def generate_dataset(n_per_class=100, img_size=32):
    """Generate synthetic dataset with image-text pairs."""
    images, text_tokens_list, labels = [], [], []

    vocab = build_vocab()
    for cls_idx, cls_name in enumerate(CLASSES):
        for _ in range(n_per_class):
            img = generate_image(cls_name, img_size)
            images.append(img)
            text_tokens_list.append(CLASS_PROMPTS[cls_idx])
            labels.append(cls_idx)

    images = torch.tensor(np.array(images), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, text_tokens_list, labels, vocab


def build_vocab():
    """Build minimal vocabulary for text encoder."""
    words = ["<pad>", "<bos>", "<eos>", "<unk>"]
    words.extend(["a", "photo", "of", "red", "green", "blue",
                  "circle", "square", "triangle"])
    return {w: i for i, w in enumerate(words)}


def tokenize_batch(texts, vocab, max_len=8):
    """Tokenize a batch of text strings."""
    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    unk_id = vocab["<unk>"]

    all_ids = []
    for text in texts:
        ids = [bos_id]
        for word in text.lower().split():
            ids.append(vocab.get(word, unk_id))
        ids.append(eos_id)
        ids = ids[:max_len]
        ids += [pad_id] * (max_len - len(ids))
        all_ids.append(ids)
    return torch.tensor(all_ids, dtype=torch.long)


# ── Image Encoder (Small CNN) ──

class ImageEncoder(nn.Module):
    """Simple CNN for 32x32 RGB images."""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return F.normalize(self.proj(h), dim=-1)


# ── Text Encoder (Small Transformer) ──

class TextEncoder(nn.Module):
    """Simple Transformer-based text encoder."""
    def __init__(self, vocab_size, embed_dim=64, n_heads=4, n_layers=2, max_len=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):
        """token_ids: (B, T) -> (B, D) normalized embeddings."""
        pad_mask = (token_ids == 0)  # <pad> id = 0
        x = self.token_embed(token_ids) + self.pos_embed[:, :token_ids.shape[1], :]
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # Mean pooling (excluding padding)
        mask = (~pad_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.proj(x), dim=-1)


# ── CLIP Model ──

class CLIPModel(nn.Module):
    """CLIP dual encoder with contrastive learning."""
    def __init__(self, vocab_size, embed_dim=64, init_temp=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        # Learnable temperature (log-parameterized)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))

    def forward(self, images, text_ids):
        """Returns similarity logits, image embeddings, text embeddings."""
        img_emb = self.image_encoder(images)    # (B, D)
        txt_emb = self.text_encoder(text_ids)   # (B, D)
        scale = self.logit_scale.exp()
        logits = scale * img_emb @ txt_emb.T    # (B, B)
        return logits, img_emb, txt_emb

    def zero_shot_classify(self, images, class_text_ids):
        """Zero-shot: compare image to all class text embeddings.

        class_text_ids: (n_classes, T) tokenized prompts for each class.
        Returns predicted class indices.
        """
        img_emb = self.image_encoder(images)      # (B, D)
        txt_emb = self.text_encoder(class_text_ids)  # (C, D)
        scale = self.logit_scale.exp()
        logits = scale * img_emb @ txt_emb.T      # (B, C)
        return logits.argmax(dim=-1), logits


# ── InfoNCE Loss ──

def clip_loss(logits):
    """Symmetric InfoNCE loss. Diagonal = positive pairs."""
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2


# ── Training ──

def train_clip(model, images, text_ids, labels, n_epochs=30, lr=3e-3, batch_size=128, device='cpu'):
    """Train CLIP with contrastive loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dataset = torch.utils.data.TensorDataset(images, text_ids, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    temps = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, bt, bl in loader:
            bx, bt = bx.to(device), bt.to(device)
            logits, _, _ = model(bx, bt)
            loss = clip_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            temps.append(model.logit_scale.exp().item())

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Temp: {1/temps[-1]:.4f}")

    return losses, temps


def train_supervised_baseline(images, labels, embed_dim=64, n_epochs=30, lr=1e-3, batch_size=128, device='cpu'):
    """Standard supervised classifier for comparison."""
    encoder = ImageEncoder(embed_dim).to(device)
    classifier = nn.Linear(embed_dim, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)

    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                emb = encoder(bx)
            logits = classifier(emb)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses, encoder, classifier


# ── Evaluation ──

def evaluate_zero_shot(model, images, labels, vocab, device='cpu'):
    """Zero-shot classification accuracy."""
    # Tokenize class prompts
    class_ids = tokenize_batch(CLASS_PROMPTS, vocab, max_len=8).to(device)
    n_classes = len(CLASSES)

    correct, total = 0, 0
    batch_size = 256
    for i in range(0, len(images), batch_size):
        bx = images[i:i+batch_size].to(device)
        by = labels[i:i+batch_size].to(device)
        pred, _ = model.zero_shot_classify(bx, class_ids)
        correct += (pred == by).sum().item()
        total += by.shape[0]

    return correct / total


def evaluate_retrieval(model, images, text_ids, labels, device='cpu'):
    """Image-to-text and text-to-image retrieval (R@1)."""
    batch_size = 256
    all_img_emb = []
    all_txt_emb = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            bx = images[i:i+batch_size].to(device)
            bt = text_ids[i:i+batch_size].to(device)
            ie = model.image_encoder(bx)
            te = model.text_encoder(bt)
            all_img_emb.append(ie.cpu())
            all_txt_emb.append(te.cpu())

    img_emb = torch.cat(all_img_emb)
    txt_emb = torch.cat(all_txt_emb)

    # For retrieval: group by class, use one text per class
    class_txt_ids = tokenize_batch(CLASS_PROMPTS, build_vocab(), max_len=8).to(device)
    with torch.no_grad():
        class_txt_emb = model.text_encoder(class_txt_ids).cpu()

    # Image-to-text retrieval
    sim = img_emb @ class_txt_emb.T  # (N, C)
    i2t_r1 = (sim.argmax(dim=1) == labels).float().mean().item()

    # Text-to-image retrieval (for each class, find top images)
    sim_t2i = class_txt_emb @ img_emb.T  # (C, N)
    t2i_r1 = 0
    for c in range(len(CLASSES)):
        mask = (labels == c)
        top_img_class = labels[sim_t2i[c].argmax().item()]
        if top_img_class == c:
            t2i_r1 += 1
    t2i_r1 /= len(CLASSES)

    return i2t_r1, t2i_r1


def evaluate_linear_probe(model, images, labels, n_epochs=10, device='cpu'):
    """Linear probe on frozen CLIP image embeddings."""
    classifier = nn.Linear(64, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for _ in range(n_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                emb = model.image_encoder(bx)
            loss = F.cross_entropy(classifier(emb), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    correct, total = 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        with torch.no_grad():
            emb = model.image_encoder(bx)
        pred = classifier(emb).argmax(dim=1)
        correct += (pred == by).sum().item()
        total += by.shape[0]
    return correct / total


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "82-clip-zeroshot"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("=== Generating Synthetic Data ===")
    images, texts, labels, vocab = generate_dataset(n_per_class=100, img_size=32)
    text_ids = tokenize_batch(texts, vocab, max_len=8)
    print(f"  {len(images)} images, {len(CLASSES)} classes, vocab size: {len(vocab)}")

    # Train CLIP
    print("\n=== Training CLIP (Contrastive) ===")
    clip_model = CLIPModel(vocab_size=len(vocab), embed_dim=64).to(device)
    clip_losses, temps = train_clip(clip_model, images, text_ids, labels,
                                     n_epochs=40, lr=3e-3, batch_size=128, device=device)

    # Train supervised baseline
    print("\n=== Training Supervised Baseline ===")
    sup_losses, sup_encoder, sup_classifier = train_supervised_baseline(
        images, labels, embed_dim=64, n_epochs=40, lr=1e-3, batch_size=128, device=device)

    # Evaluate
    print("\n=== Evaluation ===")
    zero_shot_acc = evaluate_zero_shot(clip_model, images, labels, vocab, device=device)
    print(f"  Zero-shot accuracy: {zero_shot_acc:.4f}")

    i2t_r1, t2i_r1 = evaluate_retrieval(clip_model, images, text_ids, labels, device=device)
    print(f"  Image→Text R@1: {i2t_r1:.4f}")
    print(f"  Text→Image R@1: {t2i_r1:.4f}")

    linprobe_acc = evaluate_linear_probe(clip_model, images, labels, n_epochs=10, device=device)
    print(f"  Linear probe accuracy: {linprobe_acc:.4f}")

    # Supervised accuracy
    sup_correct, sup_total = 0, 0
    with torch.no_grad():
        for i in range(0, len(images), 256):
            bx = images[i:i+256].to(device)
            by = labels[i:i+256].to(device)
            pred = sup_classifier(sup_encoder(bx)).argmax(dim=1)
            sup_correct += (pred == by).sum().item()
            sup_total += by.shape[0]
    sup_acc = sup_correct / sup_total
    print(f"  Supervised baseline accuracy: {sup_acc:.4f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(clip_losses, label='CLIP (Contrastive)', color='blue')
    axes[0].plot(sup_losses, label='Supervised (CE)', color='red')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temps, color='green')
    axes[1].axhline(y=1/0.07, color='gray', linestyle='--', alpha=0.5, label='CLIP default (1/0.07)')
    axes[1].set_title("Learned Temperature (1/τ)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Scale")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CLIP: Contrastive Language-Image Pre-training", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Embedding space visualization (PCA)
    from sklearn.decomposition import PCA

    clip_model.eval()
    with torch.no_grad():
        all_img_emb = clip_model.image_encoder(images[:300].to(device)).cpu().numpy()
        class_ids = tokenize_batch(CLASS_PROMPTS, vocab, max_len=8).to(device)
        class_txt_emb = clip_model.text_encoder(class_ids).cpu().numpy()

    combined = np.vstack([all_img_emb, class_txt_emb])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 8))
    labels_np = labels[:300].numpy()

    # Plot image embeddings by class
    for c in range(len(CLASSES)):
        mask = labels_np == c
        ax.scatter(combined_2d[:300][mask, 0], combined_2d[:300][mask, 1],
                  alpha=0.4, s=15, label=f"{CLASSES[c]} (img)")

    # Plot text embeddings (star markers)
    for c in range(len(CLASSES)):
        ax.scatter(combined_2d[300+c, 0], combined_2d[300+c, 1],
                  marker='*', s=400, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(CLASSES[c], (combined_2d[300+c, 0], combined_2d[300+c, 1]),
                   fontsize=8, fontweight='bold', ha='left', va='bottom')

    ax.set_title("CLIP Embedding Space (PCA) — Image + Text Embeddings")
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "embedding_space.png", dpi=150)
    plt.close()

    # 3. Similarity matrix: class text embeddings vs class mean image embeddings
    with torch.no_grad():
        class_mean_embs = []
        for c in range(len(CLASSES)):
            mask = (labels == c)
            mean_emb = clip_model.image_encoder(images[mask][:20].to(device)).mean(dim=0)
            class_mean_embs.append(F.normalize(mean_emb, dim=-1).cpu().unsqueeze(0))
        class_mean_embs = torch.cat(class_mean_embs)

        class_ids_dev = tokenize_batch(CLASS_PROMPTS, vocab, max_len=8).to(device)
        class_txt_emb_t = clip_model.text_encoder(class_ids_dev).cpu()

        sim_matrix = (class_mean_embs @ class_txt_emb_t.T).numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-0.3, vmax=1.0)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Text Class Embedding")
    ax.set_ylabel("Image Class Mean Embedding")
    ax.set_title("Cross-Modal Similarity Matrix")
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Annotate cells
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                   color='white' if abs(sim_matrix[i,j]) > 0.6 else 'black')

    plt.tight_layout()
    plt.savefig(results_dir / "similarity_matrix.png", dpi=150)
    plt.close()

    # 4. Zero-shot accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Zero-Shot\n(CLIP)', 'Linear Probe\n(CLIP emb)', 'Supervised\n(CE)']
    accs = [zero_shot_acc, linprobe_acc, sup_acc]
    colors = ['blue', 'cyan', 'red']
    bars = ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("CLIP: Zero-Shot vs Linear Probe vs Supervised")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    # 5. Retrieval performance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Image→Text\nR@1', 'Text→Image\nR@1'], [i2t_r1, t2i_r1],
           color=['steelblue', 'coral'], alpha=0.7)
    ax.set_ylabel("Recall@1")
    ax.set_title("CLIP: Cross-Modal Retrieval Performance")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for i, v in enumerate([i2t_r1, t2i_r1]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "retrieval_performance.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
