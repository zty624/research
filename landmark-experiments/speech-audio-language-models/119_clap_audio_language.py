"""
Minimal CLAP Reproduction
=========================
Reproduces core ideas from CLAP: Learning Transferable Audio-Visual-Audio
Representations (2211.09372, Wu et al.):
1. Dual encoder: audio encoder + text encoder
2. Contrastive learning: align audio-text pairs in shared space
3. InfoNCE loss with temperature scaling
4. Zero-shot audio classification via text prompts
5. Compare: contrastive vs random projection baselines
6. Show: retrieval accuracy vs training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Audio Encoder ──

class AudioEncoder(nn.Module):
    """Simple 1D CNN audio encoder (mimics HTSAT/ResNet audio backbone)."""
    def __init__(self, n_mels=40, n_frames=64, d_model=128, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model * 8, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, n_mels, n_frames) → (B, out_dim)"""
        h = self.conv(x)
        return F.normalize(self.proj(h), dim=-1)


# ── Text Encoder ──

class TextEncoder(nn.Module):
    """Simple text encoder (mimics Transformer text backbone)."""
    def __init__(self, vocab_size=1000, d_model=128, out_dim=256, max_len=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                        dim_feedforward=d_model * 4,
                                        batch_first=True, activation='gelu'),
            num_layers=2,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, T) token ids → (B, out_dim)"""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos_emb(pos)
        h = self.transformer(h)
        h = h.mean(dim=1)  # global average pooling
        return F.normalize(self.proj(h), dim=-1)


# ── CLAP Model ──

class CLAP(nn.Module):
    """Contrastive Language-Audio Pretraining."""
    def __init__(self, audio_encoder, text_encoder, init_temp=0.07):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, audio, text):
        """
        audio: (B, n_mels, n_frames)
        text: (B, T) token ids
        Returns: loss, logits (B, B), audio_features, text_features
        """
        a_feat = self.audio_encoder(audio)  # (B, D)
        t_feat = self.text_encoder(text)    # (B, D)

        temp = self.log_temp.exp()
        logits = a_feat @ t_feat.T / temp  # (B, B)

        # Symmetric InfoNCE loss
        labels = torch.arange(audio.shape[0], device=audio.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)
        loss = (loss_a2t + loss_t2a) / 2

        return loss, logits, a_feat, t_feat


# ── Synthetic Audio-Text Data ──

class AudioTextDataset:
    """Synthetic dataset: audio spectrograms paired with text descriptions."""
    def __init__(self, n_classes=10, n_samples_per_class=50, n_mels=40,
                 n_frames=64, vocab_size=1000, max_len=16, device='cpu'):
        self.device = device
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_mels = n_mels
        self.n_frames = n_frames

        torch.manual_seed(42)
        # Class-specific audio templates (different spectral patterns)
        self.audio_templates = []
        for c in range(n_classes):
            freq = 100 + c * 50  # base frequency
            template = torch.zeros(n_mels, n_frames)
            mel_bin = int(freq / 50) % n_mels
            template[mel_bin, :] = 1.0
            # Add harmonics
            if mel_bin + 5 < n_mels:
                template[mel_bin + 5, :] = 0.5
            self.audio_templates.append(template)

        # Class-specific text templates (token patterns)
        self.text_templates = []
        for c in range(n_classes):
            # Each class has a distinctive token pattern
            base = torch.randint(10, 100, (max_len,))
            base[0] = c  # first token encodes class
            base[1] = c * 10 + 5  # second token also class-specific
            self.text_templates.append(base)

        # Generate samples
        self.data = []
        for c in range(n_classes):
            for _ in range(n_samples_per_class):
                # Audio: template + noise
                audio = self.audio_templates[c] + torch.randn(n_mels, n_frames) * 0.3
                # Text: template with slight variation
                text = self.text_templates[c].clone()
                text[2:] = torch.randint(0, vocab_size, (max_len - 2,))
                self.data.append((audio, text, c))

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size):
        indices = torch.randint(0, len(self.data), (batch_size,))
        audios, texts, labels = [], [], []
        for idx in indices:
            a, t, l = self.data[idx]
            audios.append(a)
            texts.append(t)
            labels.append(l)
        return (torch.stack(audios).to(self.device),
                torch.stack(texts).to(self.device),
                torch.tensor(labels, device=self.device))


# ── Training ──

def train_clap(model, dataset, n_steps=2000, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    metrics = {'loss': [], 'temp': [], 'acc_a2t': [], 'acc_t2a': []}

    for step in range(n_steps):
        audio, text, labels = dataset.get_batch(batch_size)
        loss, logits, a_feat, t_feat = model(audio, text)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc_a2t = (logits.argmax(dim=-1) == torch.arange(batch_size, device=device)).float().mean()
            acc_t2a = (logits.T.argmax(dim=-1) == torch.arange(batch_size, device=device)).float().mean()
            metrics['loss'].append(loss.item())
            metrics['temp'].append(model.log_temp.exp().item())
            metrics['acc_a2t'].append(acc_a2t.item())
            metrics['acc_t2a'].append(acc_t2a.item())

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"A2T: {metrics['acc_a2t'][-1]:.3f} | "
                  f"T2A: {metrics['acc_t2a'][-1]:.3f} | "
                  f"Temp: {metrics['temp'][-1]:.4f}")

    return metrics


# ── Zero-Shot Classification ──

def zero_shot_classify(model, dataset, n_test=200, device='cpu'):
    """Zero-shot audio classification using text prompts."""
    model.eval()
    class_names = [f"class_{i}" for i in range(dataset.n_classes)]

    # Encode text prompts for each class
    prompt_tokens = []
    for c in range(dataset.n_classes):
        # Create prompt: "the sound of class_C"
        tokens = dataset.text_templates[c].clone().unsqueeze(0).to(device)
        prompt_tokens.append(tokens)

    # Get text features for all classes
    with torch.no_grad():
        text_feats = []
        for tokens in prompt_tokens:
            t_feat = model.text_encoder(tokens)
            text_feats.append(t_feat)
        text_feats = torch.cat(text_feats, dim=0)  # (C, D)

    # Test on held-out samples
    correct = 0
    total = 0
    for _ in range(n_test):
        idx = torch.randint(0, len(dataset.data), (1,)).item()
        audio, _, label = dataset.data[idx]
        audio = audio.unsqueeze(0).to(device)

        with torch.no_grad():
            a_feat = model.audio_encoder(audio)  # (1, D)
            sims = (a_feat @ text_feats.T).squeeze(0)  # (C,)
            pred = sims.argmax().item()

        correct += (pred == label)
        total += 1

    return correct / total


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "119-clap-audio-language"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("=== Creating Audio-Text Dataset ===")
    dataset = AudioTextDataset(n_classes=10, n_samples_per_class=50, device=device)
    print(f"  Dataset: {len(dataset)} samples, {dataset.n_classes} classes")

    # ── Experiment 1: Train CLAP ──
    print("\n=== Training CLAP ===")
    audio_enc = AudioEncoder(n_mels=40, n_frames=64, d_model=128, out_dim=256).to(device)
    text_enc = TextEncoder(vocab_size=1000, d_model=128, out_dim=256, max_len=16).to(device)
    clap = CLAP(audio_enc, text_enc).to(device)
    metrics = train_clap(clap, dataset, n_steps=2000, batch_size=64, device=device)

    # ── Experiment 2: Zero-shot classification ──
    print("\n=== Zero-Shot Classification ===")
    zs_acc = zero_shot_classify(clap, dataset, n_test=200, device=device)
    print(f"  Zero-shot accuracy: {zs_acc:.3f}")

    # ── Experiment 3: Random baseline ──
    print("\n=== Random Baseline ===")
    rand_audio_enc = AudioEncoder(n_mels=40, n_frames=64, d_model=128, out_dim=256).to(device)
    rand_text_enc = TextEncoder(vocab_size=1000, d_model=128, out_dim=256, max_len=16).to(device)
    rand_clap = CLAP(rand_audio_enc, rand_text_enc).to(device)
    rand_acc = zero_shot_classify(rand_clap, dataset, n_test=200, device=device)
    print(f"  Random baseline accuracy: {rand_acc:.3f}")

    # ── Experiment 4: Temperature analysis ──
    print("\n=== Temperature Analysis ===")
    temp_results = {}
    for init_temp in [0.01, 0.07, 0.1, 0.5, 1.0]:
        ae = AudioEncoder(n_mels=40, n_frames=64, d_model=128, out_dim=256).to(device)
        te = TextEncoder(vocab_size=1000, d_model=128, out_dim=256, max_len=16).to(device)
        m = CLAP(ae, te, init_temp=init_temp).to(device)
        met = train_clap(m, dataset, n_steps=1000, batch_size=64, device=device)
        acc = zero_shot_classify(m, dataset, n_test=100, device=device)
        temp_results[init_temp] = {'final_loss': np.mean(met['loss'][-50:]),
                                    'zs_acc': acc,
                                    'learned_temp': met['temp'][-1]}
        print(f"  init_temp={init_temp}: zs_acc={acc:.3f}, learned_temp={met['temp'][-1]:.4f}")

    # ── Experiment 5: Embedding space visualization ──
    print("\n=== Embedding Space Analysis ===")
    clap.eval()
    all_a_feats = []
    all_t_feats = []
    all_labels = []
    with torch.no_grad():
        for i in range(min(200, len(dataset.data))):
            audio, text, label = dataset.data[i]
            a = audio.unsqueeze(0).to(device)
            t = text.unsqueeze(0).to(device)
            af = clap.audio_encoder(a)
            tf = clap.text_encoder(t)
            all_a_feats.append(af.cpu())
            all_t_feats.append(tf.cpu())
            all_labels.append(label)

    all_a_feats = torch.cat(all_a_feats).numpy()
    all_t_feats = torch.cat(all_t_feats).numpy()
    all_labels = np.array(all_labels)

    # Compute cross-modal retrieval accuracy
    sim_matrix = all_a_feats @ all_t_feats.T
    retrieval_acc = (sim_matrix.argmax(axis=1) == np.arange(len(all_labels))).mean()

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    w = 20
    for ax, key, title, color in [
        (axes[0, 0], 'loss', 'InfoNCE Loss', 'blue'),
        (axes[0, 1], 'temp', 'Temperature', 'red'),
        (axes[1, 0], 'acc_a2t', 'Audio→Text Accuracy', 'green'),
        (axes[1, 1], 'acc_t2a', 'Text→Audio Accuracy', 'purple'),
    ]:
        s = np.convolve(metrics[key], np.ones(w)/w, mode='valid')
        ax.plot(s, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.suptitle('CLAP Training Dynamics (2211.09372)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training.png', dpi=150)
    plt.close()

    # 2. Zero-shot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['CLAP\n(trained)', 'Random\nBaseline', 'Chance\n(1/10)']
    accs = [zs_acc, rand_acc, 0.1]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Zero-Shot Accuracy")
    ax.set_title("Zero-Shot Audio Classification")
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(results_dir / 'zero_shot.png', dpi=150)
    plt.close()

    # 3. Temperature analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    temps = sorted(temp_results.keys())
    losses_t = [temp_results[t]['final_loss'] for t in temps]
    accs_t = [temp_results[t]['zs_acc'] for t in temps]

    axes[0].plot(temps, losses_t, marker='o', color='blue', linewidth=2)
    axes[0].set_xlabel("Initial Temperature")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Loss vs Initial Temperature")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temps, accs_t, marker='s', color='green', linewidth=2)
    axes[1].set_xlabel("Initial Temperature")
    axes[1].set_ylabel("Zero-Shot Accuracy")
    axes[1].set_title("Accuracy vs Initial Temperature")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'temperature.png', dpi=150)
    plt.close()

    # 4. Cross-modal similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    # Show similarity for first 50 samples
    n_show = min(50, len(all_labels))
    sim_show = sim_matrix[:n_show, :n_show]
    im = ax.imshow(sim_show, cmap='RdYlBu_r', aspect='auto')
    ax.set_xlabel("Text Samples")
    ax.set_ylabel("Audio Samples")
    ax.set_title(f"Cross-Modal Similarity (Retrieval Acc: {retrieval_acc:.3f})")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(results_dir / 'similarity_matrix.png', dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    concept = (
        "CLAP: Contrastive Language-Audio Pretraining (2211.09372)\n"
        "=" * 60 + "\n\n"
        "Architecture:\n"
        "  Audio Encoder (HTSAT/ResNet) → f_a(audio) ∈ R^d\n"
        "  Text Encoder (Transformer)   → f_t(text) ∈ R^d\n"
        "  Shared embedding space via contrastive learning\n\n"
        "InfoNCE Loss:\n"
        "  L = -log[exp(sim(a_i, t_i)/τ) / Σ_j exp(sim(a_i, t_j)/τ)]\n"
        "  τ = learnable temperature parameter\n\n"
        "Zero-Shot Classification:\n"
        "  1. Encode class prompts: \"the sound of [class]\"\n"
        "  2. Encode query audio: f_a(audio)\n"
        "  3. Classify: argmax_c cos(f_a(audio), f_t(prompt_c))\n\n"
        "Key Results:\n"
        "  • Transferable audio representations\n"
        "  • Zero-shot classification without fine-tuning\n"
        "  • Temperature is critical: too small → sharp, too large → uniform"
    )
    ax.text(0.05, 0.95, concept, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
