"""
Minimal BERT-Style MLM Pretraining Reproduction
================================================
Reproduces core ideas from BERT (1810.04805, Devlin et al., 2018):
1. Bidirectional encoder: masked language model (MLM) objective
2. [MASK] token prediction: predict masked tokens using full context
3. Next Sentence Prediction (NSP): binary classification of sentence pairs
4. Transformer encoder stack with learned position embeddings
5. Compare: masked LM vs causal LM on the same data
6. Show: MLM convergence, attention patterns, bidirectional advantage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── BERT Model ──

class BertEncoder(nn.Module):
    """BERT-style bidirectional transformer encoder."""
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=4,
                 max_len=64, d_ff=256):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(2, d_model)  # segment A/B

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, activation='gelu', batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, segments=None, mask=None):
        """x: (B, T) token indices → (B, T, d_model) hidden states."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        if segments is not None:
            h = h + self.seg_emb(segments)

        for layer in self.layers:
            h = layer(h, src_key_padding_mask=mask)
        return self.norm(h)


class BertForMLM(nn.Module):
    """BERT with MLM head and optional NSP head."""
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=4,
                 max_len=64, d_ff=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = BertEncoder(vocab_size, d_model, n_heads, n_layers,
                                   max_len, d_ff)
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        # NSP head
        self.nsp_head = nn.Linear(d_model, 2)

        # Special tokens
        self.mask_token = vocab_size - 1
        self.cls_token = vocab_size - 2
        self.sep_token = vocab_size - 3

    def forward(self, x, segments=None, mask_positions=None):
        """Forward pass returning MLM logits and NSP logits."""
        h = self.encoder(x, segments)
        mlm_logits = self.mlm_head(h)  # (B, T, vocab_size)
        cls_hidden = h[:, 0, :]  # [CLS] token
        nsp_logits = self.nsp_head(cls_hidden)  # (B, 2)
        return mlm_logits, nsp_logits


class CausalLM(nn.Module):
    """Causal (autoregressive) LM for comparison."""
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=4,
                 max_len=64, d_ff=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, activation='gelu', batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.head(self.norm(h))


# ── Synthetic Data ──

class SyntheticCorpus:
    """Synthetic text-like corpus with structure for MLM/NSP tasks."""
    def __init__(self, vocab_size=256, seq_len=32, device='cpu'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        self.special = 3  # last 3 tokens are special
        self.content_size = vocab_size - self.special

        # Create topic-specific vocab clusters
        n_topics = 8
        self.topic_vocabs = []
        for _ in range(n_topics):
            subset = np.random.choice(self.content_size, size=self.content_size // 2, replace=False)
            self.topic_vocabs.append(subset)

    def generate_sentence(self, topic=None, length=None):
        """Generate a sentence from a topic cluster."""
        if topic is None:
            topic = np.random.randint(len(self.topic_vocabs))
        if length is None:
            length = self.seq_len // 2 - 2
        vocab = self.topic_vocabs[topic]
        tokens = np.random.choice(vocab, size=length).tolist()
        return tokens, topic

    def generate_mlm_batch(self, batch_size, mask_prob=0.15):
        """Generate a batch with MLM masking and NSP labels."""
        half = self.seq_len // 2
        # Create sentence pairs (50% same topic = next sentence, 50% different)
        input_ids = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        segments = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        labels = torch.full((batch_size, self.seq_len), -100, dtype=torch.long, device=self.device)
        nsp_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            sent_a, topic_a = self.generate_sentence(length=half - 3)
            is_next = np.random.random() < 0.5
            if is_next:
                sent_b, _ = self.generate_sentence(topic=topic_a, length=half - 3)
                nsp_labels[i] = 1
            else:
                other_topic = (topic_a + np.random.randint(1, len(self.topic_vocabs))) % len(self.topic_vocabs)
                sent_b, _ = self.generate_sentence(topic=other_topic, length=half - 3)

            # Build input: [CLS] sent_a [SEP] sent_b [SEP] + padding
            tokens = [self.vocab_size - 2] + sent_a + [self.vocab_size - 3] + sent_b + [self.vocab_size - 3]
            pad_len = self.seq_len - len(tokens)
            tokens = tokens + [0] * pad_len

            input_ids[i] = torch.tensor(tokens)
            # Segments: 0 for [CLS]+sent_a+[SEP], 1 for sent_b+[SEP]+pad
            seg_a_len = len(sent_a) + 2  # [CLS] + sent_a + [SEP]
            segments[i, seg_a_len:] = 1

            # Apply MLM masking (only on content tokens)
            for j in range(1, min(len(tokens) - 1, self.seq_len - 1)):
                if tokens[j] < self.content_size and np.random.random() < mask_prob:
                    labels[i, j] = tokens[j]
                    r = np.random.random()
                    if r < 0.8:
                        input_ids[i, j] = self.vocab_size - 1  # [MASK]
                    elif r < 0.9:
                        input_ids[i, j] = np.random.randint(self.content_size)  # random
                    # else: keep original

        return input_ids, segments, labels, nsp_labels

    def generate_causal_batch(self, batch_size):
        """Generate a batch for causal LM training."""
        input_ids = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            sent, _ = self.generate_sentence(length=self.seq_len - 2)
            tokens = sent[:self.seq_len]
            pad_len = self.seq_len - len(tokens)
            tokens = tokens + [0] * pad_len
            input_ids[i] = torch.tensor(tokens)
        return input_ids


# ── Training ──

def train_bert(model, corpus, n_steps=3000, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    metrics = {'mlm_loss': [], 'nsp_loss': [], 'total': [], 'mlm_acc': []}

    for step in range(n_steps):
        input_ids, segments, labels, nsp_labels = corpus.generate_mlm_batch(batch_size)

        mlm_logits, nsp_logits = model(input_ids, segments)

        # MLM loss (only on masked positions)
        mlm_loss = F.cross_entropy(
            mlm_logits.reshape(-1, model.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # NSP loss
        nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)

        total_loss = mlm_loss + 0.5 * nsp_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # MLM accuracy on masked tokens
        with torch.no_grad():
            mask = labels != -100
            if mask.any():
                pred = mlm_logits.argmax(dim=-1)
                acc = (pred[mask] == labels[mask]).float().mean().item()
            else:
                acc = 0.0

        metrics['mlm_loss'].append(mlm_loss.item())
        metrics['nsp_loss'].append(nsp_loss.item())
        metrics['total'].append(total_loss.item())
        metrics['mlm_acc'].append(acc)

        if (step + 1) % 500 == 0:
            print(f"  [BERT] Step {step+1} | MLM: {mlm_loss.item():.4f} | "
                  f"NSP: {nsp_loss.item():.4f} | MLM Acc: {acc:.3f}")

    return metrics


def train_causal(model, corpus, n_steps=3000, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    metrics = {'loss': [], 'acc': []}

    for step in range(n_steps):
        input_ids = corpus.generate_causal_batch(batch_size)

        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]

        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                               targets.reshape(-1), ignore_index=0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = targets != 0
            acc = (pred[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0

        metrics['loss'].append(loss.item())
        metrics['acc'].append(acc)

        if (step + 1) % 500 == 0:
            print(f"  [Causal] Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    return metrics


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "105-bert-mlm"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 256
    seq_len = 32
    d_model = 128
    n_steps = 3000

    corpus = SyntheticCorpus(vocab_size, seq_len, device=device)

    # Train BERT
    print("=== Training BERT (MLM + NSP) ===")
    bert = BertForMLM(vocab_size, d_model, n_heads=4, n_layers=4, max_len=seq_len).to(device)
    bert_params = sum(p.numel() for p in bert.parameters())
    print(f"  BERT params: {bert_params:,}")
    bert_m = train_bert(bert, corpus, n_steps=n_steps, device=device)

    # Train Causal LM
    print("\n=== Training Causal LM ===")
    causal = CausalLM(vocab_size, d_model, n_heads=4, n_layers=4, max_len=seq_len).to(device)
    causal_params = sum(p.numel() for p in causal.parameters())
    print(f"  Causal LM params: {causal_params:,}")
    causal_m = train_causal(causal, corpus, n_steps=n_steps, device=device)

    # ── Visualization ──
    w = 30

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MLM loss vs Causal loss
    bert_s = np.convolve(bert_m['mlm_loss'], np.ones(w)/w, mode='valid')
    causal_s = np.convolve(causal_m['loss'], np.ones(w)/w, mode='valid')
    axes[0, 0].plot(bert_s, label='BERT MLM Loss', color='blue')
    axes[0, 0].plot(causal_s, label='Causal LM Loss', color='red')
    axes[0, 0].set_title('Training Loss: BERT vs Causal LM')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss (smoothed)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. MLM accuracy vs Causal accuracy
    bert_acc = np.convolve(bert_m['mlm_acc'], np.ones(w)/w, mode='valid')
    causal_acc = np.convolve(causal_m['acc'], np.ones(w)/w, mode='valid')
    axes[0, 1].plot(bert_acc, label='BERT MLM Acc', color='blue')
    axes[0, 1].plot(causal_acc, label='Causal LM Acc', color='red')
    axes[0, 1].set_title('Token Prediction Accuracy')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy (smoothed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. NSP loss
    nsp_s = np.convolve(bert_m['nsp_loss'], np.ones(w)/w, mode='valid')
    axes[1, 0].plot(nsp_s, color='green')
    axes[1, 0].set_title('NSP Loss (Next Sentence Prediction)')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('NSP Loss (smoothed)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Attention pattern (last layer, first head)
    bert.eval()
    with torch.no_grad():
        input_ids, segments, _, _ = corpus.generate_mlm_batch(1)
        h = bert.encoder.token_emb(input_ids) + bert.encoder.pos_emb(
            torch.arange(seq_len, device=device).unsqueeze(0))
        if segments is not None:
            h = h + bert.encoder.seg_emb(segments)
        # Get attention from first layer
        attn_out, attn_w = bert.encoder.layers[0].self_attn(
            h, h, h, need_weights=True)
    attn_np = attn_w[0].cpu().numpy()
    axes[1, 1].imshow(attn_np, cmap='Blues', aspect='auto')
    axes[1, 1].set_title('BERT Self-Attention Pattern (Bidirectional)')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')

    plt.suptitle('BERT-Style MLM Pretraining (1810.04805)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'bert_mlm_results.png', dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BERT diagram
    ax = axes[0]
    ax.axis('off')
    ax.set_title("BERT: Bidirectional + MLM", fontsize=13, fontweight='bold')
    texts = [
        ("Input", "[CLS] The [MASK] sat on [MASK] mat [SEP]", 0.5, 0.9),
        ("Encoder", "Full Self-Attention\n(every token attends\nto every other token)", 0.5, 0.6),
        ("MLM Head", "Predict masked tokens\nusing BOTH left & right context", 0.5, 0.3),
        ("NSP Head", "Is sentence B the\nnext sentence after A?", 0.5, 0.05),
    ]
    for name, desc, x, y in texts:
        ax.text(x, y, f"{name}\n{desc}", fontsize=9, ha='center', va='center',
                fontfamily='monospace', color='blue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.0)

    # Causal LM diagram
    ax = axes[1]
    ax.axis('off')
    ax.set_title("Causal LM: Autoregressive", fontsize=13, fontweight='bold')
    texts = [
        ("Input", "The cat sat on the mat", 0.5, 0.9),
        ("Decoder", "Causal Self-Attention\n(each token only attends\nto LEFT context)", 0.5, 0.6),
        ("LM Head", "Predict NEXT token\nusing only past context", 0.5, 0.3),
        ("Generation", "Autoregressive:\ngenerate one token at a time", 0.5, 0.05),
    ]
    for name, desc, x, y in texts:
        ax.text(x, y, f"{name}\n{desc}", fontsize=9, ha='center', va='center',
                fontfamily='monospace', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.0)

    plt.suptitle('BERT vs Causal LM: Architecture Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'bert_vs_causal_concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
