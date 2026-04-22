"""
Minimal BERT Reproduction
==========================
Reproduces the core ideas from "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding" (1810.04805):
1. Masked Language Model (MLM) pre-training objective
2. Next Sentence Prediction (NSP) objective
3. Bidirectional attention (vs GPT's causal mask)
4. [CLS] token for sequence-level classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── BERT Model ──

class BertEmbedding(nn.Module):
    """Token + Position + Segment embeddings."""
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(2, d_model)  # Segment A/B
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        x = self.token_emb(input_ids) + self.pos_emb(positions) + self.seg_emb(segment_ids)
        return self.norm(x)


class BertEncoder(nn.Module):
    """Bidirectional Transformer encoder (no causal mask)."""
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return x


class BertForPreTraining(nn.Module):
    """BERT with MLM and NSP heads."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.emb = BertEmbedding(vocab_size, d_model, max_len)
        self.encoder = BertEncoder(d_model, n_heads, d_ff, n_layers)
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        # NSP head
        self.nsp_head = nn.Linear(d_model, 2)

        # Special tokens
        self.PAD = 0
        self.MASK = 1
        self.CLS = 2
        self.SEP = 3

    def forward(self, input_ids, segment_ids=None, padding_mask=None):
        x = self.emb(input_ids, segment_ids)
        x = self.encoder(x, padding_mask=padding_mask)
        # MLM: predict masked tokens
        mlm_logits = self.mlm_head(x)
        # NSP: use [CLS] representation
        cls_repr = x[:, 0]
        nsp_logits = self.nsp_head(cls_repr)
        return mlm_logits, nsp_logits


# ── Synthetic Data ──

def generate_pair(vocab_size, seq_len, device='cpu'):
    """Generate a sentence pair for NSP training.
    IsNext: second sentence follows first.
    NotNext: second sentence is random.
    """
    half = seq_len // 2 - 1  # Account for [CLS] and [SEP]
    # Sentence A: random tokens
    sent_a = torch.randint(4, vocab_size, (half,), device=device)
    # Sentence B: either continuation or random
    is_next = torch.randint(0, 2, (1,)).item()
    if is_next:
        sent_b = torch.randint(4, vocab_size, (half,), device=device)
    else:
        sent_b = torch.randint(4, vocab_size, (half,), device=device)

    # Build input: [CLS] sent_a [SEP] sent_b [SEP] [PAD...]
    cls = torch.tensor([2], device=device)
    sep = torch.tensor([3], device=device)
    pad = torch.tensor([0], device=device)

    input_ids = torch.cat([cls, sent_a, sep, sent_b, sep])
    # Pad to seq_len
    if len(input_ids) < seq_len:
        input_ids = torch.cat([input_ids, pad.repeat(seq_len - len(input_ids))])
    input_ids = input_ids[:seq_len]

    # Segment IDs: 0 for [CLS] sent_a [SEP], 1 for sent_b [SEP] [PAD...]
    seg_a_len = 1 + half + 1
    segment_ids = torch.cat([
        torch.zeros(seg_a_len, device=device, dtype=torch.long),
        torch.ones(seq_len - seg_a_len, device=device, dtype=torch.long)
    ])

    return input_ids, segment_ids, is_next


def mask_input(input_ids, mask_prob=0.15, device='cpu'):
    """Apply MLM masking: 80% [MASK], 10% random, 10% unchanged."""
    mask_token = 1
    labels = input_ids.clone()
    prob = torch.rand(input_ids.shape, device=device)
    # Don't mask special tokens (PAD=0, MASK=1, CLS=2, SEP=3)
    special = (input_ids <= 3)
    mask = (prob < mask_prob) & ~special

    # 80% replace with [MASK]
    prob_replace = torch.rand(input_ids.shape, device=device)
    input_ids = input_ids.clone()
    input_ids[mask & (prob_replace < 0.8)] = mask_token
    # 10% replace with random token
    random_tokens = torch.randint(4, input_ids.max().item() + 1, input_ids.shape, device=device)
    input_ids[mask & (prob_replace >= 0.8) & (prob_replace < 0.9)] = random_tokens[mask & (prob_replace >= 0.8) & (prob_replace < 0.9)]
    # 10% unchanged (already the case)

    # Set labels for non-masked positions to -100 (ignore)
    labels[~mask] = -100
    return input_ids, labels, mask


# ── Training ──

def train_bert(model, n_steps=5000, batch_size=32, seq_len=32,
               vocab_size=200, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mlm_losses = []
    nsp_losses = []
    nsp_accs = []

    for step in range(n_steps):
        # Generate batch
        input_ids_list, seg_ids_list, is_next_list = [], [], []
        for _ in range(batch_size):
            ids, segs, is_next = generate_pair(vocab_size, seq_len, device)
            input_ids_list.append(ids)
            seg_ids_list.append(segs)
            is_next_list.append(is_next)

        input_ids = torch.stack(input_ids_list)
        seg_ids = torch.stack(seg_ids_list)
        is_next = torch.tensor(is_next_list, device=device)

        # Apply MLM masking
        masked_input, mlm_labels, mask_positions = mask_input(input_ids, device=device)

        # Padding mask
        padding_mask = (masked_input == 0)

        # Forward
        mlm_logits, nsp_logits = model(masked_input, seg_ids, padding_mask)

        # MLM loss
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, vocab_size), mlm_labels.view(-1), ignore_index=-100
        )
        # NSP loss
        nsp_loss = F.cross_entropy(nsp_logits, is_next)

        loss = mlm_loss + nsp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        mlm_losses.append(mlm_loss.item())
        nsp_losses.append(nsp_loss.item())
        nsp_pred = nsp_logits.argmax(dim=-1)
        nsp_accs.append((nsp_pred == is_next).float().mean().item())

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | MLM Loss: {mlm_loss.item():.4f} | "
                  f"NSP Loss: {nsp_loss.item():.4f} | NSP Acc: {nsp_accs[-1]:.3f}")

    return mlm_losses, nsp_losses, nsp_accs


# ── Compare with Causal (GPT-style) ──

class CausalTransformer(nn.Module):
    """GPT-style causal transformer for comparison."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.norm(self.token_emb(input_ids) + self.pos_emb(positions))
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        return self.head(x)


def train_causal(model, n_steps=5000, batch_size=32, seq_len=32,
                 vocab_size=200, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        labels[labels <= 3] = -100  # Ignore special tokens

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "06-bert"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 200
    seq_len = 32
    d_model = 128
    n_heads = 4
    n_layers = 4
    d_ff = 512

    # Train BERT
    print("=== Training BERT (MLM + NSP) ===")
    bert = BertForPreTraining(vocab_size, d_model, n_heads, d_ff, n_layers).to(device)
    mlm_losses, nsp_losses, nsp_accs = train_bert(
        bert, n_steps=5000, device=device, vocab_size=vocab_size, seq_len=seq_len
    )

    # Train Causal baseline
    print("\n=== Training Causal (GPT-style) LM ===")
    causal = CausalTransformer(vocab_size, d_model, n_heads, d_ff, n_layers).to(device)
    causal_losses = train_causal(
        causal, n_steps=5000, device=device, vocab_size=vocab_size, seq_len=seq_len
    )

    # ── Visualization ──
    window = 50

    # 1. MLM & NSP training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    mlm_s = np.convolve(mlm_losses, np.ones(window)/window, mode='valid')
    nsp_s = np.convolve(nsp_losses, np.ones(window)/window, mode='valid')
    nsp_acc_s = np.convolve(nsp_accs, np.ones(window)/window, mode='valid')

    axes[0].plot(mlm_s, color='blue')
    axes[0].set_title("MLM Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nsp_s, color='orange')
    axes[1].set_title("NSP Loss")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(nsp_acc_s, color='green')
    axes[2].set_title("NSP Accuracy")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Accuracy")
    axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("BERT Pre-training: MLM + NSP", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "bert_training.png", dpi=150)
    plt.close()

    # 2. Bidirectional vs Causal comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    mlm_full = np.convolve(mlm_losses, np.ones(window)/window, mode='valid')
    causal_s = np.convolve(causal_losses, np.ones(window)/window, mode='valid')
    min_len = min(len(mlm_full), len(causal_s))
    ax.plot(mlm_full[:min_len], label='BERT (Bidirectional MLM)', color='blue')
    ax.plot(causal_s[:min_len], label='GPT (Causal LM)', color='red')
    ax.set_title("Bidirectional vs Causal Language Modeling")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "bidirectional_vs_causal.png", dpi=150)
    plt.close()

    # 3. Attention pattern comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Extract attention from BERT (bidirectional)
    bert.eval()
    with torch.no_grad():
        input_ids = torch.randint(4, vocab_size, (1, seq_len), device=device)
        seg_ids = torch.zeros(1, seq_len, device=device, dtype=torch.long)
        x = bert.emb(input_ids, seg_ids)
        # Compute attention weights manually from first layer
        layer = bert.encoder.layers[0]
        W_q = layer.self_attn.in_proj_weight[:d_model]
        W_k = layer.self_attn.in_proj_weight[d_model:2*d_model]
        W_v = layer.self_attn.in_proj_weight[2*d_model:]
        b_q = layer.self_attn.in_proj_bias[:d_model]
        b_k = layer.self_attn.in_proj_bias[d_model:2*d_model]
        q = (x @ W_q.T + b_q)  # (1, T, d_model)
        k = (x @ W_k.T + b_k)  # (1, T, d_model)
        # Head-averaged attention for visualization
        head_dim = d_model // n_heads
        q_heads = q.view(1, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        k_heads = k.view(1, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        attn = (q_heads @ k_heads.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1).mean(dim=1).squeeze(0).cpu().numpy()  # Average over heads

    axes[0].imshow(attn, cmap='Blues')
    axes[0].set_title("BERT Attention (Bidirectional)")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")

    # Causal attention
    causal.eval()
    with torch.no_grad():
        x_c = causal.norm(causal.token_emb(input_ids) + causal.pos_emb(
            torch.arange(seq_len, device=device).unsqueeze(0)))
        layer_c = causal.layers[0]
        W_q_c = layer_c.self_attn.in_proj_weight[:d_model]
        W_k_c = layer_c.self_attn.in_proj_weight[d_model:2*d_model]
        b_q_c = layer_c.self_attn.in_proj_bias[:d_model]
        b_k_c = layer_c.self_attn.in_proj_bias[d_model:2*d_model]
        q_c = (x_c @ W_q_c.T + b_q_c)
        k_c = (x_c @ W_k_c.T + b_k_c)
        q_heads_c = q_c.view(1, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        k_heads_c = k_c.view(1, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        attn_c = (q_heads_c @ k_heads_c.transpose(-2, -1)) / (head_dim ** 0.5)
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_c = attn_c.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_c = F.softmax(attn_c, dim=-1).mean(dim=1).squeeze(0).cpu().numpy()

    axes[1].imshow(attn_c, cmap='Reds')
    axes[1].set_title("GPT Attention (Causal)")
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")

    plt.suptitle("Attention Pattern: Bidirectional vs Causal", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "attention_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
