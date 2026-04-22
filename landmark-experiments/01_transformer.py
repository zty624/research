"""
Minimal Transformer Reproduction
=================================
Reproduces the core ideas from "Attention Is All You Need" (1706.03762):
1. Scaled dot-product attention
2. Multi-head attention
3. Positional encoding (sinusoidal)
4. Encoder-decoder architecture
5. Minimal sequence transduction task (copy/reverse)

Trains on a simple task: reverse a sequence of tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from pathlib import Path

# ── Scaled Dot-Product Attention ──

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (batch, heads, seq, d_k)"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


# ── Multi-Head Attention ──

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.W_q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out), attn


# ── Position-wise FFN ──

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ── Sinusoidal Positional Encoding ──

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── Encoder Block ──

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


# ── Decoder Block ──

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out, cross_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x, cross_weights


# ── Full Transformer ──

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def make_tgt_mask(self, sz):
        return torch.tril(torch.ones(sz, sz)).unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):
        tgt_mask = self.make_tgt_mask(tgt.size(1)).to(tgt.device)

        src_emb = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))

        enc = src_emb
        for layer in self.encoder:
            enc, _ = layer(enc)

        dec = tgt_emb
        for layer in self.decoder:
            dec, _ = layer(dec, enc, tgt_mask=tgt_mask)

        return self.fc_out(dec)


# ── Training ──

def generate_reverse_task(batch_size, seq_len, vocab_size, device):
    """Input: random token sequence. Target: reversed sequence."""
    src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    tgt = src.flip(dims=[1])
    # Decoder input: <sos> + target[:-1]
    sos = torch.full((batch_size, 1), 0, device=device)
    tgt_input = torch.cat([sos, tgt[:, :-1]], dim=1)
    return src, tgt_input, tgt


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 10  # digits 0-9, 0=SOS
    seq_len = 8
    d_model = 64
    n_heads = 4
    n_layers = 2
    batch_size = 64
    epochs = 50

    model = Transformer(vocab_size, d_model, n_heads, d_ff=256, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    # Warmup + inverse sqrt schedule
    warmup_steps = 400
    step = 0

    losses = []
    accs = []

    for epoch in range(epochs):
        model.train()
        src, tgt_input, tgt = generate_reverse_task(batch_size, seq_len, vocab_size, device)

        # LR schedule
        step += 1
        lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        out = model(src, tgt_input)
        loss = criterion(out.view(-1, vocab_size), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        pred = out.argmax(dim=-1)
        acc = (pred == tgt).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | LR: {lr:.6f}")

    # ── Visualization ──
    results_dir = Path(__file__).parent / "results" / "01-transformer"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax1.set_title("Training Loss (Sequence Reversal)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(accs)
    ax2.set_title("Token Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # Visualize attention patterns
    model.eval()
    with torch.no_grad():
        src, tgt_input, tgt = generate_reverse_task(1, seq_len, vocab_size, device)
        # Get attention weights from first encoder layer
        src_emb = model.pos_enc(model.embed(src) * math.sqrt(model.d_model))
        enc, attn_weights = model.encoder[0](src_emb)
        attn = attn_weights[0, 0].cpu().numpy()  # first head

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap='Blues')
    ax.set_title("Self-Attention Pattern (Encoder, Head 0)")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(results_dir / "attention_pattern.png", dpi=150)
    plt.close()

    # Test: show a few examples
    model.eval()
    with torch.no_grad():
        src, tgt_input, tgt = generate_reverse_task(4, seq_len, vocab_size, device)
        out = model(src, tgt_input)
        pred = out.argmax(dim=-1)

    print("\n=== Sample Predictions ===")
    for i in range(4):
        src_str = src[i].cpu().tolist()
        tgt_str = tgt[i].cpu().tolist()
        pred_str = pred[i].cpu().tolist()
        print(f"  Input:    {src_str}")
        print(f"  Target:   {tgt_str}")
        print(f"  Predicted:{pred_str}")
        print()

    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    train()
