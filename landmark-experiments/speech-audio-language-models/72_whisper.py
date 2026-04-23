"""
Minimal Whisper-style ASR Reproduction
=======================================
Reproduces the core ideas from "Robust Speech Recognition via Large-Scale
Weak Supervision" (Radford et al., 2022, 2212.04356):
1. Encoder-decoder Transformer: mel spectrogram input -> text output
2. Mel spectrogram feature extraction from audio
3. Cross-attention between encoder and decoder
4. Synthetic task: sine waves at different frequencies mapped to characters
5. Demonstrates attention patterns and training convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Mel Spectrogram ──

def _mel_filterbank(n_mels, n_fft, sample_rate, device):
    """Create a mel filterbank matrix."""
    fmax = sample_rate / 2
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return torch.tensor(filterbank, dtype=torch.float32, device=device)


def mel_spectrogram(waveform, sample_rate=16000, n_fft=400, hop_length=160,
                    n_mels=80):
    """Compute log mel spectrogram from raw waveform (batched)."""
    window = torch.hann_window(n_fft, device=waveform.device)

    stft = torch.stft(waveform, n_fft, hop_length=hop_length,
                       window=window, return_complex=True)
    magnitude = stft.abs().pow(2)  # (B, n_fft//2+1, n_frames)

    mel_basis = _mel_filterbank(n_mels, n_fft, sample_rate,
                                 device=waveform.device)
    mel_spec = mel_basis @ magnitude  # (B, n_mels, n_frames)
    mel_spec = torch.log(mel_spec + 1e-8)
    return mel_spec


# ── Positional Encoding ──

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.pe[:, :x.size(1)]


# ── Whisper Encoder ──

class WhisperEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D)"""
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + self.dropout(h)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class WhisperEncoder(nn.Module):
    def __init__(self, n_mels=80, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=3000):
        super().__init__()
        # Conv feature extraction (Whisper uses 2 conv layers before Transformer)
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel):
        """mel: (B, n_mels, T_mel)"""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))  # (B, d_model, T_mel//2)
        x = x.transpose(1, 2)  # (B, T, d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)  # (B, T, d_model)


# ── Whisper Decoder ──

class WhisperDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                 batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None):
        """x: (B, T_dec, D), enc_out: (B, T_enc, D)"""
        # Masked self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=tgt_mask)
        x = x + self.dropout(h)

        # Cross-attention
        h = self.norm2(x)
        h, cross_attn_weights = self.cross_attn(h, enc_out, enc_out)
        x = x + self.dropout(h)

        # FFN
        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))

        return x, cross_attn_weights


class WhisperDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=448):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            WhisperDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, enc_out):
        """tokens: (B, T_dec), enc_out: (B, T_enc, D)"""
        x = self.token_emb(tokens)
        x = self.pos_enc(x)

        # Causal mask
        T = x.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        tgt_mask = tgt_mask.masked_fill(tgt_mask, float('-inf'))

        cross_attn_weights = None
        for layer in self.layers:
            x, cross_attn_weights = layer(x, enc_out, tgt_mask)

        x = self.norm(x)
        logits = self.output_proj(x)  # (B, T_dec, vocab_size)
        return logits, cross_attn_weights


# ── Full Whisper Model ──

class WhisperModel(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=128, n_heads=4,
                 n_layers=4, d_ff=512):
        super().__init__()
        self.encoder = WhisperEncoder(n_mels, d_model, n_heads, n_layers, d_ff)
        self.decoder = WhisperDecoder(vocab_size, d_model, n_heads, n_layers, d_ff)

    def forward(self, mel, tokens):
        """mel: (B, n_mels, T_mel), tokens: (B, T_dec)"""
        enc_out = self.encoder(mel)
        logits, cross_attn = self.decoder(tokens, enc_out)
        return logits, cross_attn


# ── Synthetic Data: Frequency -> Character Mapping ──

class SyntheticSpeechDataset:
    """
    Generate sine waves at different frequencies, each mapped to a character.
    Freq bins: 100, 200, 300, ... Hz -> characters 'a', 'b', 'c', ...
    Each "utterance" is a sequence of 3-5 frequency segments concatenated.
    """
    def __init__(self, n_chars=8, sample_rate=16000, segment_duration=0.3,
                 freq_base=100, freq_step=100, device='cpu'):
        self.n_chars = n_chars
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.freq_base = freq_base
        self.freq_step = freq_step
        self.device = device

        # Special tokens
        self.sos_token = n_chars      # <sos>
        self.eos_token = n_chars + 1  # <eos>
        self.vocab_size = n_chars + 2

    def generate_batch(self, batch_size=16, min_len=3, max_len=5):
        """Generate a batch of synthetic speech + token sequences."""
        waveforms = []
        token_seqs = []

        for _ in range(batch_size):
            seq_len = np.random.randint(min_len, max_len + 1)
            char_ids = np.random.randint(0, self.n_chars, size=seq_len)

            # Build waveform: concatenate sine segments
            waveform = np.zeros(self.segment_len * seq_len, dtype=np.float32)
            for i, cid in enumerate(char_ids):
                freq = self.freq_base + cid * self.freq_step
                t = np.arange(self.segment_len, dtype=np.float32) / self.sample_rate
                segment = np.sin(2 * np.pi * freq * t)
                # Add slight noise for realism
                segment += np.random.randn(self.segment_len).astype(np.float32) * 0.02
                start = i * self.segment_len
                waveform[start:start + self.segment_len] = segment

            waveforms.append(waveform)

            # Token sequence: <sos> char1 char2 ... charN <eos>
            tokens = [self.sos_token] + list(char_ids) + [self.eos_token]
            token_seqs.append(tokens)

        # Pad waveforms to same length
        max_wav_len = max(len(w) for w in waveforms)
        wav_tensor = torch.zeros(batch_size, max_wav_len, device=self.device)
        for i, w in enumerate(waveforms):
            wav_tensor[i, :len(w)] = torch.tensor(w)

        # Pad token sequences
        max_tok_len = max(len(t) for t in token_seqs)
        tok_tensor = torch.zeros(batch_size, max_tok_len, dtype=torch.long,
                                  device=self.device)
        tok_tensor.fill_(self.eos_token)  # pad with eos
        for i, t in enumerate(token_seqs):
            tok_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        return wav_tensor, tok_tensor


# ── Training ──

def train_whisper(model, dataset, n_steps=3000, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    accs = []

    for step in range(n_steps):
        waveform, tokens = dataset.generate_batch(batch_size=32)

        # Compute mel spectrogram
        mel = mel_spectrogram(waveform, sample_rate=dataset.sample_rate,
                              n_mels=80, n_fft=400, hop_length=160)

        # Decoder input: tokens[:-1], target: tokens[1:]
        dec_input = tokens[:, :-1]
        target = tokens[:, 1:]

        logits, _ = model(mel, dec_input)

        # Loss: cross-entropy over vocabulary
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1),
                               ignore_index=dataset.eos_token)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accuracy (exclude padding)
        pred = logits.argmax(-1)
        mask = target != dataset.eos_token
        acc = (pred[mask] == target[mask]).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs


# ── Inference (Greedy Decoding) ──

@torch.no_grad()
def greedy_decode(model, mel, dataset, max_len=20):
    """Greedy decode from mel spectrogram."""
    model.eval()
    enc_out = model.encoder(mel)

    # Start with <sos>
    B = mel.size(0)
    tokens = torch.full((B, 1), dataset.sos_token, dtype=torch.long,
                         device=mel.device)

    for _ in range(max_len):
        logits, _ = model.decoder(tokens, enc_out)
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop if all sequences produced <eos>
        if (next_token == dataset.eos_token).all():
            break

    return tokens


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "72-whisper"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ──
    print("=== Whisper-style ASR: Synthetic Frequency-to-Character ===\n")
    dataset = SyntheticSpeechDataset(n_chars=8, device=device)
    model = WhisperModel(
        vocab_size=dataset.vocab_size,
        n_mels=80, d_model=128, n_heads=4, n_layers=4, d_ff=512
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Vocabulary size: {dataset.vocab_size} (8 chars + <sos> + <eos>)")

    # ── Train ──
    print("\nTraining...")
    losses, accs = train_whisper(model, dataset, n_steps=3000, device=device)

    # ── Visualization ──

    # 1. Training convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    window = 30
    loss_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    acc_smooth = np.convolve(accs, np.ones(window)/window, mode='valid')

    ax1.plot(loss_smooth, color='blue')
    ax1.set_title("Whisper Training Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss (smoothed)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(acc_smooth, color='green')
    ax2.set_title("Whisper Token Prediction Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (smoothed)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Whisper-style ASR: Training on Synthetic Frequency-Character Data",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "training_convergence.png", dpi=150)
    plt.close()

    # 2. Attention patterns
    print("\nVisualizing attention patterns...")
    model.eval()
    waveform, tokens = dataset.generate_batch(batch_size=1)
    mel = mel_spectrogram(waveform, sample_rate=dataset.sample_rate)

    with torch.no_grad():
        enc_out = model.encoder(mel.to(device))
        dec_input = tokens[:, :-1].to(device)
        logits, cross_attn = model.decoder(dec_input, enc_out)
        # cross_attn: (B, T_dec, T_enc) from last decoder layer

    attn = cross_attn[0].cpu().numpy()  # (T_dec, T_enc)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(attn, aspect='auto', cmap='viridis')
    ax.set_title("Cross-Attention: Decoder Positions (y) attending to Encoder Frames (x)")
    ax.set_xlabel("Encoder Time Frame")
    ax.set_ylabel("Decoder Token Position")
    plt.colorbar(im, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(results_dir / "cross_attention.png", dpi=150)
    plt.close()

    # 3. Mel spectrogram of a sample
    char_names = [chr(ord('a') + i) for i in range(8)]
    sample_mel = mel[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.imshow(sample_mel, aspect='auto', origin='lower', cmap='magma')
    ax1.set_title("Mel Spectrogram of Synthetic Speech (freq -> char mapping)")
    ax1.set_xlabel("Time Frame")
    ax1.set_ylabel("Mel Bin")

    # Show waveform
    ax2.plot(waveform[0].cpu().numpy())
    ax2.set_title("Raw Waveform")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Amplitude")

    # Annotate segments
    seg_len_samples = dataset.segment_len
    n_segs = waveform.size(1) // seg_len_samples
    for i in range(n_segs):
        start = i * seg_len_samples
        mid = start + seg_len_samples // 2
        char_id = tokens[0, i + 1].item()  # +1 for <sos>
        if char_id < dataset.n_chars:
            ax2.annotate(char_names[char_id], xy=(mid, 0.8),
                        fontsize=12, fontweight='bold', ha='center',
                        color='red')

    plt.tight_layout()
    plt.savefig(results_dir / "mel_spectrogram.png", dpi=150)
    plt.close()

    # 4. Inference demo
    print("\nInference demo...")
    waveform, tokens = dataset.generate_batch(batch_size=4)
    mel = mel_spectrogram(waveform, sample_rate=dataset.sample_rate)
    pred_tokens = greedy_decode(model, mel.to(device), dataset)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    for i in range(4):
        # Ground truth
        gt_chars = []
        for t in tokens[i].cpu().numpy():
            if t == dataset.sos_token:
                continue
            elif t == dataset.eos_token:
                break
            gt_chars.append(char_names[t])
        gt_str = ''.join(gt_chars)

        # Prediction
        pred_chars = []
        for t in pred_tokens[i].cpu().numpy():
            if t == dataset.sos_token:
                continue
            elif t == dataset.eos_token:
                break
            elif t < dataset.n_chars:
                pred_chars.append(char_names[t])
        pred_str = ''.join(pred_chars)

        axes[i].plot(waveform[i].cpu().numpy())
        match = "OK" if gt_str == pred_str else "MISMATCH"
        axes[i].set_title(f"GT: {gt_str} | Pred: {pred_str} [{match}]",
                         fontsize=10)
        axes[i].set_ylabel("Amp")

    plt.suptitle("Whisper Inference: Ground Truth vs Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "inference_demo.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
