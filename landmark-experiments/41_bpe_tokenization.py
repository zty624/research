"""
Minimal BPE Tokenization Reproduction
======================================
Reproduces core ideas from BPE (1508.07909, Sennrich et al.) and SentencePiece:
1. Byte-Pair Encoding: iteratively merge most frequent byte pairs
2. Subword tokenization: between character and word level
3. Handles OOV words naturally via subword composition
4. Compare: character-level vs word-level vs BPE tokenization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict


# ── BPE Tokenizer ──

class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.merges = []  # List of (a, b) merge operations
        self.vocab = {}   # token -> id

    def train(self, text, n_merges=100):
        """Train BPE on text corpus."""
        # Split into words (pretokens)
        words = text.lower().split()
        # Represent each word as tuple of characters + end-of-word marker
        word_freqs = Counter(words)
        splits = {word: list(word) + ['</w>'] for word in word_freqs}

        self.merges = []
        for _ in range(n_merges):
            # Count pairs
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i+1])] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = pair_freqs.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Merge the pair in all words
            for word in word_freqs:
                symbols = splits[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_symbols.append(symbols[i] + symbols[i+1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

        # Build vocabulary
        self.vocab = {}
        idx = 0
        # Add character-level tokens
        all_chars = set()
        for word in word_freqs:
            for char in word:
                all_chars.add(char)
            all_chars.add('</w>')
        for char in sorted(all_chars):
            self.vocab[char] = idx
            idx += 1
        # Add merged tokens
        for a, b in self.merges:
            merged = a + b
            if merged not in self.vocab:
                self.vocab[merged] = idx
                idx += 1

        return self

    def tokenize(self, text):
        """Tokenize text using learned BPE merges."""
        words = text.lower().split()
        all_tokens = []
        for word in words:
            symbols = list(word) + ['</w>']
            # Apply merges in order
            for a, b in self.merges:
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                        new_symbols.append(a + b)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            all_tokens.extend(symbols)
        return all_tokens

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(t, 0) for t in tokens]


class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self):
        self.vocab = {}
        self.vocab['<unk>'] = 0
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz .,!?\n\'"-:;()'):
            self.vocab[c] = i + 1

    def encode(self, text):
        return [self.vocab.get(c, 0) for c in text.lower()]

    def tokenize(self, text):
        return list(text.lower())


class WordTokenizer:
    """Simple word-level tokenizer."""
    def __init__(self, max_vocab=500):
        self.max_vocab = max_vocab
        self.vocab = {}

    def train(self, text):
        word_freqs = Counter(text.lower().split())
        # Keep top words
        top_words = word_freqs.most_common(self.max_vocab - 2)
        self.vocab = {'<unk>': 0, '<pad>': 1}
        for i, (word, _) in enumerate(top_words):
            self.vocab[word] = i + 2
        return self

    def encode(self, text):
        return [self.vocab.get(w, 0) for w in text.lower().split()]

    def tokenize(self, text):
        return text.lower().split()


# ── Language Model ──

class TinyLM(nn.Module):
    """Simple LSTM language model."""
    def __init__(self, vocab_size, d_model=64, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.lstm(h)
        return self.head(h)


# ── Training ──

def train_lm(model, data, n_steps=2000, batch_size=32, seq_len=32, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n_tokens = len(data)

    for step in range(n_steps):
        starts = torch.randint(0, n_tokens - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f} | PPL: {np.exp(loss.item()):.1f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "41-bpe"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create a corpus
    print("=== Creating Corpus ===")
    corpus_sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat and a dog played",
        "the cat chased the mouse",
        "dogs and cats are friends",
        "the quick brown fox jumped",
        "the lazy dog slept all day",
        "cats love to chase mice",
        "dogs fetch the ball",
        "the fox ran across the field",
    ]
    corpus = " ".join(corpus_sentences * 200)  # repeat for volume
    print(f"  Corpus size: {len(corpus)} characters, {len(corpus.split())} words")

    # Train tokenizers
    print("\n=== Training Tokenizers ===")

    # Character-level
    char_tok = CharTokenizer()
    char_data = torch.tensor(char_tok.encode(corpus), dtype=torch.long)
    print(f"  Char tokenizer: vocab={len(char_tok.vocab)}, tokens={len(char_data)}")

    # Word-level
    word_tok = WordTokenizer(max_vocab=500)
    word_tok.train(corpus)
    word_data = torch.tensor(word_tok.encode(corpus), dtype=torch.long)
    print(f"  Word tokenizer: vocab={len(word_tok.vocab)}, tokens={len(word_data)}")

    # BPE with different merge counts
    bpe_results = {}
    for n_merges in [50, 100, 200]:
        bpe_tok = BPETokenizer()
        bpe_tok.train(corpus, n_merges=n_merges)
        bpe_data = torch.tensor(bpe_tok.encode(corpus), dtype=torch.long)
        print(f"  BPE ({n_merges} merges): vocab={len(bpe_tok.vocab)}, tokens={len(bpe_data)}")
        bpe_results[n_merges] = {
            'tokenizer': bpe_tok,
            'data': bpe_data,
            'vocab_size': len(bpe_tok.vocab),
        }

    # Experiment 1: Show tokenization examples
    print("\n=== Tokenization Examples ===")
    test_sentence = "the cat chased the dog"
    print(f"  Input: '{test_sentence}'")
    print(f"  Char: {char_tok.tokenize(test_sentence)[:20]}...")
    print(f"  Word: {word_tok.tokenize(test_sentence)}")

    for n_merges, r in bpe_results.items():
        tokens = r['tokenizer'].tokenize(test_sentence)
        print(f"  BPE ({n_merges}): {tokens}")

    # OOV handling
    oov_sentence = "the cat paraded gracefully"
    print(f"\n  OOV Input: '{oov_sentence}'")
    print(f"  Word (OOV handling): {[w if w in word_tok.vocab else '<unk>' for w in oov_sentence.split()]}")
    for n_merges, r in bpe_results.items():
        tokens = r['tokenizer'].tokenize(oov_sentence)
        print(f"  BPE ({n_merges}): {tokens}  (no <unk>!)")

    # Experiment 2: Train LMs with different tokenizations
    print("\n=== Training Language Models ===")
    lm_results = {}

    # Char-level
    print("  Char-level:")
    char_model = TinyLM(len(char_tok.vocab), d_model=64).to(device)
    char_losses = train_lm(char_model, char_data, n_steps=2000, seq_len=64, device=device)
    lm_results['char'] = {'losses': char_losses, 'final_loss': char_losses[-1],
                           'vocab_size': len(char_tok.vocab), 'token_count': len(char_data)}

    # BPE-50
    print("  BPE-50:")
    bpe50_data = bpe_results[50]['data']
    bpe50_model = TinyLM(bpe_results[50]['vocab_size'], d_model=64).to(device)
    bpe50_losses = train_lm(bpe50_model, bpe50_data, n_steps=2000, seq_len=32, device=device)
    lm_results['bpe50'] = {'losses': bpe50_losses, 'final_loss': bpe50_losses[-1],
                            'vocab_size': bpe_results[50]['vocab_size'], 'token_count': len(bpe50_data)}

    # BPE-100
    print("  BPE-100:")
    bpe100_data = bpe_results[100]['data']
    bpe100_model = TinyLM(bpe_results[100]['vocab_size'], d_model=64).to(device)
    bpe100_losses = train_lm(bpe100_model, bpe100_data, n_steps=2000, seq_len=32, device=device)
    lm_results['bpe100'] = {'losses': bpe100_losses, 'final_loss': bpe100_losses[-1],
                              'vocab_size': bpe_results[100]['vocab_size'], 'token_count': len(bpe100_data)}

    # BPE-200
    print("  BPE-200:")
    bpe200_data = bpe_results[200]['data']
    bpe200_model = TinyLM(bpe_results[200]['vocab_size'], d_model=64).to(device)
    bpe200_losses = train_lm(bpe200_model, bpe200_data, n_steps=2000, seq_len=32, device=device)
    lm_results['bpe200'] = {'losses': bpe200_losses, 'final_loss': bpe200_losses[-1],
                              'vocab_size': bpe_results[200]['vocab_size'], 'token_count': len(bpe200_data)}

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"  {'Tokenizer':>10s} | {'Vocab':>6s} | {'Tokens':>7s} | {'Loss':>6s} | {'PPL':>6s}")
    print("  " + "-" * 50)
    for name, r in lm_results.items():
        ppl = np.exp(r['final_loss'])
        print(f"  {name:>10s} | {r['vocab_size']:>6d} | {r['token_count']:>7d} | {r['final_loss']:>6.4f} | {ppl:>6.1f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'char': 'red', 'bpe50': 'orange', 'bpe100': 'green', 'bpe200': 'blue'}
    window = 20
    for name, r in lm_results.items():
        smoothed = np.convolve(r['losses'], np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=name.upper(), color=colors[name], alpha=0.8)
    axes[0].set_title("Training Loss by Tokenization")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Compression ratio
    names = list(lm_results.keys())
    token_counts = [lm_results[n]['token_count'] for n in names]
    char_count = len(corpus)
    compression = [char_count / tc for tc in token_counts]

    axes[1].bar([n.upper() for n in names], compression,
                color=[colors[n] for n in names], alpha=0.7)
    axes[1].set_ylabel("Compression Ratio (chars/tokens)")
    axes[1].set_title("Tokenization Compression")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("BPE: Subword Tokenization for Language Models", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Merge frequency analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Show merge operations learned
    bpe100 = bpe_results[100]['tokenizer']
    merge_pairs = [(f"{a}+{b}", i) for i, (a, b) in enumerate(bpe100.merges[:30])]
    if merge_pairs:
        pairs, indices = zip(*merge_pairs)
        axes[0].barh(range(len(pairs)), [1]*len(pairs), color='green', alpha=0.5)
        axes[0].set_yticks(range(len(pairs)))
        axes[0].set_yticklabels(pairs, fontsize=6)
        axes[0].set_xlabel("Merge order")
        axes[0].set_title("BPE Merge Operations (first 30)")
        axes[0].invert_yaxis()

    # Vocabulary growth
    n_merges_range = [0, 10, 25, 50, 100, 150, 200]
    vocab_sizes = []
    for nm in n_merges_range:
        bpe_tmp = BPETokenizer()
        bpe_tmp.train(corpus, n_merges=nm)
        vocab_sizes.append(len(bpe_tmp.vocab))

    axes[1].plot(n_merges_range, vocab_sizes, 'o-', color='green')
    axes[1].set_xlabel("Number of Merges")
    axes[1].set_ylabel("Vocabulary Size")
    axes[1].set_title("Vocabulary Growth with BPE Merges")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("BPE: How Merge Operations Build Vocabulary", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "bpe_analysis.png", dpi=150)
    plt.close()

    # 3. Tokenization spectrum
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Show tokenization of the same text at different granularities
    text = "the cat chased the dog"
    tokens_char = ' | '.join(list(text))
    tokens_word = ' | '.join(text.split())
    tokens_bpe = ' | '.join(bpe_results[100]['tokenizer'].tokenize(text))

    ax.text(0.5, 0.8, f"Character: {tokens_char}", fontsize=9, ha='center',
            fontfamily='monospace', color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(0.5, 0.5, f"BPE-100: {tokens_bpe}", fontsize=9, ha='center',
            fontfamily='monospace', color='green',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(0.5, 0.2, f"Word: {tokens_word}", fontsize=9, ha='center',
            fontfamily='monospace', color='blue',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Tokenization Spectrum: Character → BPE → Word", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "tokenization_spectrum.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Character\nLevel", "Each char = token\nNo OOV problem\nBut very long\nsequences\n→ Hard to learn", 0.14, 'red'),
        ("BPE\n(Subword)", "Merge frequent pairs\nVocab = chars + merges\nHandles OOV naturally\nCompresses text\n→ Best trade-off", 0.5, 'green'),
        ("Word\nLevel", "Each word = token\nShort sequences\nBut huge vocab\nOOV problem\n→ Rare words fail", 0.86, 'blue'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("BPE: The Goldilocks Tokenization (GPT, LLaMA, etc.)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "bpe_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
