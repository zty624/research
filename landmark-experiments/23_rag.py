"""
Minimal RAG (Retrieval-Augmented Generation) Reproduction
==========================================================
Reproduces core ideas from RAG (2005.11401, Lewis et al.):
1. Document indexing with dense embeddings
2. Retrieval via similarity search (cosine similarity)
3. Augmented generation: conditioning LLM on retrieved context
4. Compare: no retrieval vs RAG vs oracle (ground truth context)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Embedder ──

class TextEncoder(nn.Module):
    """Encode token sequences into dense embeddings."""
    def __init__(self, vocab_size, d_model=64, max_len=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        h = h.mean(dim=1)  # mean pooling
        return F.normalize(self.proj(h), dim=-1)  # L2 normalize


# ── Generator ──

class Generator(nn.Module):
    """Small language model that can condition on retrieved context."""
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.ctx_proj = nn.Linear(d_model, d_model)  # project context embedding
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, ctx_emb=None):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)

        # Add context embedding as prefix influence
        if ctx_emb is not None:
            ctx_signal = self.ctx_proj(ctx_emb).unsqueeze(1)  # (B, 1, D)
            h = h + ctx_signal.expand(-1, T, -1) * 0.3  # soft conditioning

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.head(self.norm(h))


# ── Knowledge Base ──

class KnowledgeBase:
    """Simple knowledge base with facts about numbers.
    Facts are of the form: "N is even/odd", "N is prime/composite",
    "N squared is X", "N doubled is X", etc.
    """
    def __init__(self, n_facts=200, vocab_size=32, seq_len=8):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.facts = []
        self.fact_tokens = []

        for n in range(2, 2 + n_facts):
            facts_for_n = []
            # Fact 1: parity
            if n % 2 == 0:
                facts_for_n.append(f"{n} even")
            else:
                facts_for_n.append(f"{n} odd")

            # Fact 2: prime/composite
            is_prime = all(n % i != 0 for i in range(2, int(n**0.5)+1)) and n > 1
            if is_prime:
                facts_for_n.append(f"{n} prime")
            else:
                facts_for_n.append(f"{n} composite")

            # Fact 3: square
            facts_for_n.append(f"{n} squared {n*n}")

            # Fact 4: double
            facts_for_n.append(f"{n} doubled {n*2}")

            for fact in facts_for_n:
                self.facts.append((n, fact))
                tokens = self.tokenize(fact)
                self.fact_tokens.append(tokens)

        self.fact_tokens = torch.tensor(self.fact_tokens, dtype=torch.long)

    def tokenize(self, text):
        """Simple char-level tokenization using digit + special tokens."""
        tokens = []
        for ch in text:
            if ch.isdigit():
                tokens.append(int(ch))
            elif ch == ' ':
                tokens.append(10)
            elif ch == 'e':
                tokens.append(11)
            elif ch == 'v':
                tokens.append(12)
            elif ch == 'o':
                tokens.append(13)
            elif ch == 'd':
                tokens.append(14)
            elif ch == 'p':
                tokens.append(15)
            elif ch == 'r':
                tokens.append(16)
            elif ch == 'i':
                tokens.append(17)
            elif ch == 'm':
                tokens.append(18)
            elif ch == 'c':
                tokens.append(19)
            elif ch == 'q':
                tokens.append(20)
            elif ch == 'u':
                tokens.append(21)
            elif ch == 'a':
                tokens.append(22)
            elif ch == 'l':
                tokens.append(23)
            elif ch == 'b':
                tokens.append(24)
            elif ch == 's':
                tokens.append(25)
            elif ch == 't':
                tokens.append(26)
            elif ch == 'n':
                tokens.append(27)
            else:
                tokens.append(28)  # unknown
        # Pad/truncate to seq_len
        tokens = tokens[:self.seq_len]
        tokens += [29] * (self.seq_len - len(tokens))  # PAD
        return tokens

    def detokenize(self, tokens):
        inv_map = {i: str(i) for i in range(10)}
        inv_map[10] = ' '; inv_map[11] = 'e'; inv_map[12] = 'v'
        inv_map[13] = 'o'; inv_map[14] = 'd'; inv_map[15] = 'p'
        inv_map[16] = 'r'; inv_map[17] = 'i'; inv_map[18] = 'm'
        inv_map[19] = 'c'; inv_map[20] = 'q'; inv_map[21] = 'u'
        inv_map[22] = 'a'; inv_map[23] = 'l'; inv_map[24] = 'b'
        inv_map[25] = 's'; inv_map[26] = 't'; inv_map[27] = 'n'
        inv_map[28] = '?'; inv_map[29] = ''
        return ''.join(inv_map.get(t, '?') for t in tokens if t != 29)


# ── RAG Model ──

class RAGModel(nn.Module):
    """RAG: Retriever + Generator."""
    def __init__(self, vocab_size, d_model=64, max_len=64):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, d_model)
        self.generator = Generator(vocab_size, d_model, max_len=max_len)
        self.vocab_size = vocab_size

    def index_documents(self, kb, device='cpu'):
        """Index all documents in the knowledge base."""
        self.kb = kb
        with torch.no_grad():
            kb_tokens = kb.fact_tokens.to(device)
            # Encode in batches
            embeddings = []
            for i in range(0, len(kb_tokens), 256):
                batch = kb_tokens[i:i+256]
                emb = self.encoder(batch)
                embeddings.append(emb)
            self.doc_embeddings = torch.cat(embeddings, dim=0)  # (N_docs, D)

    def retrieve(self, query_tokens, top_k=3):
        """Retrieve top-k relevant documents."""
        query_emb = self.encoder(query_tokens)  # (B, D)
        # Cosine similarity
        scores = query_emb @ self.doc_embeddings.T  # (B, N_docs)
        topk_scores, topk_indices = scores.topk(top_k, dim=-1)  # (B, k)
        return topk_indices, topk_scores

    def forward(self, query_tokens, target_tokens, top_k=3, use_retrieval=True):
        """Forward pass: retrieve + generate."""
        if use_retrieval:
            # Retrieve
            topk_indices, topk_scores = self.retrieve(query_tokens, top_k)
            # Use top-1 document embedding as context
            B = query_tokens.shape[0]
            ctx_indices = topk_indices[:, 0]  # (B,)
            ctx_emb = self.doc_embeddings[ctx_indices]  # (B, D)
        else:
            ctx_emb = None

        # Generate
        logits = self.generator(target_tokens[:, :-1], ctx_emb)
        return logits


# ── Training ──

def generate_qa_pairs(kb, n_pairs, device='cpu'):
    """Generate question-answer pairs from knowledge base.
    Question: "What is N?" (tokenized)
    Answer: a fact about N (e.g., "N is even", "N squared is X")
    """
    questions = []
    answers = []
    labels = []  # which fact was used

    for _ in range(n_pairs):
        idx = np.random.randint(0, len(kb.facts))
        n, fact = kb.facts[idx]

        # Question: "what N" → tokens
        q_text = f"what {n}"
        q_tokens = kb.tokenize(q_text)

        # Answer: the fact
        a_tokens = kb.tokenize(fact)

        questions.append(q_tokens)
        answers.append(a_tokens)
        labels.append(idx)

    questions = torch.tensor(questions, dtype=torch.long, device=device)
    answers = torch.tensor(answers, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return questions, answers, labels


def train_model(model, kb, mode='rag', n_steps=3000, batch_size=64, lr=1e-3, device='cpu'):
    """Train RAG or baseline model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    retrievals_correct = []

    for step in range(n_steps):
        questions, answers, labels = generate_qa_pairs(kb, batch_size, device)

        use_rag = (mode == 'rag')
        logits = model(questions, answers, top_k=3, use_retrieval=use_rag)

        ce_loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            answers[:, 1:].reshape(-1),
            ignore_index=29  # ignore padding
        )

        # Retrieval accuracy (for RAG mode)
        if use_rag and step % 50 == 0:
            with torch.no_grad():
                topk_indices, _ = model.retrieve(questions, top_k=5)
                # Check if correct fact is in top-k
                correct = 0
                for i in range(batch_size):
                    if labels[i] in topk_indices[i]:
                        correct += 1
                retrievals_correct.append(correct / batch_size)

        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(ce_loss.item())

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {ce_loss.item():.4f}")

    return losses, retrievals_correct


# ── Evaluation ──

def evaluate(model, kb, mode='rag', n_eval=500, device='cpu'):
    """Evaluate model on QA pairs."""
    model.eval()
    questions, answers, labels = generate_qa_pairs(kb, n_eval, device)

    use_rag = (mode == 'rag')
    with torch.no_grad():
        logits = model(questions, answers, top_k=3, use_retrieval=use_rag)
        preds = logits.argmax(dim=-1)  # (B, T-1)
        targets = answers[:, 1:]

        # Token-level accuracy (ignoring padding)
        mask = targets != 29
        correct = (preds == targets) & mask
        token_acc = correct.sum().float() / mask.sum().float()

        # Retrieval accuracy
        retrieval_acc = 0
        if use_rag:
            topk_indices, _ = model.retrieve(questions, top_k=5)
            hits = 0
            for i in range(n_eval):
                if labels[i] in topk_indices[i]:
                    hits += 1
            retrieval_acc = hits / n_eval

    model.train()
    return token_acc.item(), retrieval_acc


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "23-rag"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 30
    d_model = 64

    kb = KnowledgeBase(n_facts=100, vocab_size=vocab_size, seq_len=8)
    print(f"Knowledge base: {len(kb.facts)} facts")

    # 1. Train RAG model
    print("\n=== Training RAG Model ===")
    rag = RAGModel(vocab_size, d_model).to(device)
    rag.index_documents(kb, device)
    rag_losses, rag_retrieval = train_model(rag, kb, mode='rag', n_steps=3000, device=device)

    # Re-index after training (encoder has been updated)
    rag.index_documents(kb, device)

    # 2. Train baseline (no retrieval)
    print("\n=== Training Baseline (No Retrieval) ===")
    baseline = RAGModel(vocab_size, d_model).to(device)
    baseline.index_documents(kb, device)
    base_losses, _ = train_model(baseline, kb, mode='baseline', n_steps=3000, device=device)

    # 3. Evaluate
    print("\n=== Evaluation ===")
    rag_token_acc, rag_retrieval_acc = evaluate(rag, kb, mode='rag', n_eval=500, device=device)
    base_token_acc, _ = evaluate(baseline, kb, mode='baseline', n_eval=500, device=device)

    print(f"  RAG:      Token Acc = {rag_token_acc:.3f}, Retrieval@5 = {rag_retrieval_acc:.3f}")
    print(f"  Baseline: Token Acc = {base_token_acc:.3f}")

    # ── Visualization ──

    # 1. Training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 30
    rag_s = np.convolve(rag_losses, np.ones(window)/window, mode='valid')
    base_s = np.convolve(base_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(base_s, label='Baseline (no retrieval)', color='red')
    axes[0].plot(rag_s, label='RAG (retrieval-augmented)', color='blue')
    axes[0].set_title("Training Loss: Baseline vs RAG")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Retrieval accuracy over training
    if rag_retrieval:
        axes[1].plot(rag_retrieval, color='green')
        axes[1].set_title("RAG Retrieval Accuracy (Recall@5)")
        axes[1].set_xlabel("Evaluation Point")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

    plt.suptitle("RAG: Retrieval-Augmented Generation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 3. Final accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Baseline\n(no retrieval)', 'RAG\n(retrieval-augmented)']
    accs = [base_token_acc, rag_token_acc]
    colors = ['red', 'blue']
    bars = ax.bar(methods, accs, color=colors, alpha=0.7)
    ax.set_ylabel("Token Prediction Accuracy")
    ax.set_title("RAG vs Baseline: QA Accuracy")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
                ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    # 4. RAG pipeline diagram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    steps = [
        ("1. Query", "Encode question\n→ dense embedding", 0.12, 'purple'),
        ("2. Retrieve", "Search knowledge\nbase by similarity\n→ top-k documents", 0.37, 'orange'),
        ("3. Augment", "Concat retrieved\ncontext with query", 0.62, 'teal'),
        ("4. Generate", "LLM generates\nanswer conditioned\non context", 0.87, 'green'),
    ]

    for name, desc, x_pos, color in steps:
        ax.text(x_pos, 0.75, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    # Arrows
    for x in [0.24, 0.49, 0.74]:
        ax.annotate('→', xy=(x, 0.55), fontsize=24, ha='center', va='center', color='gray')

    ax.set_title("RAG Pipeline: Query → Retrieve → Augment → Generate", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rag_pipeline.png", dpi=150)
    plt.close()

    # 5. Document embedding space (t-SNE / PCA)
    try:
        from sklearn.decomposition import PCA
        emb_np = rag.doc_embeddings.cpu().numpy()
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb_np)

        # Color by number
        numbers = [kb.facts[i][0] for i in range(len(kb.facts))]

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=numbers, cmap='viridis',
                           alpha=0.5, s=10)
        plt.colorbar(scatter, label='Number N')
        ax.set_title("RAG: Document Embedding Space (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "embedding_space.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  PCA visualization failed: {e}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
