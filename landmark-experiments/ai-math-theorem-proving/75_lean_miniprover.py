"""
Minimal LeanDojo-Style Theorem Prover Reproduction
===================================================
Reproduces the core ideas from "LeanDojo: Theorem Proving with
Retrieval-Augmented Language Models" (2306.15626):
1. Formal proof state encoding: represent proof states as token sequences
2. Tactic generation: seq2seq model maps proof state → next tactic
3. Retrieval-augmented generation: retrieve similar proof states from
   a database to condition tactic generation
4. Proof search: tree search over tactic sequences with backtracking
5. Compare: no-retrieval baseline vs RAG-augmented prover

Synthetic domain: simplified natural number arithmetic in a Lean-like
syntax. Statements like "n + 0 = n", "a + b = b + a" with tactics
like `induction`, `simp`, `rw [add_comm]`, `apply congrArg`, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Synthetic Formal Language ──

# Tokens for our simplified Lean-like language
TOKENS = {
    # Special
    'PAD': 0, 'BOS': 1, 'EOS': 2, 'SEP': 3,
    # Variables
    'n': 4, 'a': 5, 'b': 6, 'm': 7, 'k': 8,
    # Numbers
    '0': 9, '1': 10, '2': 11,
    # Operators
    '+': 12, '*': 13, '=': 14,
    # Keywords (theorem/proof structure)
    'theorem': 15, 'by': 16, 'sorry': 17,
    # Tactics
    'induction': 18, 'simp': 19, 'rw': 20, 'apply': 21,
    'exact': 22, 'refl': 23, 'ring': 24,
    # Lemma names
    'add_comm': 25, 'add_zero': 26, 'zero_add': 27,
    'add_assoc': 28, 'mul_comm': 29, 'mul_zero': 30,
    'succ': 31, 'nat': 32, 'hypothesis': 33,
    # Brackets / punctuation
    '[': 34, ']': 35, '(': 36, ')': 37, ',': 38,
}
VOCAB_SIZE = len(TOKENS)
INV_TOKENS = {v: k for k, v in TOKENS.items()}

VAR_TOKENS = ['n', 'a', 'b', 'm', 'k']
TACTIC_TOKENS = ['induction', 'simp', 'rw', 'apply', 'exact', 'refl', 'ring']
LEMMA_TOKENS = ['add_comm', 'add_zero', 'zero_add', 'add_assoc', 'mul_comm', 'mul_zero']
MAX_SEQ_LEN = 32


def tokenize(text: str) -> list[int]:
    """Tokenize a Lean-like string into token IDs."""
    tokens = [TOKENS['BOS']]
    for word in text.split():
        if word in TOKENS:
            tokens.append(TOKENS[word])
        elif word.strip('[](),') in TOKENS:
            # Handle bracketed tokens like [add_comm]
            for ch in word:
                if ch in TOKENS:
                    tokens.append(TOKENS[ch])
    tokens.append(TOKENS['EOS'])
    tokens = tokens[:MAX_SEQ_LEN]
    tokens += [TOKENS['PAD']] * (MAX_SEQ_LEN - len(tokens))
    return tokens


def detokenize(ids: list[int]) -> str:
    """Convert token IDs back to string."""
    parts = []
    for i in ids:
        if i in (TOKENS['PAD'], TOKENS['BOS']):
            continue
        if i == TOKENS['EOS']:
            break
        parts.append(INV_TOKENS.get(i, '?'))
    return ' '.join(parts)


# ── Proof State & Tactic System ──

@dataclass
class Theorem:
    """A theorem statement with its correct proof (sequence of tactics)."""
    name: str
    statement: str       # e.g., "a + 0 = a"
    proof_tactics: list[str]  # e.g., ["rw add_zero"]
    difficulty: int = 1  # 1-3, number of tactic steps


@dataclass
class ProofState:
    """Current state of a proof attempt."""
    theorem: Theorem
    remaining_goals: list[str]  # subgoals still to prove
    applied_tactics: list[str] = field(default_factory=list)
    is_complete: bool = False


class ProofEnvironment:
    """Simplified proof checking environment for natural number arithmetic.

    Tactic semantics:
    - refl:   closes goal X = X
    - simp:   closes simple equational goals (a+0=a, 0+a=a, a*0=0, etc.)
    - rw [L]: rewrites using lemma L, may close or simplify goal
    - apply:  applies a constructor or hypothesis
    - exact:  provides exact witness
    - ring:   closes goals provable by ring axioms (associativity, commutativity)
    - induction: splits into base + step subgoals
    """
    LEMMAS = {
        'add_zero': {'pattern': '+ 0', 'effect': 'remove'},
        'zero_add': {'pattern': '0 +', 'effect': 'remove'},
        'add_comm': {'pattern': '+', 'effect': 'swap'},
        'add_assoc': {'pattern': '+', 'effect': 'reassociate'},
        'mul_comm': {'pattern': '*', 'effect': 'swap'},
        'mul_zero': {'pattern': '* 0', 'effect': 'replace_zero'},
    }

    def __init__(self):
        self.theorem_db: list[Theorem] = []
        self._build_theorem_database()

    def _build_theorem_database(self):
        """Create a database of theorems with known proofs."""
        # Difficulty 1: single-tactic proofs
        simple_theorems = [
            Theorem("add_zero_right", "a + 0 = a", ["simp"], 1),
            Theorem("zero_add_left", "0 + a = a", ["simp"], 1),
            Theorem("add_self", "a + a = a + a", ["refl"], 1),
            Theorem("mul_zero_right", "a * 0 = 0", ["simp"], 1),
            Theorem("refl_eq", "a = a", ["refl"], 1),
            Theorem("n_eq_n", "n = n", ["refl"], 1),
            Theorem("zero_eq_zero", "0 = 0", ["refl"], 1),
            Theorem("add_zero_n", "n + 0 = n", ["simp"], 1),
            Theorem("zero_add_n", "0 + n = n", ["simp"], 1),
        ]
        # Difficulty 2: two-tactic proofs
        medium_theorems = [
            Theorem("add_comm_simple", "a + b = b + a", ["rw add_comm"], 2),
            Theorem("add_zero_then_refl", "n + 0 + 0 = n", ["simp", "refl"], 2),
            Theorem("mul_comm_simple", "a * b = b * a", ["rw mul_comm"], 2),
            Theorem("zero_add_mul", "0 + a * 0 = 0", ["simp", "simp"], 2),
            Theorem("add_assoc_simple", "a + b + 0 = a + b", ["simp"], 2),
        ]
        # Difficulty 3: three-tactic proofs
        hard_theorems = [
            Theorem("add_comm_zero", "a + 0 = 0 + a", ["simp", "simp"], 3),
            Theorem("add_comm_assoc", "a + b + 0 = b + a",
                    ["simp", "rw add_comm"], 3),
            Theorem("mul_add_zero", "a * 0 + 0 = 0", ["simp", "simp"], 3),
            Theorem("complex_rewrite", "a + b = b + a",
                    ["rw add_comm"], 3),
        ]
        self.theorem_db = simple_theorems + medium_theorems + hard_theorems

    def check_tactic(self, state: ProofState, tactic: str) -> ProofState:
        """Apply a tactic to a proof state, return new state.

        Simplified semantics: we check if the tactic is in the correct
        proof sequence for the theorem. This simulates a proof checker.
        """
        new_state = ProofState(
            theorem=state.theorem,
            remaining_goals=list(state.remaining_goals),
            applied_tactics=list(state.applied_tactics) + [tactic],
        )
        correct_tactics = state.theorem.proof_tactics
        step = len(state.applied_tactics)

        # Check if tactic matches the expected one at this step
        if step < len(correct_tactics):
            expected = correct_tactics[step]
            if tactic.strip() == expected.strip():
                # Correct tactic applied
                if step + 1 >= len(correct_tactics):
                    new_state.is_complete = True
                    new_state.remaining_goals = []
                else:
                    new_state.remaining_goals = [f"goal_after_{tactic}"]
            else:
                # Wrong tactic: add extra subgoal (proof gets harder)
                new_state.remaining_goals.append(f"stuck_{step}")
        else:
            # Already past the proof length, can only close with refl
            if tactic.strip() == 'refl' and not new_state.remaining_goals:
                new_state.is_complete = True
            else:
                new_state.remaining_goals.append(f"extra_{step}")

        return new_state

    def is_valid_tactic(self, tactic: str) -> bool:
        """Check if a tactic string is syntactically valid."""
        t = tactic.strip()
        if t in ('refl', 'simp', 'ring', 'sorry'):
            return True
        if t.startswith('rw') or t.startswith('apply') or t.startswith('exact'):
            return True
        if t.startswith('induction'):
            return True
        return False

    def get_all_tactics(self) -> list[str]:
        """Return all possible tactic strings."""
        tactics = ['refl', 'simp', 'ring', 'sorry']
        for lemma in self.LEMMAS:
            tactics.append(f'rw {lemma}')
        tactics.append('induction n')
        tactics.append('exact 0')
        return tactics


# ── Models ──

class ProofStateEncoder(nn.Module):
    """Encode a proof state (theorem + goals) into a dense embedding."""
    def __init__(self, vocab_size, d_model=64, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        h = h.mean(dim=1)  # mean pooling
        return F.normalize(self.proj(h), dim=-1)


class TacticGenerator(nn.Module):
    """Seq2seq model: proof state → next tactic tokens.

    Can optionally condition on a retrieved context embedding (RAG).
    """
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2,
                 max_len=MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.ctx_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1, activation='gelu', batch_first=True,
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, state_tokens, tactic_tokens, ctx_emb=None):
        """Given proof state tokens and partial tactic tokens, predict next token.

        Args:
            state_tokens: (B, S) encoded proof state
            tactic_tokens: (B, T) partial tactic sequence
            ctx_emb: (B, D) optional retrieval context embedding
        """
        B, T = tactic_tokens.shape
        positions = torch.arange(T, device=tactic_tokens.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(tactic_tokens) + self.pos_emb(positions)

        # Condition on retrieval context
        if ctx_emb is not None:
            ctx_signal = self.ctx_proj(ctx_emb).unsqueeze(1)
            h = h + ctx_signal.expand(-1, T, -1) * 0.3

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tactic_tokens.device
        )
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        return self.head(self.norm(h))


class MiniProver(nn.Module):
    """LeanDojo-style prover: encoder + retrieval + generator."""
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.encoder = ProofStateEncoder(vocab_size, d_model)
        self.generator = TacticGenerator(vocab_size, d_model, n_heads, n_layers)
        self.vocab_size = vocab_size

        # Retrieval database (populated after indexing)
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.doc_tactics: list[list[int]] = []

    def index_proofs(self, env: ProofEnvironment, device='cpu'):
        """Index all theorem proof states for retrieval."""
        self.env = env
        all_state_tokens = []
        all_tactic_tokens = []

        for theorem in env.theorem_db:
            state_text = f"theorem {theorem.statement}"
            state_ids = tokenize(state_text)
            all_state_tokens.append(state_ids)

            # Correct first tactic as target
            if theorem.proof_tactics:
                tactic_ids = tokenize(theorem.proof_tactics[0])
                all_tactic_tokens.append(tactic_ids)

        state_tensor = torch.tensor(all_state_tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            embeddings = []
            for i in range(0, len(state_tensor), 64):
                batch = state_tensor[i:i+64]
                emb = self.encoder(batch)
                embeddings.append(emb)
            self.doc_embeddings = torch.cat(embeddings, dim=0)

        self.doc_tactics = all_tactic_tokens

    def retrieve(self, query_tokens, top_k=3):
        """Retrieve top-k similar proof states."""
        if self.doc_embeddings is None:
            return None, None
        query_emb = self.encoder(query_tokens)  # (B, D)
        scores = query_emb @ self.doc_embeddings.T  # (B, N)
        topk_scores, topk_indices = scores.topk(top_k, dim=-1)
        return topk_indices, topk_scores


# ── Training ──

def generate_training_data(env: ProofEnvironment, n_samples: int, device='cpu'):
    """Generate training samples: (proof_state, correct_tactic) pairs."""
    state_tokens = []
    tactic_tokens = []
    tactic_labels = []

    for _ in range(n_samples):
        theorem = env.theorem_db[np.random.randint(len(env.theorem_db))]
        # For each step in the proof, create a training sample
        for step_idx, tactic in enumerate(theorem.proof_tactics):
            # State: theorem statement (simplified — in reality would show current goals)
            state_text = f"theorem {theorem.statement}"
            state_ids = tokenize(state_text)
            state_tokens.append(state_ids)

            # Tactic as target
            tactic_ids = tokenize(tactic)
            tactic_tokens.append(tactic_ids)

            # Label: the tactic token sequence shifted (for teacher forcing)
            tactic_labels.append(tactic_ids)

    state_t = torch.tensor(state_tokens, dtype=torch.long, device=device)
    tactic_t = torch.tensor(tactic_tokens, dtype=torch.long, device=device)
    return state_t, tactic_t


def train_prover(model, env, use_rag=True, n_steps=2000, batch_size=32,
                 lr=1e-3, device='cpu'):
    """Train the prover model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    retrieval_accs = []

    for step in range(n_steps):
        # Sample training data
        idx = np.random.randint(0, len(env.theorem_db), size=batch_size)
        state_tokens = []
        tactic_tokens = []

        for i in idx:
            theorem = env.theorem_db[i]
            step_idx = np.random.randint(len(theorem.proof_tactics))
            state_text = f"theorem {theorem.statement}"
            state_tokens.append(tokenize(state_text))
            tactic_tokens.append(tokenize(theorem.proof_tactics[step_idx]))

        state_t = torch.tensor(state_tokens, dtype=torch.long, device=device)
        tactic_t = torch.tensor(tactic_tokens, dtype=torch.long, device=device)

        # Optionally retrieve context
        ctx_emb = None
        if use_rag and model.doc_embeddings is not None:
            topk_indices, topk_scores = model.retrieve(state_t, top_k=1)
            ctx_emb = model.doc_embeddings[topk_indices[:, 0]]  # (B, D)

        # Forward: predict tactic tokens
        logits = model.generator(state_t, tactic_t[:, :-1], ctx_emb)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            tactic_t[:, 1:].reshape(-1),
            ignore_index=TOKENS['PAD'],
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # Track retrieval accuracy
        if use_rag and step % 100 == 0 and model.doc_embeddings is not None:
            with torch.no_grad():
                topk_indices, _ = model.retrieve(state_t, top_k=5)
                correct = 0
                for i in range(batch_size):
                    # Check if the same theorem is retrieved
                    if idx[i] in topk_indices[i].cpu().numpy():
                        correct += 1
                retrieval_accs.append(correct / batch_size)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f}")

    # Re-index after training
    model.index_proofs(env, device)
    return losses, retrieval_accs


# ── Proof Search ──

def attempt_proof(model, theorem, env, use_rag=True, max_steps=6,
                  temperature=0.8, device='cpu'):
    """Attempt to prove a theorem using the trained model.

    Returns (success, num_steps, applied_tactics).
    """
    state_text = f"theorem {theorem.statement}"
    state_ids = tokenize(state_text)
    state_t = torch.tensor([state_ids], dtype=torch.long, device=device)

    proof_state = ProofState(theorem=theorem, remaining_goals=[theorem.statement])

    for step in range(max_steps):
        # Get context from retrieval
        ctx_emb = None
        if use_rag and model.doc_embeddings is not None:
            topk_indices, _ = model.retrieve(state_t, top_k=1)
            ctx_emb = model.doc_embeddings[topk_indices[:, 0]]

        # Generate tactic tokens autoregressively
        tactic_input = torch.tensor([[TOKENS['BOS']]], dtype=torch.long, device=device)
        generated = [TOKENS['BOS']]

        for _ in range(8):  # max tactic length
            logits = model.generator(state_t, tactic_input, ctx_emb)
            next_logits = logits[:, -1, :] / temperature
            # Suppress PAD/BOS
            next_logits[:, TOKENS['PAD']] = -1e9
            next_logits[:, TOKENS['BOS']] = -1e9
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            if next_token.item() == TOKENS['EOS']:
                break
            tactic_input = torch.cat([tactic_input, next_token], dim=1)

        # Decode tactic
        tactic_str = detokenize(generated[1:])  # skip BOS

        # Validate and apply tactic
        if not env.is_valid_tactic(tactic_str):
            tactic_str = 'sorry'  # fallback

        proof_state = env.check_tactic(proof_state, tactic_str)

        if proof_state.is_complete:
            return True, step + 1, proof_state.applied_tactics

    return False, max_steps, proof_state.applied_tactics


def evaluate_prover(model, env, use_rag=True, n_eval=None, device='cpu'):
    """Evaluate prover on theorem database."""
    if n_eval is None:
        n_eval = len(env.theorem_db)

    results = {'success': 0, 'total': 0, 'steps': [], 'by_difficulty': {1: [], 2: [], 3: []}}

    theorems = list(env.theorem_db[:n_eval])
    for theorem in theorems:
        success, steps, tactics = attempt_proof(
            model, theorem, env, use_rag=use_rag, device=device
        )
        results['total'] += 1
        if success:
            results['success'] += 1
            results['steps'].append(steps)
        results['by_difficulty'][theorem.difficulty].append(1.0 if success else 0.0)

    results['rate'] = results['success'] / max(results['total'], 1)
    return results


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "75-lean-miniprover"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = ProofEnvironment()
    print(f"Theorem database: {len(env.theorem_db)} theorems")
    for d in [1, 2, 3]:
        n = sum(1 for t in env.theorem_db if t.difficulty == d)
        print(f"  Difficulty {d}: {n} theorems")

    d_model = 64
    n_steps = 2000

    # 1. Train RAG-augmented prover
    print("\n=== Training RAG-Augmented Prover ===")
    rag_prover = MiniProver(VOCAB_SIZE, d_model).to(device)
    rag_prover.index_proofs(env, device)
    rag_losses, rag_retrieval = train_prover(
        rag_prover, env, use_rag=True, n_steps=n_steps, device=device
    )

    # 2. Train baseline prover (no retrieval)
    print("\n=== Training Baseline Prover (No Retrieval) ===")
    base_prover = MiniProver(VOCAB_SIZE, d_model).to(device)
    base_prover.index_proofs(env, device)
    base_losses, _ = train_prover(
        base_prover, env, use_rag=False, n_steps=n_steps, device=device
    )

    # 3. Evaluate both provers
    print("\n=== Evaluation ===")
    rag_results = evaluate_prover(rag_prover, env, use_rag=True, device=device)
    base_results = evaluate_prover(base_prover, env, use_rag=False, device=device)

    print(f"  RAG Prover:      {rag_results['success']}/{rag_results['total']} "
          f"({rag_results['rate']:.1%})")
    print(f"  Baseline Prover: {base_results['success']}/{base_results['total']} "
          f"({base_results['rate']:.1%})")

    # By difficulty
    for d in [1, 2, 3]:
        rag_d = np.mean(rag_results['by_difficulty'][d]) if rag_results['by_difficulty'][d] else 0
        base_d = np.mean(base_results['by_difficulty'][d]) if base_results['by_difficulty'][d] else 0
        print(f"  Difficulty {d}: RAG={rag_d:.1%}, Baseline={base_d:.1%}")

    # ── Visualization ──
    window = 20

    # 1. Training loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    base_s = np.convolve(base_losses, np.ones(window)/window, mode='valid')
    rag_s = np.convolve(rag_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(base_s, label='Baseline (no retrieval)', color='red')
    axes[0].plot(rag_s, label='RAG-augmented', color='blue')
    axes[0].set_title("Training Loss: Proof State → Tactic Generation")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Retrieval accuracy
    if rag_retrieval:
        axes[1].plot(rag_retrieval, color='green', marker='o', markersize=3)
        axes[1].set_title("Retrieval Accuracy (Recall@5)")
        axes[1].set_xlabel("Evaluation Point")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("LeanDojo-Style Theorem Prover: RAG vs Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 3. Proof success rate by difficulty
    fig, ax = plt.subplots(figsize=(8, 5))
    difficulties = [1, 2, 3]
    rag_rates = [np.mean(rag_results['by_difficulty'][d]) if rag_results['by_difficulty'][d] else 0
                 for d in difficulties]
    base_rates = [np.mean(base_results['by_difficulty'][d]) if base_results['by_difficulty'][d] else 0
                  for d in difficulties]

    x = np.arange(len(difficulties))
    width = 0.35
    ax.bar(x - width/2, base_rates, width, label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, rag_rates, width, label='RAG', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Difficulty {d}' for d in difficulties])
    ax.set_ylabel("Proof Success Rate")
    ax.set_title("Proof Success Rate by Difficulty")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(results_dir / "success_by_difficulty.png", dpi=150)
    plt.close()

    # 4. Proof length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    if rag_results['steps']:
        ax.hist(rag_results['steps'], bins=range(1, 8), alpha=0.6,
                label='RAG (successful proofs)', color='blue', edgecolor='black')
    if base_results['steps']:
        ax.hist(base_results['steps'], bins=range(1, 8), alpha=0.6,
                label='Baseline (successful proofs)', color='red', edgecolor='black')
    ax.set_xlabel("Number of Tactic Steps")
    ax.set_ylabel("Count")
    ax.set_title("Proof Length Distribution (Successful Proofs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "proof_length_distribution.png", dpi=150)
    plt.close()

    # 5. RAG pipeline diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    steps = [
        ("1. Encode\nProof State", "Tokenize goals &\nhypotheses →\ndense embedding", 0.1, 'purple'),
        ("2. Retrieve\nSimilar Proofs", "Search theorem DB\nby embedding\nsimilarity", 0.3, 'orange'),
        ("3. Condition\nGenerator", "Concat retrieved\ncontext with\nstate embedding", 0.5, 'teal'),
        ("4. Generate\nTactic", "Seq2seq decode\nnext tactic\nfrom vocabulary", 0.7, 'green'),
        ("5. Verify &\nIterate", "Check tactic with\nproof checker,\nupdate state", 0.9, 'brown'),
    ]

    for name, desc, x_pos, color in steps:
        ax.text(x_pos, 0.7, name, fontsize=11, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.25, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.2, 0.4, 0.6, 0.8]:
        ax.annotate('\u2192', xy=(x, 0.48), fontsize=20, ha='center',
                    va='center', color='gray')

    ax.set_title("LeanDojo: Retrieval-Augmented Theorem Proving Pipeline",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rag_prover_pipeline.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
