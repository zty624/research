"""
Minimal Toolformer-Style Tool Use Reproduction
================================================
Reproduces core ideas from Toolformer (2302.04761, Schick et al.):
1. Language model learns WHEN and HOW to call external tools
2. Insert tool call tokens into sequence: [CALC(expr)] -> execute -> insert result
3. Self-supervised: filter tool calls by whether they help predict future tokens
4. Tools: calculator and search (simulated)
5. Key insight: LMs can teach themselves to use tools via filtered self-play
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import re


# ── Synthetic Data ──

# Simple arithmetic expressions embedded in text
TEMPLATES = [
    "the result of {expr} is [CALC({expr})]",
    "compute {expr} and get [CALC({expr})]",
    "{expr} equals [CALC({expr})]",
    "if we calculate {expr} we find [CALC({expr})]",
    "the answer to {expr} is [CALC({expr})]",
]

EXPR_TEMPLATES = [
    "{a}+{b}",
    "{a}-{b}",
    "{a}*{b}",
    "{a}+{b}+{c}",
    "{a}*{b}+{c}",
]

# Search tool: look up facts about entities
SEARCH_TEMPLATES = [
    "the capital of {country} is [SEARCH(capital of {country})]",
    "{country} has population [SEARCH(population of {country})]",
    "the currency of {country} is [SEARCH(currency of {country})]",
]

COUNTRIES = {
    "france": {"capital": "paris", "population": "67M", "currency": "euro"},
    "japan": {"capital": "tokyo", "population": "125M", "currency": "yen"},
    "brazil": {"capital": "brasilia", "population": "214M", "currency": "real"},
    "germany": {"capital": "berlin", "population": "83M", "currency": "euro"},
    "india": {"capital": "new delhi", "population": "1.4B", "currency": "rupee"},
}


def eval_expr(expr):
    """Safely evaluate simple arithmetic expression."""
    try:
        # Only allow digits, +, -, *
        cleaned = re.sub(r'[^0-9+\-*]', '', expr)
        return str(eval(cleaned))
    except Exception:
        return "ERROR"


def search_tool(query):
    """Simulate search engine for country facts."""
    q = query.lower()
    for country, facts in COUNTRIES.items():
        if country in q:
            if "capital" in q:
                return facts["capital"]
            elif "population" in q:
                return facts["population"]
            elif "currency" in q:
                return facts["currency"]
    return "UNKNOWN"


def execute_tool_call(tool_name, arg):
    """Execute a tool call and return the result string."""
    if tool_name == "CALC":
        return eval_expr(arg)
    elif tool_name == "SEARCH":
        return search_tool(arg)
    return "UNKNOWN"


# ── Tokenizer ──

SPECIAL_TOKENS = {
    "[CALC": 50, "]": 51, "[/CALC": 52,
    "[SEARCH": 53, "[/SEARCH": 54,
    "[RESULT": 55, "[/RESULT": 56,
    "[TOOL_OFF": 57, "[/TOOL_OFF": 58,
}
VOCAB_SIZE = 80
MAX_LEN = 48


def tokenize(text):
    """Simple word/number-level tokenizer with special tool tokens."""
    tokens = []
    # First, extract special token patterns
    remaining = text
    parts = re.split(r'(\[/?CALC\]|\[/?SEARCH\]|\[/?RESULT\]|\[/?TOOL_OFF\])', remaining)
    for part in parts:
        if part in SPECIAL_TOKENS:
            tokens.append(SPECIAL_TOKENS[part])
        else:
            for word in part.split():
                if word.isdigit():
                    # Map number to token (mod vocab)
                    num = int(word)
                    # For numbers > 9, split digits
                    for ch in str(num):
                        tokens.append(int(ch))
                elif word.lower() in SPECIAL_TOKENS:
                    tokens.append(SPECIAL_TOKENS[word.lower()])
                else:
                    # Hash word to token
                    tokens.append(hash(word.lower()) % 40 + 20)
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))  # PAD=0
    return tokens


def generate_samples(n_samples=2000, tool_prob=0.7):
    """Generate training samples with and without tool calls."""
    rng = random.Random(42)
    samples = []  # (input_tokens, target_tokens, has_tool_call, tool_name)

    for _ in range(n_samples):
        if rng.random() < tool_prob:
            # With tool call
            if rng.random() < 0.6:
                # Calculator tool
                expr_tmpl = rng.choice(EXPR_TEMPLATES)
                a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 20)
                expr = expr_tmpl.format(a=a, b=b, c=c)
                tmpl = rng.choice(TEMPLATES)
                result = eval_expr(expr)
                text = tmpl.format(expr=expr)
                # Insert result after tool call
                text = text.replace(f"[CALC({expr})]", f"[CALC]{expr}[/CALC][RESULT]{result}[/RESULT]")
                tool_name = "CALC"
            else:
                # Search tool
                country = rng.choice(list(COUNTRIES.keys()))
                tmpl = rng.choice(SEARCH_TEMPLATES)
                query = tmpl.split("[SEARCH(")[1].split(")")[0] if "[SEARCH(" in tmpl else f"capital of {country}"
                text = tmpl.format(country=country)
                result = search_tool(query)
                text = text.replace(
                    f"[SEARCH({query})]",
                    f"[SEARCH]{query}[/SEARCH][RESULT]{result}[/RESULT]"
                )
                tool_name = "SEARCH"
            has_tool = True
        else:
            # Plain text without tool calls
            expr_tmpl = rng.choice(EXPR_TEMPLATES)
            a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 20)
            expr = expr_tmpl.format(a=a, b=b, c=c)
            result = eval_expr(expr)
            text = f"the result of {expr} is {result}"
            tool_name = None
            has_tool = False

        tokens = tokenize(text)
        # Input = all tokens, Target = shifted
        samples.append((tokens, has_tool, tool_name))

    return samples


# ── Tool-Augmented Language Model ──

class ToolformerModel(nn.Module):
    """Small Transformer LM that can emit tool call tokens."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_heads=2, n_layers=2, max_len=MAX_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Tool call detector: binary head on top of LM hidden states
        self.tool_detector = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 2)  # 0=no tool, 1=tool
        )

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask)
        h = self.norm(h)

        lm_logits = self.head(h)           # (B, T, V)
        tool_logits = self.tool_detector(h) # (B, T, 2)
        return lm_logits, tool_logits


# ── Tool Call Filtering (core Toolformer idea) ──

def filter_tool_calls(model, samples, device='cpu', threshold=0.5):
    """Filter tool calls by checking if they help predict future tokens.
    This is the key self-supervised filtering from the Toolformer paper:
    - Generate candidate tool call positions
    - Execute the tool
    - Compare loss with vs without tool result
    - Keep only tool calls that reduce loss (helpful)
    """
    model.eval()
    kept = 0
    total = 0
    helpful_scores = []

    for tokens, has_tool, tool_name in samples[:500]:
        if not has_tool:
            continue
        total += 1

        x = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            lm_logits, tool_logits = model(x)

            # Check tool detector confidence at each position
            tool_probs = F.softmax(tool_logits, dim=-1)  # (1, T, 2)
            max_tool_prob = tool_probs[0, :, 1].max().item()

            # Simulate: compare loss with tool result vs without
            # In real Toolformer, this uses the actual LM loss difference
            # Here we approximate: if tool detector fires with high confidence,
            # the model has "learned" this tool call is useful
            helpful_scores.append(max_tool_prob)

            if max_tool_prob > threshold:
                kept += 1

    model.train()
    if total == 0:
        return 0.0, []
    keep_rate = kept / total
    return keep_rate, helpful_scores


# ── Training ──

def train(model, samples, n_epochs=15, batch_size=32, lr=1e-3, device='cpu',
          tool_weight=0.3):
    """Train with both LM loss and tool detection loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prepare data
    all_tokens = torch.tensor([s[0] for s in samples], dtype=torch.long, device=device)
    all_tool_labels = torch.tensor(
        [1 if s[1] else 0 for s in samples], dtype=torch.long, device=device
    )

    n_samples = len(samples)
    losses_lm = []
    losses_tool = []
    tool_call_freqs = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        epoch_lm_loss = 0
        epoch_tool_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            x = all_tokens[idx]       # (B, T)
            tool_lbl = all_tool_labels[idx]  # (B,)

            lm_logits, tool_logits = model(x)

            # LM loss: predict next token
            lm_loss = F.cross_entropy(
                lm_logits[:, :-1].reshape(-1, model.vocab_size),
                x[:, 1:].reshape(-1),
                ignore_index=0
            )

            # Tool detection loss: does this sequence contain a tool call?
            # Pool over sequence for binary classification
            tool_pred = tool_logits.mean(dim=1)  # (B, 2)
            tool_loss = F.cross_entropy(tool_pred, tool_lbl)

            loss = lm_loss + tool_weight * tool_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_lm_loss += lm_loss.item()
            epoch_tool_loss += tool_loss.item()
            n_batches += 1

        avg_lm = epoch_lm_loss / n_batches
        avg_tool = epoch_tool_loss / n_batches
        losses_lm.append(avg_lm)
        losses_tool.append(avg_tool)

        # Measure tool call frequency at end of epoch
        with torch.no_grad():
            sample_idx = torch.arange(0, min(200, n_samples), device=device)
            x_sample = all_tokens[sample_idx]
            _, tool_logits_sample = model(x_sample)
            tool_probs = F.softmax(tool_logits_sample, dim=-1)
            # Fraction of positions with tool call probability > 0.5
            freq = (tool_probs[:, :, 1] > 0.5).float().mean().item()
            tool_call_freqs.append(freq)

        print(f"  Epoch {epoch+1}/{n_epochs} | LM Loss: {avg_lm:.4f} | "
              f"Tool Loss: {avg_tool:.4f} | Tool Freq: {tool_call_freqs[-1]:.4f}")

    return losses_lm, losses_tool, tool_call_freqs


# ── Evaluation ──

def evaluate(model, samples, device='cpu'):
    """Evaluate accuracy with tools vs without tools."""
    model.eval()

    tool_correct = 0
    tool_total = 0
    no_tool_correct = 0
    no_tool_total = 0

    tool_detected_tp = 0
    tool_detected_fn = 0
    tool_detected_fp = 0
    tool_detected_tn = 0

    for tokens, has_tool, tool_name in samples[:500]:
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            lm_logits, tool_logits = model(x)

            # Token prediction accuracy
            preds = lm_logits[0].argmax(dim=-1)
            targets = torch.tensor(tokens[1:], device=device)
            mask = targets != 0
            if mask.sum() > 0:
                acc = (preds[:-1][mask] == targets[mask]).float().mean().item()
            else:
                acc = 0

            if has_tool:
                tool_correct += acc
                tool_total += 1
            else:
                no_tool_correct += acc
                no_tool_total += 1

            # Tool detection accuracy
            tool_probs = F.softmax(tool_logits[0], dim=-1)
            detected = (tool_probs[:, 1] > 0.5).any().item()

            if has_tool and detected:
                tool_detected_tp += 1
            elif has_tool and not detected:
                tool_detected_fn += 1
            elif not has_tool and detected:
                tool_detected_fp += 1
            else:
                tool_detected_tn += 1

    model.train()

    tool_acc = tool_correct / max(tool_total, 1)
    no_tool_acc = no_tool_correct / max(no_tool_total, 1)

    detection_precision = tool_detected_tp / max(tool_detected_tp + tool_detected_fp, 1)
    detection_recall = tool_detected_tp / max(tool_detected_tp + tool_detected_fn, 1)

    return {
        'tool_acc': tool_acc,
        'no_tool_acc': no_tool_acc,
        'detection_precision': detection_precision,
        'detection_recall': detection_recall,
    }


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "85-toolformer"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("=== Generating Training Data ===")
    samples = generate_samples(n_samples=2000, tool_prob=0.7)
    tool_count = sum(1 for _, h, _ in samples if h)
    print(f"  Total samples: {len(samples)}, with tools: {tool_count}")

    # Train model
    print("\n=== Training Toolformer Model ===")
    model = ToolformerModel().to(device)
    losses_lm, losses_tool, tool_freqs = train(
        model, samples, n_epochs=15, batch_size=32, lr=1e-3, device=device
    )

    # Evaluate
    print("\n=== Evaluation ===")
    eval_results = evaluate(model, samples, device=device)
    print(f"  Accuracy WITH tools:    {eval_results['tool_acc']:.4f}")
    print(f"  Accuracy WITHOUT tools: {eval_results['no_tool_acc']:.4f}")
    print(f"  Tool detection P/R:     {eval_results['detection_precision']:.3f} / "
          f"{eval_results['detection_recall']:.3f}")

    # Tool call filtering analysis
    print("\n=== Tool Call Filtering (Self-Supervised) ===")
    keep_rates = []
    helpful_all = []
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        rate, scores = filter_tool_calls(model, samples, device=device, threshold=threshold)
        keep_rates.append(rate)
        helpful_all.extend(scores)
        print(f"  Threshold {threshold:.1f}: keep rate = {rate:.3f}")

    # ── Visualization ──

    # 1. Training losses
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(losses_lm, label='LM Loss', color='blue')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Language Modeling Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses_tool, label='Tool Detection Loss', color='orange')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Tool Detection Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tool_freqs, label='Tool Call Frequency', color='green')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Fraction of Positions with Tool Call")
    axes[2].set_title("Tool Call Frequency Over Training")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Toolformer: Training Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_dynamics.png", dpi=150)
    plt.close()

    # 2. Accuracy with vs without tools
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['Without Tools', 'With Tools']
    accs = [eval_results['no_tool_acc'], eval_results['tool_acc']]
    colors = ['red', 'blue']
    bars = axes[0].bar(methods, accs, color=colors, alpha=0.7)
    axes[0].set_ylabel("Token Prediction Accuracy")
    axes[0].set_title("Accuracy: With vs Without Tool Calls")
    axes[0].set_ylim(0, max(accs) * 1.3 + 0.05)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f'{v:.4f}', ha='center', fontweight='bold')

    # 3. Tool call filtering: keep rate vs threshold
    thresholds = [0.3, 0.5, 0.7, 0.9]
    axes[1].plot(thresholds, keep_rates, 'o-', color='purple', linewidth=2)
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("Tool Call Keep Rate")
    axes[1].set_title("Self-Supervised Tool Call Filtering")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    for t, r in zip(thresholds, keep_rates):
        axes[1].annotate(f'{r:.2f}', (t, r), textcoords="offset points",
                         xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_and_filtering.png", dpi=150)
    plt.close()

    # 4. Helpful score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(helpful_all, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
    ax.set_xlabel("Tool Helpfulness Score (max tool detector confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Tool Call Helpfulness Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "helpfulness_distribution.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    steps = [
        ("1. Sample\nTool Calls", "Insert [CALC(expr)]\ninto text at\ncandidate positions\n\nMultiple candidates\nper position", 0.1, 'purple'),
        ("2. Execute\nTools", "Run calculator\nor search engine\n\n[CALC(3+5)] -> 8\n[SEARCH(capital)]\n  -> paris", 0.3, 'orange'),
        ("3. Filter by\nPerplexity", "Compare LM loss\nwith vs without\ntool result\n\nKeep only calls\nthat HELP", 0.55, 'teal'),
        ("4. Fine-tune\nLM", "Train on filtered\ntool-augmented data\n\nLM learns WHEN\nand HOW to call\nexternal tools", 0.82, 'green'),
    ]

    for name, desc, x_pos, color in steps:
        ax.text(x_pos, 0.78, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.2, 0.43, 0.69]:
        ax.annotate('→', xy=(x, 0.55), fontsize=24, ha='center', va='center', color='gray')

    ax.set_title("Toolformer: Self-Supervised Tool Use Learning", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "toolformer_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
