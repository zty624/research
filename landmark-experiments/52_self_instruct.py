"""
Minimal Self-Instruct Reproduction
====================================
Reproduces core ideas from Self-Instruct (2302.04761, Wang et al.):
1. Bootstrapping instruction-following data from a seed set
2. Generate new instructions from existing ones (in-context learning)
3. Filter low-quality / redundant instructions
4. Use generated instructions to fine-tune models
5. Key insight: LLMs can improve themselves via self-generated training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import re
from collections import Counter


# ── Instruction Generation (simulated) ──

SEED_INSTRUCTIONS = [
    "Classify the sentiment of the following text",
    "Summarize the given paragraph in one sentence",
    "Translate the following sentence to French",
    "Answer the following question about history",
    "Rewrite the sentence in formal language",
    "List three key points from the passage",
    "Identify the main argument in the text",
    "Generate a title for the given article",
    "Convert the following recipe to metric units",
    "Explain the concept in simple terms",
    "Find the logical fallacy in the argument",
    "Compare and contrast the two approaches",
    "Predict the next event in the sequence",
    "Evaluate the strength of the evidence",
    "Describe the process step by step",
]

INSTRUCTION_TEMPLATES = [
    "{verb} the following {noun}",
    "How would you {verb} a {noun}?",
    "{verb} a {noun} based on the given {noun2}",
    "What is the best way to {verb} a {noun}?",
    "Given a {noun}, {verb} the {noun2}",
    "Create a {noun} that {verb}s the {noun2}",
    "{verb} between {noun} and {noun2}",
    "For the following {noun}, {verb} its {noun2}",
]

VERBS = ["analyze", "evaluate", "transform", "generate", "identify", "extract",
         "compare", "categorize", "simplify", "expand", "verify", "predict"]

NOUNS = ["text", "sentence", "paragraph", "passage", "argument", "claim",
         "question", "problem", "scenario", "data", "concept", "definition"]

NOUNS2 = ["meaning", "structure", "content", "theme", "key points", "errors",
          "pattern", "relevance", "implications", "features", "context", "purpose"]


def generate_instruction_from_seed(seed, rng):
    """Simulate instruction generation by recombining parts."""
    template = rng.choice(INSTRUCTION_TEMPLATES)
    verb = rng.choice(VERBS)
    noun = rng.choice(NOUNS)
    noun2 = rng.choice(NOUNS2)
    try:
        return template.format(verb=verb, noun=noun, noun2=noun2)
    except (KeyError, IndexError):
        return f"{verb} the {noun}"


def generate_instruction_pool(n_target=200, seed_size=15, temperature=0.8):
    """Self-Instruct pipeline: bootstrap instructions from seed set."""
    rng = random.Random(42)
    pool = list(SEED_INSTRUCTIONS[:seed_size])
    generation_log = []  # Track how each instruction was generated

    for seed_instr in pool:
        generation_log.append(('seed', seed_instr))

    attempts = 0
    while len(pool) < n_target and attempts < n_target * 5:
        attempts += 1

        # Sample 1-3 seed instructions as in-context examples
        n_examples = rng.randint(1, 3)
        examples = rng.sample(pool, min(n_examples, len(pool)))

        # Generate new instruction (simulated: recombine patterns)
        new_instr = generate_instruction_from_seed(examples[0], rng)

        # Apply temperature-like randomness (higher temp = more creative)
        if rng.random() < temperature * 0.3:
            # More creative: use multiple templates
            parts = []
            for _ in range(rng.randint(1, 2)):
                parts.append(generate_instruction_from_seed(rng.choice(pool), rng))
            new_instr = " and ".join(parts)

        # Filter: skip if too similar to existing
        if not is_duplicate(new_instr, pool):
            pool.append(new_instr)
            generation_log.append(('generated', new_instr))

    return pool, generation_log


def is_duplicate(new_instr, existing, threshold=0.5):
    """Check if instruction is too similar to existing ones (ROUGE-like)."""
    new_words = set(new_instr.lower().split())
    for existing_instr in existing:
        existing_words = set(existing_instr.lower().split())
        overlap = len(new_words & existing_words) / max(len(new_words | existing_words), 1)
        if overlap > threshold:
            return True
    return False


# ── Response Generation (simulated) ──

def generate_response_quality(instruction, model_quality=0.5):
    """Simulate response quality based on model quality.
    Higher model_quality = better responses.
    """
    rng = random.Random(hash(instruction) % 2**31)
    # Quality score: combination of model quality and instruction complexity
    complexity = len(instruction.split()) / 10.0
    quality = model_quality * (1 - 0.1 * complexity) + rng.gauss(0, 0.1)
    return max(0, min(1, quality))


# ── Instruction-Following Model (simulated fine-tuning) ──

class InstructionModel(nn.Module):
    """Simple model that maps instruction embeddings to quality predictions."""
    def __init__(self, vocab_size=100, d_model=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.emb(x).mean(dim=1)
        return self.net(h).squeeze(-1)


def tokenize_instruction(instr, vocab_size=100):
    """Simple hash-based tokenization."""
    words = instr.lower().split()
    tokens = [hash(w) % (vocab_size - 1) + 1 for w in words]
    # Pad or truncate to fixed length
    max_len = 16
    tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
    return tokens


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "52-self-instruct"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Bootstrap instruction pool
    print("=== Self-Instruct: Bootstrapping ===")
    pool, gen_log = generate_instruction_pool(n_target=200, seed_size=15, temperature=0.8)

    seed_count = sum(1 for t, _ in gen_log if t == 'seed')
    gen_count = sum(1 for t, _ in gen_log if t == 'generated')
    print(f"  Seed instructions: {seed_count}")
    print(f"  Generated instructions: {gen_count}")
    print(f"  Total pool: {len(pool)}")
    print(f"  Sample generated:")
    for instr in pool[15:20]:
        print(f"    - {instr}")

    # Experiment 2: Quality vs quantity trade-off
    print("\n=== Quality vs Quantity ===")
    quality_results = {}
    for temp in [0.3, 0.5, 0.8, 1.0]:
        pool_t, _ = generate_instruction_pool(n_target=150, seed_size=15, temperature=temp)

        # Compute diversity metrics
        unique_words = set()
        lengths = []
        for instr in pool_t:
            words = instr.lower().split()
            unique_words.update(words)
            lengths.append(len(words))

        # Compute pairwise similarity (sample)
        n_sample = min(50, len(pool_t))
        sample = random.sample(pool_t, n_sample)
        similarities = []
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                w1 = set(sample[i].lower().split())
                w2 = set(sample[j].lower().split())
                sim = len(w1 & w2) / max(len(w1 | w2), 1)
                similarities.append(sim)

        quality_results[temp] = {
            'pool_size': len(pool_t),
            'avg_length': np.mean(lengths),
            'vocabulary': len(unique_words),
            'avg_similarity': np.mean(similarities) if similarities else 0,
        }
        print(f"  temp={temp}: pool={len(pool_t)}, avg_len={np.mean(lengths):.1f}, "
              f"vocab={len(unique_words)}, avg_sim={quality_results[temp]['avg_similarity']:.3f}")

    # Experiment 3: Simulate self-training loop
    print("\n=== Self-Training Loop ===")
    n_iterations = 5
    model_qualities = [0.3]  # Start with low quality

    for iteration in range(n_iterations):
        current_quality = model_qualities[-1]

        # Generate instructions with current quality
        pool_it, _ = generate_instruction_pool(n_target=100, seed_size=15, temperature=0.8)

        # Generate responses and compute average quality
        qualities = [generate_response_quality(instr, current_quality) for instr in pool_it]
        avg_quality = np.mean(qualities)

        # Model improves from its own data
        new_quality = min(0.95, current_quality + 0.12 * avg_quality)
        model_qualities.append(new_quality)

        print(f"  Iter {iteration+1}: model_quality={current_quality:.3f} → "
              f"data_quality={avg_quality:.3f} → new_model={new_quality:.3f}")

    # ── Visualization ──

    # 1. Pool growth over generations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pool_sizes = [sum(1 for t, _ in gen_log[:i+1] if t == 'generated')
                  for i in range(len(gen_log))]
    axes[0].plot(pool_sizes, color='blue')
    axes[0].axhline(y=seed_count, color='red', linestyle='--', label='Seed count')
    axes[0].set_xlabel("Generation Attempt")
    axes[0].set_ylabel("New Instructions Generated")
    axes[0].set_title("Instruction Pool Growth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Instruction length distribution
    seed_lengths = [len(instr.split()) for t, instr in gen_log if t == 'seed']
    gen_lengths = [len(instr.split()) for t, instr in gen_log if t == 'generated']
    axes[1].hist(seed_lengths, bins=10, alpha=0.7, label='Seed', color='red')
    axes[1].hist(gen_lengths, bins=10, alpha=0.7, label='Generated', color='blue')
    axes[1].set_xlabel("Instruction Length (words)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Instruction Length Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Self-Instruct: Bootstrapping Instruction Data", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "pool_growth.png", dpi=150)
    plt.close()

    # 2. Quality vs Temperature
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    temps = sorted(quality_results.keys())
    pool_sizes_q = [quality_results[t]['pool_size'] for t in temps]
    vocab_sizes = [quality_results[t]['vocabulary'] for t in temps]
    avg_sims = [quality_results[t]['avg_similarity'] for t in temps]

    axes[0].bar(range(len(temps)), pool_sizes_q, color='blue', alpha=0.7)
    axes[0].set_xticks(range(len(temps)))
    axes[0].set_xticklabels([f'{t:.1f}' for t in temps])
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Pool Size")
    axes[0].set_title("Generated Pool Size")

    axes[1].bar(range(len(temps)), vocab_sizes, color='green', alpha=0.7)
    axes[1].set_xticks(range(len(temps)))
    axes[1].set_xticklabels([f'{t:.1f}' for t in temps])
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Unique Words")
    axes[1].set_title("Vocabulary Diversity")

    axes[2].bar(range(len(temps)), avg_sims, color='red', alpha=0.7)
    axes[2].set_xticks(range(len(temps)))
    axes[2].set_xticklabels([f'{t:.1f}' for t in temps])
    axes[2].set_xlabel("Temperature")
    axes[2].set_ylabel("Avg Pairwise Similarity")
    axes[2].set_title("Instruction Redundancy (lower=better)")

    plt.suptitle("Self-Instruct: Temperature Effects on Quality", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "temperature_quality.png", dpi=150)
    plt.close()

    # 3. Self-training loop
    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = list(range(len(model_qualities)))
    ax.plot(iterations, model_qualities, 'o-', color='blue', linewidth=2)
    ax.fill_between(iterations, [q - 0.05 for q in model_qualities],
                     [q + 0.05 for q in model_qualities], alpha=0.2, color='blue')
    ax.set_xlabel("Self-Training Iteration")
    ax.set_ylabel("Model Quality")
    ax.set_title("Self-Training: Model Improves from Its Own Data")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(results_dir / "self_training_loop.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Seed\nInstructions", "175 human-written\ntask instructions\nCover diverse\ntask types\n→ Starting point", 0.14, 'red'),
        ("Self-Instruct\nGeneration", "LLM generates new\ninstructions from seeds\nInput: k examples\nOutput: new instruction\n+ Filter duplicates", 0.5, 'blue'),
        ("Fine-tune\n& Improve", "Train on generated\ndata, model improves\nBetter model generates\nbetter data\n→ Flywheel!", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Self-Instruct: Bootstrapping Instruction-Following from LLMs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "self_instruct_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
