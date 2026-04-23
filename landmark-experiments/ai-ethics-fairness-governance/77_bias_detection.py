"""
Minimal LLM Bias Detection & Measurement via WEAT
====================================================
Reproduces core ideas from bias detection literature (2411.10915, 2404.01349):
1. WEAT (Word Embedding Association Test): measures association between
   target concepts (e.g., male/female names) and attribute words (e.g., career/family)
2. Effect size: d = (mean(X) - mean(Y)) / pooled_std, measures bias magnitude
3. Permutation test for statistical significance of bias
4. Debiasing via neutralization: project embeddings to remove bias direction
5. Compare biased vs debiased embedding spaces across demographic dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Word Embedding Space with Intentional Bias ──

def create_biased_embeddings(vocab_size=200, embed_dim=50, seed=42):
    """Create synthetic word embeddings with intentional gender/race bias.

    We construct a vocabulary where:
    - First 20 words: male-associated names (target set A)
    - Next 20 words: female-associated names (target set B)
    - Next 20 words: career words (attribute set X)
    - Next 20 words: family words (attribute set Y)
    - Next 20 words: pleasant words (attribute set P)
    - Next 20 words: unpleasant words (attribute set U)
    - Next 20 words: European-American names (target set C)
    - Next 20 words: African-American names (target set D)
    - Remaining: neutral filler words

    Bias is injected by making certain target-attribute associations stronger.
    """
    rng = np.random.RandomState(seed)

    # Base random embeddings
    embeddings = rng.randn(vocab_size, embed_dim).astype(np.float32) * 0.5

    # Define bias directions
    # Gender bias direction: dim 0
    gender_dir = np.zeros(embed_dim, dtype=np.float32)
    gender_dir[0] = 1.0

    # Career-family bias direction: correlated with gender (dim 0,1)
    career_dir = np.zeros(embed_dim, dtype=np.float32)
    career_dir[0] = 0.7
    career_dir[1] = 0.7

    family_dir = np.zeros(embed_dim, dtype=np.float32)
    family_dir[0] = -0.7
    family_dir[1] = 0.7

    # Racial bias direction: dim 2
    race_dir = np.zeros(embed_dim, dtype=np.float32)
    race_dir[2] = 1.0

    # Pleasant-unpleasant direction: correlated with race (dim 2,3)
    pleasant_dir = np.zeros(embed_dim, dtype=np.float32)
    pleasant_dir[2] = 0.6
    pleasant_dir[3] = 0.8

    unpleasant_dir = np.zeros(embed_dim, dtype=np.float32)
    unpleasant_dir[2] = -0.6
    unpleasant_dir[3] = 0.8

    # Inject gender bias: male names aligned with career, female with family
    bias_strength = 2.0
    for i in range(20):  # Male names → career direction
        embeddings[i] += bias_strength * career_dir
    for i in range(20, 40):  # Female names → family direction
        embeddings[i] += bias_strength * family_dir

    # Inject career/family attribute bias
    for i in range(40, 60):  # Career words → career direction
        embeddings[i] += bias_strength * career_dir
    for i in range(60, 80):  # Family words → family direction
        embeddings[i] += bias_strength * family_dir

    # Inject racial bias: EA names → pleasant, AA names → unpleasant
    for i in range(80, 100):  # Pleasant words
        embeddings[i] += bias_strength * pleasant_dir
    for i in range(100, 120):  # Unpleasant words
        embeddings[i] += bias_strength * unpleasant_dir
    for i in range(120, 140):  # EA names → pleasant direction
        embeddings[i] += bias_strength * pleasant_dir
    for i in range(140, 160):  # AA names → unpleasant direction
        embeddings[i] += bias_strength * unpleasant_dir

    # Vocabulary mapping
    vocab = {
        'male_names': list(range(0, 20)),
        'female_names': list(range(20, 40)),
        'career': list(range(40, 60)),
        'family': list(range(60, 80)),
        'pleasant': list(range(80, 100)),
        'unpleasant': list(range(100, 120)),
        'ea_names': list(range(120, 140)),
        'aa_names': list(range(140, 160)),
        'neutral': list(range(160, 200)),
    }

    return torch.tensor(embeddings), vocab


# ── WEAT Score Computation ──

def cosine_similarity(a, b):
    """Compute cosine similarity between vectors a and b."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze()


def mean_cosine_to_set(word_vec, attribute_vecs):
    """Mean cosine similarity of a word to all words in an attribute set."""
    word_expanded = word_vec.unsqueeze(0).expand(attribute_vecs.shape[0], -1)
    sims = F.cosine_similarity(word_expanded, attribute_vecs, dim=1)
    return sims.mean()


def weat_effect_size(target_a, target_b, attr_x, attr_y):
    """Compute WEAT effect size d.

    d = (mean(s(w, X) - s(w, Y) for w in A) - mean(s(w, X) - s(w, Y) for w in B))
        / pooled_std

    Positive d means A is more associated with X (and B with Y).
    """
    # s(w, X) - s(w, Y) for each target word
    diff_a = torch.tensor([
        mean_cosine_to_set(target_a[i], attr_x) - mean_cosine_to_set(target_a[i], attr_y)
        for i in range(target_a.shape[0])
    ])
    diff_b = torch.tensor([
        mean_cosine_to_set(target_b[i], attr_x) - mean_cosine_to_set(target_b[i], attr_y)
        for i in range(target_b.shape[0])
    ])

    pooled_std = torch.cat([diff_a, diff_b]).std()
    if pooled_std < 1e-8:
        return 0.0

    effect_size = (diff_a.mean() - diff_b.mean()) / pooled_std
    return effect_size.item()


def weat_p_value(target_a, target_b, attr_x, attr_y, n_permutations=1000, seed=42):
    """Permutation test for WEAT significance.

    Randomly swap words between target sets A and B and recompute effect size.
    p-value = fraction of permutations with effect size >= observed.
    """
    rng = np.random.RandomState(seed)
    all_targets = torch.cat([target_a, target_b], dim=0)
    n_a = target_a.shape[0]
    observed = abs(weat_effect_size(target_a, target_b, attr_x, attr_y))

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(all_targets.shape[0])
        perm_a = all_targets[perm[:n_a]]
        perm_b = all_targets[perm[n_a:]]
        perm_effect = abs(weat_effect_size(perm_a, perm_b, attr_x, attr_y))
        if perm_effect >= observed:
            count += 1

    return count / n_permutations


# ── Debiasing via Neutralization ──

def compute_bias_direction(embeddings, group_a_idx, group_b_idx):
    """Compute bias direction as the difference of group centroids."""
    centroid_a = embeddings[group_a_idx].mean(dim=0)
    centroid_b = embeddings[group_b_idx].mean(dim=0)
    bias_dir = centroid_a - centroid_b
    bias_dir = bias_dir / (bias_dir.norm() + 1e-8)
    return bias_dir


def debias_embeddings(embeddings, bias_direction, protect_idx=None):
    """Remove bias direction from embeddings via projection.

    For each embedding, subtract its projection onto the bias direction.
    Optionally preserve protected words by not debiasing them.
    """
    debiased = embeddings.clone()
    for i in range(embeddings.shape[0]):
        if protect_idx is not None and i in protect_idx:
            continue
        proj = torch.dot(debiased[i], bias_direction) * bias_direction
        debiased[i] = debiased[i] - proj
    return debiased


# ── Main ──

def main():
    torch.manual_seed(42)
    results_dir = Path(__file__).parent / "results" / "77-bias-detection"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create biased embedding space
    embeddings, vocab = create_biased_embeddings()
    print(f"Vocabulary: {embeddings.shape[0]} words, {embeddings.shape[1]} dims")
    print(f"Groups: {', '.join(vocab.keys())}")

    # ── WEAT Tests ──

    # Test 1: Gender-Career/Family bias (male→career, female→family)
    print("\n=== WEAT Test 1: Gender × Career/Family ===")
    male_emb = embeddings[vocab['male_names']]
    female_emb = embeddings[vocab['female_names']]
    career_emb = embeddings[vocab['career']]
    family_emb = embeddings[vocab['family']]

    d1 = weat_effect_size(male_emb, female_emb, career_emb, family_emb)
    p1 = weat_p_value(male_emb, female_emb, career_emb, family_emb, n_permutations=1000)
    print(f"  Effect size d = {d1:.4f}")
    print(f"  p-value       = {p1:.4f}")

    # Test 2: Race × Pleasant/Unpleasant bias
    print("\n=== WEAT Test 2: Race × Pleasant/Unpleasant ===")
    ea_emb = embeddings[vocab['ea_names']]
    aa_emb = embeddings[vocab['aa_names']]
    pleasant_emb = embeddings[vocab['pleasant']]
    unpleasant_emb = embeddings[vocab['unpleasant']]

    d2 = weat_effect_size(ea_emb, aa_emb, pleasant_emb, unpleasant_emb)
    p2 = weat_p_value(ea_emb, aa_emb, pleasant_emb, unpleasant_emb, n_permutations=1000)
    print(f"  Effect size d = {d2:.4f}")
    print(f"  p-value       = {p2:.4f}")

    # Test 3: Cross-dimension (Gender × Pleasant/Unpleasant) — should be weaker
    print("\n=== WEAT Test 3: Gender × Pleasant/Unpleasant (cross-dim) ===")
    d3 = weat_effect_size(male_emb, female_emb, pleasant_emb, unpleasant_emb)
    p3 = weat_p_value(male_emb, female_emb, pleasant_emb, unpleasant_emb, n_permutations=1000)
    print(f"  Effect size d = {d3:.4f}")
    print(f"  p-value       = {p3:.4f}")

    # Test 4: Race × Career/Family — should be weaker
    print("\n=== WEAT Test 4: Race × Career/Family (cross-dim) ===")
    d4 = weat_effect_size(ea_emb, aa_emb, career_emb, family_emb)
    p4 = weat_p_value(ea_emb, aa_emb, career_emb, family_emb, n_permutations=1000)
    print(f"  Effect size d = {d4:.4f}")
    print(f"  p-value       = {p4:.4f}")

    # ── Debiasing ──

    print("\n=== Debiasing: Gender Neutralization ===")
    gender_bias_dir = compute_bias_direction(
        embeddings,
        list(range(vocab['male_names'][0], vocab['male_names'][-1] + 1)),
        list(range(vocab['female_names'][0], vocab['female_names'][-1] + 1)),
    )
    print(f"  Gender bias direction (first 5 dims): {gender_bias_dir[:5].tolist()}")

    # Debias all words except the target group names (we keep those to re-test)
    protect_gender = set(vocab['male_names'] + vocab['female_names'])
    debiased_gender = debias_embeddings(embeddings, gender_bias_dir, protect_idx=protect_gender)

    # Re-evaluate after debiasing
    male_deb = debiased_gender[vocab['male_names']]
    female_deb = debiased_gender[vocab['female_names']]
    career_deb = debiased_gender[vocab['career']]
    family_deb = debiased_gender[vocab['family']]

    d1_deb = weat_effect_size(male_deb, female_deb, career_deb, family_deb)
    p1_deb = weat_p_value(male_deb, female_deb, career_deb, family_deb, n_permutations=1000)
    print(f"  After debiasing: d = {d1_deb:.4f}, p = {p1_deb:.4f}")
    print(f"  Reduction: {abs(d1 - d1_deb) / (abs(d1) + 1e-8) * 100:.1f}%")

    print("\n=== Debiasing: Race Neutralization ===")
    race_bias_dir = compute_bias_direction(
        embeddings,
        list(range(vocab['ea_names'][0], vocab['ea_names'][-1] + 1)),
        list(range(vocab['aa_names'][0], vocab['aa_names'][-1] + 1)),
    )
    protect_race = set(vocab['ea_names'] + vocab['aa_names'])
    debiased_race = debias_embeddings(embeddings, race_bias_dir, protect_idx=protect_race)

    ea_deb = debiased_race[vocab['ea_names']]
    aa_deb = debiased_race[vocab['aa_names']]
    pleasant_deb = debiased_race[vocab['pleasant']]
    unpleasant_deb = debiased_race[vocab['unpleasant']]

    d2_deb = weat_effect_size(ea_deb, aa_deb, pleasant_deb, unpleasant_deb)
    p2_deb = weat_p_value(ea_deb, aa_deb, pleasant_deb, unpleasant_deb, n_permutations=1000)
    print(f"  After debiasing: d = {d2_deb:.4f}, p = {p2_deb:.4f}")
    print(f"  Reduction: {abs(d2 - d2_deb) / (abs(d2) + 1e-8) * 100:.1f}%")

    # Debias both simultaneously
    print("\n=== Debiasing: Both Gender + Race ===")
    debiased_both = debias_embeddings(debiased_gender, race_bias_dir, protect_idx=protect_race)

    ea_both = debiased_both[vocab['ea_names']]
    aa_both = debiased_both[vocab['aa_names']]
    pleasant_both = debiased_both[vocab['pleasant']]
    unpleasant_both = debiased_both[vocab['unpleasant']]
    male_both = debiased_both[vocab['male_names']]
    female_both = debiased_both[vocab['female_names']]
    career_both = debiased_both[vocab['career']]
    family_both = debiased_both[vocab['family']]

    d1_both = weat_effect_size(male_both, female_both, career_both, family_both)
    d2_both = weat_effect_size(ea_both, aa_both, pleasant_both, unpleasant_both)
    print(f"  Gender bias: d = {d1_both:.4f} (was {d1:.4f})")
    print(f"  Race bias:   d = {d2_both:.4f} (was {d2:.4f})")

    # ── Per-dimension bias analysis ──

    print("\n=== Per-Dimension Bias Scores ===")
    n_dims = embeddings.shape[1]
    dim_scores = {}
    for dim in range(min(n_dims, 10)):
        # Use single dimension as "attribute"
        dim_vec = torch.zeros(1, n_dims)
        dim_vec[0, dim] = 1.0

        # Gender bias along this dimension
        male_proj = embeddings[vocab['male_names']] @ dim_vec.T
        female_proj = embeddings[vocab['female_names']] @ dim_vec.T
        gender_dim_bias = (male_proj.mean() - female_proj.mean()).item()

        # Race bias along this dimension
        ea_proj = embeddings[vocab['ea_names']] @ dim_vec.T
        aa_proj = embeddings[vocab['aa_names']] @ dim_vec.T
        race_dim_bias = (ea_proj.mean() - aa_proj.mean()).item()

        dim_scores[dim] = {'gender': gender_dim_bias, 'race': race_dim_bias}
        print(f"  Dim {dim:2d}: gender={gender_dim_bias:+.4f}, race={race_dim_bias:+.4f}")

    # ── Visualization ──

    # 1. Bias heatmap: WEAT scores across demographic dimensions × attribute sets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before debiasing
    tests = ['Gender×\nCareer/Family', 'Race×\nPleasant/Unpleasant',
             'Gender×\nPleasant/Unpleasant', 'Race×\nCareer/Family']
    before_scores = [d1, d2, d3, d4]
    after_gender_scores = [d1_deb, d2, d3, d4]  # Only gender debiased
    after_both_scores = [d1_both, d2_both, d3, d4]

    x = np.arange(len(tests))
    width = 0.25

    bars1 = axes[0].bar(x - width, before_scores, width, label='Biased', color='#e74c3c', alpha=0.8)
    bars2 = axes[0].bar(x, after_gender_scores, width, label='Gender Debias', color='#f39c12', alpha=0.8)
    bars3 = axes[0].bar(x + width, after_both_scores, width, label='Full Debias', color='#2ecc71', alpha=0.8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tests, fontsize=9)
    axes[0].set_ylabel("WEAT Effect Size (d)")
    axes[0].set_title("WEAT Bias Scores: Before vs After Debiasing")
    axes[0].legend()
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Per-dimension bias scores
    dims = list(dim_scores.keys())
    gender_vals = [dim_scores[d]['gender'] for d in dims]
    race_vals = [dim_scores[d]['race'] for d in dims]

    axes[1].bar(np.array(dims) - 0.15, gender_vals, 0.3, label='Gender bias', color='#3498db', alpha=0.8)
    axes[1].bar(np.array(dims) + 0.15, race_vals, 0.3, label='Race bias', color='#e74c3c', alpha=0.8)
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_ylabel("Bias Score (mean diff)")
    axes[1].set_title("Per-Dimension Bias: Which Dimensions Carry Bias?")
    axes[1].legend()
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "bias_heatmap.png", dpi=150)
    plt.close()

    # 3. Embedding space visualization (2D PCA)
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (emb, title) in enumerate([
        (embeddings, "Biased Embeddings"),
        (debiased_gender, "Gender-Debiased"),
        (debiased_both, "Fully Debiased"),
    ]):
        # Use only target + attribute words for clarity
        groups_to_plot = {
            'Male Names': vocab['male_names'],
            'Female Names': vocab['female_names'],
            'Career': vocab['career'],
            'Family': vocab['family'],
            'EA Names': vocab['ea_names'],
            'AA Names': vocab['aa_names'],
        }

        subset_idx = []
        for idx_list in groups_to_plot.values():
            subset_idx.extend(idx_list)
        subset_idx = sorted(set(subset_idx))

        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb[subset_idx].numpy())

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        markers = ['o', 's', '^', 'v', 'D', 'p']

        offset = 0
        for (name, idx_list), color, marker in zip(groups_to_plot.items(), colors, markers):
            n = len(idx_list)
            axes[idx].scatter(emb_2d[offset:offset + n, 0], emb_2d[offset:offset + n, 1],
                              c=color, marker=marker, label=name, alpha=0.7, s=40)
            offset += n

        axes[idx].set_title(title)
        axes[idx].legend(fontsize=7, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Word Embedding Space: Biased vs Debiased", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "embedding_pca.png", dpi=150)
    plt.close()

    # 4. Debiasing progression: effect of increasing debiasing strength
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Partial debiasing: mix biased and debiased embeddings
    alphas = np.linspace(0, 1, 11)
    gender_scores = []
    race_scores = []

    for alpha in alphas:
        partial = embeddings * (1 - alpha) + debiased_both * alpha
        m = partial[vocab['male_names']]
        f = partial[vocab['female_names']]
        c = partial[vocab['career']]
        fa = partial[vocab['family']]
        gender_scores.append(abs(weat_effect_size(m, f, c, fa)))

        ea = partial[vocab['ea_names']]
        aa = partial[vocab['aa_names']]
        pl = partial[vocab['pleasant']]
        un = partial[vocab['unpleasant']]
        race_scores.append(abs(weat_effect_size(ea, aa, pl, un)))

    axes[0].plot(alphas, gender_scores, 'o-', label='|Gender bias|', color='#3498db', linewidth=2)
    axes[0].plot(alphas, race_scores, 's-', label='|Race bias|', color='#e74c3c', linewidth=2)
    axes[0].set_xlabel("Debiasing Strength (alpha)")
    axes[0].set_ylabel("Absolute WEAT Effect Size")
    axes[0].set_title("Bias Reduction vs Debiasing Strength")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 5. Significance test results
    test_names = ['Gender×\nCareer', 'Race×\nPleasant', 'Gender×\nPleasant', 'Race×\nCareer']
    p_vals_before = [p1, p2, p3, p4]
    p_vals_after = [p1_deb, p2_deb, p3, p4]

    x = np.arange(len(test_names))
    axes[1].bar(x - 0.15, -np.log10([max(p, 1e-10) for p in p_vals_before]), 0.3,
                label='Before debiasing', color='#e74c3c', alpha=0.8)
    axes[1].bar(x + 0.15, -np.log10([max(p, 1e-10) for p in p_vals_after]), 0.3,
                label='After debiasing', color='#2ecc71', alpha=0.8)
    axes[1].axhline(y=-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(test_names, fontsize=9)
    axes[1].set_ylabel("-log10(p-value)")
    axes[1].set_title("Statistical Significance of Bias")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "debiasing_analysis.png", dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("WEAT\nMeasurement", "Target sets (A, B):\nMale/Female names\nAttribute sets (X, Y):\nCareer/Family words\n\nEffect size d measures\nstrength of association\nPermutation test for\nstatistical significance", 0.17, '#3498db'),
        ("Bias\nDirection", "Compute direction\nseparating groups:\n  dir = centroid(A) - centroid(B)\n\nThis direction encodes\nthe systematic bias\nin embedding space\n→ Identified via PCA/mean diff", 0.5, '#e74c3c'),
        ("Debiasing\nNeutralization", "Project out bias:\n  e' = e - (e·dir)·dir\nRemove bias component\nfrom each embedding\n\nPreserves non-bias\ninformation while\neliminating bias direction", 0.83, '#2ecc71'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.78, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("WEAT: Word Embedding Association Test for Bias Detection & Debiasing",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "weat_concept.png", dpi=150)
    plt.close()

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"  {'Test':<30s} | {'Before d':>9s} | {'After d':>9s} | {'Reduction':>9s}")
    print("  " + "-" * 65)
    print(f"  {'Gender × Career/Family':<30s} | {d1:>+9.4f} | {d1_both:>+9.4f} | "
          f"{abs(d1 - d1_both) / (abs(d1) + 1e-8) * 100:>8.1f}%")
    print(f"  {'Race × Pleasant/Unpleasant':<30s} | {d2:>+9.4f} | {d2_both:>+9.4f} | "
          f"{abs(d2 - d2_both) / (abs(d2) + 1e-8) * 100:>8.1f}%")
    print(f"  {'Gender × Pleasant/Unpleasant':<30s} | {d3:>+9.4f} | {'N/A':>9s} | {'N/A':>9s}")
    print(f"  {'Race × Career/Family':<30s} | {d4:>+9.4f} | {'N/A':>9s} | {'N/A':>9s}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
