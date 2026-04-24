"""
Minimal Constitutional AI (CAI) Reproduction
=============================================
Reproduces core ideas from "Constitutional AI: Harmlessness from AI Feedback"
(2212.08073, Bai et al.):
1. Self-critique: model critiques its own harmful responses
2. Revision: model revises responses based on critique
3. Constitutional principles: a set of rules guiding critique and revision
4. RL from AI Feedback (RLAIF): train on AI-generated preferences
5. Compare: standard RLHF vs Constitutional AI (self-critique + RLAIF)
6. Show: harmfulness reduction through constitutional revision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── Models ──

class TinyLM(nn.Module):
    """Small language model for generating/critiquing/revision."""
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2, max_len=24):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=12, temperature=1.0):
        B = prompts.shape[0]
        current = prompts.clone()
        for _ in range(max_new_tokens):
            if current.shape[1] >= self.max_len:
                break
            logits = self.forward(current)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            current = torch.cat([current, next_tok], dim=1)
        return current


class RewardModel(nn.Module):
    """Reward model for RLAIF preference learning."""
    def __init__(self, vocab_size=32, d_model=64, n_heads=2, n_layers=2, max_len=24):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4,
                                       batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h)[:, -1, :]).squeeze(-1)


# ── Constitutional Principles ──

PRINCIPLES = [
    "Choose the response that is most helpful and least harmful",
    "Choose the response that avoids assisting with dangerous activities",
    "Choose the response that is most respectful and least offensive",
    "Choose the response that is most honest and transparent",
    "Choose the response that promotes wellbeing and avoids harm",
]


# ── Synthetic Data ──

def generate_prompts(batch_size, vocab_size=32, prompt_len=6, device='cpu'):
    """Generate synthetic prompts — some 'harmful' (contain danger tokens)."""
    prompts = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
    # Mark some as 'harmful' by setting first token to danger zone (0-3)
    harm_mask = torch.rand(batch_size) < 0.5
    prompts[harm_mask, 0] = torch.randint(0, 4, (harm_mask.sum(),), device=device)
    return prompts, harm_mask


def score_harmfulness(sequences, vocab_size=32):
    """Score how 'harmful' a sequence is (synthetic metric).

    Harmful = contains many low-index tokens (danger zone).
    """
    # Count tokens in danger zone (0-3)
    danger_count = (sequences < 4).float().sum(dim=-1)
    total = sequences.shape[-1]
    return danger_count / total  # 0 = safe, 1 = maximally harmful


def score_helpfulness(sequences, vocab_size=32):
    """Score helpfulness (synthetic metric).

    Helpful = diverse vocabulary usage.
    """
    diversity = []
    for seq in sequences:
        diversity.append(len(seq.unique()) / seq.shape[0])
    return torch.tensor(diversity, dtype=torch.float32)


# ── Constitutional AI Pipeline ──

def constitutional_revision(lm, prompts, principles, n_revisions=2, device='cpu'):
    """CAI self-critique and revision loop.

    1. Generate initial response
    2. For each revision round:
       a. Critique: evaluate response against principles
       b. Revise: improve response based on critique
    """
    # Step 1: Initial generation
    responses = lm.generate(prompts, max_new_tokens=10, temperature=1.0)

    revision_history = [responses.cpu().clone()]

    for rev in range(n_revisions):
        # Step 2a: Self-critique (simulated by scoring)
        harm_scores = score_harmfulness(responses)
        help_scores = score_helpfulness(responses)

        # Step 2b: Revision — regenerate with lower temperature and constraint
        # In real CAI, the model sees its critique and revises.
        # Here we simulate by generating with lower temperature and penalty.
        revised = lm.generate(prompts, max_new_tokens=10, temperature=0.7)

        # Filter: only keep revisions that reduce harm
        revised_harm = score_harmfulness(revised)
        improved = revised_harm < harm_scores

        # Keep revised where it improved, else keep original
        responses = torch.where(
            improved.unsqueeze(-1).expand_as(responses),
            revised, responses
        )

        revision_history.append(responses.cpu().clone())

    return responses, revision_history


# ── RLAIF: AI Feedback Preference Learning ──

def generate_constitutional_preferences(lm, prompts, n_pairs=64, device='cpu'):
    """Generate preference pairs using constitutional principles.

    For each prompt, generate two responses and have the AI choose
    which is more constitutional (less harmful, more helpful).
    """
    chosen = []
    rejected = []

    for _ in range(n_pairs):
        # Generate two responses
        r1 = lm.generate(prompts[:1], max_new_tokens=10, temperature=1.2)
        r2 = lm.generate(prompts[:1], max_new_tokens=10, temperature=0.8)

        # AI evaluates using constitutional principles
        harm1 = score_harmfulness(r1).item()
        harm2 = score_harmfulness(r2).item()
        help1 = score_helpfulness(r1).item()
        help2 = score_helpfulness(r2).item()

        # Constitutional score: low harm + high helpfulness
        score1 = -harm1 + 0.5 * help1
        score2 = -harm2 + 0.5 * help2

        if score1 > score2:
            chosen.append(r1[0])
            rejected.append(r2[0])
        else:
            chosen.append(r2[0])
            rejected.append(r1[0])

    return torch.stack(chosen), torch.stack(rejected)


def train_rlaif(reward_model, chosen, rejected, n_steps=500, lr=1e-3, device='cpu'):
    """Train reward model on AI-generated preferences (Bradley-Terry)."""
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr)

    for step in range(n_steps):
        idx = torch.randint(0, len(chosen), (32,))
        c = chosen[idx].to(device)
        r = rejected[idx].to(device)

        r_chosen = reward_model(c)
        r_rejected = reward_model(r)

        # Bradley-Terry loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return reward_model


# ── Training ──

def train_sft(model, data_fn, n_steps=2000, batch_size=32, lr=1e-3, device='cpu'):
    """Supervised fine-tuning on 'good' responses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []

    for step in range(n_steps):
        prompts, harm_mask = data_fn(batch_size, device=device)

        # Generate target: safe responses (avoid danger tokens)
        # Target: replace danger tokens with safe ones (indices 10+)
        target = prompts.clone()
        danger = target < 4
        target[danger] = torch.randint(10, 32, (danger.sum(),), device=device)

        logits = model(target[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                               target[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | SFT Loss: {loss.item():.4f}")

    return losses


def train_rlhf(policy, ref_model, reward_model, data_fn, n_steps=1000,
               batch_size=16, kl_coef=0.1, device='cpu'):
    """Standard RLHF training."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = {'loss': [], 'reward': [], 'harm': []}

    for step in range(n_steps):
        prompts, harm_mask = data_fn(batch_size, device=device)

        # Generate
        responses = policy.generate(prompts, max_new_tokens=8)

        # Reward
        with torch.no_grad():
            rewards = reward_model(responses)
            ref_logits = ref_model(responses)
        pol_logits = policy(responses)

        # KL penalty
        pol_logp = F.log_softmax(pol_logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        kl = (pol_logp.exp() * (pol_logp - ref_logp)).sum(dim=-1).sum(dim=-1)
        total_reward = rewards - kl_coef * kl

        # Policy gradient
        log_probs = F.log_softmax(pol_logits[:, :-1], dim=-1)
        actions = responses[:, 1:]
        action_logp = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        loss = -(action_logp * (total_reward - total_reward.mean())).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            harm = score_harmfulness(responses).mean().item()

        metrics['loss'].append(loss.item())
        metrics['reward'].append(rewards.mean().item())
        metrics['harm'].append(harm)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Reward: {rewards.mean().item():.3f} | Harm: {harm:.3f}")

    return metrics


def train_constitutional(policy, ref_model, data_fn, n_steps=1000,
                          batch_size=16, kl_coef=0.1, n_revisions=2, device='cpu'):
    """Constitutional AI training: self-critique + revision + RLAIF."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    reward_model = RewardModel(vocab_size=32).to(device)
    metrics = {'loss': [], 'reward': [], 'harm': []}

    for step in range(n_steps):
        prompts, harm_mask = data_fn(batch_size, device=device)

        # Constitutional revision
        revised, _ = constitutional_revision(policy, prompts, PRINCIPLES,
                                              n_revisions=n_revisions, device=device)

        # RLAIF: generate preference data and update reward model
        if step % 50 == 0:
            chosen, rejected = generate_constitutional_preferences(
                policy, prompts[:8], n_pairs=32, device=device)
            reward_model = train_rlaif(reward_model, chosen, rejected,
                                        n_steps=100, device=device)

        # Reward from AI feedback
        with torch.no_grad():
            rewards = reward_model(revised)
            ref_logits = ref_model(revised)

        pol_logits = policy(revised)
        pol_logp = F.log_softmax(pol_logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        kl = (pol_logp.exp() * (pol_logp - ref_logp)).sum(dim=-1).sum(dim=-1)
        total_reward = rewards - kl_coef * kl

        log_probs = F.log_softmax(pol_logits[:, :-1], dim=-1)
        actions = revised[:, 1:]
        action_logp = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        loss = -(action_logp * (total_reward - total_reward.mean())).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            harm = score_harmfulness(revised).mean().item()

        metrics['loss'].append(loss.item())
        metrics['reward'].append(rewards.mean().item())
        metrics['harm'].append(harm)

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Reward: {rewards.mean().item():.3f} | Harm: {harm:.3f}")

    return metrics


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "111-constitutional-ai"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_fn = lambda bs, device='cpu': generate_prompts(bs, device=device)

    # ── SFT Phase ──
    print("=== Phase 1: Supervised Fine-Tuning ===")
    sft_model = TinyLM(vocab_size=32).to(device)
    sft_losses = train_sft(sft_model, data_fn, n_steps=1500, device=device)

    # Create reference model
    ref_model = deepcopy(sft_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Train Reward Model (for RLHF baseline) ──
    print("\n=== Training Reward Model (Human Feedback Sim) ===")
    rm_rlhf = RewardModel(vocab_size=32).to(device)
    # Simulate: preferred = low harm, high helpfulness
    for step in range(300):
        prompts, _ = data_fn(32, device=device)
        r1 = sft_model.generate(prompts, max_new_tokens=8, temperature=1.0)
        r2 = sft_model.generate(prompts, max_new_tokens=8, temperature=0.7)

        h1 = score_harmfulness(r1).to(device)
        h2 = score_harmfulness(r2).to(device)
        d1 = score_helpfulness(r1).to(device)
        d2 = score_helpfulness(r2).to(device)

        score1 = -h1 + 0.3 * d1
        score2 = -h2 + 0.3 * d2

        chosen = torch.where((score1 > score2).unsqueeze(-1).expand_as(r1), r1, r2)
        rejected = torch.where((score1 > score2).unsqueeze(-1).expand_as(r1), r2, r1)

        rc = rm_rlhf(chosen)
        rr = rm_rlhf(rejected)
        loss = -F.logsigmoid(rc - rr).mean()

        rm_rlhf.zero_grad()
        loss.backward()
        torch.optim.Adam(rm_rlhf.parameters(), lr=1e-3).step()

    # ── RLHF Baseline ──
    print("\n=== Phase 2a: Standard RLHF ===")
    rlhf_policy = deepcopy(sft_model)
    rlhf_metrics = train_rlhf(rlhf_policy, ref_model, rm_rlhf, data_fn,
                                n_steps=800, device=device)

    # ── Constitutional AI ──
    print("\n=== Phase 2b: Constitutional AI (Self-Critique + RLAIF) ===")
    cai_policy = deepcopy(sft_model)
    cai_metrics = train_constitutional(cai_policy, ref_model, data_fn,
                                        n_steps=800, n_revisions=2, device=device)

    # ── Evaluation ──
    print("\n=== Final Evaluation ===")
    n_eval = 200
    prompts, harm_mask = data_fn(n_eval, device=device)

    with torch.no_grad():
        sft_resp = sft_model.generate(prompts, max_new_tokens=10)
        rlhf_resp = rlhf_policy.generate(prompts, max_new_tokens=10)
        cai_resp, _ = constitutional_revision(cai_policy, prompts, PRINCIPLES, n_revisions=2, device=device)

    sft_harm = score_harmfulness(sft_resp).mean().item()
    rlhf_harm = score_harmfulness(rlhf_resp).mean().item()
    cai_harm = score_harmfulness(cai_resp).mean().item()

    sft_help = score_helpfulness(sft_resp).mean().item()
    rlhf_help = score_helpfulness(rlhf_resp).mean().item()
    cai_help = score_helpfulness(cai_resp).mean().item()

    print(f"  SFT:  harm={sft_harm:.3f}, helpfulness={sft_help:.3f}")
    print(f"  RLHF: harm={rlhf_harm:.3f}, helpfulness={rlhf_help:.3f}")
    print(f"  CAI:  harm={cai_harm:.3f}, helpfulfulness={cai_help:.3f}")

    # ── Visualization ──

    # 1. Harmfulness over training
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    w = 30

    rlhf_harm_s = np.convolve(rlhf_metrics['harm'], np.ones(w)/w, mode='valid')
    cai_harm_s = np.convolve(cai_metrics['harm'], np.ones(w)/w, mode='valid')
    axes[0].plot(rlhf_harm_s, label='RLHF', color='red')
    axes[0].plot(cai_harm_s, label='CAI', color='blue')
    axes[0].set_title("Harmfulness During Training")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Harm Score (lower = safer)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rlhf_r_s = np.convolve(rlhf_metrics['reward'], np.ones(w)/w, mode='valid')
    cai_r_s = np.convolve(cai_metrics['reward'], np.ones(w)/w, mode='valid')
    axes[1].plot(rlhf_r_s, label='RLHF', color='red')
    axes[1].plot(cai_r_s, label='CAI', color='blue')
    axes[1].set_title("Reward During Training")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    rlhf_l_s = np.convolve(rlhf_metrics['loss'], np.ones(w)/w, mode='valid')
    cai_l_s = np.convolve(cai_metrics['loss'], np.ones(w)/w, mode='valid')
    axes[2].plot(rlhf_l_s, label='RLHF', color='red')
    axes[2].plot(cai_l_s, label='CAI', color='blue')
    axes[2].set_title("Training Loss")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('RLHF vs Constitutional AI: Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training_comparison.png', dpi=150)
    plt.close()

    # 2. Final comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ['SFT', 'RLHF', 'CAI']
    harm_vals = [sft_harm, rlhf_harm, cai_harm]
    help_vals = [sft_help, rlhf_help, cai_help]
    colors = ['#95a5a6', '#e74c3c', '#3498db']

    axes[0].bar(methods, harm_vals, color=colors, alpha=0.7)
    axes[0].set_ylabel("Harm Score (lower = safer)")
    axes[0].set_title("Harmfulness")
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(methods, help_vals, color=colors, alpha=0.7)
    axes[1].set_ylabel("Helpfulness Score")
    axes[1].set_title("Helpfulness")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Final Evaluation: SFT vs RLHF vs CAI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'final_comparison.png', dpi=150)
    plt.close()

    # 3. Constitutional revision effect
    print("\n=== Constitutional Revision Effect ===")
    prompts_small, _ = data_fn(50, device=device)
    _, rev_history = constitutional_revision(cai_policy, prompts_small, PRINCIPLES,
                                              n_revisions=3, device=device)

    rev_harms = [score_harmfulness(r).mean().item() for r in rev_history]
    rev_helps = [score_helpfulness(r).mean().item() for r in rev_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rev_labels = [f"Rev {i}" for i in range(len(rev_history))]
    axes[0].plot(rev_labels, rev_harms, marker='o', color='red', linewidth=2)
    axes[0].set_title("Harmfulness After Each Revision")
    axes[0].set_ylabel("Harm Score")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rev_labels, rev_helps, marker='o', color='green', linewidth=2)
    axes[1].set_title("Helpfulness After Each Revision")
    axes[1].set_ylabel("Helpfulness Score")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Constitutional AI: Self-Critique Revision', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'revision_effect.png', dpi=150)
    plt.close()

    # 4. Pipeline concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    concept = (
        "Constitutional AI Pipeline (2212.08073)\n"
        "=" * 55 + "\n\n"
        "Phase 1: Supervised Learning (SFT)\n"
        "  → Train on helpful demonstrations\n\n"
        "Phase 2: Self-Critique & Revision (SL)\n"
        "  1. Generate response to prompt\n"
        "  2. Critique: 'Is this response harmful?'\n"
        "  3. Revise: 'Please rewrite to be less harmful'\n"
        "  4. Repeat for N revision rounds\n"
        "  → Fine-tune on revised (constitutional) responses\n\n"
        "Phase 3: RL from AI Feedback (RLAIF)\n"
        "  1. Generate two responses per prompt\n"
        "  2. AI evaluates using constitutional principles\n"
        "  3. Train reward model on AI preferences\n"
        "  4. PPO with AI reward model + KL constraint\n\n"
        "Key insight: No human labelers needed for harmlessness.\n"
        "The AI critiques itself using constitutional principles."
    )
    ax.text(0.05, 0.95, concept, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
