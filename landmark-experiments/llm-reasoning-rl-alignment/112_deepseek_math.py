"""
Minimal DeepSeekMath / GRPO Reproduction
=========================================
Reproduces core ideas from "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (2402.03300, Shao et al.):
1. GRPO (Group Relative Policy Optimization): no critic/value network needed
2. Group-normalized advantages: baseline from group mean, not value function
3. Math reasoning: multi-step arithmetic with chain-of-thought
4. Compare: REINFORCE vs PPO vs GRPO on math tasks
5. Show: GRPO achieves similar performance with fewer parameters
6. Demonstrate: advantage normalization effect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── Policy Model ──

class MathPolicy(nn.Module):
    """Small transformer for generating math solutions."""
    def __init__(self, vocab_size=20, d_model=64, n_heads=2, n_layers=2, max_len=32):
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
    def generate(self, prompts, max_new_tokens=16, temperature=1.0):
        B = prompts.shape[0]
        current = prompts.clone()
        for _ in range(max_new_tokens):
            if current.shape[1] >= self.max_len:
                break
            logits = self.forward(current)[:, -1, :] / temperature
            logits = logits.clamp(-10, 10)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            current = torch.cat([current, next_tok], dim=1)
        return current


class ValueNetwork(nn.Module):
    """Value network for PPO baseline (not needed for GRPO)."""
    def __init__(self, vocab_size=20, d_model=64, n_heads=2, n_layers=2, max_len=32):
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


# ── Math Problem Generator ──

class MathProblemGenerator:
    """Generate synthetic math problems and verify solutions.

    Token vocabulary:
      0-9: digits
      10: '+'
      11: '-'
      12: '='
      13: step separator
      14: end of answer
      15-19: reserved
    """
    def __init__(self, vocab_size=20, max_len=32, device='cpu'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = device

    def generate_problem(self):
        """Generate a simple addition/subtraction problem."""
        a = np.random.randint(1, 50)
        b = np.random.randint(1, 50)
        op = np.random.choice(['+', '-'])

        if op == '+':
            answer = a + b
        else:
            answer = a - b

        # Encode as tokens: [a_digits, op, b_digits, =, answer_digits, END]
        tokens = list(str(a)) + [op] + list(str(b)) + ['='] + list(str(answer)) + ['<end>']
        token_ids = self._encode(tokens)
        return token_ids, answer

    def _encode(self, tokens):
        """Encode text tokens to IDs."""
        ids = []
        for t in tokens:
            if t.isdigit():
                ids.append(int(t))
            elif t == '+':
                ids.append(10)
            elif t == '-':
                ids.append(11)
            elif t == '=':
                ids.append(12)
            elif t == '<end>':
                ids.append(14)
            elif t == '<step>':
                ids.append(13)
            else:
                ids.append(0)
        return ids

    def generate_batch(self, batch_size, prompt_len=6):
        """Generate a batch of math problems."""
        problems = []
        answers = []
        for _ in range(batch_size):
            ids, ans = self.generate_problem()
            # Pad/truncate to prompt_len
            ids = ids[:prompt_len]
            ids = ids + [0] * (prompt_len - len(ids))
            problems.append(ids)
            answers.append(ans)
        return torch.tensor(problems, dtype=torch.long, device=self.device), answers

    def verify_answer(self, response, expected_answer):
        """Check if response contains the correct answer.

        Extracts the number after '=' sign.
        """
        # response is 1D: (T,)
        eq_positions = (response == 12).nonzero(as_tuple=False)
        if eq_positions.numel() == 0:
            return False

        # Get tokens after first '='
        eq_pos = eq_positions[0].item()
        after_eq = response[eq_pos + 1:]
        # Extract digits until non-digit
        answer_tokens = []
        for t in after_eq:
            if t < 10:
                answer_tokens.append(str(t.item()))
            else:
                break
        if answer_tokens:
            try:
                parsed = int(''.join(answer_tokens))
                return parsed == expected_answer
            except ValueError:
                pass
        return False


# ── Reward Functions ──

def compute_reward(responses, answers, math_gen):
    """Compute reward: 1 if correct answer, 0 otherwise.
    Small bonus for format (contains '=' sign).
    """
    batch_size = responses.shape[0]
    rewards = torch.zeros(batch_size, device=responses.device)

    for i in range(batch_size):
        if math_gen.verify_answer(responses[i], answers[i]):
            rewards[i] = 1.0
        elif (responses[i] == 12).any():  # has '=' sign
            rewards[i] = 0.1

    return rewards


# ── Training Algorithms ──

def train_grpo(policy, ref_policy, math_gen, n_steps=2000, batch_size=16,
               group_size=4, lr=1e-4, kl_coef=0.05, clip_eps=0.2, device='cpu'):
    """GRPO: Group Relative Policy Optimization.

    Key idea: instead of a value network, compute advantages relative to
    the group mean. For each prompt, generate `group_size` responses,
    compute rewards, and use group statistics as baseline.
    """
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    metrics = {'loss': [], 'reward': [], 'kl': [], 'correct': []}

    for step in range(n_steps):
        prompts, answers = math_gen.generate_batch(batch_size, prompt_len=6)

        # Generate group_size responses per prompt
        all_responses = []
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []

        for g in range(group_size):
            responses = policy.generate(prompts, max_new_tokens=12, temperature=0.8)
            rewards = compute_reward(responses, answers, math_gen)

            # Current policy log probs
            logits = policy(responses[:, :-1])
            log_p = F.log_softmax(logits, dim=-1)
            actions = responses[:, 1:]
            log_probs = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

            # Reference log probs
            with torch.no_grad():
                ref_logits = ref_policy(responses[:, :-1])
                ref_log_p = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs_batch = ref_log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

            all_responses.append(responses)
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
            all_ref_log_probs.append(ref_log_probs_batch)

        # Stack: (group_size, batch_size)
        rewards_stack = torch.stack(all_rewards)
        log_probs_stack = torch.stack(all_log_probs)
        ref_log_probs_stack = torch.stack(all_ref_log_probs)

        # GRPO advantage: group-relative normalization
        # advantage_i = (r_i - mean(r_group)) / std(r_group)
        group_mean = rewards_stack.mean(dim=0)
        group_std = rewards_stack.std(dim=0, correction=0) + 1e-8  # population std for small groups
        advantages = (rewards_stack - group_mean) / group_std  # (G, B)

        # KL penalty
        kl = log_probs_stack - ref_log_probs_stack  # (G, B)

        # PPO-style clipped objective
        # For simplicity, use old_log_probs from current generation
        with torch.no_grad():
            old_log_probs = log_probs_stack.clone()

        ratio = torch.exp(log_probs_stack - old_log_probs)
        adv_expanded = advantages

        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_expanded
        loss = -torch.min(surr1, surr2).mean() + kl_coef * (kl ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            avg_reward = rewards_stack.mean().item()
            correct = (rewards_stack == 1.0).float().mean().item()

        metrics['loss'].append(loss.item())
        metrics['reward'].append(avg_reward)
        metrics['kl'].append(kl.mean().item())
        metrics['correct'].append(correct)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Reward: {avg_reward:.3f} | Correct: {correct:.3f}")

    return metrics


def train_reinforce(policy, ref_policy, math_gen, n_steps=2000, batch_size=16,
                    lr=1e-4, kl_coef=0.05, device='cpu'):
    """REINFORCE with baseline (mean reward)."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    metrics = {'loss': [], 'reward': [], 'kl': [], 'correct': []}

    for step in range(n_steps):
        prompts, answers = math_gen.generate_batch(batch_size, prompt_len=6)
        responses = policy.generate(prompts, max_new_tokens=12, temperature=0.8)
        rewards = compute_reward(responses, answers, math_gen)

        logits = policy(responses[:, :-1])
        log_p = F.log_softmax(logits, dim=-1)
        actions = responses[:, 1:]
        log_probs = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        with torch.no_grad():
            ref_logits = ref_policy(responses[:, :-1])
            ref_log_p = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        kl = log_probs - ref_log_probs

        # REINFORCE: advantage = reward - baseline (mean)
        advantage = rewards - rewards.mean()
        if rewards.std() > 1e-8:
            advantage = advantage / (rewards.std() + 1e-8)

        loss = -(log_probs * advantage).mean() + kl_coef * (kl ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            correct = (rewards == 1.0).float().mean().item()

        metrics['loss'].append(loss.item())
        metrics['reward'].append(rewards.mean().item())
        metrics['kl'].append(kl.mean().item())
        metrics['correct'].append(correct)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Reward: {rewards.mean().item():.3f} | Correct: {correct:.3f}")

    return metrics


def train_ppo(policy, ref_policy, value_net, math_gen, n_steps=2000,
              batch_size=16, lr=1e-4, kl_coef=0.05, clip_eps=0.2, device='cpu'):
    """PPO with value network baseline."""
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(value_net.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    metrics = {'loss': [], 'reward': [], 'kl': [], 'correct': []}

    for step in range(n_steps):
        prompts, answers = math_gen.generate_batch(batch_size, prompt_len=6)
        responses = policy.generate(prompts, max_new_tokens=12, temperature=0.8)
        rewards = compute_reward(responses, answers, math_gen)

        # Policy
        logits = policy(responses[:, :-1])
        log_p = F.log_softmax(logits, dim=-1)
        actions = responses[:, 1:]
        log_probs = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # Value
        values = value_net(responses)

        # Advantage = reward - value
        advantage = rewards - values.detach()
        if advantage.std() > 1e-8:
            advantage = advantage / (advantage.std() + 1e-8)

        # PPO clipped
        with torch.no_grad():
            old_log_probs = log_probs.clone()
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, rewards)

        # KL
        with torch.no_grad():
            ref_logits = ref_policy(responses[:, :-1])
            ref_log_p = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        kl = log_probs - ref_log_probs

        loss = policy_loss + 0.5 * value_loss + kl_coef * (kl ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            correct = (rewards == 1.0).float().mean().item()

        metrics['loss'].append(loss.item())
        metrics['reward'].append(rewards.mean().item())
        metrics['kl'].append(kl.mean().item())
        metrics['correct'].append(correct)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Reward: {rewards.mean().item():.3f} | Correct: {correct:.3f}")

    return metrics


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "112-deepseek-math"
    results_dir.mkdir(parents=True, exist_ok=True)

    math_gen = MathProblemGenerator(vocab_size=20, max_len=32, device=device)

    # ── Train REINFORCE ──
    print("=== Training REINFORCE ===")
    reinforce_policy = MathPolicy(vocab_size=20).to(device)
    ref_r = deepcopy(reinforce_policy)
    ref_r.eval()
    for p in ref_r.parameters():
        p.requires_grad = False
    r_metrics = train_reinforce(reinforce_policy, ref_r, math_gen,
                                 n_steps=1000, device=device)

    # ── Train PPO ──
    print("\n=== Training PPO (with Value Network) ===")
    ppo_policy = MathPolicy(vocab_size=20).to(device)
    value_net = ValueNetwork(vocab_size=20).to(device)
    ref_p = deepcopy(ppo_policy)
    ref_p.eval()
    for p in ref_p.parameters():
        p.requires_grad = False
    ppo_params = sum(p.numel() for p in ppo_policy.parameters()) + sum(p.numel() for p in value_net.parameters())
    print(f"  Total params (policy + value): {ppo_params:,}")
    p_metrics = train_ppo(ppo_policy, ref_p, value_net, math_gen,
                           n_steps=1000, device=device)

    # ── Train GRPO ──
    print("\n=== Training GRPO (no Value Network) ===")
    grpo_policy = MathPolicy(vocab_size=20).to(device)
    ref_g = deepcopy(grpo_policy)
    ref_g.eval()
    for p in ref_g.parameters():
        p.requires_grad = False
    grpo_params = sum(p.numel() for p in grpo_policy.parameters())
    print(f"  Total params (policy only): {grpo_params:,} (vs PPO: {ppo_params:,})")
    g_metrics = train_grpo(grpo_policy, ref_g, math_gen,
                            n_steps=1000, group_size=4, device=device)

    # ── Visualization ──

    # 1. Correctness comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    w = 30

    r_c = np.convolve(r_metrics['correct'], np.ones(w)/w, mode='valid')
    p_c = np.convolve(p_metrics['correct'], np.ones(w)/w, mode='valid')
    g_c = np.convolve(g_metrics['correct'], np.ones(w)/w, mode='valid')

    axes[0].plot(r_c, label='REINFORCE', color='gray')
    axes[0].plot(p_c, label='PPO (w/ value net)', color='red')
    axes[0].plot(g_c, label='GRPO (no value net)', color='blue')
    axes[0].set_title("Math Correctness")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Correct Rate (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    r_r = np.convolve(r_metrics['reward'], np.ones(w)/w, mode='valid')
    p_r = np.convolve(p_metrics['reward'], np.ones(w)/w, mode='valid')
    g_r = np.convolve(g_metrics['reward'], np.ones(w)/w, mode='valid')

    axes[1].plot(r_r, label='REINFORCE', color='gray')
    axes[1].plot(p_r, label='PPO', color='red')
    axes[1].plot(g_r, label='GRPO', color='blue')
    axes[1].set_title("Average Reward")
    axes[1].set_xlabel("Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Parameter efficiency
    axes[2].bar(['REINFORCE', 'PPO', 'GRPO'],
                [grpo_params, ppo_params, grpo_params],
                color=['gray', 'red', 'blue'], alpha=0.7)
    axes[2].set_ylabel("Total Parameters")
    axes[2].set_title("Parameter Efficiency")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('DeepSeekMath: GRPO vs PPO vs REINFORCE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training_comparison.png', dpi=150)
    plt.close()

    # 2. GRPO advantage distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    prompts, answers = math_gen.generate_batch(64, prompt_len=6)
    with torch.no_grad():
        responses = grpo_policy.generate(prompts, max_new_tokens=12)
        rewards = compute_reward(responses, answers, math_gen)

    # Simulate GRPO advantage computation
    group_size = 4
    n_groups = 64 // group_size
    rewards_grouped = rewards.reshape(n_groups, group_size)
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
    grpo_advantages = (rewards_grouped - group_mean) / group_std

    # REINFORCE advantages
    reinforce_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    axes[0].hist(grpo_advantages.flatten().cpu().numpy(), bins=20, color='blue', alpha=0.7, label='GRPO')
    axes[0].hist(reinforce_advantages.cpu().numpy(), bins=20, color='gray', alpha=0.5, label='REINFORCE')
    axes[0].set_title("Advantage Distribution")
    axes[0].set_xlabel("Advantage")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # GRPO variance reduction
    grpo_var = grpo_advantages.var().item()
    reinforce_var = reinforce_advantages.var().item()
    axes[1].bar(['REINFORCE', 'GRPO'], [reinforce_var, grpo_var],
                color=['gray', 'blue'], alpha=0.7)
    axes[1].set_ylabel("Advantage Variance")
    axes[1].set_title("Variance Reduction: GRPO vs REINFORCE")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / 'advantage_analysis.png', dpi=150)
    plt.close()

    # 3. Group size study
    fig, ax = plt.subplots(figsize=(10, 5))
    group_sizes = [2, 4, 8, 16]
    final_correct = []

    for gs in group_sizes:
        test_policy = MathPolicy(vocab_size=20).to(device)
        ref_t = deepcopy(test_policy)
        ref_t.eval()
        for p in ref_t.parameters():
            p.requires_grad = False
        m = train_grpo(test_policy, ref_t, math_gen, n_steps=500,
                        group_size=gs, device=device)
        final_correct.append(np.mean(m['correct'][-100:]))

    ax.plot(group_sizes, final_correct, marker='o', color='blue', linewidth=2)
    ax.set_xlabel("Group Size")
    ax.set_ylabel("Final Correct Rate")
    ax.set_title("GRPO: Effect of Group Size on Performance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'group_size_study.png', dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    concept = (
        "DeepSeekMath: GRPO (2402.03300)\n"
        "=" * 50 + "\n\n"
        "Problem: PPO needs a value network (extra parameters, instability)\n\n"
        "GRPO Solution: Group Relative advantages\n"
        "  For each prompt, generate G responses:\n"
        "    r_1, r_2, ..., r_G\n\n"
        "  Advantage_i = (r_i - mean(r)) / std(r)\n\n"
        "  No value network needed! The group serves as baseline.\n\n"
        "Benefits:\n"
        "  • Fewer parameters (no critic network)\n"
        "  • Lower variance than REINFORCE\n"
        "  • Simpler implementation than PPO\n"
        "  • Scales well with group size G\n\n"
        "Tradeoff:\n"
        "  • More computation per step (G generations per prompt)\n"
        "  • Larger batch needed for stable group statistics"
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
