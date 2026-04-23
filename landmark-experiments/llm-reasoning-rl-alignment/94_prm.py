"""
Process Reward Model (PRM) — "Let's Verify Step by Step"
=========================================================
Reproduces the core ideas from "Let's Verify Step by Step"
(Lightman et al., 2023, 2305.20050):
1. Train a step-level reward model (Process RM) instead of outcome-level (Outcome RM)
2. Each reasoning step receives its own correctness reward
3. Compare: Outcome RM vs Process RM for solution selection
4. Synthetic: multi-step arithmetic with per-step ground-truth rewards
5. Visualise: step reward heatmap, selection accuracy vs number of solution candidates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Multi-Step Arithmetic ──

class MultiStepArithmetic:
    """Generate multi-step addition problems with per-step correctness labels.

    Problem format:  compute a + b + c + d in sequential steps.
    Step k adds the k-th number to the running total.
    Each step can be correct or incorrect; the final answer depends on all steps.
    """

    def __init__(self, n_numbers=4, max_val=50, n_steps=None):
        self.n_numbers = n_numbers
        self.max_val = max_val
        self.n_steps = n_steps or n_numbers  # one step per addend after the first

    def generate_problem(self, n_problems=1, mistake_prob=0.3):
        """Generate problems with noisy solution traces.

        Returns:
            problems:  (n_problems, n_numbers) ints to add
            traces:    (n_problems, n_steps) intermediate results
            step_correct: (n_problems, n_steps) bool — was each step correct?
            outcome_correct: (n_problems,) bool — was the final answer right?
        """
        problems = np.random.randint(1, self.max_val, (n_problems, self.n_numbers))
        traces = np.zeros((n_problems, self.n_steps), dtype=int)
        step_correct = np.zeros((n_problems, self.n_steps), dtype=bool)

        for i in range(n_problems):
            running = problems[i, 0]
            for s in range(self.n_steps):
                correct_val = running + problems[i, s + 1] if s + 1 < self.n_numbers else running
                if np.random.random() < mistake_prob:
                    # Introduce a mistake
                    error = np.random.choice([-3, -2, -1, 1, 2, 3])
                    traces[i, s] = correct_val + error
                    step_correct[i, s] = False
                else:
                    traces[i, s] = correct_val
                    step_correct[i, s] = True
                running = traces[i, s]

        true_sum = problems.sum(axis=1)
        outcome_correct = traces[:, -1] == true_sum

        return problems, traces, step_correct, outcome_correct

    def generate_solutions(self, problems, n_solutions=8, mistake_prob=0.3):
        """Generate multiple solution traces for the same problem.

        Returns:
            traces:    (n_problems, n_solutions, n_steps)
            step_correct: (n_problems, n_solutions, n_steps)
            outcome_correct: (n_problems, n_solutions)
        """
        N = problems.shape[0]
        traces = np.zeros((N, n_solutions, self.n_steps), dtype=int)
        step_correct = np.zeros((N, n_solutions, self.n_steps), dtype=bool)

        for j in range(n_solutions):
            for i in range(N):
                running = problems[i, 0]
                for s in range(self.n_steps):
                    correct_val = running + problems[i, s + 1] if s + 1 < self.n_numbers else running
                    if np.random.random() < mistake_prob:
                        error = np.random.choice([-3, -2, -1, 1, 2, 3])
                        traces[i, j, s] = correct_val + error
                        step_correct[i, j, s] = False
                    else:
                        traces[i, j, s] = correct_val
                        step_correct[i, j, s] = True
                    running = traces[i, j, s]

        true_sum = problems.sum(axis=1)
        outcome_correct = traces[:, :, -1] == true_sum[:, None]

        return traces, step_correct, outcome_correct


# ── Reward Models ──

class OutcomeRewardModel(nn.Module):
    """Predicts a single scalar reward for the entire solution trace."""

    def __init__(self, d_model=64, n_steps=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(n_steps, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

    def forward(self, traces):
        """
        traces: (B, n_steps) integer intermediate values
        Returns: (B,) scalar outcome reward
        """
        B, T = traces.shape
        x = self.embed(traces.float().unsqueeze(-1))  # (B, T, d)
        pos = torch.arange(T, device=traces.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        x = self.encoder(x)
        pooled = x.mean(dim=1)  # (B, d)
        return self.head(pooled).squeeze(-1)


class ProcessRewardModel(nn.Module):
    """Predicts a reward for each step in the solution trace."""

    def __init__(self, d_model=64, n_steps=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(n_steps, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

    def forward(self, traces):
        """
        traces: (B, n_steps) integer intermediate values
        Returns: (B, n_steps) per-step rewards
        """
        B, T = traces.shape
        x = self.embed(traces.float().unsqueeze(-1))
        pos = torch.arange(T, device=traces.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        x = self.encoder(x)
        return self.head(x).squeeze(-1)  # (B, n_steps)


# ── Training ──

def train_outcome_rm(model, env, n_steps=1000, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        problems, traces, step_ok, outcome_ok = env.generate_problem(batch_size)
        traces_t = torch.tensor(traces, dtype=torch.long, device=device)
        labels = torch.tensor(outcome_ok, dtype=torch.float32, device=device)

        rewards = model(traces_t)
        loss = F.binary_cross_entropy_with_logits(rewards, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 250 == 0:
            acc = ((rewards > 0) == labels).float().mean().item()
            print(f"  [Outcome RM] Step {step+1} | Loss {loss.item():.4f} | Acc {acc:.3f}")

    return losses


def train_process_rm(model, env, n_steps=1000, batch_size=64, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        problems, traces, step_ok, outcome_ok = env.generate_problem(batch_size)
        traces_t = torch.tensor(traces, dtype=torch.long, device=device)
        labels = torch.tensor(step_ok, dtype=torch.float32, device=device)

        rewards = model(traces_t)  # (B, n_steps)
        loss = F.binary_cross_entropy_with_logits(rewards, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 250 == 0:
            preds = (rewards > 0)
            acc = (preds == labels).float().mean().item()
            print(f"  [Process RM] Step {step+1} | Loss {loss.item():.4f} | Step Acc {acc:.3f}")

    return losses


# ── Evaluation: Best-of-N Selection ──

def evaluate_selection(outcome_rm, process_rm, env, n_problems=200,
                       n_solutions_list=(1, 2, 4, 8, 16, 32),
                       mistake_prob=0.35, device='cpu'):
    """For each problem, generate N solutions, select the best via each RM,
    and check if the selected solution is correct.

    Outcome RM: pick solution with highest outcome reward.
    Process RM: pick solution with highest total process reward (sum of step rewards).
    """
    outcome_rm.eval()
    process_rm.eval()

    results = {'orm': [], 'prm': [], 'random': []}

    for n_sol in n_solutions_list:
        # Generate fresh problems
        problems, _, _, _ = env.generate_problem(n_problems, mistake_prob=0)
        # Generate solutions
        all_traces, all_step_ok, all_outcome_ok = env.generate_solutions(
            problems, n_solutions=n_sol, mistake_prob=mistake_prob
        )
        # all_traces: (n_problems, n_sol, n_steps)

        orm_correct = 0
        prm_correct = 0
        rand_correct = 0

        for i in range(n_problems):
            traces_np = all_traces[i]  # (n_sol, n_steps)
            outcome_ok_np = all_outcome_ok[i]  # (n_sol,)

            traces_t = torch.tensor(traces_np, dtype=torch.long, device=device)

            # Outcome RM selection
            with torch.no_grad():
                orm_scores = outcome_rm(traces_t)  # (n_sol,)
            orm_pick = orm_scores.argmax().item()
            if outcome_ok_np[orm_pick]:
                orm_correct += 1

            # Process RM selection
            with torch.no_grad():
                prm_scores = process_rm(traces_t).sum(dim=1)  # (n_sol,)
            prm_pick = prm_scores.argmax().item()
            if outcome_ok_np[prm_pick]:
                prm_correct += 1

            # Random baseline
            rand_pick = np.random.randint(n_sol)
            if outcome_ok_np[rand_pick]:
                rand_correct += 1

        n_p = n_problems
        results['orm'].append(orm_correct / n_p)
        results['prm'].append(prm_correct / n_p)
        results['random'].append(rand_correct / n_p)

        print(f"  N={n_sol:>2} | ORM {orm_correct/n_p:.3f} | "
              f"PRM {prm_correct/n_p:.3f} | Rand {rand_correct/n_p:.3f}")

    return results, n_solutions_list


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "94-prm"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_numbers = 4
    n_steps = n_numbers
    d_model = 64
    env = MultiStepArithmetic(n_numbers=n_numbers, max_val=50, n_steps=n_steps)

    # ── Train models ──
    print("=== Training Outcome RM ===")
    orm = OutcomeRewardModel(d_model, n_steps).to(device)
    orm_losses = train_outcome_rm(orm, env, n_steps=1500, device=device)

    print("\n=== Training Process RM ===")
    prm = ProcessRewardModel(d_model, n_steps).to(device)
    prm_losses = train_process_rm(prm, env, n_steps=1500, device=device)

    # ── Evaluate Best-of-N selection ──
    print("\n=== Best-of-N Selection Accuracy ===")
    sel_results, n_sol_list = evaluate_selection(
        orm, prm, env, n_problems=200,
        n_solutions_list=(1, 2, 4, 8, 16, 32),
        device=device,
    )

    # ── Plot 1: Training loss ──
    w = 20
    fig, ax = plt.subplots(figsize=(8, 5))
    orm_s = np.convolve(orm_losses, np.ones(w)/w, mode='valid')
    prm_s = np.convolve(prm_losses, np.ones(w)/w, mode='valid')
    ax.plot(orm_s, label='Outcome RM', color='red')
    ax.plot(prm_s, label='Process RM', color='blue')
    ax.set_title('Training Loss: Outcome RM vs Process RM')
    ax.set_xlabel('Step')
    ax.set_ylabel('BCE Loss (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'training_loss.png', dpi=150)
    plt.close()

    # ── Plot 2: Step reward heatmap ──
    print("\n=== Step Reward Heatmap ===")
    problems, traces, step_ok, outcome_ok = env.generate_problem(12, mistake_prob=0.4)
    traces_t = torch.tensor(traces, dtype=torch.long, device=device)
    with torch.no_grad():
        prm_rewards = torch.sigmoid(prm(traces_t)).cpu().numpy()  # (12, n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})

    # Heatmap of step rewards
    im = axes[0].imshow(prm_rewards, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Sample')
    axes[0].set_title('PRM: Per-Step Reward Heatmap')
    axes[0].set_xticks(range(n_steps))
    plt.colorbar(im, ax=axes[0], label='Predicted P(correct)')

    # Ground-truth overlay
    gt = step_ok.astype(float)
    axes[1].imshow(gt, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Sample')
    axes[1].set_title('Ground-Truth Step Correctness')
    axes[1].set_xticks(range(n_steps))

    plt.suptitle('Process Reward Model: Step-Level Predictions vs Ground Truth',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'step_reward_heatmap.png', dpi=150)
    plt.close()

    # ── Plot 3: Best-of-N selection accuracy ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_sol_list, sel_results['orm'], 'o-', label='Outcome RM', color='red', linewidth=2)
    ax.plot(n_sol_list, sel_results['prm'], 's-', label='Process RM', color='blue', linewidth=2)
    ax.plot(n_sol_list, sel_results['random'], '^--', label='Random', color='gray', linewidth=1.5)
    ax.set_xlabel('Number of Solution Candidates (N)')
    ax.set_ylabel('Selection Accuracy')
    ax.set_title('Best-of-N: Outcome RM vs Process RM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(results_dir / 'best_of_n.png', dpi=150)
    plt.close()

    # ── Plot 4: PRM advantage analysis — how early errors propagate ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-step accuracy of PRM
    problems, traces, step_ok, outcome_ok = env.generate_problem(500, mistake_prob=0.35)
    traces_t = torch.tensor(traces, dtype=torch.long, device=device)
    with torch.no_grad():
        step_rewards = torch.sigmoid(prm(traces_t)).cpu().numpy()
    step_preds = step_rewards > 0.5
    per_step_acc = (step_preds == step_ok).mean(axis=0)
    per_step_auc_approx = []
    for s in range(n_steps):
        # Approximate AUC: fraction of correct rankings
        pos_r = step_rewards[step_ok[:, s], s]
        neg_r = step_rewards[~step_ok[:, s], s]
        # Pairwise comparison
        auc = 0
        count = 0
        for pr in pos_r[:50]:
            for nr in neg_r[:50]:
                auc += (pr > nr) + 0.5 * (pr == nr)
                count += 1
        per_step_auc_approx.append(auc / max(count, 1))

    axes[0].bar(range(n_steps), per_step_acc, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Step Index')
    axes[0].set_ylabel('Binary Accuracy')
    axes[0].set_title('PRM: Per-Step Prediction Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(range(n_steps), per_step_auc_approx, color='darkorange', alpha=0.8)
    axes[1].set_xlabel('Step Index')
    axes[1].set_ylabel('Approx. AUC')
    axes[1].set_title('PRM: Per-Step Reward Quality (AUC)')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Process Reward Model: Step-Level Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'per_step_analysis.png', dpi=150)
    plt.close()

    # ── Plot 5: Conceptual comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    boxes = [
        ('Outcome RM', 0.17,
         'Score entire solution\n→ Single reward\n→ Cannot locate errors\n→ Misses partial progress', 'red'),
        ('Process RM', 0.5,
         'Score each step\n→ Per-step rewards\n→ Detects where errors occur\n→ Better credit assignment', 'blue'),
        ('Key Insight', 0.83,
         'Correct process\n→ Correct outcome\n(but not vice versa)\n→ PRM is more informative', 'green'),
    ]
    for name, xp, desc, color in boxes:
        ax.text(xp, 0.72, name, fontsize=14, fontweight='bold', ha='center', color=color)
        ax.text(xp, 0.35, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.85))

    ax.set_title('"Let\'s Verify Step by Step": Outcome RM vs Process RM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'conceptual_comparison.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
