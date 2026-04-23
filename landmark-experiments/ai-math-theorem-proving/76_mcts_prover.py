"""
Minimal MCTS-Based Proof Search Reproduction
=============================================
Reproduces the core ideas from "DeepSeek-Prover-V1.5: Harnessing
Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo
Tree Search" (2408.08152):
1. Monte Carlo Tree Search (MCTS) for proof exploration
2. Value network to evaluate proof states (how close to completion)
3. Policy network to propose candidate tactics at each state
4. UCB selection: balance exploitation vs exploration
5. Compare: greedy search vs beam search vs MCTS
6. Show success rate vs compute budget (number of simulations)

Synthetic domain: propositional logic theorems with AND, OR, IMPLIES,
NOT. Tactic language: `intro`, `apply`, `exact`, `split`, `left`,
`right`, `assumption`, `constructor`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import math


# ── Propositional Logic Proof System ──

# Token vocabulary for propositional logic
TOKENS = {
    'PAD': 0, 'BOS': 1, 'EOS': 2,
    # Variables
    'P': 3, 'Q': 4, 'R': 5, 'S': 6,
    # Connectives
    'AND': 7, 'OR': 8, 'IMPLIES': 9, 'NOT': 10,
    # Constants
    'TRUE': 11, 'FALSE': 12,
    # Delimiters
    '(': 13, ')': 14, ',': 15,
    # Tactics
    'intro': 16, 'apply': 17, 'exact': 18,
    'split': 19, 'left': 20, 'right': 21,
    'assumption': 22, 'constructor': 23, 'trivial': 24,
    # Hypotheses labels
    'h1': 25, 'h2': 26, 'h3': 27,
    # Goal markers
    'goal': 28, 'turnstile': 29,
}
VOCAB_SIZE = len(TOKENS)
INV_TOKENS = {v: k for k, v in TOKENS.items()}
MAX_SEQ_LEN = 24

TACTIC_IDS = [TOKENS[t] for t in
              ['intro', 'apply', 'exact', 'split', 'left', 'right',
               'assumption', 'constructor', 'trivial']]
N_TACTICS = len(TACTIC_IDS)


def tokenize_logic(text: str) -> list[int]:
    """Tokenize a propositional logic expression."""
    tokens = [TOKENS['BOS']]
    for word in text.split():
        if word in TOKENS:
            tokens.append(TOKENS[word])
    tokens.append(TOKENS['EOS'])
    tokens = tokens[:MAX_SEQ_LEN]
    tokens += [TOKENS['PAD']] * (MAX_SEQ_LEN - len(tokens))
    return tokens


def detokenize_logic(ids: list[int]) -> str:
    """Convert token IDs back to string."""
    parts = []
    for i in ids:
        if i in (TOKENS['PAD'], TOKENS['BOS']):
            continue
        if i == TOKENS['EOS']:
            break
        parts.append(INV_TOKENS.get(i, '?'))
    return ' '.join(parts)


@dataclass
class LogicTheorem:
    """A propositional logic theorem: hypotheses ⊢ goal."""
    hypotheses: list[str]   # e.g., ["P IMPLIES Q"]
    goal: str               # e.g., "P AND Q IMPLIES Q AND P"
    proof: list[int]        # sequence of tactic IDs
    difficulty: int = 1


@dataclass
class LogicProofState:
    """Current state in a proof attempt."""
    hypotheses: list[str]
    goal: str
    applied: list[int] = field(default_factory=list)
    solved: bool = False
    depth: int = 0


class LogicEnvironment:
    """Propositional logic proof checker.

    Tactics:
    - trivial:    closes trivially true goals (P IMPLIES P)
    - assumption: closes goal that matches a hypothesis
    - intro:      transforms (P IMPLIES Q) into hypothesis P, goal Q
    - split:      transforms goal (P AND Q) into subgoals P, Q
    - left:       transforms goal (P OR Q) into goal P
    - right:      transforms goal (P OR Q) into goal Q
    - constructor: same as split for AND goals
    - exact h:    closes goal matching hypothesis h
    - apply h:    applies hypothesis h (modus ponens style)
    """
    def __init__(self):
        self.theorems: list[LogicTheorem] = []
        self._build_theorem_database()

    def _build_theorem_database(self):
        # Difficulty 1: single-step proofs
        self.theorems += [
            LogicTheorem(["P"], "P", [TOKENS['assumption']], 1),
            LogicTheorem(["Q"], "Q", [TOKENS['assumption']], 1),
            LogicTheorem([], "P IMPLIES P", [TOKENS['intro']], 1),
            LogicTheorem([], "Q IMPLIES Q", [TOKENS['intro']], 1),
            LogicTheorem(["P", "Q"], "P", [TOKENS['assumption']], 1),
            LogicTheorem(["P"], "P AND P IMPLIES P", [TOKENS['assumption']], 1),
            LogicTheorem(["P"], "NOT NOT P IMPLIES P", [TOKENS['assumption']], 1),
        ]
        # Difficulty 2: two-step proofs
        self.theorems += [
            LogicTheorem([], "P AND Q IMPLIES P",
                         [TOKENS['intro'], TOKENS['left']], 2),
            LogicTheorem([], "P AND Q IMPLIES Q",
                         [TOKENS['intro'], TOKENS['right']], 2),
            LogicTheorem(["P IMPLIES Q", "P"], "Q",
                         [TOKENS['apply']], 2),
            LogicTheorem([], "P IMPLIES Q IMPLIES P",
                         [TOKENS['intro'], TOKENS['intro']], 2),
            LogicTheorem(["P", "Q"], "P AND Q",
                         [TOKENS['split']], 2),
        ]
        # Difficulty 3: three-step proofs
        self.theorems += [
            LogicTheorem([], "P AND Q IMPLIES Q AND P",
                         [TOKENS['intro'], TOKENS['split'], TOKENS['right']], 3),
            LogicTheorem(["P IMPLIES Q", "Q IMPLIES R", "P"], "R",
                         [TOKENS['apply'], TOKENS['apply']], 3),
            LogicTheorem([], "P IMPLIES P AND P",
                         [TOKENS['intro'], TOKENS['split']], 3),
            LogicTheorem(["P IMPLIES Q"], "NOT Q IMPLIES NOT P",
                         [TOKENS['intro'], TOKENS['apply']], 3),
        ]

    def apply_tactic(self, state: LogicProofState, tactic_id: int) -> LogicProofState:
        """Apply a tactic to a proof state.

        Simplified semantics: check if tactic matches the expected proof step.
        If correct, advance; if not, mark state as stuck.
        """
        # Find matching theorem
        for thm in self.theorems:
            if (thm.hypotheses == state.hypotheses and
                    thm.goal == state.goal and
                    not state.solved):
                step = len(state.applied)
                if step < len(thm.proof):
                    if thm.proof[step] == tactic_id:
                        new_applied = state.applied + [tactic_id]
                        solved = (step + 1 >= len(thm.proof))
                        return LogicProofState(
                            hypotheses=state.hypotheses,
                            goal=state.goal,
                            applied=new_applied,
                            solved=solved,
                            depth=state.depth + 1,
                        )
                break

        # Tactic doesn't help
        return LogicProofState(
            hypotheses=state.hypotheses,
            goal=state.goal,
            applied=state.applied + [tactic_id],
            solved=False,
            depth=state.depth + 1,
        )

    def state_to_tokens(self, state: LogicProofState) -> list[int]:
        """Encode a proof state as token sequence."""
        text = "goal " + state.goal
        for h in state.hypotheses:
            text += " " + h
        return tokenize_logic(text)


# ── Neural Networks ──

class StateEncoder(nn.Module):
    """Encode proof state into a dense vector."""
    def __init__(self, vocab_size, d_model=64, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        h = h.mean(dim=1)
        return self.proj(h)


class PolicyNetwork(nn.Module):
    """Predict tactic distribution given a proof state."""
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.encoder = StateEncoder(vocab_size, d_model)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, N_TACTICS),
        )

    def forward(self, state_tokens):
        h = self.encoder(state_tokens)
        return F.log_softmax(self.policy_head(h), dim=-1)


class ValueNetwork(nn.Module):
    """Predict value (probability of successful proof) given a proof state."""
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.encoder = StateEncoder(vocab_size, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_tokens):
        h = self.encoder(state_tokens)
        return self.value_head(h).squeeze(-1)


# ── MCTS ──

class MCTSNode:
    """A node in the MCTS search tree."""
    def __init__(self, state: LogicProofState, parent=None, tactic_id=None):
        self.state = state
        self.parent = parent
        self.tactic_id = tactic_id  # tactic that led to this node
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.total_value = 0.0
        self.prior = 0.0  # policy prior probability

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, c_puct=1.5):
        """Upper confidence bound score for selection."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.q_value
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return exploitation + exploration

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state.solved or self.state.depth >= 8


class MCTS:
    """Monte Carlo Tree Search for proof exploration.

    Inspired by DeepSeek-Prover-V1.5:
    1. Select: traverse tree using UCB until reaching a leaf
    2. Expand: use policy network to propose tactics, create child nodes
    3. Evaluate: use value network to estimate leaf value
    4. Backpropagate: update visit counts and values up the tree
    """
    def __init__(self, policy_net, value_net, env, c_puct=1.5, device='cpu'):
        self.policy_net = policy_net
        self.value_net = value_net
        self.env = env
        self.c_puct = c_puct
        self.device = device

    def search(self, initial_state, n_simulations=50):
        """Run MCTS from an initial proof state."""
        root = MCTSNode(initial_state)

        for _ in range(n_simulations):
            node = root

            # 1. Selection: traverse tree using UCB
            while not node.is_leaf() and not node.is_terminal():
                node = max(node.children, key=lambda c: c.ucb_score(self.c_puct))

            # 2. Check if terminal
            if node.is_terminal():
                value = 1.0 if node.state.solved else 0.0
            else:
                # 3. Expansion: use policy network to expand
                state_tokens = torch.tensor(
                    [self.env.state_to_tokens(node.state)],
                    dtype=torch.long, device=self.device
                )

                with torch.no_grad():
                    log_policy = self.policy_net(state_tokens)  # (1, N_TACTICS)
                    policy = torch.exp(log_policy).squeeze(0).cpu().numpy()

                # Create children for each tactic
                for i, tactic_id in enumerate(TACTIC_IDS):
                    new_state = self.env.apply_tactic(node.state, tactic_id)
                    child = MCTSNode(new_state, parent=node, tactic_id=tactic_id)
                    child.prior = policy[i]
                    node.children.append(child)

                # 4. Evaluation: use value network
                with torch.no_grad():
                    value = self.value_net(state_tokens).item()

            # 5. Backpropagate
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

        return root

    def get_best_tactic(self, root):
        """Select the best tactic from the root node (most visited)."""
        if not root.children:
            return TACTIC_IDS[0]  # fallback
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.tactic_id


# ── Search Strategies ──

def greedy_search(policy_net, env, theorem, max_depth=8, device='cpu'):
    """Greedy: always pick the highest-probability tactic."""
    state = LogicProofState(
        hypotheses=theorem.hypotheses, goal=theorem.goal
    )

    for _ in range(max_depth):
        if state.solved:
            return True, len(state.applied)
        state_tokens = torch.tensor(
            [env.state_to_tokens(state)], dtype=torch.long, device=device
        )
        with torch.no_grad():
            log_probs = policy_net(state_tokens)
            tactic_id = TACTIC_IDS[log_probs.argmax(dim=-1).item()]
        state = env.apply_tactic(state, tactic_id)

    return state.solved, len(state.applied)


def beam_search(policy_net, env, theorem, beam_width=3, max_depth=8, device='cpu'):
    """Beam search: keep top-k partial proofs."""
    initial = LogicProofState(
        hypotheses=theorem.hypotheses, goal=theorem.goal
    )
    beams = [(0.0, initial)]  # (log_prob, state)

    for _ in range(max_depth):
        new_beams = []
        for log_prob, state in beams:
            if state.solved:
                new_beams.append((log_prob, state))
                continue

            state_tokens = torch.tensor(
                [env.state_to_tokens(state)], dtype=torch.long, device=device
            )
            with torch.no_grad():
                log_probs = policy_net(state_tokens).squeeze(0)  # (N_TACTICS,)

            # Top-k tactics
            topk = torch.topk(log_probs, min(beam_width, N_TACTICS))
            for idx, prob in zip(topk.indices, topk.values):
                tactic_id = TACTIC_IDS[idx.item()]
                new_state = env.apply_tactic(state, tactic_id)
                new_beams.append((log_prob + prob.item(), new_state))

        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

        # Check if any beam is solved
        if any(state.solved for _, state in beams):
            for log_p, state in beams:
                if state.solved:
                    return True, len(state.applied)

    return False, max_depth


def mcts_search(mcts, env, theorem, n_simulations=50, max_depth=8, device='cpu'):
    """MCTS search: use tree search with policy + value networks."""
    initial = LogicProofState(
        hypotheses=theorem.hypotheses, goal=theorem.goal
    )

    for _ in range(max_depth):
        if initial.solved:
            return True, len(initial.applied)
        root = mcts.search(initial, n_simulations=n_simulations)
        best_tactic = mcts.get_best_tactic(root)
        initial = env.apply_tactic(initial, best_tactic)

    return initial.solved, len(initial.applied)


# ── Training ──

def generate_training_data(env, n_samples, device='cpu'):
    """Generate (state, tactic, value) training data from theorem proofs."""
    state_tokens = []
    tactic_labels = []
    value_labels = []

    for _ in range(n_samples):
        thm = env.theorems[np.random.randint(len(env.theorems))]
        for step, tactic_id in enumerate(thm.proof):
            state = LogicProofState(
                hypotheses=thm.hypotheses, goal=thm.goal,
                applied=thm.proof[:step], depth=step,
            )
            s_tokens = env.state_to_tokens(state)
            state_tokens.append(s_tokens)
            tactic_labels.append(TACTIC_IDS.index(tactic_id))
            # Value: 1.0 if remaining steps are small, decaying otherwise
            remaining = len(thm.proof) - step
            value_labels.append(1.0 / remaining)

    state_t = torch.tensor(state_tokens, dtype=torch.long, device=device)
    tactic_t = torch.tensor(tactic_labels, dtype=torch.long, device=device)
    value_t = torch.tensor(value_labels, dtype=torch.float32, device=device)
    return state_t, tactic_t, value_t


def train_networks(policy_net, value_net, env, n_steps=2000,
                   batch_size=64, lr=1e-3, device='cpu'):
    """Train policy and value networks with supervised learning."""
    optimizer = torch.optim.AdamW(
        list(policy_net.parameters()) + list(value_net.parameters()), lr=lr
    )
    losses = []
    policy_accs = []

    for step in range(n_steps):
        state_t, tactic_t, value_t = generate_training_data(env, batch_size, device)

        # Policy loss
        log_policy = policy_net(state_t)  # (B, N_TACTICS)
        policy_loss = F.nll_loss(log_policy, tactic_t)

        # Value loss
        pred_value = value_net(state_t)  # (B,)
        value_loss = F.mse_loss(pred_value, value_t)

        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(policy_net.parameters()) + list(value_net.parameters()), 1.0
        )
        optimizer.step()

        losses.append(loss.item())

        # Policy accuracy
        preds = log_policy.argmax(dim=-1)
        acc = (preds == tactic_t).float().mean().item()
        policy_accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | "
                  f"Policy Acc: {acc:.3f}")

    return losses, policy_accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "76-mcts-prover"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = LogicEnvironment()
    print(f"Theorem database: {len(env.theorems)} theorems")
    for d in [1, 2, 3]:
        n = sum(1 for t in env.theorems if t.difficulty == d)
        print(f"  Difficulty {d}: {n} theorems")

    d_model = 64
    n_steps = 2000

    # 1. Train policy and value networks
    print("\n=== Training Policy & Value Networks ===")
    policy_net = PolicyNetwork(VOCAB_SIZE, d_model).to(device)
    value_net = ValueNetwork(VOCAB_SIZE, d_model).to(device)
    losses, accs = train_networks(
        policy_net, value_net, env, n_steps=n_steps, device=device
    )

    # 2. Create MCTS
    mcts = MCTS(policy_net, value_net, env, c_puct=1.5, device=device)

    # 3. Compare search strategies at different compute budgets
    print("\n=== Comparing Search Strategies ===")
    sim_counts = [10, 25, 50, 100]

    greedy_results = []
    beam_results = []
    mcts_results = {n: [] for n in sim_counts}

    n_eval_runs = 3  # average over multiple runs

    for run in range(n_eval_runs):
        print(f"\n  Run {run+1}/{n_eval_runs}")

        # Greedy
        greedy_success = 0
        greedy_steps = []
        for thm in env.theorems:
            success, steps = greedy_search(policy_net, env, thm, device=device)
            greedy_success += int(success)
            if success:
                greedy_steps.append(steps)
        greedy_results.append({
            'rate': greedy_success / len(env.theorems),
            'steps': greedy_steps,
        })
        print(f"    Greedy: {greedy_success}/{len(env.theorems)} "
              f"({greedy_success/len(env.theorems):.1%})")

        # Beam search (width=3)
        beam_success = 0
        beam_steps = []
        for thm in env.theorems:
            success, steps = beam_search(
                policy_net, env, thm, beam_width=3, device=device
            )
            beam_success += int(success)
            if success:
                beam_steps.append(steps)
        beam_results.append({
            'rate': beam_success / len(env.theorems),
            'steps': beam_steps,
        })
        print(f"    Beam (w=3): {beam_success}/{len(env.theorems)} "
              f"({beam_success/len(env.theorems):.1%})")

        # MCTS with different simulation counts
        for n_sim in sim_counts:
            mcts_success = 0
            mcts_steps = []
            for thm in env.theorems:
                success, steps = mcts_search(
                    mcts, env, thm, n_simulations=n_sim, device=device
                )
                mcts_success += int(success)
                if success:
                    mcts_steps.append(steps)
            mcts_results[n_sim].append({
                'rate': mcts_success / len(env.theorems),
                'steps': mcts_steps,
            })
            print(f"    MCTS (sim={n_sim}): {mcts_success}/{len(env.theorems)} "
                  f"({mcts_success/len(env.theorems):.1%})")

    # Average results
    greedy_rate = np.mean([r['rate'] for r in greedy_results])
    beam_rate = np.mean([r['rate'] for r in beam_results])
    mcts_rates = {n: np.mean([r['rate'] for r in mcts_results[n]])
                  for n in sim_counts}

    print(f"\n  Average results:")
    print(f"    Greedy:       {greedy_rate:.1%}")
    print(f"    Beam (w=3):   {beam_rate:.1%}")
    for n in sim_counts:
        print(f"    MCTS (sim={n}): {mcts_rates[n]:.1%}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 20
    loss_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    acc_smooth = np.convolve(accs, np.ones(window)/window, mode='valid')

    axes[0].plot(loss_smooth, color='blue')
    axes[0].set_title("Training Loss (Policy + Value)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(acc_smooth, color='green')
    axes[1].set_title("Policy Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy (smoothed)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("MCTS Prover: Policy & Value Network Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Success rate vs compute budget
    fig, ax = plt.subplots(figsize=(9, 6))

    # Compute budgets (total tactic evaluations)
    greedy_budget = 1
    beam_budget = 3
    mcts_budgets = sim_counts

    methods = ['Greedy', 'Beam\n(w=3)'] + [f'MCTS\n(sim={n})' for n in mcts_budgets]
    rates = [greedy_rate, beam_rate] + [mcts_rates[n] for n in mcts_budgets]
    budgets = [greedy_budget, beam_budget] + mcts_budgets
    colors = ['red', 'orange'] + ['blue'] * len(mcts_budgets)

    bars = ax.bar(methods, rates, color=colors, alpha=0.7, edgecolor='black')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02,
                f'{rate:.1%}', ha='center', fontweight='bold', fontsize=10)

    ax.set_ylabel("Proof Success Rate")
    ax.set_title("Search Strategy Comparison: Success Rate vs Compute Budget")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / "success_vs_compute.png", dpi=150)
    plt.close()

    # 3. MCTS scaling curve
    fig, ax = plt.subplots(figsize=(8, 5))
    all_budgets = [greedy_budget, beam_budget] + mcts_budgets
    all_rates = [greedy_rate, beam_rate] + [mcts_rates[n] for n in mcts_budgets]
    all_labels = ['Greedy', 'Beam (w=3)'] + ['MCTS'] * len(mcts_budgets)

    ax.scatter([greedy_budget], [greedy_rate], color='red', s=100,
               zorder=5, label='Greedy')
    ax.scatter([beam_budget], [beam_rate], color='orange', s=100,
               zorder=5, label='Beam (w=3)')
    ax.plot(mcts_budgets, [mcts_rates[n] for n in mcts_budgets],
            'bo-', markersize=8, label='MCTS')

    ax.set_xlabel("Compute Budget (tactic evaluations)")
    ax.set_ylabel("Proof Success Rate")
    ax.set_title("MCTS: Scaling Success Rate with Compute")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(results_dir / "mcts_scaling.png", dpi=150)
    plt.close()

    # 4. Value prediction visualization
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sample some states and show value predictions
    sample_values = []
    true_values = []
    labels = []

    for thm in env.theorems[:8]:
        state = LogicProofState(
            hypotheses=thm.hypotheses, goal=thm.goal
        )
        state_tokens = torch.tensor(
            [env.state_to_tokens(state)], dtype=torch.long, device=device
        )
        with torch.no_grad():
            pred_v = value_net(state_tokens).item()

        # True value: inverse of proof length
        true_v = 1.0 / max(len(thm.proof), 1)

        sample_values.append(pred_v)
        true_values.append(true_v)
        labels.append(f"d={thm.difficulty}")

    x = np.arange(len(sample_values))
    width = 0.35
    ax.bar(x - width/2, true_values, width, label='True Value (1/steps)',
           color='green', alpha=0.7)
    ax.bar(x + width/2, sample_values, width, label='Predicted Value',
           color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Value Network: Predicted vs True Proof State Values")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / "value_predictions.png", dpi=150)
    plt.close()

    # 5. MCTS algorithm diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    phases = [
        ("1. SELECT",
         "Traverse tree using\nUCB score:\nQ(s,a) + c*p*sqrt(N_parent)/(1+N)",
         0.1, 'purple'),
        ("2. EXPAND",
         "Use policy network\nto propose tactics:\npi(a|s) from PolicyNet",
         0.3, 'orange'),
        ("3. EVALUATE",
         "Use value network\nto estimate leaf:\nv(s) from ValueNet",
         0.5, 'teal'),
        ("4. BACKPROP",
         "Update visit counts\nand values up tree:\nN += 1, W += v",
         0.7, 'green'),
        ("5. SELECT BEST",
         "Pick most-visited\naction from root:\nargmax_a N(s,a)",
         0.9, 'brown'),
    ]

    for name, desc, x_pos, color in phases:
        ax.text(x_pos, 0.75, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.2, 0.4, 0.6, 0.8]:
        ax.annotate('\u2192', xy=(x, 0.52), fontsize=20, ha='center',
                    va='center', color='gray')

    # Loop arrow from step 5 back to 1
    ax.annotate('', xy=(0.15, 0.9), xytext=(0.85, 0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(0.5, 0.98, 'repeat for n_simulations', fontsize=9,
            ha='center', va='center', color='red', fontstyle='italic')

    ax.set_title("MCTS for Theorem Proving (DeepSeek-Prover-V1.5 Style)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "mcts_algorithm.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
