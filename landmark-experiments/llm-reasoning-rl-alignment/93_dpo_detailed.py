"""
DPO with Detailed Analysis
===========================
Reproduces the core ideas from "Direct Preference Optimization:
Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023, 2305.18290):
1. DPO eliminates the reward model — closed-form optimal policy under KL constraint
2. Loss: L_DPO = -log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
3. Compare DPO vs simplified PPO on a toy preference learning task
4. Analyze: reward margin, KL divergence from reference, win rate over training
5. Study: effect of β (temperature) on optimization dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Policy Model ──

class PolicyNet(nn.Module):
    """Small autoregressive policy for token sequences."""
    def __init__(self, vocab_size=16, d_model=64, n_heads=2, n_layers=2, max_len=20):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, activation='gelu', batch_first=True,
            )
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

    def get_log_probs(self, sequences):
        """Sum of log-probs for each token given predecessors."""
        logits = self.forward(sequences[:, :-1])
        log_p = F.log_softmax(logits, dim=-1)
        tok_lp = log_p.gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        return tok_lp.sum(dim=1)

    def sample(self, prompt, max_new=12, temperature=1.0):
        x = prompt.clone()
        for _ in range(max_new):
            logits = self.forward(x)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
            if x.shape[1] >= self.max_len:
                break
        return x


# ── Reward Model (for PPO baseline) ──

class RewardNet(nn.Module):
    """Scalar reward for a sequence (mean-pool + MLP)."""
    def __init__(self, vocab_size=16, d_model=64, n_layers=2, max_len=20):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h.mean(dim=1))).squeeze(-1)


# ── Preference Environment ──

class PreferenceEnv:
    """Synthetic preference pairs on token sequences.

    Quality of a sequence = weighted sum of token values.
    Chosen sequences have higher quality than rejected ones.
    This gives us a controlled setting to measure win-rate, KL, etc.
    """
    def __init__(self, vocab_size=16, seq_len=10, device='cpu'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        # Tokens 0-1 are special; 2..vocab_size-1 carry increasing quality
        self.offset = 2
        # Weights per position so reward is non-trivial
        self.weights = torch.randn(seq_len).abs()
        self.weights /= self.weights.sum()

    def quality(self, seqs):
        """Weighted token-value sum — ground-truth reward."""
        vals = seqs.float() - self.offset  # higher token id → higher value
        return (vals * self.weights.to(seqs.device)).sum(dim=1)

    def generate_pair(self, batch_size):
        """Return (chosen, rejected) with chosen quality > rejected quality."""
        chosen = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        rejected = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            for j in range(self.seq_len):
                chosen[i, j] = torch.randint(self.offset + 4, self.vocab_size, (1,)).item()
                rejected[i, j] = torch.randint(self.offset, self.vocab_size - 4, (1,)).item()
        return chosen, rejected


# ── DPO Training ──

def train_dpo(policy, ref_policy, env, n_steps=1500, beta=0.1,
              batch_size=32, lr=1e-4, device='cpu'):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    metrics = {'loss': [], 'margin': [], 'kl': [], 'win': []}

    for step in range(n_steps):
        chosen, rejected = env.generate_pair(batch_size)

        log_pi_w = policy.get_log_probs(chosen)
        log_pi_l = policy.get_log_probs(rejected)
        with torch.no_grad():
            log_ref_w = ref_policy.get_log_probs(chosen)
            log_ref_l = ref_policy.get_log_probs(rejected)

        log_r_w = log_pi_w - log_ref_w
        log_r_l = log_pi_l - log_ref_l
        logits = beta * (log_r_w - log_r_l)
        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            margin = (log_r_w - log_r_l).mean().item()
            # KL ≈ E[log(π/π_ref)] on a held-out batch
            sample_seqs = torch.randint(env.offset, env.vocab_size,
                                        (batch_size, env.seq_len), device=env.device)
            kl = (policy.get_log_probs(sample_seqs) - ref_policy.get_log_probs(sample_seqs)).mean().item()
            # Win rate: fraction where implicit reward of chosen > rejected
            win = (logits > 0).float().mean().item()

        metrics['loss'].append(loss.item())
        metrics['margin'].append(margin)
        metrics['kl'].append(kl)
        metrics['win'].append(win)

        if (step + 1) % 500 == 0:
            print(f"  [DPO] Step {step+1} | Loss {loss.item():.4f} | "
                  f"Margin {margin:.4f} | KL {kl:.4f} | Win {win:.3f}")

    return metrics


# ── Simplified PPO Training ──

def train_ppo(policy, reward_model, ref_policy, env, n_steps=1500,
              kl_coeff=0.1, clip_eps=0.2, batch_size=32, lr=1e-4):
    # Phase 1: train reward model on preference data
    print("  [PPO] Training reward model ...")
    rm_opt = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)
    for _ in range(500):
        chosen, rejected = env.generate_pair(batch_size)
        r_w = reward_model(chosen)
        r_l = reward_model(rejected)
        loss = -F.logsigmoid(r_w - r_l).mean()
        rm_opt.zero_grad()
        loss.backward()
        rm_opt.step()

    # Phase 2: PPO-style policy optimisation
    print("  [PPO] Policy optimisation ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    metrics = {'loss': [], 'margin': [], 'kl': [], 'win': []}
    prompt = torch.full((batch_size, 1), 1, dtype=torch.long, device=env.device)

    for step in range(n_steps):
        # Sample sequences from current policy
        with torch.no_grad():
            seqs = policy.sample(prompt, max_new=env.seq_len - 1, temperature=1.0)
            seqs = seqs[:, :env.seq_len]
            rewards = reward_model(seqs)
            old_lp = policy.get_log_probs(seqs)
            ref_lp = ref_policy.get_log_probs(seqs)

        kl = old_lp - ref_lp
        advantages = rewards - kl_coeff * kl

        # PPO clipped update
        new_lp = policy.get_log_probs(seqs)
        ratio = torch.exp(new_lp - old_lp)
        clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        ppo_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

        optimizer.zero_grad()
        ppo_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            cur_lp = policy.get_log_probs(seqs)
            kl_val = (cur_lp - ref_lp).mean().item()
            # Win rate: use reward model to compare with random rejected
            _, rejected = env.generate_pair(batch_size)
            r_gen = reward_model(seqs)
            r_rej = reward_model(rejected)
            win = (r_gen > r_rej).float().mean().item()
            margin = (r_gen - r_rej).mean().item()

        metrics['loss'].append(ppo_loss.item())
        metrics['margin'].append(margin)
        metrics['kl'].append(kl_val)
        metrics['win'].append(win)

        if (step + 1) % 500 == 0:
            print(f"  [PPO] Step {step+1} | Loss {ppo_loss.item():.4f} | "
                  f"Margin {margin:.4f} | KL {kl_val:.4f} | Win {win:.3f}")

    return metrics


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "93-dpo-detailed"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 16
    seq_len = 10
    d_model = 64
    n_steps = 1500
    env = PreferenceEnv(vocab_size, seq_len, device=device)

    # ── Train DPO ──
    print("=== DPO ===")
    policy_dpo = PolicyNet(vocab_size, d_model, max_len=seq_len).to(device)
    ref_dpo = PolicyNet(vocab_size, d_model, max_len=seq_len).to(device)
    ref_dpo.eval()
    dpo_m = train_dpo(policy_dpo, ref_dpo, env, n_steps=n_steps, beta=0.1, device=device)

    # ── Train PPO ──
    print("\n=== PPO (simplified RLHF) ===")
    policy_ppo = PolicyNet(vocab_size, d_model, max_len=seq_len).to(device)
    ref_ppo = PolicyNet(vocab_size, d_model, max_len=seq_len).to(device)
    ref_ppo.load_state_dict(policy_ppo.state_dict())
    ref_ppo.eval()
    rm = RewardNet(vocab_size, d_model, max_len=seq_len).to(device)
    ppo_m = train_ppo(policy_ppo, rm, ref_ppo, env, n_steps=n_steps, device=device)

    # ── Plot 1: Four-metric comparison ──
    w = 30
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        ('loss', 'Training Loss'),
        ('margin', 'Reward Margin (chosen − rejected)'),
        ('kl', 'KL(π ‖ π_ref)'),
        ('win', 'Win Rate (chosen > rejected)'),
    ]
    for ax, (key, title) in zip(axes.flat, pairs):
        d = np.convolve(dpo_m[key], np.ones(w)/w, mode='valid')
        p = np.convolve(ppo_m[key], np.ones(w)/w, mode='valid')
        ax.plot(d, label='DPO', color='blue')
        ax.plot(p, label='PPO', color='red')
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if key == 'win':
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color='gray', ls='--', alpha=0.4)
        if key == 'kl':
            ax.axhline(0, color='gray', ls='--', alpha=0.4)

    plt.suptitle('DPO vs PPO: Detailed Comparison (β=0.1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'dpo_vs_ppo_four_metrics.png', dpi=150)
    plt.close()

    # ── Plot 2: Beta sensitivity — multiple β values ──
    print("\n=== β Sensitivity ===")
    betas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    beta_metrics = {}
    for beta in betas:
        policy = PolicyNet(vocab_size, d_model, max_len=seq_len).to(device)
        ref = PolicyNet(vocab_size, d_model, max_len=seq_len).to(device)
        ref.eval()
        m = train_dpo(policy, ref, env, n_steps=1000, beta=beta, device=device)
        beta_metrics[beta] = m
        print(f"  β={beta:<5} final win={np.mean(m['win'][-50:]):.3f} "
              f"KL={np.mean(m['kl'][-50:]):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))

    # Win rate vs step for each beta
    for beta, c in zip(betas, colors):
        s = np.convolve(beta_metrics[beta]['win'], np.ones(w)/w, mode='valid')
        axes[0].plot(s, label=f'β={beta}', color=c)
    axes[0].set_title('Win Rate vs Step')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Win Rate')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # KL divergence vs step
    for beta, c in zip(betas, colors):
        s = np.convolve(beta_metrics[beta]['kl'], np.ones(w)/w, mode='valid')
        axes[1].plot(s, label=f'β={beta}', color=c)
    axes[1].set_title('KL Divergence vs Step')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('KL(π ‖ π_ref)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Final metrics vs beta
    final_wins = [np.mean(beta_metrics[b]['win'][-50:]) for b in betas]
    final_kls = [np.mean(beta_metrics[b]['kl'][-50:]) for b in betas]
    final_margins = [np.mean(beta_metrics[b]['margin'][-50:]) for b in betas]

    ax2 = axes[2]
    ax2.plot(betas, final_wins, 'o-', label='Win Rate', color='blue')
    ax2.plot(betas, [m / max(abs(v) for v in final_margins) for m in final_margins],
             's--', label='Margin (norm)', color='orange', alpha=0.7)
    ax2.set_xlabel('β')
    ax2.set_xscale('log')
    ax2.set_title('Final Metrics vs β')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DPO: Effect of β (Temperature)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'beta_sensitivity.png', dpi=150)
    plt.close()

    # ── Plot 3: DPO loss landscape illustration ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # How the DPO loss changes with the log-ratio difference
    z = np.linspace(-3, 3, 200)
    for beta, c in zip([0.05, 0.1, 0.5, 1.0], ['green', 'blue', 'orange', 'red']):
        loss_curve = -np.log(1 / (1 + np.exp(-beta * z)))
        axes[0].plot(z, loss_curve, label=f'β={beta}', color=c)
    axes[0].set_xlabel('log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)')
    axes[0].set_ylabel('L_DPO')
    axes[0].set_title('DPO Loss vs Log-Ratio Difference')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Implicit reward r(x,y) = β log π(y|x)/π_ref(y|x) + const
    axes[1].barh(['RLHF (PPO)', 'DPO'], [4, 2], color=['red', 'blue'], alpha=0.7)
    axes[1].set_xlabel('Number of Models to Train')
    axes[1].set_title('Model Complexity: RLHF vs DPO')
    for i, v in enumerate([4, 2]):
        axes[1].text(v + 0.1, i, str(v), va='center', fontweight='bold')
    axes[1].set_xlim(0, 6)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('DPO: Loss Landscape & Efficiency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'dpo_loss_landscape.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
