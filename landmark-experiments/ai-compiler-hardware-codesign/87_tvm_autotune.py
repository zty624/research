"""
TVM-style Auto-tuning for Compilation Schedules
=================================================
Reproduces core ideas from TVM (1802.04799, Chen et al.) and Ansor (2006.06762, Zheng et al.):
1. Search over compilation schedules (tile sizes, loop order, unrolling)
2. Cost model: predict execution time from schedule features
3. Bayesian optimization (GP-based) for efficient schedule search
4. Synthetic: simulated cost model for matmul with different tile configs
5. Show: search trajectory, best schedule found, cost model accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Schedule Representation ──

@dataclass
class ScheduleConfig:
    """A compilation schedule for matmul C[M,N] = A[M,K] x B[K,N].
    Parameters: tile sizes for each loop dimension and unroll flag.
    """
    tile_m: int
    tile_n: int
    tile_k: int
    loop_order: int  # 0=mnk, 1=mkn, 2=nmk, 3=nkm, 4=kmn, 5=knm
    unroll: bool

    def to_features(self) -> np.ndarray:
        """Extract features for the cost model."""
        return np.array([
            self.tile_m / 128.0,
            self.tile_n / 128.0,
            self.tile_k / 64.0,
            self.loop_order / 5.0,
            float(self.unroll),
            # Interaction features
            (self.tile_m * self.tile_n) / (128.0 * 128.0),
            (self.tile_m * self.tile_k) / (128.0 * 64.0),
            (self.tile_n * self.tile_k) / (128.0 * 64.0),
            # Log features
            np.log2(self.tile_m) / 7.0,
            np.log2(self.tile_n) / 7.0,
            np.log2(self.tile_k) / 6.0,
        ], dtype=np.float32)

    def __repr__(self):
        orders = ['mnk', 'mkn', 'nmk', 'nkm', 'kmn', 'knm']
        unroll_s = 'T' if self.unroll else 'F'
        return f"Schedule(tm={self.tile_m},tn={self.tile_n},tk={self.tile_k},ord={orders[self.loop_order]},unroll={unroll_s})"


# ── Simulated Cost Model (ground truth) ──

class SimulatedCostModel:
    """Simulates execution time for a matmul schedule.
    Models real hardware effects: cache locality, unrolling overhead, loop order.
    """
    def __init__(self, M=512, N=512, K=512, seed=42):
        self.M, self.N, self.K = M, N, K
        rng = np.random.RandomState(seed)
        # Simulate cache line size and L1/L2 sizes
        self.cache_line = 64  # bytes
        self.l1_size = 32768  # 32KB
        self.l2_size = 262144  # 256KB
        # Add some fixed noise
        self.noise_scale = 0.03

    def evaluate(self, config: ScheduleConfig) -> float:
        """Return simulated execution time in ms."""
        tm, tn, tk = config.tile_m, config.tile_n, config.tile_k
        M, N, K = self.M, self.N, self.K

        # Base compute time (proportional to FLOPs)
        base_flops = 2.0 * M * N * K  # multiply-add = 2 FLOPs
        # Assume 100 GFLOP/s peak, base time in ms
        base_time = base_flops / 100e9 * 1000  # ms

        # Cache efficiency: working set must fit in L1 for best performance
        # Working set for one tile: tm*tk + tk*tn + tm*tn (in floats, 4 bytes each)
        working_set_bytes = (tm * tk + tk * tn + tm * tn) * 4

        if working_set_bytes <= self.l1_size:
            cache_factor = 0.6  # best: L1 resident
        elif working_set_bytes <= self.l2_size:
            cache_factor = 0.85  # L2 resident
        else:
            cache_factor = 1.0 + 0.3 * (working_set_bytes / self.l2_size - 1)  # spill

        # Tile alignment: tiles should evenly divide dimensions
        alignment_penalty = 0.0
        if M % tm != 0:
            alignment_penalty += 0.05
        if N % tn != 0:
            alignment_penalty += 0.05
        if K % tk != 0:
            alignment_penalty += 0.03

        # Loop order: some orders are better for cache locality
        # kmn and knm are generally good (iterate k innermost for register reuse)
        order_scores = {
            0: 1.0,   # mnk: baseline
            1: 0.95,  # mkn: slightly better (k inner)
            2: 1.05,  # nmk: similar
            3: 1.0,   # nkm
            4: 0.88,  # kmn: good for register blocking
            5: 0.85,  # knm: best for register reuse
        }
        order_factor = order_scores[config.loop_order]

        # Unrolling: helps for small tiles, hurts for large (code bloat)
        if config.unroll:
            tile_product = tm * tn * tk
            if tile_product <= 64:
                unroll_factor = 0.85  # good: small unrolled loops
            elif tile_product <= 256:
                unroll_factor = 0.92
            else:
                unroll_factor = 1.05 + 0.1 * (tile_product / 1024)  # code bloat
        else:
            unroll_factor = 1.0

        # Tile size sweet spot: too small = overhead, too large = cache miss
        # Power-of-2 tiles are generally better
        def tile_score(t):
            s = 0.0
            if t in [8, 16, 32, 64, 128]:
                s -= 0.03  # bonus for power of 2
            if t < 4:
                s += 0.15  # too small: loop overhead
            elif t > 128:
                s += 0.1  # too large
            return s

        tile_penalty = tile_score(tm) + tile_score(tn) + tile_score(tk)

        # Final time
        time = base_time * cache_factor * order_factor * unroll_factor
        time *= (1.0 + alignment_penalty + tile_penalty)

        # Add small noise
        noise = np.random.randn() * self.noise_scale * base_time
        return max(time + noise, base_time * 0.5)  # floor at 50% of base


# ── Learned Cost Model ──

class CostModel(nn.Module):
    """Neural network that predicts execution time from schedule features."""
    def __init__(self, n_features=11, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_cost_model(model, X_train, y_train, n_epochs=200, lr=1e-3):
    """Train the cost model on observed (schedule, time) pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    losses = []

    for epoch in range(n_epochs):
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ── Bayesian Optimization (GP-based) ──

class GaussianProcess:
    """Simple GP with RBF kernel for Bayesian optimization."""
    def __init__(self, length_scale=0.3, noise=1e-4):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None

    def rbf_kernel(self, X1, X2):
        """RBF kernel between X1 (n1, d) and X2 (n2, d)."""
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sq_dist / self.length_scale**2)

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()

    def predict(self, X):
        """Return mean and std predictions."""
        K = self.rbf_kernel(self.X_train, self.X_train)
        K_s = self.rbf_kernel(X, self.X_train)
        K_ss = self.rbf_kernel(X, X)

        # Add noise to diagonal for numerical stability
        K += self.noise * np.eye(len(K))
        K += 1e-6 * np.eye(len(K))  # jitter

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            mean = K_s @ alpha

            v = np.linalg.solve(L, K_s.T)
            var = np.diag(K_ss - v.T @ v)
            std = np.sqrt(np.maximum(var, 1e-8))
        except np.linalg.LinAlgError:
            # Fallback: use training mean
            mean = np.full(len(X), np.mean(self.y_train))
            std = np.full(len(X), np.std(self.y_train) + 1.0)

        return mean, std

    def acquisition_ei(self, X, best_y):
        """Expected Improvement acquisition function."""
        mean, std = self.predict(X)
        # EI = (mean - best_y) * Phi(z) + std * phi(z)
        # where z = (mean - best_y) / std
        z = (mean - best_y) / (std + 1e-8)
        # Use negative because we minimize cost
        improvement = best_y - mean
        from scipy.stats import norm
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei[std < 1e-6] = 0.0  # no uncertainty, no exploration value
        return ei


# ── Schedule Search Space ──

def generate_search_space():
    """Generate the full discrete search space of schedule configs."""
    tile_sizes = [4, 8, 16, 32, 64, 128]
    loop_orders = list(range(6))
    unroll_options = [False, True]

    configs = []
    for tm in tile_sizes:
        for tn in tile_sizes:
            for tk in [4, 8, 16, 32, 64]:
                for lo in loop_orders:
                    for unroll in unroll_options:
                        configs.append(ScheduleConfig(tm, tn, tk, lo, unroll))

    return configs


def random_search(cost_fn, configs, n_trials, seed=42):
    """Random search baseline."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(configs), size=min(n_trials, len(configs)), replace=False)
    results = []
    best_time = float('inf')
    best_config = None
    trajectory = []

    for i, idx in enumerate(indices):
        cfg = configs[idx]
        t = cost_fn(cfg)
        results.append((cfg, t))
        if t < best_time:
            best_time = t
            best_config = cfg
        trajectory.append(best_time)

    return results, best_config, best_time, trajectory


def bayesian_search(cost_fn, configs, n_trials, seed=42):
    """GP-based Bayesian optimization search."""
    rng = np.random.RandomState(seed)
    n_features = len(configs[0].to_features())

    # Initial random samples to bootstrap GP
    n_init = min(10, n_trials // 3)
    init_indices = rng.choice(len(configs), size=n_init, replace=False)

    observed_X = []
    observed_y = []
    observed_configs = []
    best_time = float('inf')
    best_config = None
    trajectory = []

    # Initial random evaluations
    for idx in init_indices:
        cfg = configs[idx]
        t = cost_fn(cfg)
        observed_X.append(cfg.to_features())
        observed_y.append(t)
        observed_configs.append(cfg)
        if t < best_time:
            best_time = t
            best_config = cfg
        trajectory.append(best_time)

    # GP-guided search
    gp = GaussianProcess(length_scale=0.3, noise=1e-3)
    remaining = n_trials - n_init

    # Precompute all candidate features for acquisition
    all_features = np.array([c.to_features() for c in configs])

    for i in range(remaining):
        # Fit GP on observed data
        X_arr = np.array(observed_X)
        # Normalize y for GP stability
        y_mean = np.mean(observed_y)
        y_std = np.std(observed_y) + 1e-6
        y_arr = (np.array(observed_y) - y_mean) / y_std

        gp.fit(X_arr, y_arr)

        # Compute acquisition function over all configs
        ei = gp.acquisition_ei(all_features, (best_time - y_mean) / y_std)

        # Exclude already-observed configs
        observed_set = set(id(c) for c in observed_configs)
        for j, c in enumerate(configs):
            if id(c) in observed_set:
                ei[j] = -1

        # Select best acquisition
        next_idx = np.argmax(ei)
        if ei[next_idx] < 0:
            # All observed, pick random unobserved
            remaining_indices = [j for j in range(len(configs)) if id(configs[j]) not in observed_set]
            if not remaining_indices:
                break
            next_idx = rng.choice(remaining_indices)

        cfg = configs[next_idx]
        t = cost_fn(cfg)
        observed_X.append(cfg.to_features())
        observed_y.append(t)
        observed_configs.append(cfg)

        if t < best_time:
            best_time = t
            best_config = cfg
        trajectory.append(best_time)

    return list(zip(observed_configs, observed_y)), best_config, best_time, trajectory


def exhaustive_search(cost_fn, configs):
    """Evaluate all configs (ground truth best)."""
    best_time = float('inf')
    best_config = None
    all_times = []

    for cfg in configs:
        t = cost_fn(cfg)
        all_times.append(t)
        if t < best_time:
            best_time = t
            best_config = cfg

    return best_config, best_time, all_times


# ── Main ──

def main():
    results_dir = Path(__file__).parent / "results" / "87-tvm-autotune"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Problem size
    M, N, K = 512, 512, 512
    print(f"=== TVM-style Auto-tuning for Matmul [{M}x{K}] x [{K}x{N}] ===\n")

    cost_fn = SimulatedCostModel(M=M, N=N, K=K)
    configs = generate_search_space()
    print(f"Search space size: {len(configs)} schedules\n")

    # 1. Exhaustive search (ground truth)
    print("=== Exhaustive Search (Ground Truth) ===")
    best_exhaust, best_time_exhaust, all_times = exhaustive_search(cost_fn.evaluate, configs)
    sorted_times = sorted(all_times)
    print(f"  Best schedule: {best_exhaust}")
    print(f"  Best time:     {best_time_exhaust:.3f} ms")
    print(f"  Worst time:    {sorted_times[-1]:.3f} ms")
    print(f"  Median time:   {sorted_times[len(sorted_times)//2]:.3f} ms\n")

    # 2. Random search
    n_trials = 80
    print(f"=== Random Search ({n_trials} trials) ===")
    rand_results, rand_best, rand_time, rand_traj = random_search(
        cost_fn.evaluate, configs, n_trials, seed=42
    )
    print(f"  Best schedule: {rand_best}")
    print(f"  Best time:     {rand_time:.3f} ms")
    print(f"  Gap from opt:  {(rand_time / best_time_exhaust - 1) * 100:.1f}%\n")

    # 3. Bayesian optimization search
    print(f"=== Bayesian Optimization Search ({n_trials} trials) ===")
    bo_results, bo_best, bo_time, bo_traj = bayesian_search(
        cost_fn.evaluate, configs, n_trials, seed=42
    )
    print(f"  Best schedule: {bo_best}")
    print(f"  Best time:     {bo_time:.3f} ms")
    print(f"  Gap from opt:  {(bo_time / best_time_exhaust - 1) * 100:.1f}%\n")

    # 4. Cost model training and evaluation
    print("=== Cost Model Training ===")
    # Use all BO observations as training data
    X_train = np.array([cfg.to_features() for cfg, _ in bo_results])
    y_train = np.array([t for _, t in bo_results])

    # Also evaluate on a held-out set
    n_eval = 200
    eval_indices = np.random.choice(len(configs), size=n_eval, replace=False)
    X_eval = np.array([configs[i].to_features() for i in eval_indices])
    y_eval = np.array([cost_fn.evaluate(configs[i]) for i in eval_indices])

    cost_model = CostModel(n_features=11, hidden=64)
    train_losses = train_cost_model(cost_model, X_train, y_train, n_epochs=300, lr=1e-3)

    cost_model.eval()
    with torch.no_grad():
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
        y_pred = cost_model(X_eval_t).numpy()

    mae = np.mean(np.abs(y_pred - y_eval))
    rmse = np.sqrt(np.mean((y_pred - y_eval)**2))
    rel_err = np.mean(np.abs(y_pred - y_eval) / y_eval) * 100
    print(f"  MAE:          {mae:.3f} ms")
    print(f"  RMSE:         {rmse:.3f} ms")
    print(f"  Relative err: {rel_err:.1f}%\n")

    # ── Visualization ──

    # 1. Search trajectory comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(rand_traj) + 1), rand_traj, 'o-', label='Random Search',
            color='gray', alpha=0.7, markersize=3)
    ax.plot(range(1, len(bo_traj) + 1), bo_traj, 's-', label='Bayesian Optimization',
            color='blue', alpha=0.7, markersize=3)
    ax.axhline(y=best_time_exhaust, color='red', linestyle='--', label=f'Optimal ({best_time_exhaust:.2f} ms)')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Time Found (ms)")
    ax.set_title("Auto-tuning: Search Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "search_trajectory.png", dpi=150)
    plt.close()

    # 2. Cost model accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].plot(train_losses, color='blue')
    axes[0].set_title("Cost Model Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True, alpha=0.3)

    # Predicted vs actual
    axes[1].scatter(y_eval, y_pred, alpha=0.4, s=20, color='blue')
    min_t, max_t = y_eval.min(), y_eval.max()
    axes[1].plot([min_t, max_t], [min_t, max_t], 'r--', label='Perfect')
    axes[1].set_xlabel("Actual Time (ms)")
    axes[1].set_ylabel("Predicted Time (ms)")
    axes[1].set_title(f"Cost Model: Predicted vs Actual (MAE={mae:.2f} ms)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("TVM-style Cost Model for Schedule Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "cost_model_accuracy.png", dpi=150)
    plt.close()

    # 3. Schedule space analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution of times across all schedules
    axes[0, 0].hist(all_times, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=best_time_exhaust, color='red', linestyle='--', label=f'Optimal={best_time_exhaust:.2f}')
    axes[0, 0].axvline(x=bo_time, color='blue', linestyle='--', label=f'BO={bo_time:.2f}')
    axes[0, 0].set_title("Distribution of Schedule Performance")
    axes[0, 0].set_xlabel("Execution Time (ms)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Effect of tile_m on performance (aggregated)
    tile_m_times = {}
    for cfg, t in zip(configs, all_times):
        tile_m_times.setdefault(cfg.tile_m, []).append(t)
    tile_ms = sorted(tile_m_times.keys())
    means = [np.mean(tile_m_times[t]) for t in tile_ms]
    stds = [np.std(tile_m_times[t]) for t in tile_ms]
    axes[0, 1].bar(range(len(tile_ms)), means, yerr=stds, color='steelblue',
                    alpha=0.7, capsize=3)
    axes[0, 1].set_xticks(range(len(tile_ms)))
    axes[0, 1].set_xticklabels(tile_ms)
    axes[0, 1].set_title("Effect of Tile M on Performance")
    axes[0, 1].set_xlabel("Tile M")
    axes[0, 1].set_ylabel("Mean Execution Time (ms)")
    axes[0, 1].grid(True, alpha=0.3)

    # Effect of loop order
    order_times = {}
    order_names = ['mnk', 'mkn', 'nmk', 'nkm', 'kmn', 'knm']
    for cfg, t in zip(configs, all_times):
        order_times.setdefault(cfg.loop_order, []).append(t)
    order_ids = sorted(order_times.keys())
    means = [np.mean(order_times[o]) for o in order_ids]
    stds = [np.std(order_times[o]) for o in order_ids]
    axes[1, 0].bar(range(len(order_ids)), means, yerr=stds, color='coral',
                    alpha=0.7, capsize=3)
    axes[1, 0].set_xticks(range(len(order_ids)))
    axes[1, 0].set_xticklabels([order_names[o] for o in order_ids])
    axes[1, 0].set_title("Effect of Loop Order on Performance")
    axes[1, 0].set_xlabel("Loop Order")
    axes[1, 0].set_ylabel("Mean Execution Time (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    # Effect of unrolling
    unroll_times = {True: [], False: []}
    for cfg, t in zip(configs, all_times):
        unroll_times[cfg.unroll].append(t)
    positions = [0, 1]
    bp = axes[1, 1].boxplot(
        [unroll_times[False], unroll_times[True]],
        positions=positions, patch_artist=True
    )
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(['No Unroll', 'Unroll'])
    axes[1, 1].set_title("Effect of Unrolling on Performance")
    axes[1, 1].set_ylabel("Execution Time (ms)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Schedule Space Analysis for Matmul", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "schedule_analysis.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("TVM\nAutoTVM", "Expert-designed\ntemplate schedules\nManual tuning space\nRule-based search", 0.14, 'gray'),
        ("Ansor\n(2020)", "Automated schedule\ngeneration\nSketch + annotation\nMuch larger search space", 0.5, 'blue'),
        ("Cost Model\n+ BO", "Predict perf from\nschedule features\nGP acquisition fn\n→ Sample-efficient search", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Arrows
    for x1, x2 in [(0.28, 0.36), (0.64, 0.72)]:
        ax.annotate('', xy=(x2, 0.55), xytext=(x1, 0.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_title("TVM/Ansor: From Manual Templates to Automated Schedule Search", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "tvm_concept.png", dpi=150)
    plt.close()

    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
