"""
Minimal TIES Model Merging Reproduction
=========================================
Reproduces core ideas from "TIES-Merging: Resolving Interference When
Merging Models" (2306.01708, Yadav et al.):

1. Problem: naively averaging multiple task-specific finetuned models causes
   interference -- parameters that help one task hurt another.
2. TIES three-step procedure:
   a. TRIM: For each task vector, zero out the smallest-magnitude delta values,
      keeping only the top-k% most significant updates (k=20% default).
   b. ELECT: For each parameter position, elect a sign (+/-) based on which
      sign has greater aggregate magnitude across all (trimmed) task vectors.
   c. MERGE: Average only the values whose sign matches the elected sign;
      discard values that disagree with consensus.
3. Compared to: simple averaging (baseline), task arithmetic (sum all task
   vectors without trimming), and TIES.

This experiment trains small MLPs on synthetic classification tasks with
different label mappings, then merges them. We measure per-task accuracy
and visualize parameter-level interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── Model ──

class SmallMLP(nn.Module):
    def __init__(self, in_dim=64, hidden=128, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Task Vector Utilities ──

def get_task_vector(finetuned, pretrained):
    """Compute task vector: delta = theta_ft - theta_pre."""
    tv = {}
    for (name, p_ft), (_, p_pre) in zip(
        finetuned.named_parameters(), pretrained.named_parameters()
    ):
        tv[name] = p_ft.data - p_pre.data
    return tv


def apply_task_vector(pretrained, task_vector, scale=1.0):
    """Return a new model = pretrained + scale * task_vector."""
    merged = deepcopy(pretrained)
    for name, param in merged.named_parameters():
        if name in task_vector:
            param.data += scale * task_vector[name]
    return merged


# ── Merging Strategies ──

def merge_averaging(pretrained, task_vectors):
    """Simple averaging: pretrained + mean(task_vectors)."""
    merged = deepcopy(pretrained)
    for name, param in merged.named_parameters():
        avg_delta = sum(tv[name] for tv in task_vectors) / len(task_vectors)
        param.data += avg_delta
    return merged


def merge_task_arithmetic(pretrained, task_vectors, scale=1.0):
    """Task Arithmetic: pretrained + scale * sum(task_vectors)."""
    return apply_task_vector(pretrained,
                             {n: sum(tv[n] for tv in task_vectors)
                              for n in task_vectors[0]},
                             scale=scale)


def merge_ties(pretrained, task_vectors, k=0.2):
    """TIES Merging: Trim, Elect sign, Merge with consensus.

    1. TRIM: keep only top-k% magnitude values in each task vector.
    2. ELECT: for each parameter position, choose sign with greater aggregate
       absolute magnitude across trimmed task vectors.
    3. MERGE: average only the values whose sign matches the elected sign.
    """
    merged = deepcopy(pretrained)
    n_tasks = len(task_vectors)

    for name, param in merged.named_parameters():
        # Stack task vectors: (n_tasks, *param_shape)
        tv_stacked = torch.stack([tv[name] for tv in task_vectors])
        flat = tv_stacked.flatten(start_dim=1)  # (n_tasks, n_params)
        n_params = flat.shape[1]
        k_count = max(1, int(k * n_params))

        # Step 1: TRIM -- keep only top-k% magnitude per task
        trimmed = torch.zeros_like(flat)
        for i in range(n_tasks):
            _, top_idx = flat[i].abs().topk(k_count)
            trimmed[i, top_idx] = flat[i, top_idx]

        # Step 2: ELECT -- sign with greater aggregate magnitude wins
        pos_sum = trimmed.clamp(min=0).sum(dim=0)
        neg_sum = (-trimmed).clamp(min=0).sum(dim=0)
        elected_sign = torch.where(
            pos_sum >= neg_sum,
            torch.ones_like(pos_sum),
            -torch.ones_like(pos_sum),
        )

        # Step 3: MERGE -- average only values matching elected sign
        mask = (trimmed * elected_sign.unsqueeze(0)) >= 0
        merged_vals = (trimmed * mask.float()).sum(dim=0)
        count = mask.sum(dim=0).clamp(min=1)
        merged_vals = merged_vals / count

        param.data += merged_vals.reshape(param.shape)

    return merged


# ── Synthetic Multi-Task Data ──

def make_task_data(n_samples, in_dim, n_classes, task_id, n_tasks):
    """Generate classification data for a specific task.

    Each task has a different rotation of the input space and different
    decision boundaries, creating genuine parameter interference.
    """
    torch.manual_seed(task_id * 1000 + 42)

    # Different random projection per task
    proj = torch.randn(in_dim, in_dim)
    proj, _ = torch.linalg.qr(proj)  # orthogonal rotation

    # Generate data on unit sphere, then rotate
    x = torch.randn(n_samples, in_dim)
    x = x @ proj.T

    # Different decision boundary per task (hyperplane classification)
    w = torch.randn(in_dim, n_classes) / (in_dim ** 0.5)
    logits = x @ w
    y = logits.argmax(dim=1)

    return x, y


# ── Training & Evaluation ──

def train_on_task(model, x, y, n_epochs=20, lr=1e-3, device='cpu'):
    """Finetune a model on a single task."""
    model = deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, x, y, device='cpu'):
    """Return accuracy on given data."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        pred = logits.argmax(dim=1)
        acc = (pred == y.to(device)).float().mean().item()
    return acc


# ── Interference Analysis ──

def compute_interference_matrix(task_vectors, pretrained):
    """Compute pairwise interference between task vectors.

    Interference[i,j] = how much task_vector_i points against task_vector_j,
    measured as the fraction of parameters where signs disagree (among
    non-zero values).
    """
    n = len(task_vectors)
    interference = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            total_disagree = 0
            total_nonzero = 0
            for name in task_vectors[i]:
                vi = task_vectors[i][name].flatten()
                vj = task_vectors[j][name].flatten()
                # Only count parameters where both are non-zero
                nonzero = (vi.abs() > 1e-8) & (vj.abs() > 1e-8)
                total_nonzero += nonzero.sum().item()
                if total_nonzero > 0:
                    disagree = nonzero & (vi.sign() != vj.sign())
                    total_disagree += disagree.sum().item()
            if total_nonzero > 0:
                interference[i, j] = total_disagree / total_nonzero

    return interference


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "92-model-merging-ties"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Config
    n_tasks = 5
    in_dim = 64
    n_classes = 10
    n_train = 2000
    n_test = 500
    finetune_epochs = 20

    # Generate task data
    print("=== Generating task data ===")
    task_data = {}
    for t in range(n_tasks):
        x_train, y_train = make_task_data(n_train, in_dim, n_classes, t, n_tasks)
        x_test, y_test = make_task_data(n_test, in_dim, n_classes, t, n_tasks)
        task_data[t] = (x_train, y_train, x_test, y_test)
        print(f"  Task {t}: train={n_train}, test={n_test}")

    # Train pretrained model on a generic mix (all tasks concatenated)
    print("\n=== Training pretrained model ===")
    all_x = torch.cat([task_data[t][0] for t in range(n_tasks)])
    all_y = torch.cat([task_data[t][1] for t in range(n_tasks)])
    pretrained = SmallMLP(in_dim, 128, n_classes).to(device)
    pretrained = train_on_task(pretrained, all_x, all_y, n_epochs=15, lr=1e-3,
                               device=device)

    # Evaluate pretrained on each task
    pre_accs = []
    for t in range(n_tasks):
        _, _, x_test, y_test = task_data[t]
        acc = evaluate(pretrained, x_test, y_test, device)
        pre_accs.append(acc)
    print(f"  Pretrained per-task acc: {[f'{a:.3f}' for a in pre_accs]}")

    # Finetune on each task individually
    print("\n=== Finetuning on individual tasks ===")
    finetuned_models = []
    ft_accs_matrix = []  # (n_tasks, n_tasks) -- model_i on task_j

    for t in range(n_tasks):
        x_train, y_train, _, _ = task_data[t]
        model_t = train_on_task(pretrained, x_train, y_train,
                                n_epochs=finetune_epochs, device=device)
        finetuned_models.append(model_t)

        row = []
        for t2 in range(n_tasks):
            _, _, x_test, y_test = task_data[t2]
            acc = evaluate(model_t, x_test, y_test, device)
            row.append(acc)
        ft_accs_matrix.append(row)
        print(f"  Task {t} model accs: {[f'{a:.3f}' for a in row]}")

    # Compute task vectors
    print("\n=== Computing task vectors ===")
    task_vectors = []
    for model_t in finetuned_models:
        tv = get_task_vector(model_t, pretrained)
        task_vectors.append(tv)

    # ── Merging Strategies ──
    print("\n=== Merging Strategies ===")

    merge_methods = {
        'Averaging': merge_averaging(pretrained, task_vectors),
        'Task Arithmetic': merge_task_arithmetic(pretrained, task_vectors, scale=1.0),
        'TIES (k=20%)': merge_ties(pretrained, task_vectors, k=0.2),
        'TIES (k=30%)': merge_ties(pretrained, task_vectors, k=0.3),
        'TIES (k=10%)': merge_ties(pretrained, task_vectors, k=0.1),
    }

    results = {'Pretrained': pre_accs}
    for t in range(n_tasks):
        results[f'FT-Task{t}'] = ft_accs_matrix[t]

    for method_name, merged_model in merge_methods.items():
        accs = []
        for t in range(n_tasks):
            _, _, x_test, y_test = task_data[t]
            acc = evaluate(merged_model, x_test, y_test, device)
            accs.append(acc)
        results[method_name] = accs
        avg = np.mean(accs)
        print(f"  {method_name}: per-task={[f'{a:.3f}' for a in accs]}, avg={avg:.3f}")

    # Task Arithmetic scaling sweep
    print("\n=== Task Arithmetic Scaling Sweep ===")
    ta_scale_results = {}
    for scale in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        ta_model = merge_task_arithmetic(pretrained, task_vectors, scale=scale)
        accs = []
        for t in range(n_tasks):
            _, _, x_test, y_test = task_data[t]
            accs.append(evaluate(ta_model, x_test, y_test, device))
        avg = np.mean(accs)
        ta_scale_results[scale] = (accs, avg)
        print(f"  Scale={scale:.1f}: avg={avg:.3f}")

    # TIES k sweep
    print("\n=== TIES Trim Ratio Sweep ===")
    ties_k_results = {}
    for k in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]:
        ties_model = merge_ties(pretrained, task_vectors, k=k)
        accs = []
        for t in range(n_tasks):
            _, _, x_test, y_test = task_data[t]
            accs.append(evaluate(ties_model, x_test, y_test, device))
        avg = np.mean(accs)
        ties_k_results[k] = (accs, avg)
        print(f"  k={k:.2f}: avg={avg:.3f}")

    # Interference analysis
    print("\n=== Interference Analysis ===")
    interference = compute_interference_matrix(task_vectors, pretrained)
    print(f"  Pairwise sign disagreement:\n{np.array2string(interference, precision=3)}")

    # ── Visualization ──

    # 1. Per-task accuracy grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_methods = ['Pretrained', 'Averaging', 'Task Arithmetic',
                    'TIES (k=20%)', 'TIES (k=30%)', 'TIES (k=10%)']
    x_pos = np.arange(n_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods

    for i, method in enumerate(plot_methods):
        offset = (i - n_methods / 2 + 0.5) * width
        accs = results[method]
        color = '#2196F3' if 'Pre' in method else (
            '#FF9800' if 'Aver' in method else (
                '#9C27B0' if 'Arith' in method else '#4CAF50'))
        ax.bar(x_pos + offset, accs, width, label=method, alpha=0.8, color=color)

    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_title('TIES Merging: Per-Task Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Task {t}' for t in range(n_tasks)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / "per_task_accuracy.png", dpi=150)
    plt.close()

    # 2. Average accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_results = {}
    for method in plot_methods:
        avg_results[method] = np.mean(results[method])
    # Also add individual FT models
    for t in range(n_tasks):
        avg_results[f'FT-Task{t}'] = np.mean(results[f'FT-Task{t}'])

    names = list(avg_results.keys())
    values = list(avg_results.values())
    colors = []
    for n in names:
        if 'Pre' in n:
            colors.append('#2196F3')
        elif 'FT' in n:
            colors.append('#F44336')
        elif 'Aver' in n:
            colors.append('#FF9800')
        elif 'Arith' in n:
            colors.append('#9C27B0')
        else:
            colors.append('#4CAF50')

    bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Average Accuracy')
    ax.set_title('TIES Merging: Average Accuracy Across Tasks')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(results_dir / "avg_accuracy.png", dpi=150)
    plt.close()

    # 3. Interference heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(interference, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='equal')
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f'Task {t}' for t in range(n_tasks)])
    ax.set_yticklabels([f'Task {t}' for t in range(n_tasks)])
    ax.set_title('Task Vector Sign Disagreement\n(fraction of params with conflicting signs)')
    for i in range(n_tasks):
        for j in range(n_tasks):
            ax.text(j, i, f'{interference[i,j]:.3f}', ha='center', va='center',
                    fontsize=9, color='white' if interference[i, j] > 0.25 else 'black')
    plt.colorbar(im, label='Sign disagreement fraction')
    plt.tight_layout()
    plt.savefig(results_dir / "interference_heatmap.png", dpi=150)
    plt.close()

    # 4. Task Arithmetic scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    scales = sorted(ta_scale_results.keys())
    avg_accs = [ta_scale_results[s][1] for s in scales]
    # Also show per-task
    for t in range(n_tasks):
        task_accs = [ta_scale_results[s][0][t] for s in scales]
        ax.plot(scales, task_accs, '--', alpha=0.4, color=f'C{t}', linewidth=1)
    ax.plot(scales, avg_accs, 'o-', color='black', linewidth=2, markersize=8,
            label='Average')
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, label='Default scale=1.0')
    ax.set_xlabel('Scaling Factor (lambda)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task Arithmetic: Effect of Scaling Factor\n(dashed = per-task)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "task_arithmetic_scaling.png", dpi=150)
    plt.close()

    # 5. TIES trim ratio sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(ties_k_results.keys())
    avg_accs_k = [ties_k_results[k][1] for k in ks]
    for t in range(n_tasks):
        task_accs_k = [ties_k_results[k][0][t] for k in ks]
        ax.plot(ks, task_accs_k, '--', alpha=0.4, color=f'C{t}', linewidth=1)
    ax.plot(ks, avg_accs_k, 'o-', color='black', linewidth=2, markersize=8,
            label='Average')
    ax.axvline(x=0.2, color='red', linestyle=':', alpha=0.5, label='Default k=0.2')
    ax.set_xlabel('Trim Ratio (k)')
    ax.set_ylabel('Accuracy')
    ax.set_title('TIES: Effect of Trim Ratio\n(dashed = per-task)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "ties_trim_sweep.png", dpi=150)
    plt.close()

    # 6. Task vector magnitude distributions (per task)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4))
    if n_tasks == 1:
        axes = [axes]
    for t in range(n_tasks):
        mags = []
        for name, vec in task_vectors[t].items():
            mags.extend(vec.flatten().cpu().numpy())
        axes[t].hist(mags, bins=100, alpha=0.7, color=f'C{t}', density=True)
        axes[t].set_title(f'Task {t}')
        axes[t].set_xlabel('Delta Magnitude')
        axes[t].set_ylabel('Density')
        axes[t].set_yscale('log')
    plt.suptitle('Task Vector Parameter Magnitude Distributions', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "task_vector_distributions.png", dpi=150)
    plt.close()

    # 7. Trimmed vs untrimmed task vector comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k = 0.2
    for t in range(min(2, n_tasks)):
        flat = torch.cat([v.flatten() for v in task_vectors[t].values()]).cpu().numpy()
        n_params = len(flat)
        k_count = max(1, int(k * n_params))
        top_idx = np.abs(flat).argsort()[::-1][:k_count]

        trimmed = np.zeros_like(flat)
        trimmed[top_idx] = flat[top_idx]

        axes[t].hist(flat, bins=100, alpha=0.5, label='Full', color='blue', density=True)
        trimmed_nz = trimmed[trimmed != 0]
        axes[t].hist(trimmed_nz, bins=50, alpha=0.7, label=f'Top {k*100:.0f}%', color='red',
                     density=True)
        axes[t].set_title(f'Task {t} Vector: Full vs Trimmed')
        axes[t].legend()
        axes[t].set_yscale('log')
        axes[t].set_xlabel('Delta Value')
    plt.suptitle('TIES Trim Step: Keeping Only Significant Updates', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "trim_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
