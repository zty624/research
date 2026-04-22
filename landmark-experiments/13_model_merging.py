"""
Minimal Model Merging Reproduction
====================================
Reproduces core ideas from model merging literature:
1. Model Soups (2203.05482): greedy selection over checkpoints
2. Task Arithmetic (2212.04089): adding task vectors
3. TIES Merging (2306.01708): trimming + sign consensus
4. Compare merging strategies on MNIST multi-task setup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── Simple MLP ──

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


# ── Task Vector Arithmetic ──

def get_task_vector(finetuned_model, pretrained_model):
    """Task vector = θ_finetuned - θ_pretrained."""
    tv = {}
    for (name, p_ft), (_, p_pre) in zip(finetuned_model.named_parameters(),
                                          pretrained_model.named_parameters()):
        tv[name] = p_ft.data - p_pre.data
    return tv


def add_task_vector(model, task_vector, scale=1.0):
    """Add scaled task vector to model."""
    merged = deepcopy(model)
    for name, param in merged.named_parameters():
        if name in task_vector:
            param.data += scale * task_vector[name]
    return merged


# ── TIES Merging ──

def ties_merge(pretrained_model, task_vectors, k=0.2):
    """TIES: TrIm, Elect, and Sign merge.
    1. Trim: keep only top-k% magnitude values per task vector
    2. Elect: for each param, use sign with highest aggregate magnitude
    3. Sign: merge only values with the elected sign
    """
    merged = deepcopy(pretrained_model)

    for name, param in merged.named_parameters():
        tv_stacked = torch.stack([tv[name] for tv in task_vectors])  # (n_tasks, *param_shape)
        flat = tv_stacked.flatten(start_dim=1)  # (n_tasks, num_params)

        n_params = flat.shape[1]
        k_count = max(1, int(k * n_params))

        # Step 1: Trim — keep top-k% magnitude per task
        trimmed = torch.zeros_like(flat)
        for i in range(len(task_vectors)):
            _, top_idx = flat[i].abs().topk(k_count)
            trimmed[i, top_idx] = flat[i, top_idx]

        # Step 2: Elect — sign with highest aggregate magnitude
        pos_sum = trimmed.clamp(min=0).sum(dim=0)
        neg_sum = (-trimmed).clamp(min=0).sum(dim=0)
        elected_sign = torch.where(pos_sum >= neg_sum,
                                    torch.ones_like(pos_sum),
                                    -torch.ones_like(pos_sum))

        # Step 3: Merge — sum values that match elected sign
        mask = (trimmed * elected_sign.unsqueeze(0)) >= 0
        merged_flat = (trimmed * mask.float()).sum(dim=0)
        non_zero_count = mask.sum(dim=0).clamp(min=1)
        merged_flat = merged_flat / non_zero_count

        # Reshape and apply
        merged_param = merged_flat.reshape(param.shape)
        param.data += merged_param

    return merged


# ── Model Soups ──

def greedy_soup(pretrained_model, checkpoints, val_loader_fn, device='cpu'):
    """Greedy Model Soup: iteratively add checkpoints if they improve validation."""
    soup = deepcopy(checkpoints[0])
    best_acc = val_loader_fn(soup, device)
    ingredients = [0]

    for i in range(1, len(checkpoints)):
        # Try adding this checkpoint
        candidate = deepcopy(soup)
        for (name, p_cand), (_, p_new) in zip(candidate.named_parameters(),
                                                  checkpoints[i].named_parameters()):
            p_cand.data = (p_cand.data + p_new.data) / 2

        acc = val_loader_fn(candidate, device)
        if acc > best_acc:
            soup = candidate
            best_acc = acc
            ingredients.append(i)
            print(f"  Added checkpoint {i} → acc = {acc:.3f}")

    return soup, best_acc, ingredients


# ── Multi-task Setup ──

def create_permuted_mnist_task(n_tasks=3, seed_base=0):
    """Create multiple permuted MNIST tasks for multi-task evaluation."""
    permutations = []
    for t in range(n_tasks):
        rng = np.random.RandomState(seed_base + t)
        perm = rng.permutation(784)
        permutations.append(torch.tensor(perm, dtype=torch.long))
    return permutations


def apply_permutation(x, perm):
    """Apply pixel permutation to MNIST images."""
    B, C, H, W = x.shape
    x_flat = x.view(B, -1)
    x_perm = x_flat[:, perm]
    return x_perm.view(B, C, H, W)


# ── Training ──

def train_on_task(model, train_loader, perm, n_epochs=3, lr=1e-3, device='cpu'):
    """Train model on a permuted MNIST task."""
    model = deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = apply_permutation(batch_x, perm)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_on_task(model, test_loader, perm, device='cpu'):
    """Evaluate model on a permuted MNIST task."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = apply_permutation(batch_x, perm)

            logits = model(batch_x)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.shape[0]

    return correct / total


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "13-model-merging"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load MNIST
    print("=== Loading MNIST ===")
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use subsets for speed
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=256)

    # Create 3 permuted MNIST tasks
    n_tasks = 3
    perms = create_permuted_mnist_task(n_tasks)

    # Train pretrained model (on original MNIST)
    print("\n=== Training Pretrained Model ===")
    pretrained = SimpleMLP().to(device)
    optimizer = torch.optim.AdamW(pretrained.parameters(), lr=1e-3)
    for epoch in range(3):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            loss = F.cross_entropy(pretrained(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"  Pretrained acc: {evaluate_on_task(pretrained, test_loader, perms[0], device):.3f}")

    # Finetune on each task
    print("\n=== Finetuning on Tasks ===")
    finetuned_models = []
    task_accuracies = {}

    for t in range(n_tasks):
        print(f"  Task {t}...")
        model_t = train_on_task(pretrained, train_loader, perms[t], n_epochs=3, device=device)
        finetuned_models.append(model_t)

        # Evaluate on all tasks
        for t2 in range(n_tasks):
            acc = evaluate_on_task(model_t, test_loader, perms[t2], device)
            task_accuracies[(f'task{t}_model', f'task{t2}')] = acc

    # Compute task vectors
    task_vectors = []
    for model_t in finetuned_models:
        tv = get_task_vector(model_t, pretrained)
        task_vectors.append(tv)

    # ── Merging Strategies ──

    # 1. Simple averaging
    print("\n=== Simple Averaging ===")
    avg_model = deepcopy(pretrained)
    for name, param in avg_model.named_parameters():
        param.data = sum(tv[name] for tv in task_vectors) / len(task_vectors) + \
                     pretrained.state_dict()[name]

    # 2. Task Arithmetic (scale=1.0)
    print("=== Task Arithmetic (scale=1.0) ===")
    ta_model = deepcopy(pretrained)
    for name, param in ta_model.named_parameters():
        param.data = pretrained.state_dict()[name] + sum(tv[name] for tv in task_vectors)

    # 3. TIES Merging
    print("=== TIES Merging ===")
    ties_model = ties_merge(pretrained, task_vectors, k=0.2)

    # Evaluate all strategies on all tasks
    strategies = {
        'Pretrained': pretrained,
        'Avg': avg_model,
        'Task Arith': ta_model,
        'TIES': ties_model,
    }

    # Add individual finetuned models
    for t in range(n_tasks):
        strategies[f'FT-Task{t}'] = finetuned_models[t]

    results = {}
    for strat_name, model in strategies.items():
        model = model.to(device)
        task_accs = []
        for t in range(n_tasks):
            acc = evaluate_on_task(model, test_loader, perms[t], device)
            task_accs.append(acc)
        results[strat_name] = task_accs
        avg_acc = np.mean(task_accs)
        print(f"  {strat_name}: per-task={[f'{a:.3f}' for a in task_accs]}, avg={avg_acc:.3f}")

    # ── Task Arithmetic scaling analysis ──
    print("\n=== Task Arithmetic Scaling ===")
    scales = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    ta_scale_results = {}

    for scale in scales:
        scaled_model = deepcopy(pretrained)
        for name, param in scaled_model.named_parameters():
            param.data = pretrained.state_dict()[name] + scale * sum(tv[name] for tv in task_vectors)

        scaled_model = scaled_model.to(device)
        accs = [evaluate_on_task(scaled_model, test_loader, perms[t], device) for t in range(n_tasks)]
        ta_scale_results[scale] = np.mean(accs)
        print(f"  Scale={scale:.1f}: avg_acc={ta_scale_results[scale]:.3f}")

    # ── Visualization ──

    # 1. Per-task accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    strat_names = [s for s in strategies if s != 'Pretrained']
    x = np.arange(n_tasks)
    width = 0.12

    for i, name in enumerate(strat_names):
        accs = results[name]
        offset = (i - len(strat_names)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=name, alpha=0.8)

    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Merging: Per-Task Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t}" for t in range(n_tasks)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / "per_task_accuracy.png", dpi=150)
    plt.close()

    # 2. Average accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_accs = {name: np.mean(accs) for name, accs in results.items() if name != 'Pretrained'}
    names = list(avg_accs.keys())
    values = list(avg_accs.values())
    colors = ['red' if 'FT' in n else ('blue' if n == 'TIES' else 'orange' if n == 'Task Arith' else 'gray')
              for n in names]
    ax.barh(names, values, color=colors, alpha=0.7)
    ax.set_xlabel("Average Accuracy")
    ax.set_title("Model Merging: Average Accuracy Across Tasks")
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(results_dir / "avg_accuracy.png", dpi=150)
    plt.close()

    # 3. Task Arithmetic scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(ta_scale_results.keys()), list(ta_scale_results.values()),
            'o-', color='blue', linewidth=2, markersize=8)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Default scale=1.0')
    ax.set_xlabel("Scaling Factor λ")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Task Arithmetic: Effect of Scaling Factor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "task_arithmetic_scaling.png", dpi=150)
    plt.close()

    # 4. Task vector magnitude distribution
    fig, axes = plt.subplots(1, n_tasks, figsize=(15, 4))
    for t in range(n_tasks):
        magnitudes = []
        for name, vec in task_vectors[t].items():
            magnitudes.extend(vec.flatten().cpu().numpy())
        axes[t].hist(magnitudes, bins=100, alpha=0.7, color=f'C{t}')
        axes[t].set_title(f"Task {t} Vector Magnitudes")
        axes[t].set_xlabel("Magnitude")
        axes[t].set_ylabel("Count")
        axes[t].set_yscale('log')

    plt.suptitle("Task Vector Parameter Magnitude Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "task_vector_distribution.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
