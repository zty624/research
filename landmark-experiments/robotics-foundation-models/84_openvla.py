"""
Minimal OpenVLA-style Vision-Language-Action Model Reproduction
================================================================
Inspired by OpenVLA: "OpenVLA: An Open-Source Vision-Language-Action Model"
(Kim et al., 2024) and the VLA paradigm more broadly.

Core ideas:
1. Language-conditioned policy: given an instruction + image → action
2. FiLM conditioning: language embedding modulates visual features
   via feature-wise linear transformation (scale + shift)
3. Multi-task: same visual scene, different language instructions
   produce different actions
4. Compare: unconditional policy (no language) vs language-conditioned
   policy — demonstrating that language enables task-specific behavior

Synthetic task: 16x16 grid with colored objects rendered as 3x3 blobs
plus distance heatmaps. Different instructions (e.g., "reach red",
"reach blue", "push left") produce different optimal actions from the
same visual scene. 5-action vocabulary (4 directions + stay).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic Multi-Task Environment ──

TASKS = [
    "reach red",      # Move toward the red object
    "reach blue",     # Move toward the blue object
    "reach green",    # Move toward the green object
    "push left",      # Move left regardless of objects
    "push right",     # Move right regardless of objects
    "stay still",     # Don't move
]
N_TASKS = len(TASKS)


class MultiTaskGridEnv:
    """16x16 grid with colored objects rendered as 3x3 blobs + heatmaps.
    Task determines which action is correct from the same visual scene.
    """
    def __init__(self, grid_size=16, max_steps=20):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_colors = 3  # red, blue, green
        self.blob_radius = 1
        # 5 actions: up, down, left, right, stay
        self.action_map = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (0, 0),   # stay
        ]
        self.action_names = ['up', 'down', 'left', 'right', 'stay']
        self.n_actions = len(self.action_map)

    def _draw_blob(self, img, channel, center, value=1.0):
        """Draw a 3x3 blob centered at (row, col)."""
        r, c = int(center[0]), int(center[1])
        for dr in range(-self.blob_radius, self.blob_radius + 1):
            for dc in range(-self.blob_radius, self.blob_radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    img[channel, rr, cc] = value

    def render(self, agent_pos, object_positions):
        """Render: ch0=red obj, ch1=blue obj, ch2=green obj, ch3=agent+heatmaps."""
        C = 4
        img = np.zeros((C, self.grid_size, self.grid_size), dtype=np.float32)
        # Colored objects as 3x3 blobs
        for color_idx, pos in object_positions.items():
            self._draw_blob(img, color_idx, pos)
        # Agent as 3x3 blob in channel 3
        self._draw_blob(img, 3, agent_pos, 0.8)
        # Distance heatmaps to each object in their respective channels
        for color_idx, pos in object_positions.items():
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    d = np.sqrt((i - pos[0])**2 + (j - pos[1])**2)
                    heatmap = max(0, 1.0 - d / (self.grid_size * 0.5))
                    img[color_idx, i, j] = max(img[color_idx, i, j], heatmap * 0.3)
        return img

    def oracle_action(self, agent_pos, object_positions, task_idx):
        """Compute the correct action for a given task."""
        if task_idx <= 2:
            # "reach color_X": move toward that color object
            color_idx = task_idx
            if color_idx in object_positions:
                target = object_positions[color_idx]
                dx = int(np.clip(target[0] - agent_pos[0], -1, 1))
                dy = int(np.clip(target[1] - agent_pos[1], -1, 1))
                # Pick dominant axis for deterministic action
                if abs(dx) >= abs(dy):
                    return self.action_map.index((dx, 0))
                else:
                    return self.action_map.index((0, dy))
            return self.action_map.index((0, 0))  # stay if no target
        elif task_idx == 3:
            return self.action_map.index((0, -1))  # push left
        elif task_idx == 4:
            return self.action_map.index((0, 1))   # push right
        else:
            return self.action_map.index((0, 0))   # stay still

    def generate_dataset(self, n_samples=5000):
        """Generate (image, task_idx, action) triplets."""
        images = []
        task_indices = []
        actions = []

        for _ in range(n_samples):
            agent_pos = np.array([
                np.random.randint(3, self.grid_size - 3),
                np.random.randint(3, self.grid_size - 3)
            ], dtype=np.float32)

            # Place colored objects at random positions (ensure not too close to agent)
            object_positions = {}
            for color_idx in range(self.n_colors):
                if np.random.random() < 0.8:
                    obj_pos = np.array([
                        np.random.randint(2, self.grid_size - 2),
                        np.random.randint(2, self.grid_size - 2)
                    ], dtype=np.float32)
                    object_positions[color_idx] = obj_pos

            # Pick a valid task
            task_idx = np.random.randint(0, N_TASKS)
            if task_idx <= 2 and task_idx not in object_positions:
                valid = [t for t in range(3) if t in object_positions] + [3, 4, 5]
                task_idx = np.random.choice(valid)

            img = self.render(agent_pos, object_positions)
            action_idx = self.oracle_action(agent_pos, object_positions, task_idx)

            images.append(img)
            task_indices.append(task_idx)
            actions.append(action_idx)

        return (np.array(images), np.array(task_indices), np.array(actions))


# ── Vision Encoder ──

class SimpleVisionEncoder(nn.Module):
    """CNN producing a flat feature vector from top-down images."""
    def __init__(self, in_channels=4, d_model=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.proj(h)


# ── Language Encoder ──

class InstructionEncoder(nn.Module):
    """Simple embedding for task instructions (treated as discrete tokens)."""
    def __init__(self, n_tasks, d_model):
        super().__init__()
        self.embed = nn.Embedding(n_tasks, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, task_idx):
        h = self.embed(task_idx)
        return self.proj(h)


# ── FiLM Layer ──

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * features + beta
    where gamma, beta are predicted from a conditioning signal (language).
    This is the key mechanism for language-conditioned action prediction.
    """
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.gamma_net = nn.Linear(d_cond, d_model)
        self.beta_net = nn.Linear(d_cond, d_model)

    def forward(self, features, cond):
        gamma = self.gamma_net(cond)
        beta = self.beta_net(cond)
        return gamma * features + beta


# ── Unconditional Policy (no language) ──

class UnconditionalPolicy(nn.Module):
    """Baseline: predict action from image only, ignoring task instruction."""
    def __init__(self, in_channels=4, d_model=128, n_actions=5):
        super().__init__()
        self.vision = SimpleVisionEncoder(in_channels, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, n_actions)
        )

    def forward(self, image, task_idx=None):
        h = self.vision(image)
        return self.head(h)


# ── Language-Conditioned Policy (FiLM) ──

class FiLMPolicy(nn.Module):
    """VLA-style policy: language modulates visual features via FiLM.
    Given instruction → language embedding → FiLM parameters (gamma, beta)
    FiLM transforms visual features → task-specific representation → action.
    """
    def __init__(self, in_channels=4, d_model=128, n_actions=5, n_tasks=N_TASKS):
        super().__init__()
        self.vision = SimpleVisionEncoder(in_channels, d_model)
        self.language = InstructionEncoder(n_tasks, d_model)
        self.film1 = FiLMLayer(d_model, d_model)
        self.film2 = FiLMLayer(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, n_actions)
        )

    def forward(self, image, task_idx):
        vis_h = self.vision(image)
        lang_h = self.language(task_idx)
        h = self.film1(vis_h, lang_h)
        h = F.relu(h)
        h = self.film2(h, lang_h)
        h = F.relu(h)
        return self.head(h)


# ── Concatenation Policy (alternative conditioning) ──

class ConcatPolicy(nn.Module):
    """Simple alternative: concatenate language + vision features."""
    def __init__(self, in_channels=4, d_model=128, n_actions=5, n_tasks=N_TASKS):
        super().__init__()
        self.vision = SimpleVisionEncoder(in_channels, d_model)
        self.language = InstructionEncoder(n_tasks, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, n_actions)
        )

    def forward(self, image, task_idx):
        vis_h = self.vision(image)
        lang_h = self.language(task_idx)
        h = torch.cat([vis_h, lang_h], dim=-1)
        return self.head(h)


# ── Training ──

def train_model(model, images, task_indices, actions, n_epochs=120,
                lr=3e-4, batch_size=256, device='cpu', name=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    images_t = torch.FloatTensor(images).to(device)
    tasks_t = torch.LongTensor(task_indices).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    N = len(images_t)

    losses = []
    accs = []

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randint(0, N, (batch_size,))
        x = images_t[idx]
        t = tasks_t[idx]
        y = actions_t[idx]

        logits = model(x, t) if not isinstance(model, UnconditionalPolicy) else model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(-1) == y).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accs.append(acc)

        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs


def evaluate_per_task(model, images, task_indices, actions, device='cpu'):
    """Evaluate accuracy broken down by task."""
    model.eval()
    images_t = torch.FloatTensor(images).to(device)
    tasks_t = torch.LongTensor(task_indices).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    results = {}
    for task_id in range(N_TASKS):
        mask = task_indices == task_id
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        x = images_t[idx]
        t = tasks_t[idx]
        y = actions_t[idx]

        with torch.no_grad():
            logits = model(x, t) if not isinstance(model, UnconditionalPolicy) else model(x)
            acc = (logits.argmax(-1) == y).float().mean().item()
        results[TASKS[task_id]] = acc

    return results


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "84-openvla"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = MultiTaskGridEnv(grid_size=16, max_steps=20)

    print("=== Generating Multi-Task Data ===")
    images, task_indices, actions = env.generate_dataset(n_samples=6000)
    print(f"  Dataset: {len(images)} samples, {N_TASKS} tasks")
    for t in range(N_TASKS):
        count = (task_indices == t).sum()
        print(f"    Task {t} ({TASKS[t]}): {count} samples ({count/len(actions)*100:.1f}%)")

    print("\n=== Training Unconditional Policy (no language) ===")
    uncond = UnconditionalPolicy(in_channels=4, d_model=128, n_actions=env.n_actions).to(device)
    uncond_losses, uncond_accs = train_model(
        uncond, images, task_indices, actions, n_epochs=120, device=device, name="Uncond"
    )

    print("\n=== Training Concat Policy (vision + language concat) ===")
    concat = ConcatPolicy(in_channels=4, d_model=128, n_actions=env.n_actions, n_tasks=N_TASKS).to(device)
    concat_losses, concat_accs = train_model(
        concat, images, task_indices, actions, n_epochs=120, device=device, name="Concat"
    )

    print("\n=== Training FiLM Policy (VLA-style) ===")
    film = FiLMPolicy(in_channels=4, d_model=128, n_actions=env.n_actions, n_tasks=N_TASKS).to(device)
    film_losses, film_accs = train_model(
        film, images, task_indices, actions, n_epochs=120, device=device, name="FiLM"
    )

    # Evaluate per task
    print("\n=== Per-Task Evaluation ===")
    uncond_per_task = evaluate_per_task(uncond, images, task_indices, actions, device=device)
    concat_per_task = evaluate_per_task(concat, images, task_indices, actions, device=device)
    film_per_task = evaluate_per_task(film, images, task_indices, actions, device=device)

    print(f"{'Task':<15} {'Uncond':>8} {'Concat':>8} {'FiLM':>8}")
    print("-" * 42)
    for task_name in TASKS:
        u = uncond_per_task.get(task_name, 0)
        c = concat_per_task.get(task_name, 0)
        f = film_per_task.get(task_name, 0)
        print(f"{task_name:<15} {u:>8.3f} {c:>8.3f} {f:>8.3f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 5

    for losses, name, color in [
        (uncond_losses, "Unconditional", "red"),
        (concat_losses, "Concat", "orange"),
        (film_losses, "FiLM (VLA)", "blue"),
    ]:
        s = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[0].plot(s, label=name, color=color)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for accs_list, name, color in [
        (uncond_accs, "Unconditional", "red"),
        (concat_accs, "Concat", "orange"),
        (film_accs, "FiLM (VLA)", "blue"),
    ]:
        s = np.convolve(accs_list, np.ones(window)/window, mode='valid')
        axes[1].plot(s, label=name, color=color)

    axes[1].set_title("Action Prediction Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("OpenVLA: Language-Conditioned Action Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Per-task accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    task_names = [t for t in TASKS if t in uncond_per_task]
    x = np.arange(len(task_names))
    width = 0.25

    uncond_vals = [uncond_per_task[t] for t in task_names]
    concat_vals = [concat_per_task[t] for t in task_names]
    film_vals = [film_per_task[t] for t in task_names]

    ax.bar(x - width, uncond_vals, width, label='Unconditional', color='red', alpha=0.7)
    ax.bar(x, concat_vals, width, label='Concat', color='orange', alpha=0.7)
    ax.bar(x + width, film_vals, width, label='FiLM (VLA)', color='blue', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=15, ha='right')
    ax.set_ylabel("Action Prediction Accuracy")
    ax.set_title("Per-Task Accuracy: Unconditional vs Concat vs FiLM")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(results_dir / "per_task_accuracy.png", dpi=150)
    plt.close()

    # 3. FiLM modulation visualization
    film.eval()
    agent_pos = np.array([8.0, 8.0])
    object_positions = {
        0: np.array([3.0, 4.0]),   # red (top-left area)
        1: np.array([12.0, 10.0]), # blue (bottom-right area)
        2: np.array([4.0, 12.0]),  # green (top-right area)
    }
    test_img = env.render(agent_pos, object_positions)
    test_img_t = torch.FloatTensor(test_img).unsqueeze(0).expand(N_TASKS, -1, -1, -1).to(device)
    test_tasks_t = torch.arange(N_TASKS).to(device)

    with torch.no_grad():
        vis_h = film.vision(test_img_t)
        lang_h = film.language(test_tasks_t)
        gamma1 = film.film1.gamma_net(lang_h)
        beta1 = film.film1.beta_net(lang_h)
        logits = film(test_img_t, test_tasks_t)
        pred_actions = logits.argmax(-1).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx in range(N_TASKS):
        ax = axes[idx // 3, idx % 3]
        g = gamma1[idx].cpu().numpy()[:32]
        b = beta1[idx].cpu().numpy()[:32]
        x_pos = np.arange(len(g))
        ax.bar(x_pos - 0.2, g, 0.4, label='gamma (scale)', color='blue', alpha=0.7)
        ax.bar(x_pos + 0.2, b, 0.4, label='beta (shift)', color='red', alpha=0.7)
        action_name = env.action_names[pred_actions[idx]]
        ax.set_title(f'"{TASKS[idx]}" -> {action_name}', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("FiLM Parameters per Instruction (Same Visual Scene)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "film_parameters.png", dpi=150)
    plt.close()

    # 4. Task-specific behavior: same scene, different actions
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx in range(N_TASKS):
        ax = axes[idx // 3, idx % 3]
        # Show scene (use first 3 channels as RGB-ish)
        ax.imshow(test_img[:3].transpose(1, 2, 0).clip(0, 1),
                  extent=[-0.5, env.grid_size-0.5, env.grid_size-0.5, -0.5])

        dx, dy = env.action_map[pred_actions[idx]]
        if dx != 0 or dy != 0:
            ax.annotate('', xy=(8 + dy * 2, 8 + dx * 2), xytext=(8, 8),
                        arrowprops=dict(arrowstyle='->', color='white', lw=3))
        else:
            ax.plot(8, 8, 'o', color='white', markersize=15)

        ax.set_title(f'Instruction: "{TASKS[idx]}"\nAction: {env.action_names[pred_actions[idx]]}',
                     fontsize=10)
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(env.grid_size - 0.5, -0.5)

    plt.suptitle("Same Scene, Different Instructions -> Different Actions (FiLM Policy)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "task_specific_behavior.png", dpi=150)
    plt.close()

    # 5. Task group accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    reach_tasks = [TASKS[i] for i in range(3)]
    direction_tasks = [TASKS[i] for i in range(3, 5)]
    still_task = [TASKS[5]]

    categories = ['Reach\n(colored objects)', 'Direction\n(push L/R)', 'Stay\nStill']
    uncond_grouped = [
        np.mean([uncond_per_task.get(t, 0) for t in reach_tasks]),
        np.mean([uncond_per_task.get(t, 0) for t in direction_tasks]),
        np.mean([uncond_per_task.get(t, 0) for t in still_task]),
    ]
    film_grouped = [
        np.mean([film_per_task.get(t, 0) for t in reach_tasks]),
        np.mean([film_per_task.get(t, 0) for t in direction_tasks]),
        np.mean([film_per_task.get(t, 0) for t in still_task]),
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, uncond_grouped, width, label='Unconditional', color='red', alpha=0.7)
    ax.bar(x + width/2, film_grouped, width, label='FiLM (VLA)', color='blue', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Language Conditioning Helps Most for Object-Specific Tasks")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    for i in range(len(categories)):
        ax.text(i - width/2, uncond_grouped[i] + 0.02, f'{uncond_grouped[i]:.2f}',
                ha='center', fontsize=9, color='red')
        ax.text(i + width/2, film_grouped[i] + 0.02, f'{film_grouped[i]:.2f}',
                ha='center', fontsize=9, color='blue')

    plt.tight_layout()
    plt.savefig(results_dir / "task_group_accuracy.png", dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Unconditional\nPolicy", "Image -> CNN -> Action\nNo task information\nSame scene -> same action\nFails multi-task", 0.15, 'red'),
        ("FiLM-VLA\nPolicy", "Image -> CNN -> features\nLanguage -> gamma, beta\nfeatures = gamma*vis + beta\nTask-specific actions!", 0.50, 'blue'),
        ("OpenVLA\n(VLM-based)", "Pre-trained VLM\nFine-tune for actions\nLanguage = action tokens\nWeb -> Robot transfer", 0.85, 'purple'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.28, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.annotate('', xy=(0.32, 0.5), xytext=(0.22, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(0.68, 0.5), xytext=(0.58, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax.set_title("From Unconditional to Language-Conditioned: VLA Architecture",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "vla_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
