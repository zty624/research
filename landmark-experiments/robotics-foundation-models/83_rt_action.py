"""
Minimal RT-1/RT-2 Action Prediction Reproduction
==================================================
Reproduces core ideas from:
- RT-1: "RT-1: Robotics Transformer for Real-World Control at Scale"
  (2212.04356, Brohan et al.)
- RT-2: "RT-2: Vision-Language-Action Models Transfer Web Knowledge
  to Robotic Control" (2307.15818, Google DeepMind)

Core ideas:
1. Vision encoder → Transformer → Discretized action tokens
2. Tokenize the action space (like a vocabulary) so robot actions
   become "words" the transformer can predict
3. Learn mapping from visual observations to action sequences
4. RT-2 extends this by pre-training on web-scale VLM data, then
   fine-tuning for action prediction — showing transfer from language

Synthetic task: 16x16 grid world with reaching tasks. Agent and
target are rendered as 3x3 blobs so spatial structure is visible to
the CNN. Discretized 7-action vocabulary (4 directions + stay +
grip-open + grip-close). Compare MLP baseline vs Transformer-based
RT-style architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Synthetic 2D Grid World ──

class GridWorldEnv:
    """2D grid world: agent must reach a target.
    Observation: 16x16 top-down image with agent/target as 3x3 blobs.
    Action: 7-token vocabulary (4 directions + stay + grip-open/close).
    """
    def __init__(self, grid_size=16, n_channels=3, max_steps=25):
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.max_steps = max_steps
        self.blob_radius = 1  # 3x3 blob

        # Action vocabulary: 7 tokens
        self.action_vocab = [
            (0, -1, 0),   # 0: move left
            (0, 1, 0),    # 1: move right
            (-1, 0, 0),   # 2: move up
            (1, 0, 0),    # 3: move down
            (0, 0, 0),    # 4: stay still
            (0, 0, 0),    # 5: grip open (no movement, label differs)
            (0, 0, 1),    # 6: grip close
        ]
        self.action_names = ['left', 'right', 'up', 'down', 'stay', 'grip-open', 'grip-close']
        self.n_actions = len(self.action_vocab)

    def _draw_blob(self, img, channel, center, value=1.0):
        """Draw a 3x3 blob centered at (row, col)."""
        r, c = int(center[0]), int(center[1])
        for dr in range(-self.blob_radius, self.blob_radius + 1):
            for dc in range(-self.blob_radius, self.blob_radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    img[channel, rr, cc] = value

    def render(self, agent_pos, target_pos):
        """Render top-down view: ch0=agent, ch1=target, ch2=distance map."""
        img = np.zeros((self.n_channels, self.grid_size, self.grid_size), dtype=np.float32)
        self._draw_blob(img, 0, agent_pos, 1.0)    # Agent in red
        self._draw_blob(img, 1, target_pos, 1.0)    # Target in green
        # Distance heatmap to target (helps spatial reasoning)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                d = np.sqrt((i - target_pos[0])**2 + (j - target_pos[1])**2)
                img[2, i, j] = max(0, 1.0 - d / (self.grid_size * 0.7))
        return img

    def oracle_action(self, agent_pos, target_pos):
        """Oracle: move toward target, close grip when there."""
        dx = int(np.clip(target_pos[0] - agent_pos[0], -1, 1))
        dy = int(np.clip(target_pos[1] - agent_pos[1], -1, 1))

        # When at target, close gripper
        if dx == 0 and dy == 0:
            return 6  # grip-close
        # Map direction to action index
        dir_map = {(-1, 0): 2, (1, 0): 3, (0, -1): 0, (0, 1): 1}
        # Diagonal: pick the dominant axis
        if abs(dx) >= abs(dy):
            return dir_map.get((dx, 0), 4)
        else:
            return dir_map.get((0, dy), 4)

    def generate_episode(self):
        """Generate expert episode using oracle."""
        # Ensure agent and target are not too close (makes task non-trivial)
        while True:
            agent_pos = np.array([
                np.random.randint(2, self.grid_size - 2),
                np.random.randint(2, self.grid_size - 2)
            ], dtype=np.float32)
            target_pos = np.array([
                np.random.randint(2, self.grid_size - 2),
                np.random.randint(2, self.grid_size - 2)
            ], dtype=np.float32)
            if np.abs(agent_pos - target_pos).sum() > 3:
                break

        images = []
        actions = []

        for step in range(self.max_steps):
            img = self.render(agent_pos, target_pos)
            images.append(img)
            action_idx = self.oracle_action(agent_pos, target_pos)
            actions.append(action_idx)

            dx, dy, grip = self.action_vocab[action_idx]
            agent_pos[0] = np.clip(agent_pos[0] + dx, 0, self.grid_size - 1)
            agent_pos[1] = np.clip(agent_pos[1] + dy, 0, self.grid_size - 1)

            if action_idx == 6:  # grip-close = reached target
                break

        return images, actions

    def generate_dataset(self, n_episodes=800):
        """Generate dataset of (image, action) pairs."""
        all_images = []
        all_actions = []

        for _ in range(n_episodes):
            images, actions = self.generate_episode()
            all_images.extend(images)
            all_actions.extend(actions)

        return np.array(all_images), np.array(all_actions)


# ── Vision Encoder ──

class VisionEncoder(nn.Module):
    """CNN that produces a sequence of visual tokens from images."""
    def __init__(self, in_channels=3, d_model=64, n_tokens=16):
        super().__init__()
        self.n_tokens = n_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(128, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

    def forward(self, x):
        """x: (B, C, H, W) → (B, n_tokens, d_model)"""
        h = self.conv(x)  # (B, 128, H', W')
        B, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Pool or pad to n_tokens
        if H * W > self.n_tokens:
            idx = torch.linspace(0, H * W - 1, self.n_tokens, device=h.device).long()
            h = h[:, idx]
        elif H * W < self.n_tokens:
            pad = torch.zeros(B, self.n_tokens - H * W, C, device=h.device)
            h = torch.cat([h, pad], dim=1)
        return self.proj(h) + self.pos_embed


# ── MLP Baseline Policy ──

class MLPPolicy(nn.Module):
    """Simple MLP: flatten image → predict action token."""
    def __init__(self, in_channels=3, grid_size=16, n_actions=7, hidden=256):
        super().__init__()
        flat_dim = in_channels * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x.flatten(1))


# ── RT-1 Style: Transformer over Visual Tokens → Action Token ──

class RT1Policy(nn.Module):
    """RT-1 style: Vision → Transformer → Discretized action.

    Key idea: treat action prediction like language modeling.
    Visual tokens are like a "sentence" describing the scene,
    and the model predicts the next action "word" from a vocabulary.
    """
    def __init__(self, in_channels=3, d_model=64, n_heads=4,
                 n_layers=3, n_actions=7, n_visual_tokens=16):
        super().__init__()
        self.d_model = d_model
        self.vision_encoder = VisionEncoder(in_channels, d_model, n_visual_tokens)

        # Learnable [ACTION] query token (like [CLS])
        self.action_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer decoder: action query cross-attends to visual tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, n_actions)
        self.attn_weights = None

    def forward(self, x):
        """x: (B, C, H, W) → (B, n_actions)"""
        visual_tokens = self.vision_encoder(x)
        B = visual_tokens.shape[0]
        query = self.action_query.expand(B, -1, -1)

        # Manual decoding to capture attention weights
        attn_maps = []
        h = query
        D = self.d_model
        for layer in self.transformer.layers:
            q = layer.self_attn(h, h, h)[0]
            q = layer.norm1(h + q)

            # Capture cross-attention weights
            nhead = layer.multihead_attn.num_heads
            d_k = D // nhead
            q_proj_w = layer.multihead_attn.in_proj_weight[:D]
            k_proj_w = layer.multihead_attn.in_proj_weight[D:2*D]
            Q = F.linear(q, q_proj_w).view(B, 1, nhead, d_k).transpose(1, 2)
            K = F.linear(visual_tokens, k_proj_w).view(B, -1, nhead, d_k).transpose(1, 2)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
            attn_w = F.softmax(attn_scores, dim=-1)
            attn_maps.append(attn_w.detach())

            ca_out = layer.multihead_attn(q, visual_tokens, visual_tokens)[0]
            h = layer.norm2(q + ca_out)
            h2 = layer.linear2(layer.activation(layer.linear1(h)))
            h = layer.norm3(h + layer.dropout2(h2))

        self.attn_weights = attn_maps
        return self.action_head(h.squeeze(1))


# ── Training ──

def train_model(model, images, actions, n_epochs=100, lr=3e-4,
                batch_size=256, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    images_t = torch.FloatTensor(images).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    N = len(images_t)

    losses = []
    accs = []

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randint(0, N, (batch_size,))
        x = images_t[idx]
        y = actions_t[idx]

        logits = model(x)
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
            print(f"  Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs


def evaluate(model, images, actions, env, n_rollouts=50, device='cpu'):
    """Evaluate action accuracy and task success rate via rollouts."""
    model.eval()
    images_t = torch.FloatTensor(images).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    with torch.no_grad():
        logits = model(images_t)
        acc = (logits.argmax(-1) == actions_t).float().mean().item()

    successes = 0
    for _ in range(n_rollouts):
        while True:
            agent_pos = np.array([
                np.random.randint(2, env.grid_size - 2),
                np.random.randint(2, env.grid_size - 2)
            ], dtype=np.float32)
            target_pos = np.array([
                np.random.randint(2, env.grid_size - 2),
                np.random.randint(2, env.grid_size - 2)
            ], dtype=np.float32)
            if np.abs(agent_pos - target_pos).sum() > 3:
                break

        for step in range(env.max_steps):
            img = env.render(agent_pos, target_pos)
            img_t = torch.FloatTensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                action_idx = model(img_t).argmax(-1).item()
            dx, dy, grip = env.action_vocab[action_idx]
            agent_pos[0] = np.clip(agent_pos[0] + dx, 0, env.grid_size - 1)
            agent_pos[1] = np.clip(agent_pos[1] + dy, 0, env.grid_size - 1)

            if action_idx == 6:  # grip-close = reached
                successes += 1
                break

    return acc, successes / n_rollouts


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "83-rt-action"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = GridWorldEnv(grid_size=16, n_channels=3, max_steps=25)

    print("=== Generating Grid World Data ===")
    images, actions = env.generate_dataset(n_episodes=800)
    print(f"  Dataset: {len(images)} (image, action) pairs")
    print(f"  Action vocabulary size: {env.n_actions}")
    for i, name in enumerate(env.action_names):
        count = (actions == i).sum()
        print(f"    {name}: {count} samples ({count/len(actions)*100:.1f}%)")

    print("\n=== Training MLP Baseline ===")
    mlp = MLPPolicy(in_channels=3, grid_size=16, n_actions=env.n_actions, hidden=256).to(device)
    mlp_losses, mlp_accs = train_model(mlp, images, actions, n_epochs=150, device=device)

    print("\n=== Training RT-1 (Transformer) ===")
    rt1 = RT1Policy(
        in_channels=3, d_model=64, n_heads=4, n_layers=3,
        n_actions=env.n_actions, n_visual_tokens=16
    ).to(device)
    rt1_losses, rt1_accs = train_model(rt1, images, actions, n_epochs=150, device=device)

    print("\n=== Evaluating ===")
    mlp_action_acc, mlp_success = evaluate(mlp, images, actions, env, n_rollouts=50, device=device)
    rt1_action_acc, rt1_success = evaluate(rt1, images, actions, env, n_rollouts=50, device=device)

    print(f"  MLP:   Action Acc={mlp_action_acc:.4f}  Success Rate={mlp_success:.4f}")
    print(f"  RT-1:  Action Acc={rt1_action_acc:.4f}  Success Rate={rt1_success:.4f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 5

    mlp_loss_s = np.convolve(mlp_losses, np.ones(window)/window, mode='valid')
    rt1_loss_s = np.convolve(rt1_losses, np.ones(window)/window, mode='valid')
    axes[0].plot(mlp_loss_s, label='MLP Baseline', color='red')
    axes[0].plot(rt1_loss_s, label='RT-1 (Transformer)', color='blue')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    mlp_acc_s = np.convolve(mlp_accs, np.ones(window)/window, mode='valid')
    rt1_acc_s = np.convolve(rt1_accs, np.ones(window)/window, mode='valid')
    axes[1].plot(mlp_acc_s, label='MLP Baseline', color='red')
    axes[1].plot(rt1_acc_s, label='RT-1 (Transformer)', color='blue')
    axes[1].set_title("Action Prediction Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("RT-1: Vision-Transformer Action Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Evaluation comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(['MLP', 'RT-1'], [mlp_action_acc, rt1_action_acc],
                color=['red', 'blue'], alpha=0.7)
    axes[0].set_title("Action Prediction Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate([mlp_action_acc, rt1_action_acc]):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(['MLP', 'RT-1'], [mlp_success, rt1_success],
                color=['red', 'blue'], alpha=0.7)
    axes[1].set_title("Task Success Rate (Rollout)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate([mlp_success, rt1_success]):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("RT-1 vs MLP: Action Accuracy & Task Success", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "evaluation_comparison.png", dpi=150)
    plt.close()

    # 3. Attention over visual tokens
    rt1.eval()
    test_agent = np.array([4.0, 4.0])
    test_target = np.array([12.0, 10.0])
    test_img = env.render(test_agent, test_target)
    test_img_t = torch.FloatTensor(test_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = rt1(test_img_t)
        pred_action = logits.argmax(-1).item()

    if rt1.attn_weights:
        attn = rt1.attn_weights[-1].cpu().numpy()  # (1, nhead, 1, T)
        attn_avg = attn[0].mean(axis=0).squeeze(0)  # (T,)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(test_img.transpose(1, 2, 0).clip(0, 1))
        axes[0].set_title("Grid World Observation\n(Red=Agent, Green=Target, Blue=Dist)")
        axes[0].axis('off')

        n_tokens = len(attn_avg)
        axes[1].bar(range(n_tokens), attn_avg, color='blue', alpha=0.7)
        axes[1].set_title("Attention Weights over Visual Tokens")
        axes[1].set_xlabel("Visual Token Index")
        axes[1].set_ylabel("Attention Weight")
        axes[1].grid(True, alpha=0.3)

        grid_size = int(np.sqrt(n_tokens))
        if grid_size * grid_size <= n_tokens:
            attn_spatial = attn_avg[:grid_size * grid_size].reshape(grid_size, grid_size)
            axes[2].imshow(attn_spatial, cmap='hot', interpolation='nearest')
            axes[2].set_title("Attention Map (Spatial)\nBright = More Attention")
            axes[2].axis('off')
            plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046)

        pred_name = env.action_names[pred_action]
        plt.suptitle(f"RT-1 Attention: Predicted Action = '{pred_name}'", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / "attention_visualization.png", dpi=150)
        plt.close()

    # 4. Action vocabulary visualization
    fig, axes = plt.subplots(1, 7, figsize=(18, 3))
    for idx in range(env.n_actions):
        ax = axes[idx]
        dx, dy, grip = env.action_vocab[idx]
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        if dx != 0 or dy != 0:
            ax.arrow(0, 0, dy * 0.8, dx * 0.8, head_width=0.2,
                     head_length=0.15, fc='blue', ec='black')
        elif grip == 1:
            ax.plot([-0.4, 0, 0.4], [-0.3, 0.1, -0.3], 'r-o', linewidth=2)
        else:
            ax.plot([-0.4, 0, 0.4], [0.3, -0.1, 0.3], 'g-o', linewidth=2)
        ax.set_title(f"{idx}: {env.action_names[idx]}", fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle("RT-1: Discretized Action Vocabulary (7 Tokens)", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "action_vocabulary.png", dpi=150)
    plt.close()

    # 5. Rollout trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model, name, color in [
        (axes[0], mlp, "MLP", "red"),
        (axes[1], rt1, "RT-1", "blue")
    ]:
        model.eval()
        while True:
            agent_pos = np.array([
                np.random.randint(2, env.grid_size - 2),
                np.random.randint(2, env.grid_size - 2)
            ], dtype=np.float32)
            target_pos = np.array([
                np.random.randint(2, env.grid_size - 2),
                np.random.randint(2, env.grid_size - 2)
            ], dtype=np.float32)
            if np.abs(agent_pos - target_pos).sum() > 3:
                break

        trajectory = [agent_pos.copy()]
        for step in range(env.max_steps):
            img = env.render(agent_pos, target_pos)
            img_t = torch.FloatTensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                action_idx = model(img_t).argmax(-1).item()
            dx, dy, _ = env.action_vocab[action_idx]
            agent_pos[0] = np.clip(agent_pos[0] + dx, 0, env.grid_size - 1)
            agent_pos[1] = np.clip(agent_pos[1] + dy, 0, env.grid_size - 1)
            trajectory.append(agent_pos.copy())
            if action_idx == 6:
                break

        traj = np.array(trajectory)
        ax.imshow(test_img[2] if step == 0 else env.render(agent_pos, target_pos)[2],
                  cmap='Blues', alpha=0.2, extent=[-0.5, env.grid_size-0.5, env.grid_size-0.5, -0.5])
        ax.plot(traj[:, 1], traj[:, 0], '-o', color=color, markersize=3,
                linewidth=1.5, label='Agent')
        ax.plot(target_pos[1], target_pos[0], '*', color='green', markersize=20, label='Target')
        ax.plot(traj[0, 1], traj[0, 0], 's', color='orange', markersize=12, label='Start')
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(env.grid_size - 0.5, -0.5)
        ax.set_title(f"{name} Rollout ({len(trajectory)-1} steps)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle("RT-1 vs MLP: Reaching Task Rollouts", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "rollout_trajectories.png", dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Traditional\nRobotics", "Hand-crafted\nperception pipeline\n→ Separate planner\n→ No generalization", 0.15, 'red'),
        ("RT-1\n(2212.04356)", "Vision → Transformer\n→ Discretized actions\nTreat actions as tokens\nLike language modeling", 0.50, 'blue'),
        ("RT-2\n(2307.15818)", "VLM pre-training\n→ Action fine-tuning\nWeb knowledge → Robot\nTransfer from language", 0.85, 'purple'),
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

    ax.set_title("From Traditional Robotics to RT-1/RT-2: Action Tokenization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "rt_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
