"""
Minimal Classifier-Free Guidance Reproduction
===============================================
Reproduces core ideas from Classifier-Free Guidance (2207.12598, Ho & Salimans):
1. Conditional generation without a separate classifier model
2. Train with condition dropout: randomly drop condition during training
3. At inference: interpolate between conditional and unconditional predictions
4. ê(x) = (1+w)·ε_θ(x,c) - w·ε_θ(x,∅)
5. w > 1 amplifies the effect of the condition (sharper, more faithful generation)
6. Used in DALL-E 2, Stable Diffusion, Imagen, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Models ──

class ConditionalDenoiser(nn.Module):
    """Simple denoising network with class conditioning."""
    def __init__(self, data_dim=2, n_classes=10, hidden=128):
        super().__init__()
        self.class_embed = nn.Embedding(n_classes + 1, hidden)  # +1 for null class
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim)
        )
        self.null_class = n_classes  # Index for null/unconditional

    def forward(self, x, t, c=None, drop_prob=0.0):
        """x: noisy data, t: timestep, c: class labels, drop_prob: condition dropout rate."""
        t_emb = self.time_embed(t.unsqueeze(-1))

        # Class embedding with dropout
        if c is not None:
            if self.training and drop_prob > 0:
                # Randomly drop condition
                mask = torch.rand(c.shape[0], device=c.device) > drop_prob
                c_drop = torch.where(mask, c, torch.full_like(c, self.null_class))
                c_emb = self.class_embed(c_drop)
            else:
                c_emb = self.class_embed(c)
        else:
            # Unconditional
            c_emb = self.class_embed(torch.full((x.shape[0],), self.null_class,
                                                  dtype=torch.long, device=x.device))

        h = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(h)


# ── Diffusion Process ──

class DiffusionProcess:
    def __init__(self, T=100, beta_start=0.0001, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.betas = betas

    def add_noise(self, x, t):
        """Add noise to x at timestep t."""
        alpha_bar = self.alpha_bar.to(x.device)[t].unsqueeze(-1)
        noise = torch.randn_like(x)
        noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
        return noisy, noise


# ── Training ──

def train_diffusion(model, data, labels, diffusion, n_steps=5000, lr=2e-4,
                    drop_prob=0.1, device='cpu'):
    """Train with classifier-free guidance (condition dropout)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, len(data), (256,))
        x = data[idx].to(device)
        c = labels[idx].to(device)

        # Sample timestep
        t = torch.randint(0, diffusion.T, (len(x),), device=device)

        # Add noise
        noisy_x, noise = diffusion.add_noise(x, t)

        # Predict noise (with condition dropout during training)
        pred = model(noisy_x, t.float() / diffusion.T, c, drop_prob=drop_prob)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f}")

    return losses


# ── Sampling with Classifier-Free Guidance ──

@torch.no_grad()
def sample_cfg(model, diffusion, n_samples=200, class_label=0, guidance_w=1.0,
               n_steps=50, device='cpu'):
    """Sample with classifier-free guidance.
    guidance_w: guidance scale. w=1 is normal conditional, w>1 amplifies condition.
    """
    model.eval()
    x = torch.randn(n_samples, 2, device=device)

    for t_idx in reversed(range(0, diffusion.T, diffusion.T // n_steps)):
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.float32) / diffusion.T

        # Conditional prediction
        c = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
        eps_cond = model(x, t, c)

        if guidance_w != 1.0:
            # Unconditional prediction
            eps_uncond = model(x, t, c=None)

            # Classifier-free guidance: ê = (1+w)·ε_cond - w·ε_uncond
            eps_pred = (1 + guidance_w) * eps_cond - guidance_w * eps_uncond
        else:
            eps_pred = eps_cond

        # DDPM-style update
        alpha = diffusion.alphas[t_idx].to(device)
        alpha_bar = diffusion.alpha_bar[t_idx].to(device)

        x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) * eps_pred) / (torch.sqrt(alpha) + 1e-8)
        if t_idx > 0:
            x = x + torch.randn_like(x) * 0.05

    model.train()
    return x.cpu().numpy()


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "55-cfg"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create conditional 2D data: 4 classes, each a different Gaussian
    print("=== Generating Conditional Data ===")
    np.random.seed(42)
    n_per_class = 750
    class_centers = {0: [1, 1], 1: [-1, 1], 2: [-1, -1], 3: [1, -1]}
    class_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}

    data_list = []
    labels_list = []
    for cls, center in class_centers.items():
        pts = np.random.randn(n_per_class, 2) * 0.3 + center
        data_list.append(pts)
        labels_list.extend([cls] * n_per_class)

    data = torch.tensor(np.vstack(data_list), dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.long)

    print(f"  Data: {data.shape}, Classes: {len(class_centers)}")

    # Train diffusion model
    print("\n=== Training Conditional Diffusion ===")
    diffusion = DiffusionProcess(T=100, beta_start=0.0001, beta_end=0.02)
    model = ConditionalDenoiser(data_dim=2, n_classes=4, hidden=128).to(device)
    train_losses = train_diffusion(model, data, labels, diffusion,
                                    n_steps=5000, drop_prob=0.1, device=device)

    # Experiment 1: Different guidance scales
    print("\n=== Sampling with Different Guidance Scales ===")
    guidance_scales = [0.0, 1.0, 2.0, 3.0, 5.0]
    target_class = 0

    sample_results = {}
    for w in guidance_scales:
        print(f"  w={w}:")
        if w == 0.0:
            # Pure unconditional
            samples = sample_cfg(model, diffusion, n_samples=300, class_label=target_class,
                                guidance_w=0.0, device=device)
        else:
            samples = sample_cfg(model, diffusion, n_samples=300, class_label=target_class,
                                guidance_w=w, device=device)
        sample_results[w] = samples

        # Compute statistics
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        target = np.array(class_centers[target_class])
        dist = np.linalg.norm(mean - target)
        print(f"    Mean: ({mean[0]:.2f}, {mean[1]:.2f}), Std: ({std[0]:.2f}, {std[1]:.2f}), "
              f"Dist from target: {dist:.2f}")

    # Experiment 2: Class-conditional generation
    print("\n=== Class-Conditional Generation (w=3.0) ===")
    class_samples = {}
    for cls in range(4):
        samples = sample_cfg(model, diffusion, n_samples=200, class_label=cls,
                            guidance_w=3.0, device=device)
        class_samples[cls] = samples
        mean = samples.mean(axis=0)
        target = np.array(class_centers[cls])
        print(f"  Class {cls}: mean=({mean[0]:.2f}, {mean[1]:.2f}), target=({target[0]}, {target[1]})")

    # ── Visualization ──

    # 1. Guidance scale effect
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(20, 4))

    # Plot training data in background
    for idx, w in enumerate(guidance_scales):
        ax = axes[idx]
        for cls, center in class_centers.items():
            mask = labels.numpy() == cls
            ax.scatter(data[mask, 0].numpy(), data[mask, 1].numpy(),
                      alpha=0.05, s=1, color=class_colors[cls])

        # Plot generated samples
        samples = sample_results[w]
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5,
                  color='black', label=f'w={w}')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f"Guidance w={w}")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Classifier-Free Guidance: Target Class {target_class}", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "guidance_scale.png", dpi=150)
    plt.close()

    # 2. Class-conditional generation
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for cls in range(4):
        ax = axes[cls]
        for c, center in class_centers.items():
            mask = labels.numpy() == c
            ax.scatter(data[mask, 0].numpy(), data[mask, 1].numpy(),
                      alpha=0.05, s=1, color=class_colors[c])

        samples = class_samples[cls]
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5,
                  color=class_colors[cls])

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f"Generated Class {cls}")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Classifier-Free Guidance: Conditional Generation (w=3.0)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "class_conditional.png", dpi=150)
    plt.close()

    # 3. Training loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, alpha=0.3, color='gray')
    # Smoothed
    window = 50
    smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='blue', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (with condition dropout p=0.1)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_loss.png", dpi=150)
    plt.close()

    # 4. Guidance scale vs quality metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distance from target
    ws = guidance_scales
    dists = []
    stds = []
    for w in ws:
        s = sample_results[w]
        target = np.array(class_centers[target_class])
        dists.append(np.linalg.norm(s.mean(axis=0) - target))
        stds.append(s.std(axis=0).mean())

    axes[0].plot(ws, dists, 'o-', color='red')
    axes[0].set_xlabel("Guidance Scale (w)")
    axes[0].set_ylabel("Distance from Target Center")
    axes[0].set_title("Generation Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    axes[1].plot(ws, stds, 'o-', color='blue')
    axes[1].set_xlabel("Guidance Scale (w)")
    axes[1].set_ylabel("Sample Std Dev")
    axes[1].set_title("Sample Diversity (lower = more concentrated)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Classifier-Free Guidance: Accuracy vs Diversity Trade-off", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "guidance_tradeoff.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Training\n(Dropout)", "Randomly drop condition\nwith probability p\nModel learns both:\nε(x,c) and ε(x,∅)\n→ Joint training", 0.14, 'red'),
        ("Inference\n(Guidance)", "ê = (1+w)·ε(x,c) - w·ε(x,∅)\nInterpolate conditional\nand unconditional\nw=1: normal cond.\nw>1: amplified cond.", 0.5, 'blue'),
        ("Effect of\nw > 1", "Sharper generation\nMore faithful to prompt\nLess diversity\nBetter image-text\n   alignment\n→ SD/DALL-E standard", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Classifier-Free Guidance: No Classifier Needed for Conditional Generation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "cfg_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
