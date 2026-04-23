"""
Minimal EMA (Exponential Moving Average) Reproduction
======================================================
Reproduces the EMA technique used in diffusion models, BYOL, and stable training:
1. EMA: θ_ema = β * θ_ema + (1 - β) * θ (smooth averaging of weights)
2. Used in DDPM, BYOL, MoCo, etc. for stable generation / representation learning
3. EMA model produces smoother, higher-quality outputs
4. Compare: raw model vs EMA model at different β values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy


# ── EMA ──

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        """Apply EMA parameters to model."""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name])


# ── Models ──

class DiffusionNet(nn.Module):
    """Simple denoising network for 1D data."""
    def __init__(self, data_dim=2, hidden=128, n_layers=4):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


class Classifier(nn.Module):
    """Simple MLP classifier."""
    def __init__(self, in_dim=784, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x.flatten(1))


# ── Diffusion Process ──

def add_noise(x, t, noise_schedule):
    """Add noise according to schedule."""
    alpha = noise_schedule[t]
    noise = torch.randn_like(x)
    return alpha * x + (1 - alpha) * noise, noise


# ── Training ──

def train_diffusion(model, ema, data, n_steps=3000, lr=2e-4, beta=0.999, device='cpu'):
    """Train diffusion model with EMA."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses_raw = []
    losses_ema = []

    # Noise schedule (linear)
    T = 100
    noise_schedule = torch.linspace(1.0, 0.01, T, device=device)

    eval_every = 100

    for step in range(n_steps):
        idx = torch.randint(0, len(data), (256,))
        x = data[idx].to(device)
        t = torch.randint(0, T, (len(x),), device=device)

        # Add noise
        alpha = noise_schedule[t].unsqueeze(-1)
        noise = torch.randn_like(x)
        noisy_x = alpha * x + (1 - alpha) * noise

        # Predict noise
        pred_noise = model(noisy_x, t.float() / T)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA
        ema.update(model)

        losses_raw.append(loss.item())

        # Evaluate EMA loss periodically
        if (step + 1) % eval_every == 0:
            backup = ema.apply(model)
            with torch.no_grad():
                idx2 = torch.randint(0, len(data), (256,))
                x2 = data[idx2].to(device)
                t2 = torch.randint(0, T, (len(x2),), device=device)
                alpha2 = noise_schedule[t2].unsqueeze(-1)
                noise2 = torch.randn_like(x2)
                noisy_x2 = alpha2 * x2 + (1 - alpha2) * noise2
                pred2 = model(noisy_x2, t2.float() / T)
                ema_loss = F.mse_loss(pred2, noise2).item()
            ema.restore(model, backup)
            losses_ema.append((step, ema_loss))

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Raw Loss: {loss.item():.4f}")

    return losses_raw, losses_ema


def train_classifier(model, ema, train_loader, test_loader, n_epochs=15, lr=1e-3, device='cpu'):
    """Train classifier with EMA tracking."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    raw_accs = []
    ema_accs = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate raw model
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        raw_accs.append(correct / total)

        # Evaluate EMA model
        backup = ema.apply(model)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        ema_accs.append(correct / total)
        ema.restore(model, backup)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | "
                  f"Raw: {raw_accs[-1]:.4f} | EMA: {ema_accs[-1]:.4f}")

    return train_losses, raw_accs, ema_accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "49-ema"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Diffusion model with EMA
    print("=== Diffusion Model with EMA ===")
    np.random.seed(42)
    # Create 2D mixture of Gaussians
    n_data = 3000
    centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    data = []
    for c in centers:
        data.append(np.random.randn(n_data // 4, 2) * 0.3 + c)
    data = torch.tensor(np.vstack(data), dtype=torch.float32)

    diff_model = DiffusionNet(data_dim=2, hidden=128).to(device)
    ema = EMA(diff_model, decay=0.999)
    raw_losses, ema_losses = train_diffusion(diff_model, ema, data, n_steps=3000, device=device)

    # Generate samples from raw and EMA models
    print("\n  Generating samples...")
    T = 100
    noise_schedule = torch.linspace(1.0, 0.01, T, device=device)

    @torch.no_grad()
    def generate(model, n_samples=500, n_steps=50):
        x = torch.randn(n_samples, 2, device=device)
        for t_idx in reversed(range(0, T, T // n_steps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.float32) / T
            pred_noise = model(x, t)
            alpha = noise_schedule[t_idx]
            x = (x - (1 - alpha) / torch.sqrt(1 - alpha**2 + 1e-8) * pred_noise) / (alpha + 1e-8)
            if t_idx > 0:
                x = x + torch.randn_like(x) * 0.05
        return x.cpu().numpy()

    raw_samples = generate(diff_model)

    backup = ema.apply(diff_model)
    ema_samples = generate(diff_model)
    ema.restore(diff_model, backup)

    # Experiment 2: Classifier with EMA
    print("\n=== Classifier with EMA ===")
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(10000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    clf = Classifier(hidden=256).to(device)
    ema_clf = EMA(clf, decay=0.999)
    clf_losses, raw_accs, ema_accs = train_classifier(clf, ema_clf, train_loader, test_loader,
                                                       n_epochs=15, device=device)

    # Experiment 3: EMA decay sensitivity
    print("\n=== EMA Decay Sensitivity ===")
    decay_results = {}
    for decay in [0.9, 0.99, 0.999, 0.9999]:
        print(f"  β={decay}:")
        m = Classifier(hidden=256).to(device)
        ema_m = EMA(m, decay=decay)
        _, raw_a, ema_a = train_classifier(m, ema_m, train_loader, test_loader,
                                            n_epochs=15, device=device)
        decay_results[decay] = {'raw': raw_a, 'ema': ema_a, 'final_raw': raw_a[-1], 'final_ema': ema_a[-1]}
        print(f"    Raw: {raw_a[-1]:.4f}, EMA: {ema_a[-1]:.4f}")

    # ── Visualization ──

    # 1. Generated samples comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.1, s=1, color='gray')
    axes[0].set_title("Training Data")

    axes[1].scatter(raw_samples[:, 0], raw_samples[:, 1], alpha=0.3, s=3, color='red')
    axes[1].set_title("Raw Model Samples")

    axes[2].scatter(ema_samples[:, 0], ema_samples[:, 1], alpha=0.3, s=3, color='blue')
    axes[2].set_title("EMA Model Samples")

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle("EMA: Smoother, Higher-Quality Generation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "diffusion_ema.png", dpi=150)
    plt.close()

    # 2. Classifier accuracy: raw vs EMA
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(raw_accs, label='Raw Model', color='red')
    axes[0].plot(ema_accs, label='EMA Model', color='blue')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Classifier: Raw vs EMA Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Decay sensitivity
    decays = sorted(decay_results.keys())
    raw_finals = [decay_results[d]['final_raw'] for d in decays]
    ema_finals = [decay_results[d]['final_ema'] for d in decays]

    x = np.arange(len(decays))
    width = 0.35
    axes[1].bar(x - width/2, raw_finals, width, label='Raw', color='red', alpha=0.7)
    axes[1].bar(x + width/2, ema_finals, width, label='EMA', color='blue', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'β={d}' for d in decays])
    axes[1].set_ylabel("Final Test Accuracy")
    axes[1].set_title("EMA Decay Sensitivity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("EMA: Exponential Moving Average for Stable Models", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "ema_comparison.png", dpi=150)
    plt.close()

    # 3. EMA weight tracking
    fig, ax = plt.subplots(figsize=(10, 5))

    # Show how EMA smooths a weight trajectory
    n_steps = 200
    raw_weight = np.sin(np.linspace(0, 8*np.pi, n_steps)) + np.random.randn(n_steps) * 0.3
    for beta in [0.9, 0.99, 0.999]:
        ema_weight = np.zeros(n_steps)
        ema_weight[0] = raw_weight[0]
        for i in range(1, n_steps):
            ema_weight[i] = beta * ema_weight[i-1] + (1 - beta) * raw_weight[i]
        ax.plot(ema_weight, label=f'EMA β={beta}', linewidth=2)

    ax.plot(raw_weight, alpha=0.3, label='Raw weight', color='gray')
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Weight Value")
    ax.set_title("EMA: Smoothing Weight Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "ema_smoothing.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Training\nNoise", "Weights oscillate\nduring training\nNoisy outputs\n→ Inconsistent\n   quality", 0.14, 'red'),
        ("EMA\nSmoothing", "θ_ema = βθ_ema + (1-β)θ\nSmooth averaging\nof weight history\nβ=0.999 typical\n→ Stable outputs", 0.5, 'blue'),
        ("Where Used", "DDPM/DiT:\n  denoiser quality\nBYOL/MoCo:\n  target network\nStable Diffusion:\n  generation quality", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("EMA: Simple but Essential for Stable Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "ema_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
