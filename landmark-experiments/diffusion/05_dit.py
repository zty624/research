"""
Minimal DiT (Diffusion Transformer) Reproduction
==================================================
Reproduces the core ideas from "Scalable Diffusion Models with Transformers" (2212.09748):
1. Patchify latent into tokens
2. Transformer backbone with adaLN-Zero for condition injection
3. Compare adaLN-Zero vs in-context conditioning
4. Train on 2D point clouds for visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── adaLN-Zero Block ──

class DiTBlockAdaLN(nn.Module):
    """DiT block with adaptive layer norm + zero initialization."""
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )
        # adaLN modulation: 6 outputs per block (γ1, β1, α1, γ2, β2, α2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )
        # Zero init for the last layer of adaLN
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        """
        x: (B, T, D) - token sequence
        c: (B, D) - conditioning (timestep embedding)
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        # All (B, D) → unsqueeze to (B, 1, D)
        shift1, scale1, gate1 = shift1.unsqueeze(1), scale1.unsqueeze(1), gate1.unsqueeze(1)
        shift2, scale2, gate2 = shift2.unsqueeze(1), scale2.unsqueeze(1), gate2.unsqueeze(1)

        # Modulated norm: (1 + scale) * norm(x) + shift
        h = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(h, h, h)
        x = x + gate1 * attn_out

        h = self.norm2(x) * (1 + scale2) + shift2
        mlp_out = self.mlp(h)
        x = x + gate2 * mlp_out

        return x


class DiTBlockInContext(nn.Module):
    """DiT block with in-context conditioning (concatenate c as a token)."""
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ── Full DiT Model ──

class DiT(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=4, cond_type='adaln'):
        super().__init__()
        self.cond_type = cond_type
        self.d_model = d_model

        # Timestep embedding (sinusoidal + MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Patchify: project 2D points to d_model
        self.input_proj = nn.Linear(2, d_model)

        if cond_type == 'adaln':
            self.blocks = nn.ModuleList([
                DiTBlockAdaLN(d_model, n_heads) for _ in range(n_layers)
            ])
        else:  # in-context
            self.blocks = nn.ModuleList([
                DiTBlockInContext(d_model, n_heads) for _ in range(n_layers)
            ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)  # Predict noise (2D)

    def timestep_embedding(self, t, max_period=10000):
        half = self.d_model // 2
        freqs = torch.exp(-torch.arange(half) * (np.log(max_period) / half)).to(t.device)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, t):
        """
        x: (B, N, 2) - noisy points as tokens
        t: (B,) - timesteps
        """
        t_emb = self.time_mlp(self.timestep_embedding(t))
        h = self.input_proj(x)  # (B, N, 2) → (B, N, d_model)

        if self.cond_type == 'adaln':
            for block in self.blocks:
                h = block(h, t_emb)
        else:
            # In-context: prepend timestep token
            t_token = t_emb.unsqueeze(1)  # (B, 1, D)
            h = torch.cat([t_token, h], dim=1)
            for block in self.blocks:
                h = block(h)
            h = h[:, 1:]  # Remove conditioning token

        return self.output_proj(self.final_norm(h))


# ── Data: 2D Mixture ──

def sample_2d_data(n, device='cpu'):
    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2
    c1 = torch.randn(n1, 2, device=device) * 0.3 + torch.tensor([2.0, 0.0], device=device)
    c2 = torch.randn(n2, 2, device=device) * 0.3 + torch.tensor([-1.0, 1.7], device=device)
    c3 = torch.randn(n3, 2, device=device) * 0.3 + torch.tensor([-1.0, -1.7], device=device)
    return torch.cat([c1, c2, c3])


# ── Training (DDPM-style) ──

def train_dit(model, n_steps=5000, batch_size=256, n_points=32, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        # Sample data points and add noise
        data = sample_2d_data(batch_size * n_points, device).reshape(batch_size, n_points, 2)
        t = torch.randint(0, 1000, (batch_size,), device=device).float() / 1000

        # Forward diffusion: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        alpha = (1 - t.unsqueeze(-1).unsqueeze(-1)).clamp(min=0.01)
        noise = torch.randn_like(data)
        x_t = torch.sqrt(alpha) * data + torch.sqrt(1 - alpha) * noise

        # Predict noise
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.6f}")

    return losses


def sample_dit(model, n_samples=500, n_steps=50, device='cpu'):
    """DDPM-style sampling."""
    x = torch.randn(n_samples, 1, 2, device=device)
    for i in reversed(range(n_steps)):
        t = torch.full((n_samples,), i / n_steps, device=device)
        with torch.no_grad():
            pred_noise = model(x, t)
            # Simple denoising step
            alpha = (1 - t.unsqueeze(-1).unsqueeze(-1)).clamp(min=0.01)
            x = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            if i > 0:
                x = x + 0.1 * torch.randn_like(x)
    return x.squeeze(1)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "05-dit"
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = 64
    n_heads = 4
    n_layers = 4

    # Train adaLN-Zero DiT
    print("=== Training DiT with adaLN-Zero ===")
    model_adaln = DiT(d_model, n_heads, n_layers, cond_type='adaln').to(device)
    losses_adaln = train_dit(model_adaln, n_steps=5000, device=device)

    # Train in-context DiT
    print("\n=== Training DiT with In-Context Conditioning ===")
    model_inctx = DiT(d_model, n_heads, n_layers, cond_type='in_context').to(device)
    losses_inctx = train_dit(model_inctx, n_steps=5000, device=device)

    # ── Visualization ──

    # Loss comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    window = 50
    loss_adaln_s = np.convolve(losses_adaln, np.ones(window)/window, mode='valid')
    loss_inctx_s = np.convolve(losses_inctx, np.ones(window)/window, mode='valid')
    ax.plot(loss_adaln_s, label='adaLN-Zero', color='blue')
    ax.plot(loss_inctx_s, label='In-Context', color='red')
    ax.set_title("DiT: adaLN-Zero vs In-Context Conditioning")
    ax.set_xlabel("Step")
    ax.set_ylabel("Noise Prediction Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "loss_comparison.png", dpi=150)
    plt.close()

    # Generated samples
    target = sample_2d_data(1000, device).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(target[:, 0], target[:, 1], alpha=0.2, s=3)
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    for ax, model, title in [(axes[1], model_adaln, 'DiT + adaLN-Zero'),
                              (axes[2], model_inctx, 'DiT + In-Context')]:
        samples = sample_dit(model, n_samples=1000, n_steps=50, device=device).cpu().numpy()
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=3, color='green')
        ax.set_title(title)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "generated_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
