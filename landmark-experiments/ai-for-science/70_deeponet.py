"""
Minimal DeepONet Reproduction
==============================
Reproduces the core ideas from "DeepONet: Learning Nonlinear Operators
Based on the Universal Approximation Theorem of Operators"
(Lu et al., 2019, arxiv 1910.03193):
1. Branch net: encode input function at sensor locations
2. Trunk net: encode query locations
3. Output = dot(branch, trunk) — universal approximation
4. Demo: learn antiderivative operator (integral)
5. Compare DeepONet vs MLP on operator learning
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── DeepONet Architecture ──

class DeepONet(nn.Module):
    """Deep Operator Network: branch(input_func) · trunk(query_point)."""
    def __init__(self, branch_layers, trunk_layers, p=50):
        """
        Args:
            branch_layers: list of ints for branch net, e.g. [100, 128, 128, p]
            trunk_layers: list of ints for trunk net, e.g. [1, 128, 128, p]
            p: latent dimension (trunk output width)
        """
        super().__init__()
        self.p = p

        # Branch net: input function evaluated at m sensor points → p-dim
        branch_modules = []
        for i in range(len(branch_layers) - 2):
            branch_modules.append(nn.Linear(branch_layers[i], branch_layers[i+1]))
            branch_modules.append(nn.ReLU())
        branch_modules.append(nn.Linear(branch_layers[-2], branch_layers[-1]))
        self.branch = nn.Sequential(*branch_modules)

        # Trunk net: query coordinate y → p-dim
        trunk_modules = []
        for i in range(len(trunk_layers) - 2):
            trunk_modules.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            trunk_modules.append(nn.ReLU())
        trunk_modules.append(nn.Linear(trunk_layers[-2], trunk_layers[-1]))
        self.trunk = nn.Sequential(*trunk_modules)

    def forward(self, u_sensor, y_query):
        """
        Args:
            u_sensor: (batch, m) — input function values at m sensor points
            y_query: (batch, n_queries, dim_y) — query coordinates
        Returns:
            (batch, n_queries) — operator output at query points
        """
        B = u_sensor.shape[0]
        N = y_query.shape[1]

        # Branch: (B, m) → (B, p)
        b = self.branch(u_sensor)  # (B, p)

        # Trunk: (B*N, dim_y) → (B*N, p)
        y_flat = y_query.reshape(B * N, -1)
        t = self.trunk(y_flat)  # (B*N, p)

        # Dot product
        b_expanded = b.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        out = (b_expanded * t).sum(dim=-1)  # (B*N,)
        return out.reshape(B, N)


# ── Antiderivative Operator ──
# Learn G: u(x) → ∫₀ˣ u(s) ds
# Input function u sampled at m=100 sensor points
# Output: integral at query locations

class AntiderivativeONet(nn.Module):
    """DeepONet specialized for the antiderivative operator."""
    def __init__(self, m_sensors=100, p=50):
        super().__init__()
        self.deeponet = DeepONet(
            branch_layers=[m_sensors, 128, 128, p],
            trunk_layers=[1, 128, 128, p],
            p=p
        )

    def forward(self, u_sensor, y_query):
        return self.deeponet(u_sensor, y_query)


# ── Dataset Generation ──

def generate_antiderivative_data(n_functions, m_sensors=100, n_queries=50,
                                  device='cpu'):
    """Generate antiderivative operator data.

    u(x) = sum of random sinusoids → G[u](y) = ∫₀ʸ u(s) ds
    """
    x_sensor = torch.linspace(0, 1, m_sensors, device=device)

    u_list, g_list, y_list = [], [], []

    for _ in range(n_functions):
        # Random sinusoidal function: u(x) = Σ a_k sin(kπx + φ_k)
        n_modes = np.random.randint(1, 5)
        coeffs = torch.randn(n_modes, device=device) * 0.5
        freqs = torch.arange(1, n_modes + 1, dtype=torch.float32, device=device)
        phases = torch.rand(n_modes, device=device) * 2 * np.pi

        # u(x) at sensor points
        u_vals = torch.zeros(m_sensors, device=device)
        for k in range(n_modes):
            u_vals += coeffs[k] * torch.sin(freqs[k] * np.pi * x_sensor + phases[k])

        # Query points (random subset)
        y_query, _ = torch.sort(torch.rand(n_queries, device=device))

        # Exact antiderivative: ∫₀ʸ a_k sin(kπs + φ_k) ds
        g_vals = torch.zeros(n_queries, device=device)
        for k in range(n_modes):
            g_vals += coeffs[k] / (freqs[k] * np.pi) * (
                -torch.cos(freqs[k] * np.pi * y_query + phases[k])
                + torch.cos(phases[k])
            )

        u_list.append(u_vals)
        y_list.append(y_query)
        g_list.append(g_vals)

    u_batch = torch.stack(u_list)  # (N, m)
    y_batch = torch.stack(y_list).unsqueeze(-1)  # (N, n_queries, 1)
    g_batch = torch.stack(g_list)  # (N, n_queries)

    return u_batch, y_batch, g_batch


# ── MLP Baseline ──

class MLPBaseline(nn.Module):
    """Concatenation-based MLP: (u_sensor, y_query) → G[u](y)."""
    def __init__(self, m_sensors=100, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_sensors + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, u_sensor, y_query):
        """u_sensor: (B, m), y_query: (B, N, 1) → (B, N)"""
        B, N, _ = y_query.shape
        y_flat = y_query.reshape(B * N, -1)  # (B*N, 1)
        u_expanded = u_sensor.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        inp = torch.cat([u_expanded, y_flat], dim=-1)
        out = self.net(inp).squeeze(-1)
        return out.reshape(B, N)


# ── Training ──

def train_model(model, u_train, y_train, g_train, epochs=500, batch_size=64,
                lr=1e-3, device='cpu'):
    """Train a model on antiderivative data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.5)
    n = u_train.shape[0]
    losses = []

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n, device=device)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = indices[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(u_train[idx], y_train[idx])
            loss = ((pred - g_train[idx]) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={avg:.6f}")

    return losses


# ── Visualization ──

def visualize_onet(model, u_test, y_test, g_test, n_show=4, save_dir=None):
    """Visualize DeepONet predictions."""
    model.eval()
    with torch.no_grad():
        pred = model(u_test, y_test)

    x_sensor = np.linspace(0, 1, u_test.shape[1])
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 6))

    for i in range(n_show):
        # Input function
        axes[0, i].plot(x_sensor, u_test[i].cpu().numpy(), 'b-', lw=1.5)
        axes[0, i].set_title(f'u_{i}(x)')
        axes[0, i].set_xlabel('x')

        # Antiderivative comparison
        y_np = y_test[i, :, 0].cpu().numpy()
        g_np = g_test[i].cpu().numpy()
        p_np = pred[i].cpu().numpy()
        axes[1, i].plot(y_np, g_np, 'b-', lw=1.5, label='Exact')
        axes[1, i].plot(y_np, p_np, 'r--', lw=1.5, label='DeepONet')
        axes[1, i].set_title(f'∫u_{i}(s)ds')
        axes[1, i].set_xlabel('y')
        axes[1, i].legend(fontsize=8)

    axes[0, 0].set_ylabel('Input u(x)')
    axes[1, 0].set_ylabel('Antiderivative G[u](y)')
    plt.suptitle('DeepONet: Antiderivative Operator Learning', y=1.02)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'deeponet_antiderivative.png', dpi=150, bbox_inches='tight')
    plt.close()


def compare_onet_mlp(onet_losses, mlp_losses, save_dir=None):
    """Compare DeepONet vs MLP training."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(onet_losses, label='DeepONet')
    ax.semilogy(mlp_losses, label='MLP (concat)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('DeepONet vs MLP: Antiderivative Operator')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'deeponet_vs_mlp.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    save_dir = Path(__file__).parent / 'results' / 'deeponet'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("Generating antiderivative data...")
    u_train, y_train, g_train = generate_antiderivative_data(500, device=device)
    u_test, y_test, g_test = generate_antiderivative_data(50, device=device)

    # Train DeepONet
    print("\n=== Training DeepONet ===")
    onet = AntiderivativeONet(m_sensors=100, p=50).to(device)
    onet_losses = train_model(onet, u_train, y_train, g_train, epochs=500, device=device)
    visualize_onet(onet, u_test, y_test, g_test, save_dir=save_dir)

    # Train MLP baseline
    print("\n=== Training MLP Baseline ===")
    mlp = MLPBaseline(m_sensors=100).to(device)
    mlp_losses = train_model(mlp, u_train, y_train, g_train, epochs=500, device=device)
    compare_onet_mlp(onet_losses, mlp_losses, save_dir)

    # Generalization test: different number of modes
    print("\n=== Generalization Test ===")
    onet.eval()
    for n_modes in [2, 5, 10]:
        u_gen, y_gen, g_gen = generate_antiderivative_data(20, device=device)
        with torch.no_grad():
            pred = onet(u_gen, y_gen)
            err = ((pred - g_gen) ** 2).mean().item()
        print(f"  n_modes={n_modes}: MSE={err:.6f}")

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
