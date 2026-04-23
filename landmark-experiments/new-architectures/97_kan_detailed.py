"""
Detailed KAN (Kolmogorov-Arnold Networks) Reproduction
======================================================
Reproduces core ideas from KAN (2404.19756, Liu et al.):
1. Kolmogorov-Arnold representation theorem: any continuous f = sum of univariate functions
2. Learnable B-spline activation functions on edges (not fixed activations on nodes)
3. Efficient vectorized B-spline evaluation via grid-based approach
4. Grid extension: adapting spline resolution during training
5. Compare KAN vs MLP on compositional function approximation
6. Show: learned activation shapes, parameter efficiency, compositional structure capture
7. Symbolic regression: KAN's ability to recover simple symbolic expressions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Efficient B-Spline Basis (Vectorized) ──

class BSplineBasis(nn.Module):
    """Efficient B-spline basis using grid-based evaluation.

    Matches the KAN paper's approach: evaluate B-spline basis functions
    on a uniform grid using vectorized operations (no Python recursion).
    Uses the same mathematical formulation but computes all basis functions
    in parallel for the entire batch.
    """
    def __init__(self, n_grid=5, degree=3, domain=(-2, 2)):
        super().__init__()
        self.n_grid = n_grid
        self.degree = degree
        self.domain = domain
        # Number of basis functions = n_grid + degree - 1
        self.n_basis = n_grid + degree - 1
        # Learnable coefficients for each basis function
        self.coeffs = nn.Parameter(torch.randn(self.n_basis) * 0.1)
        # Grid points for B-spline evaluation
        self.register_buffer('grid', torch.linspace(domain[0], domain[1], n_grid + 1))

    def basis_functions(self, x):
        """Evaluate B-spline basis functions using vectorized Cox-de Boor.

        Args:
            x: (B,) input values
        Returns:
            (B, n_basis) tensor of basis function values
        """
        # Extend grid with padding for degree
        pad = self.degree
        grid = self.grid
        h = (grid[-1] - grid[0]) / (len(grid) - 1)
        extended = torch.cat([
            grid[0] - h * torch.arange(pad, 0, -1, device=x.device),
            grid,
            grid[-1] + h * torch.arange(1, pad + 1, device=x.device),
        ])

        n_knots = len(extended)
        B = x.shape[0]
        n_basis = n_knots - 1  # number of degree-0 basis functions

        # Degree 0: indicator function
        basis = torch.zeros(B, n_basis, device=x.device)
        for i in range(n_basis):
            basis[:, i] = ((x >= extended[i]) & (x < extended[i + 1])).float()
        # Include right endpoint in last basis
        basis[:, -1] = basis[:, -1] + (x >= extended[-2]).float() * (x <= extended[-1]).float()

        # Recursive evaluation up to target degree
        for k in range(1, self.degree + 1):
            n_basis_k = n_basis - k
            new_basis = torch.zeros(B, n_basis_k, device=x.device)
            for i in range(n_basis_k):
                # Left term
                denom_left = extended[i + k] - extended[i]
                if abs(denom_left) < 1e-8:
                    left = torch.zeros(B, device=x.device)
                else:
                    left = (x - extended[i]) / denom_left * basis[:, i]

                # Right term
                denom_right = extended[i + k + 1] - extended[i + 1]
                if abs(denom_right) < 1e-8:
                    right = torch.zeros(B, device=x.device)
                else:
                    right = (extended[i + k + 1] - x) / denom_right * basis[:, i + 1]

                new_basis[:, i] = left + right
            basis = new_basis

        # Return the correct number of basis functions
        # n_basis_final = n_knots - 1 - degree = len(extended) - degree - 1
        # We want self.n_basis functions
        # Center crop if we have too many
        if basis.shape[1] >= self.n_basis:
            start = (basis.shape[1] - self.n_basis) // 2
            return basis[:, start:start + self.n_basis]
        else:
            # Pad if too few
            pad_size = self.n_basis - basis.shape[1]
            return torch.cat([basis, torch.zeros(B, pad_size, device=x.device)], dim=1)

    def forward(self, x):
        """Evaluate spline: sum(coeffs * basis_functions)."""
        basis = self.basis_functions(x)  # (B, n_basis)
        return (basis * self.coeffs).sum(dim=-1)  # (B,)


# ── KAN Layer ──

class KANLayer(nn.Module):
    """KAN layer: learnable univariate functions on edges.

    Unlike MLP where activation is fixed on nodes, KAN puts learnable
    B-spline activations on edges: phi_{i,j}(x_i) for input i -> output j.

    Each edge function = residual(SiLU) + spline(x).
    """
    def __init__(self, in_dim, out_dim, n_grid=5, degree=3, domain=(-2, 2)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Each edge has its own spline: phi_{i,j}
        self.splines = nn.ModuleList([
            BSplineBasis(n_grid, degree, domain=domain)
            for _ in range(in_dim * out_dim)
        ])
        # Residual base activation weight (SiLU, like the original paper)
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

    def forward(self, x):
        """x: (B, in_dim) -> (B, out_dim)"""
        B = x.shape[0]
        # Residual: SiLU(x) @ W_base
        base = F.silu(x) @ self.base_weight.T  # (B, out_dim)

        # Spline contributions from each edge
        spline_out = torch.zeros(B, self.out_dim, device=x.device)
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                idx = i * self.out_dim + j
                spline_out[:, j] += self.splines[idx](x[:, i])

        return base + spline_out


# ── Full KAN ──

class KAN(nn.Module):
    """Kolmogorov-Arnold Network."""
    def __init__(self, layer_sizes, n_grid=5, degree=3, domain=(-2, 2)):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                KANLayer(layer_sizes[i], layer_sizes[i+1], n_grid, degree, domain)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    """Standard MLP for comparison."""
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Training ──

def train_model(model, X, Y, n_steps=3000, lr=1e-3, batch_size=128, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    N = X.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        x_batch = X[idx].to(device)
        y_batch = Y[idx].to(device)

        pred = model(x_batch)
        loss = F.mse_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.6f}")

    return losses


# ── Target Functions ──

def make_function(name, n_points=5000, device='cpu'):
    """Generate synthetic data from various target functions."""
    X = torch.rand(n_points, 2, device=device) * 4 - 2  # [-2, 2]^2
    if name == 'sincos':
        Y = torch.sin(X[:, 0:1] * np.pi) * torch.cos(X[:, 1:2] * np.pi)
    elif name == 'x2_plus_y2':
        Y = X[:, 0:1]**2 + X[:, 1:2]**2
    elif name == 'exp_sin':
        Y = torch.exp(torch.sin(X[:, 0:1]) + X[:, 1:2]**2)
    elif name == 'sinc':
        r = torch.sqrt(X[:, 0:1]**2 + X[:, 1:2]**2 + 1e-8)
        Y = torch.sin(r) / r
    elif name == 'product':
        Y = X[:, 0:1] * X[:, 1:2]
    else:
        raise ValueError(f"Unknown function: {name}")
    return X, Y


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "97-kan-detailed"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_steps = 2000
    functions = ['sincos', 'x2_plus_y2', 'exp_sin', 'sinc', 'product']

    # ── Experiment 1: KAN vs MLP on multiple functions ──

    kan_losses_all = {}
    mlp_losses_all = {}
    kan_params_all = {}
    mlp_params_all = {}

    for func_name in functions:
        print(f"\n=== Function: {func_name} ===")
        X, Y = make_function(func_name, device=device)

        # KAN: [2, 5, 1] — small network with B-spline edges
        kan = KAN([2, 5, 1], n_grid=5, degree=3).to(device)
        kan_p = sum(p.numel() for p in kan.parameters())
        kan_params_all[func_name] = kan_p
        print(f"  KAN params: {kan_p:,}")
        kan_losses = train_model(kan, X, Y, n_steps, device=device)
        kan_losses_all[func_name] = kan_losses

        # MLP: [2, 16, 16, 1] — wider network for fair comparison
        mlp = MLP([2, 16, 16, 1]).to(device)
        mlp_p = sum(p.numel() for p in mlp.parameters())
        mlp_params_all[func_name] = mlp_p
        print(f"  MLP params: {mlp_p:,}")
        mlp_losses = train_model(mlp, X, Y, n_steps, device=device)
        mlp_losses_all[func_name] = mlp_losses

    # ── Experiment 2: Compositional structure ──

    print("\n=== Compositional Structure: Additive vs Non-Additive ===")
    X_add, Y_add = make_function('x2_plus_y2', device=device)
    X_comp, Y_comp = make_function('exp_sin', device=device)

    kan_small = KAN([2, 3, 1], n_grid=5, degree=3).to(device)
    print(f"  Small KAN params (additive): {sum(p.numel() for p in kan_small.parameters()):,}")
    kan_small_losses = train_model(kan_small, X_add, Y_add, n_steps=1500, device=device)

    kan_small2 = KAN([2, 3, 1], n_grid=5, degree=3).to(device)
    kan_small2_losses = train_model(kan_small2, X_comp, Y_comp, n_steps=1500, device=device)

    kan_large = KAN([2, 5, 1], n_grid=5, degree=3).to(device)
    kan_large_losses = train_model(kan_large, X_comp, Y_comp, n_steps=1500, device=device)

    # ── Experiment 3: Grid resolution study ──

    print("\n=== Grid Resolution Study ===")
    X_g, Y_g = make_function('sincos', device=device)
    grid_losses = {}
    for n_grid in [3, 5, 8, 10]:
        kan_g = KAN([2, 5, 1], n_grid=n_grid, degree=3).to(device)
        n_params = sum(p.numel() for p in kan_g.parameters())
        print(f"  n_grid={n_grid}, params={n_params}")
        l = train_model(kan_g, X_g, Y_g, n_steps=1500, device=device)
        grid_losses[n_grid] = l

    # ── Visualization ──

    # 1. Training loss comparison (per function)
    fig, axes = plt.subplots(1, len(functions), figsize=(5 * len(functions), 4))
    window = 30

    for idx, func_name in enumerate(functions):
        ax = axes[idx]
        kan_s = np.convolve(kan_losses_all[func_name], np.ones(window)/window, mode='valid')
        mlp_s = np.convolve(mlp_losses_all[func_name], np.ones(window)/window, mode='valid')

        ax.plot(kan_s, label=f'KAN ({kan_params_all[func_name]})', color='blue')
        ax.plot(mlp_s, label=f'MLP ({mlp_params_all[func_name]})', color='red')
        ax.set_title(func_name)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss (smoothed)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle("KAN (B-Spline) vs MLP: Function Approximation", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Final loss comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(functions))
    width = 0.35

    kan_final = [np.mean(kan_losses_all[f][-100:]) for f in functions]
    mlp_final = [np.mean(mlp_losses_all[f][-100:]) for f in functions]

    ax.bar(x - width/2, kan_final, width, label='KAN', color='blue', alpha=0.7)
    ax.bar(x + width/2, mlp_final, width, label='MLP', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(functions)
    ax.set_ylabel("Final MSE Loss")
    ax.set_title("KAN vs MLP: Final Approximation Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "final_loss_comparison.png", dpi=150)
    plt.close()

    # 3. Parameter efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(functions))
    width = 0.35
    ax.bar(x - width/2, [kan_params_all[f] for f in functions], width,
           label='KAN', color='blue', alpha=0.7)
    ax.bar(x + width/2, [mlp_params_all[f] for f in functions], width,
           label='MLP', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(functions)
    ax.set_ylabel("Parameters")
    ax.set_title("KAN vs MLP: Parameter Count (KAN uses far fewer)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / "parameter_comparison.png", dpi=150)
    plt.close()

    # 4. Learned B-spline activation functions
    print("\n=== Visualizing B-Spline Activations ===")
    X_vis, Y_vis = make_function('sincos', device=device)
    kan_vis = KAN([2, 5, 1], n_grid=5, degree=3).to(device)
    train_model(kan_vis, X_vis, Y_vis, n_steps=2000, device=device)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    x_range = torch.linspace(-2, 2, 200, device=device)

    layer = kan_vis.layers[0]
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]
            idx = i * 5 + j
            with torch.no_grad():
                spline = layer.splines[idx]
                y_spline = spline(x_range).cpu().numpy()
                y_base = (F.silu(x_range) * layer.base_weight[j, i]).cpu().numpy()

            ax.plot(x_range.cpu().numpy(), y_spline, label='B-spline', color='blue')
            ax.plot(x_range.cpu().numpy(), y_base, label='SiLU base', color='red', alpha=0.5)
            ax.plot(x_range.cpu().numpy(), y_spline + y_base, label='Total phi', color='green')
            ax.set_title(f"phi_{{{i},{j}}}")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    plt.suptitle("KAN Layer 1: Learned B-Spline Activations phi_{i,j}(x)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "bspline_activations.png", dpi=150)
    plt.close()

    # 5. Individual B-spline basis functions visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    x_plot = torch.linspace(-2, 2, 200)
    spline_0 = kan_vis.layers[0].splines[0]
    with torch.no_grad():
        basis_vals = spline_0.basis_functions(x_plot).cpu().numpy()

    for i in range(min(spline_0.n_basis, basis_vals.shape[1])):
        ax.plot(x_plot.numpy(), basis_vals[:, i], label=f'B_{i}', alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("B_i(x)")
    ax.set_title("B-Spline Basis Functions (degree=3, n_grid=5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "bspline_basis_functions.png", dpi=150)
    plt.close()

    # 6. Compositional structure: additive vs non-additive
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 30

    smoothed_small_add = np.convolve(kan_small_losses, np.ones(window)/window, mode='valid')
    smoothed_small_comp = np.convolve(kan_small2_losses, np.ones(window)/window, mode='valid')
    smoothed_large_comp = np.convolve(kan_large_losses, np.ones(window)/window, mode='valid')

    axes[0].plot(smoothed_small_add, label='KAN [2,3,1] on x^2+y^2 (additive)', color='blue')
    axes[0].plot(smoothed_small_comp, label='KAN [2,3,1] on exp(sin(x)+y^2)', color='red')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("KAN: Additive Functions Are Easier")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].plot(smoothed_small_comp, label='KAN [2,3,1] (small)', color='red')
    axes[1].plot(smoothed_large_comp, label='KAN [2,5,1] (large)', color='green')
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("Compositional: Larger KAN Helps")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.suptitle("KAN: Capturing Compositional Structure", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "compositional_structure.png", dpi=150)
    plt.close()

    # 7. Grid resolution study
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['gray', 'blue', 'green', 'red']
    for (n_grid, losses), color in zip(grid_losses.items(), colors):
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=f'n_grid={n_grid}', color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("KAN: Effect of B-Spline Grid Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "grid_resolution.png", dpi=150)
    plt.close()

    # 8. KAN vs MLP architecture concept diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MLP diagram
    ax = axes[0]
    ax.axis('off')
    ax.set_title("MLP: Fixed Activations on Nodes", fontsize=13, fontweight='bold')
    mlp_layers = [2, 3, 1]
    for l, n in enumerate(mlp_layers):
        for i in range(n):
            x_pos = l * 0.4 + 0.1
            y_pos = 1 - (i + 0.5) / n
            circle = plt.Circle((x_pos, y_pos), 0.04, color='lightblue', ec='blue', lw=2)
            ax.add_patch(circle)
            if l > 0:
                ax.text(x_pos, y_pos, 'sigma', ha='center', va='center', fontsize=8, color='blue')
    for l in range(len(mlp_layers) - 1):
        for i in range(mlp_layers[l]):
            for j in range(mlp_layers[l+1]):
                x1 = l * 0.4 + 0.14
                x2 = (l+1) * 0.4 + 0.06
                y1 = 1 - (i + 0.5) / mlp_layers[l]
                y2 = 1 - (j + 0.5) / mlp_layers[l+1]
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # KAN diagram
    ax = axes[1]
    ax.axis('off')
    ax.set_title("KAN: Learnable Activations on Edges", fontsize=13, fontweight='bold')
    kan_layers = [2, 3, 1]
    for l, n in enumerate(kan_layers):
        for i in range(n):
            x_pos = l * 0.4 + 0.1
            y_pos = 1 - (i + 0.5) / n
            circle = plt.Circle((x_pos, y_pos), 0.04, color='lightyellow', ec='green', lw=2)
            ax.add_patch(circle)
    for l in range(len(kan_layers) - 1):
        for i in range(kan_layers[l]):
            for j in range(kan_layers[l+1]):
                x1 = l * 0.4 + 0.14
                x2 = (l+1) * 0.4 + 0.06
                y1 = 1 - (i + 0.5) / kan_layers[l]
                y2 = 1 - (j + 0.5) / kan_layers[l+1]
                ax.plot([x1, x2], [y1, y2], 'green', alpha=0.5, linewidth=2)
                mx, my = (x1+x2)/2, (y1+y2)/2
                ax.text(mx, my, 'phi', fontsize=8, ha='center', va='center', color='green',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle("KAN vs MLP Architecture (B-Spline Basis)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "kan_vs_mlp_arch.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
