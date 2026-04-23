"""
Minimal Physics-Informed Neural Networks (PINN) Reproduction
============================================================
Reproduces the core ideas from "Physics-Informed Neural Networks"
(Raissi et al., 2017, arxiv 1711.10561):
1. Embed PDE constraints into the loss function via automatic differentiation
2. Solve PDEs without data — purely from physics
3. Compare: data-driven MLP vs physics-informed MLP
4. Demo: 1D Burgers equation (nonlinear PDE)
5. Demo: 2D Poisson equation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Network ──

class MLP(nn.Module):
    """Simple fully-connected network for PINN."""
    def __init__(self, layers=[2, 64, 64, 64, 1], activation=nn.Tanh):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(activation())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# ── 1D Burgers Equation ──
# u_t + u * u_x = nu * u_xx
# Domain: x in [-1, 1], t in [0, 1]
# IC: u(x, 0) = -sin(pi*x)
# BC: u(-1, t) = u(1, t) = 0

def burgers_pinn_train(n_collocation=10000, n_boundary=200, n_initial=200,
                       nu=0.01 / np.pi, epochs=5000, lr=1e-3, device='cpu'):
    """Train PINN for 1D Burgers equation."""
    model = MLP(layers=[2, 64, 64, 64, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, 0.5)

    # Initial condition points (static)
    x_ic = torch.rand(n_initial, device=device) * 2 - 1
    t_ic = torch.zeros(n_initial, device=device)
    u_ic = -torch.sin(np.pi * x_ic)

    # Boundary condition points (static)
    t_bc = torch.rand(n_boundary, device=device)
    x_bc_left = -torch.ones(n_boundary, device=device)
    x_bc_right = torch.ones(n_boundary, device=device)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Resample collocation points each epoch (avoids graph accumulation)
        x_col = torch.rand(n_collocation, device=device) * 2 - 1
        t_col = torch.rand(n_collocation, device=device)
        x_col.requires_grad_(True)
        t_col.requires_grad_(True)

        # Interior: PDE residual
        xt = torch.stack([x_col, t_col], dim=-1)
        u = model(xt)

        # Automatic differentiation for derivatives
        u_x = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_col, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]

        # PDE residual: u_t + u*u_x - nu*u_xx = 0
        pde_residual = u_t + u * u_x - nu * u_xx
        loss_pde = (pde_residual ** 2).mean()

        # IC loss
        u_ic_pred = model(torch.stack([x_ic, t_ic], dim=-1))
        loss_ic = ((u_ic_pred.squeeze() - u_ic) ** 2).mean()

        # BC loss
        u_bc_left = model(torch.stack([x_bc_left, t_bc], dim=-1))
        u_bc_right = model(torch.stack([x_bc_right, t_bc], dim=-1))
        loss_bc = (u_bc_left ** 2).mean() + (u_bc_right ** 2).mean()

        loss = loss_pde + 10 * loss_ic + 10 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            losses.append(loss.item())
            print(f"  Epoch {epoch}: loss={loss.item():.6f} "
                  f"(pde={loss_pde.item():.6f}, ic={loss_ic.item():.6f}, bc={loss_bc.item():.6f})")

    return model, losses


def burgers_pure_data_train(n_samples=2000, epochs=5000, lr=1e-3, device='cpu'):
    """Train a purely data-driven MLP on Burgers equation (no physics)."""
    # Generate synthetic training data from analytical-ish solution (small nu regime)
    nu = 0.01 / np.pi
    model = MLP(layers=[2, 64, 64, 64, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create training data: sample (x, t) and compute u using finite differences
    # For simplicity, use a coarse numerical solution as ground truth
    nx, nt = 100, 50
    x_arr = np.linspace(-1, 1, nx)
    t_arr = np.linspace(0, 1, nt)
    dx = x_arr[1] - x_arr[0]
    dt_fd = t_arr[1] - t_arr[0]

    # Simple explicit FD
    u_fd = np.zeros((nt, nx))
    u_fd[0] = -np.sin(np.pi * x_arr)
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u_fd[n+1, i] = u_fd[n, i] + dt_fd * (
                nu * (u_fd[n, i+1] - 2*u_fd[n, i] + u_fd[n, i-1]) / dx**2
                - u_fd[n, i] * (u_fd[n, i+1] - u_fd[n, i-1]) / (2*dx)
            )

    # Sample from FD solution
    x_data = torch.tensor(x_arr, dtype=torch.float32, device=device)
    t_data = torch.tensor(t_arr, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x_data, t_data, indexing='ij')
    XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
    U = torch.tensor(u_fd.T.reshape(-1), dtype=torch.float32, device=device).unsqueeze(-1)

    # Subsample
    idx = torch.randperm(XT.shape[0])[:n_samples]
    XT_train, U_train = XT[idx], U[idx]

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(XT_train)
        loss = ((pred - U_train) ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            losses.append(loss.item())
            print(f"  [Data-driven] Epoch {epoch}: loss={loss.item():.6f}")

    return model, losses


# ── 2D Poisson Equation ──
# u_xx + u_yy = f(x, y)
# Domain: (x, y) in [-1, 1]^2
# BC: u = 0 on boundary
# Exact: u = sin(pi*x)*sin(pi*y), f = -2*pi^2*sin(pi*x)*sin(pi*y)

def poisson_pinn_train(n_collocation=5000, n_boundary=400, epochs=5000,
                       lr=1e-3, device='cpu'):
    """Train PINN for 2D Poisson equation."""
    model = MLP(layers=[2, 64, 64, 64, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Boundary points (4 edges) (static)
    n_b = n_boundary // 4
    t_param = torch.rand(n_b, device=device) * 2 - 1
    xy_bc = []
    xy_bc.append(torch.stack([-torch.ones(n_b, device=device), t_param], dim=-1))
    xy_bc.append(torch.stack([torch.ones(n_b, device=device), t_param], dim=-1))
    xy_bc.append(torch.stack([t_param, -torch.ones(n_b, device=device)], dim=-1))
    xy_bc.append(torch.stack([t_param, torch.ones(n_b, device=device)], dim=-1))
    xy_bc = torch.cat(xy_bc, dim=0)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Resample collocation points each epoch (avoids graph accumulation)
        x_col = torch.rand(n_collocation, device=device) * 2 - 1
        y_col = torch.rand(n_collocation, device=device) * 2 - 1
        x_col.requires_grad_(True)
        y_col.requires_grad_(True)

        # Source term (recomputed from fresh collocation points)
        f_source = -2 * np.pi**2 * torch.sin(np.pi * x_col) * torch.sin(np.pi * y_col)

        # Interior residual
        xy = torch.stack([x_col, y_col], dim=-1)
        u = model(xy)

        u_x = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_col, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_col, grad_outputs=torch.ones_like(u_y),
                                    create_graph=True)[0]

        residual = u_xx + u_yy - f_source
        loss_pde = (residual ** 2).mean()

        # BC loss
        u_bc = model(xy_bc)
        loss_bc = (u_bc ** 2).mean()

        loss = loss_pde + 10 * loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            losses.append(loss.item())
            print(f"  [Poisson] Epoch {epoch}: loss={loss.item():.6f}")

    return model, losses


# ── Visualization ──

def visualize_burgers(model, device='cpu', save_dir=None):
    """Visualize PINN solution for Burgers equation."""
    nx, nt = 100, 100
    x = torch.linspace(-1, 1, nx, device=device)
    t = torch.linspace(0, 1, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)

    model.eval()
    with torch.no_grad():
        U = model(XT).reshape(nx, nt).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Full solution
    im = axes[0].imshow(U, extent=[0, 1, -1, 1], aspect='auto', origin='lower',
                         cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    axes[0].set_title('PINN: Burgers Equation Solution')
    plt.colorbar(im, ax=axes[0])

    # Snapshots
    t_snap = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_np = x.cpu().numpy()
    for ti in t_snap:
        idx = int(ti * (nt - 1))
        axes[1].plot(x_np, U[:, idx], label=f't={ti:.2f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].set_title('Solution Snapshots')
    axes[1].legend(fontsize=8)

    # t=0 comparison with IC
    axes[2].plot(x_np, U[:, 0], label='PINN t=0')
    axes[2].plot(x_np, -np.sin(np.pi * x_np), '--', label='Exact IC')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('u')
    axes[2].set_title('Initial Condition Match')
    axes[2].legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'burgers_pinn.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_poisson(model, device='cpu', save_dir=None):
    """Visualize PINN solution for Poisson equation."""
    n = 100
    x = torch.linspace(-1, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    model.eval()
    with torch.no_grad():
        U_pred = model(XY).reshape(n, n).cpu().numpy()

    U_exact = np.sin(np.pi * X.cpu().numpy()) * np.sin(np.pi * Y.cpu().numpy())
    error = np.abs(U_pred - U_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(U_pred, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    axes[0].set_title('PINN Solution')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_exact, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    axes[1].set_title('Exact Solution')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle('2D Poisson Equation: u_xx + u_yy = -2π²sin(πx)sin(πy)', y=1.02)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'poisson_pinn.png', dpi=150, bbox_inches='tight')
    plt.close()


def compare_pinn_vs_data(pinn_losses, data_losses, save_dir=None):
    """Compare PINN vs pure data-driven training convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(len(pinn_losses)), pinn_losses, label='PINN (physics-informed)')
    ax.semilogy(range(len(data_losses)), data_losses, label='Data-driven MLP')
    ax.set_xlabel('Evaluation Step (x500)')
    ax.set_ylabel('Loss')
    ax.set_title('PINN vs Data-Driven: Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'pinn_vs_data_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    save_dir = Path(__file__).parent / 'results' / 'pinn'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Burgers equation PINN
    print("\n=== Training PINN for 1D Burgers Equation ===")
    pinn_model, pinn_losses = burgers_pinn_train(epochs=5000, device=device)
    visualize_burgers(pinn_model, device, save_dir)

    # 2. Pure data-driven comparison
    print("\n=== Training Data-Driven MLP for Burgers Equation ===")
    data_model, data_losses = burgers_pure_data_train(epochs=5000, device=device)
    compare_pinn_vs_data(pinn_losses, data_losses, save_dir)

    # 3. 2D Poisson equation
    print("\n=== Training PINN for 2D Poisson Equation ===")
    poisson_model, _ = poisson_pinn_train(epochs=5000, device=device)
    visualize_poisson(poisson_model, device, save_dir)

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
