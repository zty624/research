"""
Minimal Neural ODE Reproduction
================================
Reproduces core ideas from Neural ODE (1806.07366, Chen et al.):
1. Replace discrete ResNet layers with continuous dynamics: dx/dt = f_θ(x, t)
2. Adjoint method: O(1) memory backprop via augmented ODE solved backwards
3. Continuous Normalizing Flows (CNF): change of variables via ODE + Hutchinson trace
4. Applications: classification, density estimation, time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── ODE Solvers ──

def euler_step(func, t, x, dt):
    """Single Euler step: x(t+dt) = x(t) + dt * f(x, t)."""
    return x + dt * func(t, x)


def rk4_step(func, t, x, dt):
    """Single RK4 step."""
    k1 = func(t, x)
    k2 = func(t + dt / 2, x + dt * k1 / 2)
    k3 = func(t + dt / 2, x + dt * k2 / 2)
    k4 = func(t + dt, x + dt * k3)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def ode_solve(func, x0, t_span, method='rk4', n_steps=20):
    """Solve ODE dx/dt = f(x,t) from t_span[0] to t_span[1].

    Args:
        func: callable(t: float, x: Tensor) -> Tensor
        x0: initial state (batch, dim)
        t_span: (t_start, t_end)
        method: 'euler' or 'rk4'
        n_steps: number of integration steps

    Returns:
        x: final state (batch, dim)
        trajectory: list of (t, x) pairs including intermediate states
    """
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    step_fn = euler_step if method == 'euler' else rk4_step

    t = t0
    x = x0
    trajectory = [(t, x.detach())]
    for _ in range(n_steps):
        x = step_fn(func, t, x, dt)
        t = t + dt
        trajectory.append((t, x.detach()))

    return x, trajectory


# ── ODE Function Network ──

class ODEFunc(nn.Module):
    """MLP parameterizing dx/dt = f_θ(x, t). Time t concatenated to input."""

    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )
        # Initialize last layer near zero for near-identity dynamics
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, x):
        # t is scalar, broadcast to (batch, 1)
        t_vec = torch.full_like(x[..., :1], t)
        return self.net(torch.cat([x, t_vec], dim=-1))


# ── Neural ODE with Adjoint ──

class NeuralODE(nn.Module):
    """Neural ODE module with optional adjoint method for memory-efficient backprop."""

    def __init__(self, func, method='rk4', n_steps=20, adjoint=False):
        super().__init__()
        self.func = func
        self.method = method
        self.n_steps = n_steps
        self.adjoint = adjoint

    def forward(self, x, t_span=(0.0, 1.0)):
        if self.adjoint:
            return self._forward_adjoint(x, t_span)
        else:
            return self._forward_standard(x, t_span)

    def _forward_standard(self, x, t_span):
        """Standard forward: solve ODE, autograd through solver steps."""
        x_out, _ = ode_solve(self.func, x, t_span, self.method, self.n_steps)
        return x_out

    def _forward_adjoint(self, x, t_span):
        """Adjoint method: solve augmented ODE backwards for O(1) memory.

        We implement a custom autograd Function that:
        1. Forward: solve ODE, store final state (NO intermediate states)
        2. Backward: solve adjoint ODE backwards to compute gradients
        """
        return _NeuralODEAdjoint.apply(
            x, t_span[0], t_span[1],
            self.func, self.method, self.n_steps
        )

    def get_trajectory(self, x, t_span=(0.0, 1.0), n_steps=40):
        """Solve ODE and return full trajectory for visualization."""
        with torch.no_grad():
            _, trajectory = ode_solve(self.func, x, t_span, self.method, n_steps)
        return trajectory


class _NeuralODEAdjoint(torch.autograd.Function):
    """Custom autograd function implementing the adjoint sensitivity method.

    Instead of storing all intermediate states (O(N) memory), we:
    1. Forward: solve ODE from t0 to t1, only save x(t1)
    2. Backward: solve augmented ODE [x, a, a_θ] backwards from t1 to t0
       where a = ∂L/∂x (adjoint) and a_θ = ∂L/∂θ (parameter gradient)
    """

    @staticmethod
    def forward(ctx, x, t0, t1, func, method, n_steps):
        # Solve forward ODE, only save final state
        dt = (t1 - t0) / n_steps
        step_fn = euler_step if method == 'euler' else rk4_step
        t = t0
        with torch.no_grad():
            state = x.clone()
            for _ in range(n_steps):
                state = step_fn(func, t, state, dt)
                t = t + dt

        ctx.save_for_backward(state)
        ctx.func = func
        ctx.method = method
        ctx.n_steps = n_steps
        ctx.t0 = t0
        ctx.t1 = t1
        return state

    @staticmethod
    def backward(ctx, grad_output):
        """Solve adjoint ODE backwards.

        Augmented state: [x, a, a_θ]
        - a = ∂L/∂x: adjoint state, initialized from grad_output
        - a_θ = ∂L/∂θ: parameter gradient accumulator

        Dynamics (solved backwards from t1 to t0):
          da/dt = -a^T ∂f/∂x
          da_θ/dt = -a^T ∂f/∂θ
        """
        x_final, = ctx.saved_tensors
        func = ctx.func
        n_steps = ctx.n_steps
        dt = (ctx.t1 - ctx.t0) / n_steps

        # Initialize augmented state
        x = x_final.detach().requires_grad_(True)
        a = grad_output.clone()  # adjoint: ∂L/∂x(t1)

        # Parameter gradient accumulator
        a_theta = []
        for p in func.parameters():
            a_theta.append(torch.zeros_like(p))
        a_theta = a_theta

        # Solve backwards
        t = ctx.t1
        all_params = list(func.parameters())
        for _ in range(n_steps):
            # Need ∂f/∂x and ∂f/∂θ at current (t, x)
            x = x.detach().requires_grad_(True)

            with torch.enable_grad():
                f_val = func(t, x)  # (batch, dim)

            # Compute both gradients in one call to avoid graph freed after first
            grads = torch.autograd.grad(
                f_val, [x] + all_params, grad_outputs=a,
                create_graph=False, allow_unused=True
            )
            df_dx = grads[0]

            # Accumulate parameter gradients
            for i, g in enumerate(grads[1:]):
                if g is not None:
                    a_theta[i] = a_theta[i] + dt * g

            # Update adjoint: a(t-dt) = a(t) + dt * (-da/dt) = a(t) - dt * df_dx
            a = a - dt * df_dx

            # Update x backwards: x(t-dt) ≈ x(t) - dt * f(x(t), t)
            with torch.no_grad():
                x = x.detach() - dt * func(t, x).detach()

            t = t - dt

        # Set parameter gradients
        for p, ag in zip(func.parameters(), a_theta):
            if p.grad is not None:
                p.grad = p.grad + ag
            else:
                p.grad = ag

        # Return gradient w.r.t. x (input)
        return a, None, None, None, None, None


# ── Continuous Normalizing Flow (CNF) ──

class CNF(nn.Module):
    """Continuous Normalizing Flow using Neural ODE.

    Log-density via change of variables:
      log p(x₁) = log p(x₀) - ∫₀¹ Tr(∂f/∂x) dt

    Trace estimated with Hutchinson's method:
      Tr(∂f/∂x) ≈ E[v^T (∂f/∂x) v] with v ~ N(0, I)
    """

    def __init__(self, dim, hidden_dim=64, method='rk4', n_steps=20):
        super().__init__()
        self.func = ODEFunc(dim, hidden_dim)
        self.node = NeuralODE(self.func, method, n_steps, adjoint=False)
        self.dim = dim

    def forward(self, x):
        """Flow x through ODE and compute log p(x).

        Returns:
            log_prob: log p(x) under the model
        """
        training = self.training
        # Augment state with running log-det: [x, log_det]
        batch = x.shape[0]
        log_det = torch.zeros(batch, 1, device=x.device)
        aug = torch.cat([x, log_det], dim=-1)  # (batch, dim+1)

        # Define augmented dynamics
        def aug_func(t, aug_state):
            x_part = aug_state[:, :self.dim].detach().requires_grad_(True)
            f = self.func(t, x_part)

            # Hutchinson trace estimator: v^T (∂f/∂x) v
            v = torch.randn_like(x_part)
            vjp = torch.autograd.grad(
                f, x_part, grad_outputs=v,
                create_graph=training  # only build graph during training
            )[0]
            trace = (vjp * v).sum(dim=-1, keepdim=True)  # (batch, 1)

            return torch.cat([f, trace], dim=-1)

        # Solve augmented ODE
        aug_out, _ = ode_solve(aug_func, aug, (0.0, 1.0), self.node.method, self.node.n_steps)
        x_out = aug_out[:, :self.dim]
        log_det_out = aug_out[:, self.dim:]

        # Log probability under standard normal prior
        log_p0 = -0.5 * (x_out ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        log_prob = log_p0 + log_det_out.squeeze(-1)

        return log_prob, x_out

    def sample(self, n_samples, device='cpu'):
        """Sample from the CNF by sampling from prior and flowing forward."""
        z = torch.randn(n_samples, self.dim, device=device)
        with torch.no_grad():
            x, _ = ode_solve(self.func, z, (0.0, 1.0), self.node.method, self.node.n_steps)
        return x


# ── Discrete Normalizing Flow (RealNVP-style) for Comparison ──

class AffineCoupling(nn.Module):
    """Single affine coupling layer."""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )
        # Init near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x1, x2 = x[:, :self.dim // 2], x[:, self.dim // 2:]
        params = self.net(x1)
        s, t = params[:, :self.dim // 2], params[:, self.dim // 2:]
        s = torch.tanh(s) * 0.5
        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det


class DiscreteNF(nn.Module):
    """Discrete Normalizing Flow with affine coupling layers."""
    def __init__(self, dim, n_layers=4, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x):
        total_log_det = 0
        h = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 1:
                h = h.flip(-1)  # alternate which half is transformed
            h, ld = layer(h)
            total_log_det = total_log_det + ld

        log_p0 = -0.5 * (h ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        log_prob = log_p0 + total_log_det
        return log_prob, h


# ── Baseline: Equivalent-Depth ResNet ──

class ResNetBaseline(nn.Module):
    """ResNet with discrete layers, comparable depth to Neural ODE."""
    def __init__(self, dim, hidden_dim=64, n_layers=6):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, dim),
            ))
        self.layers = nn.ModuleList(layers)
        # Init near identity for fair comparison
        for layer in self.layers:
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # ResNet skip connection
        return x


# ── Data Generation ──

def make_spiral(n_samples, noise=0.1):
    """Two-class spiral dataset."""
    n = n_samples // 2
    theta = np.linspace(0, 4 * np.pi, n)
    r = theta / (4 * np.pi) + 0.1

    x0 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x1 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)

    x0 += np.random.randn(n, 2) * noise
    x1 += np.random.randn(n, 2) * noise

    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_moons(n_samples, noise=0.1):
    """Two moons dataset."""
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)

    x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1 = np.stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5], axis=1)

    x0 += np.random.randn(n, 2) * noise
    x1 += np.random.randn(n, 2) * noise

    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_2d_gaussian_mixture(n_samples, n_modes=8, radius=2.0, noise=0.15):
    """Ring of Gaussian modes for density estimation."""
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    samples_per_mode = n_samples // n_modes
    X = []
    for c in centers:
        X.append(c + np.random.randn(samples_per_mode, 2) * noise)
    X = np.concatenate(X, axis=0)
    return torch.tensor(X, dtype=torch.float32)


# ── Classification Training ──

def train_classification(model_type, X, y, device='cpu', n_steps=3000, lr=1e-3):
    """Train a classifier (neural_ode or resnet)."""
    X, y = X.to(device), y.to(device)
    dim = X.shape[1]

    if model_type == 'neural_ode':
        func = ODEFunc(dim, hidden_dim=128).to(device)
        node = NeuralODE(func, method='rk4', n_steps=20, adjoint=False).to(device)
        head = nn.Linear(dim, 2).to(device)
        params = list(func.parameters()) + list(head.parameters())
    else:
        resnet = ResNetBaseline(dim, hidden_dim=128, n_layers=6).to(device)
        head = nn.Linear(dim, 2).to(device)
        params = list(resnet.parameters()) + list(head.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)

    losses = []
    accs = []
    for step in range(n_steps):
        if model_type == 'neural_ode':
            h = node(X)
        else:
            h = resnet(X)

        logits = head(h)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc)

    model_dict = {'head': head}
    if model_type == 'neural_ode':
        model_dict['node'] = node
        model_dict['func'] = func
    else:
        model_dict['resnet'] = resnet

    return losses, accs, model_dict


# ── CNF Training ──

def train_cnf(model, X, device='cpu', n_steps=3000, lr=1e-3):
    """Train CNF via maximum likelihood."""
    X = X.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        log_prob, _ = model(X)
        loss = -log_prob.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


def train_discrete_nf(model, X, device='cpu', n_steps=3000, lr=1e-3):
    """Train discrete NF via maximum likelihood."""
    X = X.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        log_prob, _ = model(X)
        loss = -log_prob.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


# ── Visualization Helpers ──

def plot_decision_boundary(ax, model_dict, model_type, X, y, title, device='cpu'):
    """Plot decision boundary for classification model."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        if model_type == 'neural_ode':
            h = model_dict['node'](grid)
        else:
            h = model_dict['resnet'](grid)
        logits = model_dict['head'](h)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    zz = probs.reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap='RdBu', alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=20, alpha=0.8)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def plot_density(ax, model, model_type, device='cpu', bounds=4.0, n_grid=80):
    """Plot density contours for flow model."""
    xx, yy = np.meshgrid(np.linspace(-bounds, bounds, n_grid),
                          np.linspace(-bounds, bounds, n_grid))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    # CNF needs autograd for Hutchinson trace estimator, so use enable_grad
    if model_type == 'cnf':
        with torch.enable_grad():
            log_prob, _ = model(grid)
        prob = log_prob.detach().exp().cpu().numpy()
    else:
        with torch.no_grad():
            log_prob, _ = model(grid)
        prob = log_prob.exp().cpu().numpy()

    zz = prob.reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap='viridis')
    ax.set_title(f'{model_type.upper()} Density')


def plot_trajectories(ax, func, X, t_span=(0.0, 1.0), n_steps=40, n_show=50):
    """Plot trajectories of points flowing through ODE."""
    X_show = X[:n_show].clone()
    with torch.no_grad():
        _, trajectory = ode_solve(func, X_show, t_span, 'rk4', n_steps)

    # Plot trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    for i in range(n_show):
        xs = [traj[1][i, 0].item() for _, traj in zip(range(len(trajectory)), trajectory)]
        ys = [traj[1][i, 1].item() for _, traj in zip(range(len(trajectory)), trajectory)]
        ax.plot(xs, ys, '-', alpha=0.3, linewidth=1)

    # Mark start and end
    x0 = trajectory[0][1]
    x1 = trajectory[-1][1]
    ax.scatter(x0[:, 0], x0[:, 1], c='blue', s=10, alpha=0.5, label='t=0')
    ax.scatter(x1[:, 0], x1[:, 1], c='red', s=10, alpha=0.5, label='t=1')
    ax.legend(fontsize=8)
    ax.set_title('Neural ODE Trajectories')


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "67-neural-ode"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # ══════════════════════════════════════════════
    # Experiment 1: Classification (Neural ODE vs ResNet)
    # ══════════════════════════════════════════════
    print("=== Experiment 1: Classification on Spiral Data ===")

    X_spiral, y_spiral = make_spiral(500, noise=0.15)

    print("  Training Neural ODE classifier...")
    node_losses, node_accs, node_model = train_classification(
        'neural_ode', X_spiral, y_spiral, device, n_steps=5000, lr=1e-3)
    print(f"    Final acc: {node_accs[-1]:.3f}")

    print("  Training ResNet classifier...")
    resnet_losses, resnet_accs, resnet_model = train_classification(
        'resnet', X_spiral, y_spiral, device, n_steps=5000, lr=1e-3)
    print(f"    Final acc: {resnet_accs[-1]:.3f}")

    # ══════════════════════════════════════════════
    # Experiment 2: Continuous Normalizing Flow
    # ══════════════════════════════════════════════
    print("\n=== Experiment 2: CNF Density Estimation ===")

    X_gmm = make_2d_gaussian_mixture(800, n_modes=8, radius=2.0, noise=0.2)

    print("  Training CNF...")
    cnf = CNF(dim=2, hidden_dim=64, method='rk4', n_steps=10).to(device)
    cnf_losses = train_cnf(cnf, X_gmm, device, n_steps=3000, lr=1e-3)
    print(f"    Final NLL: {cnf_losses[-1]:.3f}")

    print("  Training Discrete NF (RealNVP-style)...")
    dnf = DiscreteNF(dim=2, n_layers=6, hidden_dim=64).to(device)
    dnf_losses = train_discrete_nf(dnf, X_gmm, device, n_steps=3000, lr=1e-3)
    print(f"    Final NLL: {dnf_losses[-1]:.3f}")

    # ══════════════════════════════════════════════
    # Experiment 3: Trajectory Visualization
    # ══════════════════════════════════════════════
    print("\n=== Experiment 3: Trajectory Visualization ===")

    # Use CNF's learned dynamics for trajectory
    X_prior = torch.randn(100, 2)
    func_cnf = cnf.func

    # ══════════════════════════════════════════════
    # Experiment 4: ODE Solver Step Count vs Accuracy
    # ══════════════════════════════════════════════
    print("\n=== Experiment 4: Solver Steps vs Accuracy ===")

    step_counts = [1, 2, 5, 10, 20, 40]
    step_accs_euler = []
    step_accs_rk4 = []

    for n_steps in step_counts:
        # Euler
        func_e = ODEFunc(2, hidden_dim=64).to(device)
        node_e = NeuralODE(func_e, method='euler', n_steps=n_steps, adjoint=False).to(device)
        head_e = nn.Linear(2, 2).to(device)
        opt_e = torch.optim.Adam(list(func_e.parameters()) + list(head_e.parameters()), lr=1e-3)
        Xd, yd = X_spiral.to(device), y_spiral.to(device)
        for _ in range(1000):
            h = node_e(Xd)
            loss = F.cross_entropy(head_e(h), yd)
            opt_e.zero_grad(); loss.backward(); opt_e.step()
        with torch.no_grad():
            acc_e = (head_e(node_e(Xd)).argmax(-1) == yd).float().mean().item()
        step_accs_euler.append(acc_e)

        # RK4
        func_r = ODEFunc(2, hidden_dim=64).to(device)
        node_r = NeuralODE(func_r, method='rk4', n_steps=n_steps, adjoint=False).to(device)
        head_r = nn.Linear(2, 2).to(device)
        opt_r = torch.optim.Adam(list(func_r.parameters()) + list(head_r.parameters()), lr=1e-3)
        for _ in range(1000):
            h = node_r(Xd)
            loss = F.cross_entropy(head_r(h), yd)
            opt_r.zero_grad(); loss.backward(); opt_r.step()
        with torch.no_grad():
            acc_r = (head_r(node_r(Xd)).argmax(-1) == yd).float().mean().item()
        step_accs_rk4.append(acc_r)

        print(f"    Steps={n_steps:2d}: Euler acc={acc_e:.3f}, RK4 acc={acc_r:.3f}")

    # ══════════════════════════════════════════════
    # Experiment 5: Memory Comparison (Adjoint vs Standard)
    # ══════════════════════════════════════════════
    print("\n=== Experiment 5: Memory Comparison ===")

    mem_results = {}
    for method_name, use_adjoint in [('standard', False), ('adjoint', True)]:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        func_m = ODEFunc(2, hidden_dim=64).to(device)
        node_m = NeuralODE(func_m, method='rk4', n_steps=50, adjoint=use_adjoint).to(device)
        head_m = nn.Linear(2, 2).to(device)
        opt_m = torch.optim.Adam(list(func_m.parameters()) + list(head_m.parameters()), lr=1e-3)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        Xd, yd = X_spiral.to(device), y_spiral.to(device)
        for _ in range(50):
            h = node_m(Xd)
            loss = F.cross_entropy(head_m(h), yd)
            opt_m.zero_grad(); loss.backward(); opt_m.step()

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024
            mem_results[method_name] = peak_mem
            print(f"    {method_name}: peak GPU memory = {peak_mem:.1f} KB")
        else:
            # Estimate memory from parameter count of saved tensors
            # Standard: saves all n_steps intermediate states
            # Adjoint: saves only final state
            n_steps = 50
            batch = X_spiral.shape[0]
            bytes_per_float = 4
            # Standard backprop stores intermediate activations
            standard_mem = n_steps * batch * 2 * bytes_per_float  # n_steps * batch * dim * 4
            adjoint_mem = 1 * batch * 2 * bytes_per_float  # 1 * batch * dim * 4
            mem_results[method_name] = {'standard': standard_mem, 'adjoint': adjoint_mem}
            print(f"    {method_name}: estimated activation memory = "
                  f"{(standard_mem if not use_adjoint else adjoint_mem) / 1024:.1f} KB")

    # ══════════════════════════════════════════════
    # Visualizations
    # ══════════════════════════════════════════════
    print("\n=== Generating Visualizations ===")

    # 1. Classification decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    X_np, y_np = X_spiral.numpy(), y_spiral.numpy()
    plot_decision_boundary(axes[0], node_model, 'neural_ode', X_spiral, y_spiral,
                           'Neural ODE', device)
    plot_decision_boundary(axes[1], resnet_model, 'resnet', X_spiral, y_spiral,
                           'ResNet (6 layers)', device)
    plt.suptitle("Neural ODE vs ResNet: Spiral Classification", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "classification_boundary.png", dpi=150)
    plt.close()

    # 2. CNF density contours
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_density(axes[0], cnf, 'cnf', device, bounds=4.0)
    plot_density(axes[1], dnf, 'dnf', device, bounds=4.0)
    axes[2].scatter(X_gmm[:, 0].cpu(), X_gmm[:, 1].cpu(), s=5, alpha=0.5)
    axes[2].set_title('Training Data')
    axes[2].set_xlim(-4, 4); axes[2].set_ylim(-4, 4)
    plt.suptitle("CNF vs Discrete NF: Density Estimation", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "density_estimation.png", dpi=150)
    plt.close()

    # 3. Trajectory snapshots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    X_traj = torch.randn(200, 2, device=device)
    with torch.no_grad():
        _, traj = ode_solve(func_cnf, X_traj, (0.0, 1.0), 'rk4', 40)

    snapshot_indices = [0, 10, 20, 30, 40]
    for ax, idx in zip(axes, snapshot_indices):
        t_val, x_val = traj[idx]
        x_val = x_val.cpu()
        ax.scatter(x_val[:, 0], x_val[:, 1], s=8, alpha=0.5, c='blue')
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_title(f't = {t_val:.2f}')
        ax.set_aspect('equal')
    plt.suptitle("CNF: Flow from Prior (t=0) to Data (t=1)", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "trajectory_snapshots.png", dpi=150)
    plt.close()

    # 4. ODE solver steps vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(step_counts, step_accs_euler, 'o-', label='Euler', color='blue')
    ax.plot(step_counts, step_accs_rk4, 's-', label='RK4', color='red')
    ax.set_xlabel('Number of ODE Solver Steps')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Solver Accuracy vs Step Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(results_dir / "solver_steps.png", dpi=150)
    plt.close()

    # 5. Memory comparison + training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss curves
    w = 30
    for losses, label, color in [(node_losses, 'Neural ODE', 'blue'),
                                  (resnet_losses, 'ResNet', 'green')]:
        smoothed = np.convolve(losses, np.ones(w) / w, mode='valid')
        axes[0].plot(smoothed, label=label, color=color, linewidth=2)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Classification Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CNF vs DNF training curves
    for losses, label, color in [(cnf_losses, 'CNF', 'blue'),
                                  (dnf_losses, 'Discrete NF', 'green')]:
        smoothed = np.convolve(losses, np.ones(w) / w, mode='valid')
        axes[1].plot(smoothed, label=label, color=color, linewidth=2)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('NLL')
    axes[1].set_title('Density Estimation Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Memory comparison bar chart
    if isinstance(list(mem_results.values())[0], dict):
        # CPU mode: use theoretical estimates
        labels = ['Standard', 'Adjoint']
        vals = [mem_results['standard']['standard'] / 1024,
                mem_results['adjoint']['adjoint'] / 1024]
    else:
        labels = ['Standard', 'Adjoint']
        vals = [mem_results['standard'], mem_results['adjoint']]
    axes[2].bar(labels, vals, color=['blue', 'red'], alpha=0.7)
    axes[2].set_ylabel('Memory (KB)')
    axes[2].set_title('Memory: Standard vs Adjoint Backprop')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Neural ODE: Training & Memory Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "training_analysis.png", dpi=150)
    plt.close()

    # 6. Concept diagram: ResNet -> Neural ODE -> CNF
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    concepts = [
        ("ResNet\n(Discrete)", "x_{l+1} = x_l + f(x_l)\n\n"
         "Fixed number of layers\nBackprop stores all\nintermediate activations\nO(L) memory",
         0.15, 'gray'),
        ("Neural ODE\n(Continuous)", "dx/dt = f_θ(x, t)\n\n"
         "Continuous depth\nAdjoint method:\nsolve ODE backwards\nO(1) memory!",
         0.5, 'blue'),
        ("CNF\n(Density)", "log p(x₁) = log p(x₀)\n  - ∫Tr(∂f/∂x)dt\n\n"
         "Instantaneous change\nof variables\nHutchinson trace estimator",
         0.85, 'green'),
    ]

    for name, desc, x_pos, color in concepts:
        ax.text(x_pos, 0.8, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.32, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Arrows
    ax.annotate('', xy=(0.32, 0.55), xytext=(0.25, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.67, 0.55), xytext=(0.60, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.text(0.285, 0.62, 'continuous\nlimit', fontsize=9, ha='center',
            style='italic', color='gray')
    ax.text(0.635, 0.62, 'change of\nvariables', fontsize=9, ha='center',
            style='italic', color='gray')

    ax.set_title("Neural ODE: From Discrete Layers to Continuous Dynamics (1806.07366)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "concept_diagram.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
