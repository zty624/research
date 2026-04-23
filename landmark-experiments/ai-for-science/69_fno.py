"""
Minimal Fourier Neural Operator (FNO) Reproduction
===================================================
Reproduces the core ideas from "Fourier Neural Operator for Parametric
Partial Differential Equations" (Li et al., 2020, arxiv 2010.08895):
1. Learn mapping between infinite-dimensional function spaces
2. Spectral convolution: mix in Fourier space, truncate high frequencies
3. Compare FNO vs MLP on parametric PDE (Darcy flow)
4. Resolution invariance: train at low res, test at high res
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Spectral Convolution ──

class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear transform → IFFT."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # max modes in dim 1
        self.modes2 = modes2  # max modes in dim 2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """Complex multiplication: (B, C_in, M1, M2) x (C_in, C_out, M1, M2) → (B, C_out, M1, M2)."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # IFFT
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D PDEs."""
    def __init__(self, modes1=12, modes2=12, width=32, in_channels=1, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for coordinate grid

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, H, W, C) — input field
        grid = self._get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=-1)  # (B, H, W, C+2)

        x = self.fc0(x)  # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)

        # 4 Fourier layers with residual connections
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)  # (B, H, W, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def _get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        gridx = gridx.unsqueeze(0).repeat(batchsize, 1, 1)
        gridy = gridy.unsqueeze(0).repeat(batchsize, 1, 1)
        return torch.stack([gridx, gridy], dim=-1)


# ── Darcy Flow Dataset (Synthetic) ──

def generate_darcy_data(n_samples, resolution, device='cpu'):
    """Generate synthetic Darcy flow data: a(x) → u(x).

    Simplified: a(x) is a random field, u(x) is the solution to
    -∇·(a(x)∇u(x)) = f(x) with Dirichlet BC.
    We approximate u with a smoothed version of f/a for speed.
    """
    # Random coefficient field a(x) with spatial correlation
    noise = torch.randn(n_samples, resolution, resolution, 1, device=device)
    # Smooth with Gaussian-like filter
    kernel_size = max(3, resolution // 16)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 4
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel_1d = torch.exp(-ax**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

    # Apply smoothing per sample
    a = F.conv2d(noise.permute(0, 3, 1, 2), kernel_2d, padding=kernel_size // 2)
    a = 1 + 0.5 * a.permute(0, 2, 3, 1)  # a ∈ [0.5, 1.5]
    a = a.clamp(0.1, 3.0)

    # Source term: constant
    f = torch.ones(n_samples, resolution, resolution, 1, device=device)

    # Approximate solution: u ≈ smooth(f / a) (not exact but OK for demo)
    u_approx = f / a
    # Smooth again
    u_approx = F.conv2d(u_approx.permute(0, 3, 1, 2), kernel_2d, padding=kernel_size // 2)
    u_approx = u_approx.permute(0, 2, 3, 1)

    # Apply Dirichlet BC (zero on boundary)
    u_approx[:, 0, :, :] = 0
    u_approx[:, -1, :, :] = 0
    u_approx[:, :, 0, :] = 0
    u_approx[:, :, -1, :] = 0

    return a, u_approx


# ── Baseline MLP ──

class BaselineMLP(nn.Module):
    """Pointwise MLP baseline (no spatial structure)."""
    def __init__(self, in_channels=3, hidden=128, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, a, grid):
        """a: (B, H, W, 1), grid: (B, H, W, 2)"""
        x = torch.cat([a, grid], dim=-1)
        return self.net(x)


# ── Training ──

def train_fno(n_train=500, n_test=100, resolution=64, epochs=200,
              batch_size=16, lr=1e-3, device='cpu'):
    """Train FNO on Darcy flow."""
    print(f"Generating Darcy flow data (res={resolution})...")
    a_train, u_train = generate_darcy_data(n_train, resolution, device)
    a_test, u_test = generate_darcy_data(n_test, resolution, device)

    model = FNO2d(modes1=12, modes2=12, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = indices[i:i+batch_size]
            a_batch, u_batch = a_train[idx], u_train[idx]

            optimizer.zero_grad()
            pred = model(a_batch)
            loss = ((pred - u_batch) ** 2).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)

        # Test
        model.eval()
        with torch.no_grad():
            pred_test = model(a_test)
            test_loss = ((pred_test - u_test) ** 2).mean().item()
            test_losses.append(test_loss)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train={avg_train:.6f}, test={test_loss:.6f}")

    return model, train_losses, test_losses, (a_test, u_test)


def train_mlp_baseline(n_train=500, n_test=100, resolution=64, epochs=200,
                       batch_size=16, lr=1e-3, device='cpu'):
    """Train MLP baseline on Darcy flow."""
    a_train, u_train = generate_darcy_data(n_train, resolution, device)
    a_test, u_test = generate_darcy_data(n_test, resolution, device)

    model = BaselineMLP(in_channels=3, hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Precompute grid
    gridx = torch.linspace(0, 1, resolution, device=device)
    gridy = torch.linspace(0, 1, resolution, device=device)
    GX, GY = torch.meshgrid(gridx, gridy, indexing='ij')
    grid = torch.stack([GX, GY], dim=-1).unsqueeze(0).expand(n_train, -1, -1, -1)
    grid_test = grid[:n_test]

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = indices[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(a_train[idx], grid[idx])
            loss = ((pred - u_train[idx]) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            pred_test = model(a_test, grid_test)
            test_loss = ((pred_test - u_test) ** 2).mean().item()
            test_losses.append(test_loss)

        if epoch % 20 == 0:
            print(f"  [MLP] Epoch {epoch}: train={avg_train:.6f}, test={test_loss:.6f}")

    return model, train_losses, test_losses


# ── Resolution Invariance Test ──

def test_resolution_invariance(model, train_res=64, test_res=128, device='cpu'):
    """Test FNO at different resolution than training."""
    a_test, u_test = generate_darcy_data(20, test_res, device)
    model.eval()
    with torch.no_grad():
        pred = model(a_test)
        error = ((pred - u_test) ** 2).mean().item()
    print(f"  Train res={train_res}, Test res={test_res}: error={error:.6f}")
    return error


# ── Visualization ──

def visualize_results(model, a_test, u_test, save_dir=None):
    """Visualize FNO predictions."""
    model.eval()
    with torch.no_grad():
        pred = model(a_test)

    idx = 0
    a_np = a_test[idx, :, :, 0].cpu().numpy()
    u_np = u_test[idx, :, :, 0].cpu().numpy()
    p_np = pred[idx, :, :, 0].cpu().numpy()
    err = np.abs(u_np - p_np)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    im0 = axes[0].imshow(a_np, cmap='viridis')
    axes[0].set_title('Input a(x)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(u_np, cmap='viridis')
    axes[1].set_title('Ground Truth u(x)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(p_np, cmap='viridis')
    axes[2].set_title('FNO Prediction')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(err, cmap='hot')
    axes[3].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.suptitle('Fourier Neural Operator: Darcy Flow', y=1.02)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'fno_darcy.png', dpi=150, bbox_inches='tight')
    plt.close()


def compare_fno_mlp(fno_train, fno_test, mlp_train, mlp_test, save_dir=None):
    """Compare FNO vs MLP convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(fno_train, label='FNO (train)')
    axes[0].semilogy(mlp_train, label='MLP (train)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(fno_test, label='FNO (test)')
    axes[1].semilogy(mlp_test, label='MLP (test)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Test Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('FNO vs MLP: Darcy Flow Prediction')
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'fno_vs_mlp.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    save_dir = Path(__file__).parent / 'results' / 'fno'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train FNO
    print("\n=== Training FNO ===")
    fno_model, fno_train, fno_test, (a_test, u_test) = train_fno(
        n_train=500, epochs=200, device=device)
    visualize_results(fno_model, a_test, u_test, save_dir)

    # Train MLP baseline
    print("\n=== Training MLP Baseline ===")
    _, mlp_train, mlp_test = train_mlp_baseline(n_train=500, epochs=200, device=device)
    compare_fno_mlp(fno_train, fno_test, mlp_train, mlp_test, save_dir)

    # Resolution invariance
    print("\n=== Resolution Invariance Test ===")
    for res in [32, 64, 128]:
        test_resolution_invariance(fno_model, test_res=res, device=device)

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
