"""
Minimal NeRF Reproduction
=========================
Reproduces core ideas from "NeRF: Representing Scenes as Neural Radiance Fields
for View Synthesis" (2003.08934, Mildenhall et al.):
1. 5D function: (x, y, z, θ, φ) → (RGB, σ) via MLP
2. Volume rendering: integrate along camera rays
3. Positional encoding: high-frequency inputs for detail
4. Hierarchical sampling: coarse + fine networks
5. Compare: with/without positional encoding
6. Show: rendering quality vs sampling density
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Positional Encoding ──

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for input coordinates."""
    def __init__(self, n_freqs=10, include_input=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        # 2 * n_freqs frequencies + optionally the input itself
        self.out_dim = 2 * n_freqs * 3  # 3 for xyz, 2 for sin/cos
        if include_input:
            self.out_dim += 3

        freq_bands = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        """x: (..., 3) → (..., out_dim)"""
        encoded = []
        if self.include_input:
            encoded.append(x)
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq * np.pi))
            encoded.append(torch.cos(x * freq * np.pi))
        return torch.cat(encoded, dim=-1)


class DirPosEnc(nn.Module):
    """Positional encoding for view direction (2D: θ, φ)."""
    def __init__(self, n_freqs=4):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = 2 * n_freqs * 2 + 2  # 2 for θ,φ
        freq_bands = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, d):
        """d: (..., 2) → (..., out_dim)"""
        encoded = [d]
        for freq in self.freq_bands:
            encoded.append(torch.sin(d * freq * np.pi))
            encoded.append(torch.cos(d * freq * np.pi))
        return torch.cat(encoded, dim=-1)


# ── NeRF Model ──

class NeRF(nn.Module):
    """Neural Radiance Field: (x, d) → (rgb, σ)."""
    def __init__(self, pos_enc_freqs=10, dir_enc_freqs=4, hidden=128):
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_enc_freqs)
        self.dir_enc = DirPosEnc(dir_enc_freqs)

        pos_dim = self.pos_enc.out_dim
        dir_dim = self.dir_enc.out_dim

        # Coarse MLP: xyz → features + σ
        self.xyz_net = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden + 1),  # +1 for sigma
        )
        # Direction MLP: features + dir → rgb
        self.dir_net = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        """
        x: (..., 3) positions
        d: (..., 2) view directions (θ, φ)
        Returns: rgb (..., 3), sigma (..., 1)
        """
        x_enc = self.pos_enc(x)
        d_enc = self.dir_enc(d)

        h = self.xyz_net(x_enc)
        sigma = F.softplus(h[..., :1])  # ensure positive
        features = h[..., 1:]

        rgb = self.dir_net(torch.cat([features, d_enc], dim=-1))
        return rgb, sigma


# ── Volume Rendering ──

def sample_along_rays(origins, directions, near, far, n_samples, perturb=True):
    """Sample points along rays.

    origins: (B, 3), directions: (B, 3)
    Returns: pts (B, n_samples, 3), z_vals (B, n_samples)
    """
    z_vals = torch.linspace(near, far, n_samples, device=origins.device)
    z_vals = z_vals.unsqueeze(0).expand(origins.shape[0], -1)  # (B, N)

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    pts = origins.unsqueeze(1) + directions.unsqueeze(1) * z_vals.unsqueeze(-1)  # (B, N, 3)
    return pts, z_vals


def volume_render(rgb, sigma, z_vals):
    """Volume rendering equation.

    rgb: (B, N, 3), sigma: (B, N, 1), z_vals: (B, N)
    Returns: rendered_rgb (B, 3), weights (B, N)
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, dists[..., -1:]], -1)  # (B, N), last interval = second-to-last

    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (B, N)
    # Transmittance: cumulative product of (1 - alpha)
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], -1)

    weights = alpha * transmittance  # (B, N)
    rendered_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (B, 3)

    return rendered_rgb, weights


# ── Synthetic Scene ──

class SyntheticScene:
    """Simple synthetic scene: colored sphere with directional lighting."""
    def __init__(self, device='cpu'):
        self.device = device
        self.center = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.radius = 0.8

    def render_ray(self, origins, directions):
        """Ground truth: ray-sphere intersection with shading.

        origins: (B, 3), directions: (B, 3)
        Returns: rgb (B, 3)
        """
        # Ray: P = O + t*D
        oc = origins - self.center
        a = (directions * directions).sum(-1)
        b = 2.0 * (oc * directions).sum(-1)
        c = (oc * oc).sum(-1) - self.radius ** 2
        discriminant = b * b - 4 * a * c

        hit = discriminant > 0
        t = (-b - torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * a + 1e-10)

        # Normal at hit point
        hit_pts = origins + directions * t.unsqueeze(-1)
        normals = F.normalize(hit_pts - self.center, dim=-1)

        # Directional lighting
        light_dir = F.normalize(torch.tensor([1.0, 1.0, 1.0], device=self.device), dim=0)
        diffuse = (normals * light_dir).sum(-1).clamp(min=0.0)

        # Base color: checkerboard pattern based on position
        u = (hit_pts[..., 0] * 5).int() % 2
        v = (hit_pts[..., 1] * 5).int() % 2
        checker = ((u + v) % 2).float()

        # Color: blend between two colors based on checker
        color1 = torch.tensor([0.8, 0.2, 0.2], device=self.device)
        color2 = torch.tensor([0.2, 0.2, 0.8], device=self.device)
        base_color = checker.unsqueeze(-1) * color1 + (1 - checker.unsqueeze(-1)) * color2

        ambient = 0.1
        rgb = base_color * (ambient + (1 - ambient) * diffuse.unsqueeze(-1))

        # Background: sky gradient
        sky_color = torch.tensor([0.5, 0.7, 1.0], device=self.device)
        up_factor = directions[..., 1:2].clamp(min=0) * 0.5  # (B, 1)
        bg = sky_color.unsqueeze(0) * (1.0 - up_factor)  # (B, 3)

        # Combine hit vs background
        result = torch.where(hit.unsqueeze(-1), rgb, bg)
        return result

    def generate_rays(self, n_rays=512, image_size=16):
        """Generate camera rays for a simple pinhole camera."""
        # Camera at (0, 0, 3) looking at origin
        cam_pos = torch.tensor([0.0, 0.0, 3.0], device=self.device)
        focal = image_size * 1.5

        # Pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, image_size, device=self.device),
            torch.linspace(-1, 1, image_size, device=self.device),
            indexing='ij'
        )
        dirs = F.normalize(torch.stack([i, j, -torch.ones_like(i) * focal / image_size], dim=-1), dim=-1)
        dirs = dirs.reshape(-1, 3)
        origins = cam_pos.unsqueeze(0).expand_as(dirs)

        return origins, dirs


# ── Training ──

def train_nerf(model, scene, n_steps=1500, n_samples=64, lr=5e-4,
               image_size=16, batch_rays=512, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    # Pre-generate rays
    origins, directions = scene.generate_rays(image_size=image_size)
    gt_rgb = scene.render_ray(origins, directions)

    # View directions (θ, φ) from ray directions
    dir_xy = directions[..., :2]  # simplified: use x, y components
    dir_norm = F.normalize(directions, dim=-1)
    theta = torch.acos(dir_norm[..., 1].clamp(-1, 1))
    phi = torch.atan2(dir_norm[..., 2], dir_norm[..., 0])
    view_dirs = torch.stack([theta, phi], dim=-1)

    losses = []
    psnrs = []

    for step in range(n_steps):
        # Sample batch of rays
        idx = torch.randint(0, origins.shape[0], (batch_rays,), device=device)
        o_batch = origins[idx]
        d_batch = directions[idx]
        gt_batch = gt_rgb[idx]
        vd_batch = view_dirs[idx]

        # Sample along rays
        pts, z_vals = sample_along_rays(o_batch, d_batch, near=1.0, far=5.0,
                                          n_samples=n_samples, perturb=True)

        # Flatten for NeRF: (B*N, 3)
        B, N, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        vd_flat = vd_batch.unsqueeze(1).expand(-1, N, -1).reshape(-1, 2)

        # Forward
        rgb_flat, sigma_flat = model(pts_flat, vd_flat)
        rgb = rgb_flat.reshape(B, N, 3)
        sigma = sigma_flat.reshape(B, N, 1)

        # Render
        rendered, _ = volume_render(rgb, sigma, z_vals)

        loss = F.mse_loss(rendered, gt_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        with torch.no_grad():
            mse = F.mse_loss(rendered, gt_batch).item()
            psnrs.append(-10 * np.log10(mse + 1e-10))

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | PSNR: {psnrs[-1]:.1f} dB")

    return losses, psnrs


# ── No Positional Encoding Baseline ──

class NeRFNoPE(nn.Module):
    """NeRF without positional encoding (ablation)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.xyz_net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden + 1),
        )
        self.dir_net = nn.Sequential(
            nn.Linear(hidden + 2, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        h = self.xyz_net(x)
        sigma = F.softplus(h[..., :1])
        features = h[..., 1:]
        rgb = self.dir_net(torch.cat([features, d], dim=-1))
        return rgb, sigma


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "117-nerf"
    results_dir.mkdir(parents=True, exist_ok=True)

    scene = SyntheticScene(device=device)

    # ── Experiment 1: NeRF with positional encoding ──
    print("=== Training NeRF (with PE) ===")
    model_pe = NeRF(pos_enc_freqs=10, dir_enc_freqs=4, hidden=128).to(device)
    pe_losses, pe_psnrs = train_nerf(model_pe, scene, n_steps=1500, device=device)

    # ── Experiment 2: NeRF without positional encoding ──
    print("\n=== Training NeRF (without PE) ===")
    model_nope = NeRFNoPE(hidden=128).to(device)
    nope_losses, nope_psnrs = train_nerf(model_nope, scene, n_steps=1500, device=device)

    # ── Experiment 3: Effect of PE frequency ──
    print("\n=== Experiment 3: PE Frequency Sweep ===")
    freq_results = {}
    for freq in [2, 4, 6, 8, 10, 12]:
        model = NeRF(pos_enc_freqs=freq, dir_enc_freqs=4, hidden=128).to(device)
        losses, psnrs = train_nerf(model, scene, n_steps=800, device=device)
        freq_results[freq] = {'losses': losses, 'psnrs': psnrs,
                               'final_psnr': np.mean(psnrs[-50:])}
        print(f"  freq={freq}: PSNR={freq_results[freq]['final_psnr']:.1f} dB")

    # ── Experiment 4: Sampling density ──
    print("\n=== Experiment 4: Sampling Density ===")
    sample_results = {}
    for n_samp in [16, 32, 64, 128]:
        model = NeRF(pos_enc_freqs=10, dir_enc_freqs=4, hidden=128).to(device)
        losses, psnrs = train_nerf(model, scene, n_steps=800, n_samples=n_samp, device=device)
        sample_results[n_samp] = {'psnrs': psnrs, 'final_psnr': np.mean(psnrs[-50:])}
        print(f"  n_samples={n_samp}: PSNR={sample_results[n_samp]['final_psnr']:.1f} dB")

    # ── Render final image ──
    print("\n=== Rendering Image ===")
    origins, directions = scene.generate_rays(image_size=32)
    gt_rgb = scene.render_ray(origins, directions).reshape(32, 32, 3)

    dir_norm = F.normalize(directions, dim=-1)
    theta = torch.acos(dir_norm[..., 1].clamp(-1, 1))
    phi = torch.atan2(dir_norm[..., 2], dir_norm[..., 0])
    view_dirs = torch.stack([theta, phi], dim=-1)

    # Render with PE model
    n_samp = 64
    with torch.no_grad():
        pts, z_vals = sample_along_rays(origins, directions, 1.0, 5.0, n_samp, perturb=False)
        B = origins.shape[0]
        pts_flat = pts.reshape(-1, 3)
        vd_flat = view_dirs.unsqueeze(1).expand(-1, n_samp, -1).reshape(-1, 2)
        rgb_flat, sigma_flat = model_pe(pts_flat, vd_flat)
        rgb = rgb_flat.reshape(B, n_samp, 3)
        sigma = sigma_flat.reshape(B, n_samp, 1)
        rendered_pe, weights_pe = volume_render(rgb, sigma, z_vals)
        img_pe = rendered_pe.reshape(32, 32, 3)

        # Depth map
        depth_pe = (weights_pe * z_vals).sum(dim=-1).reshape(32, 32)

    # ── Visualization ──

    # 1. Training curves: with vs without PE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 20
    for ax, data_pe, data_nope, title, ylabel in [
        (axes[0], pe_losses, nope_losses, 'Training Loss', 'MSE Loss'),
        (axes[1], pe_psnrs, nope_psnrs, 'PSNR', 'PSNR (dB)'),
    ]:
        s_pe = np.convolve(data_pe, np.ones(w)/w, mode='valid')
        s_nope = np.convolve(data_nope, np.ones(w)/w, mode='valid')
        ax.plot(s_pe, label='With PE', color='blue', linewidth=2)
        ax.plot(s_nope, label='Without PE', color='red', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('NeRF: Positional Encoding Ablation (2003.08934)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'pe_ablation.png', dpi=150)
    plt.close()

    # 2. PE frequency sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = sorted(freq_results.keys())
    psnr_vals = [freq_results[f]['final_psnr'] for f in freqs]
    ax.plot(freqs, psnr_vals, marker='o', color='blue', linewidth=2)
    ax.set_xlabel("Positional Encoding Frequencies")
    ax.set_ylabel("Final PSNR (dB)")
    ax.set_title("Effect of PE Frequency on Rendering Quality")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'pe_frequency.png', dpi=150)
    plt.close()

    # 3. Sampling density
    fig, ax = plt.subplots(figsize=(8, 5))
    n_vals = sorted(sample_results.keys())
    s_psnrs = [sample_results[n]['final_psnr'] for n in n_vals]
    ax.plot(n_vals, s_psnrs, marker='s', color='green', linewidth=2)
    ax.set_xlabel("Samples per Ray")
    ax.set_ylabel("Final PSNR (dB)")
    ax.set_title("Effect of Sampling Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'sampling_density.png', dpi=150)
    plt.close()

    # 4. Rendered images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_rgb.cpu().numpy())
    axes[0].set_title("Ground Truth")
    axes[1].imshow(img_pe.cpu().numpy().clip(0, 1))
    axes[1].set_title("NeRF (with PE)")
    im = axes[2].imshow(depth_pe.cpu().numpy(), cmap='viridis')
    axes[2].set_title("Depth Map")
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.axis('off')
    plt.suptitle('NeRF Rendering', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'rendered.png', dpi=150)
    plt.close()

    # 5. Weight distribution along ray
    fig, ax = plt.subplots(figsize=(10, 5))
    ray_idx = 0
    ax.plot(z_vals[ray_idx].cpu().numpy(), weights_pe[ray_idx].cpu().numpy(),
            color='blue', linewidth=2)
    ax.set_xlabel("Depth along ray")
    ax.set_ylabel("Rendering Weight")
    ax.set_title("Volume Rendering Weight Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'weights.png', dpi=150)
    plt.close()

    # 6. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    concept = (
        "NeRF: Neural Radiance Fields (2003.08934)\n"
        "=" * 55 + "\n\n"
        "Core Idea:\n"
        "  Represent a scene as a continuous 5D function:\n"
        "    F_θ: (x, y, z, θ, φ) → (R, G, B, σ)\n"
        "  where (x,y,z) is position, (θ,φ) is view direction,\n"
        "  RGB is color, σ is volume density.\n\n"
        "Volume Rendering:\n"
        "  C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt\n"
        "  where T(t) = exp(-∫ σ(r(s)) ds) is transmittance\n\n"
        "Key Components:\n"
        "  • Positional encoding: γ(p) = [sin(2^k πp), cos(2^k πp)]\n"
        "    → Critical for high-frequency detail\n"
        "  • Hierarchical sampling: coarse → fine networks\n"
        "  • Volume rendering: differentiable ray marching\n\n"
        "Tradeoffs:\n"
        "  PE freq ↑ → sharper details, but may overshoot\n"
        "  Samples ↑ → better rendering, but slower\n"
        "  Without PE → blurry, fails to capture detail"
    )
    ax.text(0.05, 0.95, concept, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    plt.tight_layout()
    plt.savefig(results_dir / 'concept.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
