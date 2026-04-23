"""
Minimal 3D Gaussian Splatting Reproduction
==========================================
Reproduces core ideas from "3D Gaussian Splatting for Real-Time Radiance
Field Rendering" (Kerbl et al., 2023, arxiv 2308.04079):
1. Represent scene as a set of 3D Gaussians (position, covariance, color, opacity)
2. Differentiable splatting: project 3D Gaussians to 2D via EWA splatting
3. Optimize Gaussian parameters to match target multi-view images
4. Synthetic scene: colored sphere rendered from multiple viewpoints
5. Show: rendered views over training, Gaussian position evolution, PSNR convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi


# ── Camera & Projection ──

def look_at(eye, target, up):
    """Build view matrix (world-to-camera)."""
    eye = torch.tensor(eye, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    fwd = F.normalize(target - eye, dim=0)
    right = F.normalize(fwd.cross(up), dim=0)
    up2 = right.cross(fwd)
    R = torch.stack([right, up2, -fwd], dim=0)  # 3x3
    t = R @ eye
    V = torch.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -t
    return V


def perspective(fov_deg, aspect, near=0.1, far=10.0):
    """Build perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = 2 * far * near / (near - far)
    P[3, 2] = -1
    return P


# ── Synthetic Scene ──

def render_sphere_groundtruth(cam_pos, img_size=64, radius=0.8):
    """Render a shaded sphere from a given camera position."""
    eye = cam_pos
    target = [0.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]

    V = look_at(eye, target, up)
    P = perspective(50, 1.0)

    # Generate pixel rays
    u = torch.linspace(-1, 1, img_size)
    v = torch.linspace(1, -1, img_size)  # flip y
    U, Vg = torch.meshgrid(u, v, indexing='ij')
    pixels = torch.stack([U, Vg, torch.ones_like(U)], dim=-1).reshape(-1, 3)  # N,3

    # Ray directions in camera space -> world space
    R_wc = V[:3, :3].T
    cam_origin = V[:3, 3].clone()
    # Undo the negative in V construction
    cam_origin_w = -R_wc @ V[:3, 3]

    # Simple ray-sphere intersection for ground truth
    # Sphere at origin, radius r
    r = radius
    sphere_center = torch.tensor([0.0, 0.0, 0.0])

    image = torch.zeros(img_size * img_size, 3)
    depth = torch.full((img_size * img_size,), float('inf'))

    for i in range(pixels.shape[0]):
        # Ray in camera space
        d_cam = pixels[i]  # (x, y, 1)
        d_cam_h = torch.tensor([d_cam[0], d_cam[1], 1.0, 1.0])

        # Transform to world
        P_inv = torch.eye(4)
        P_inv[0, 0] = 1.0 / P[0, 0]
        P_inv[1, 1] = 1.0 / P[1, 1]
        P_inv[2, 2] = 0
        P_inv[2, 3] = 1.0 / (-P[3, 2])
        P_inv[3, 2] = (P[2, 2] / P[3, 2])
        P_inv[3, 3] = P[2, 3] / (-P[3, 2])

        d_world = R_wc @ d_cam  # approximate direction in world space

        # Ray-sphere intersection
        o = cam_origin_w
        d = F.normalize(d_world, dim=0)
        oc = o - sphere_center
        a = d @ d
        b = 2.0 * (oc @ d)
        c = oc @ oc - r * r
        disc = b * b - 4 * a * c

        if disc >= 0:
            t = (-b - torch.sqrt(disc)) / (2 * a)
            if t > 0:
                hit = o + t * d
                normal = F.normalize(hit - sphere_center, dim=0)
                # Simple shading: diffuse + ambient
                light_dir = F.normalize(torch.tensor([1.0, 1.0, 1.0]), dim=0)
                diffuse = max(normal @ light_dir, 0.0)
                # Color: blue-ish sphere with shading
                base_color = torch.tensor([0.2, 0.4, 0.9])
                image[i] = base_color * (0.3 + 0.7 * diffuse)
                depth[i] = t

    return image.reshape(img_size, img_size, 3), depth.reshape(img_size, img_size)


def generate_training_views(n_views=6, img_size=64, radius=0.8):
    """Generate synthetic multi-view images of a sphere."""
    images = []
    depths = []
    cam_positions = []
    for i in range(n_views):
        angle = 2 * pi * i / n_views
        eye = [3.0 * np.cos(angle), 1.5, 3.0 * np.sin(angle)]
        img, dep = render_sphere_groundtruth(eye, img_size, radius)
        images.append(img)
        depths.append(dep)
        cam_positions.append(eye)
    return images, depths, cam_positions


# ── 3D Gaussian Model ──

class GaussianSplatModel:
    """Collection of 3D Gaussians with differentiable rendering."""

    def __init__(self, n_gaussians=500, init_range=1.5, device='cpu'):
        self.n = n_gaussians
        self.device = device

        # Positions: randomly initialize around origin
        self.positions = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device) * init_range
        )
        # Log-scaling for covariance (3D: sx, sy, sz)
        self.log_scales = nn.Parameter(
            torch.zeros(n_gaussians, 3, device=device) + 0.5
        )
        # Rotation as quaternion (w, x, y, z) — keep simple with identity-ish
        self.quats = nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_gaussians, device=device)
        )
        # RGB colors
        self.colors = nn.Parameter(
            torch.rand(n_gaussians, 3, device=device)
        )
        # Log-opacity
        self.log_opacities = nn.Parameter(
            torch.zeros(n_gaussians, 1, device=device)
        )

    def parameters(self):
        return [self.positions, self.log_scales, self.quats,
                self.colors, self.log_opacities]

    def build_covariance_3d(self):
        """Build 3D covariance from scales and rotation."""
        s = torch.exp(self.log_scales)  # (N, 3)
        S = torch.diag_embed(s)  # (N, 3, 3)

        # Normalize quaternions
        q = F.normalize(self.quats, dim=-1)  # (N, 4)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Quaternion to rotation matrix
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)  # (N, 3, 3)

        # Sigma = R S S^T R^T
        RS = torch.bmm(R, S)
        cov = torch.bmm(RS, RS.transpose(1, 2))
        return cov

    def project_to_2d(self, V, P, img_size=64):
        """Project 3D Gaussians to 2D with EWA splatting approximation."""
        N = self.n
        # Transform positions to camera space
        pos_h = torch.cat([self.positions, torch.ones(N, 1, device=self.device)], dim=-1)  # (N, 4)
        pos_cam = (V @ pos_h.T).T[:, :3]  # (N, 3)

        # Only keep Gaussians in front of camera
        valid = pos_cam[:, 2] < -0.1
        if not valid.any():
            return None, None, None, valid

        # Project to screen
        pos_clip = (P @ (V @ pos_h.T)).T  # (N, 4)
        # NDC to pixel
        pos_2d = pos_clip[:, :2] / (pos_clip[:, 3:4] + 1e-8)  # (N, 2) in [-1, 1]
        depth = pos_cam[:, 2]

        # Project 3D covariance to 2D (Jacobian approximation)
        cov_3d = self.build_covariance_3d()
        J = torch.zeros(N, 2, 3, device=self.device)
        # Jacobian of perspective projection: [fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]
        fx = P[0, 0]
        fy = P[1, 1]
        xc = pos_cam[:, 0]
        yc = pos_cam[:, 1]
        zc = pos_cam[:, 2]
        zc_sq = zc * zc + 1e-6
        J[:, 0, 0] = fx / (zc + 1e-6)
        J[:, 0, 2] = -fx * xc / zc_sq
        J[:, 1, 1] = fy / (zc + 1e-6)
        J[:, 1, 2] = -fy * yc / zc_sq

        # 2D covariance: J * Sigma * J^T
        JC = torch.bmm(J, cov_3d)  # (N, 2, 3)
        cov_2d = torch.bmm(JC, J.transpose(1, 2))  # (N, 2, 2)

        # Add small value for stability
        cov_2d = cov_2d + 1e-4 * torch.eye(2, device=self.device).unsqueeze(0)

        return pos_2d, cov_2d, depth, valid

    def render(self, V, P, img_size=64):
        """Differentiable splatting: render image from Gaussians."""
        pos_2d, cov_2d, depth, valid = self.project_to_2d(V, P, img_size)
        if pos_2d is None:
            return torch.ones(img_size, img_size, 3, device=self.device) * 0.5

        opacities = torch.sigmoid(self.log_opacities).squeeze(-1)  # (N,)
        colors = torch.sigmoid(self.colors)  # (N, 3)

        # Sort by depth (back-to-front)
        sort_idx = torch.argsort(-depth)  # far to near

        # Create pixel grid
        u = torch.linspace(-1, 1, img_size, device=self.device)
        v = torch.linspace(1, -1, img_size, device=self.device)
        U, Vg = torch.meshgrid(u, v, indexing='ij')
        pixels = torch.stack([U, Vg], dim=-1).reshape(-1, 2)  # (M, 2)

        # For efficiency, only render Gaussians within the frustum
        image = torch.zeros(img_size * img_size, 3, device=self.device)
        alpha_acc = torch.zeros(img_size * img_size, 1, device=self.device)

        # Batch rendering: evaluate each Gaussian's contribution
        n_valid = valid.sum().item()
        # Subsample Gaussians if too many for memory
        max_render = min(n_valid, 300)
        valid_indices = torch.where(valid)[0]
        if len(valid_indices) > max_render:
            perm = torch.randperm(len(valid_indices), device=self.device)[:max_render]
            valid_indices = valid_indices[perm]

        for idx in valid_indices:
            mu = pos_2d[idx]  # (2,)
            S = cov_2d[idx]   # (2, 2)
            alpha = opacities[idx]
            col = colors[idx]  # (3,)

            # Skip if Gaussian center is far outside image
            if mu[0].abs() > 2.0 or mu[1].abs() > 2.0:
                continue

            # Compute screen-space extent (2 sigma)
            try:
                eigvals = torch.linalg.eigvalsh(S)
                if (eigvals <= 0).any():
                    continue
                radius = 3.0 * torch.sqrt(eigvals.max())
            except Exception:
                continue

            if radius > 2.0:
                continue

            # Mahalanobis distance for all pixels
            diff = pixels - mu.unsqueeze(0)  # (M, 2)

            # Inverse of cov_2d
            try:
                S_inv = torch.inverse(S)
            except Exception:
                continue

            # Gaussian weight: exp(-0.5 * diff^T S_inv diff)
            mahal = (diff @ S_inv * diff).sum(dim=-1)  # (M,)
            weight = torch.exp(-0.5 * mahal)  # (M,)

            # Alpha compositing (front-to-back)
            T = 1.0 - alpha_acc.squeeze(-1)  # (M,) transmittance
            contrib = T * weight * alpha  # (M,)

            image += contrib.unsqueeze(-1) * col.unsqueeze(0)
            alpha_acc += contrib.unsqueeze(-1)

            # Early termination
            if (alpha_acc > 0.99).all():
                break

        # Background
        T_bg = (1.0 - alpha_acc)  # (M, 1) transmittance for background
        bg_mask = (T_bg > 0.01).squeeze(-1)  # (M,) pixels needing background
        image[bg_mask] += T_bg[bg_mask] * 0.8  # gray bg

        return image.reshape(img_size, img_size, 3)


# ── Training ──

def train_gaussian_splatting(n_gaussians=500, n_views=6, img_size=64,
                              epochs=1500, lr=5e-3, device='cpu'):
    """Optimize 3D Gaussians to reproduce multi-view target images."""
    model = GaussianSplatModel(n_gaussians, device=device)

    # Generate target views
    target_images, _, cam_positions = generate_training_views(n_views, img_size)
    targets = [img.to(device) for img in target_images]

    # Build camera matrices
    cameras = []
    for eye in cam_positions:
        V = look_at(eye, [0, 0, 0], [0, 1, 0]).to(device)
        P = perspective(50, 1.0).to(device)
        cameras.append((V, P))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.5)

    losses = []
    psnrs = []

    for epoch in range(epochs):
        total_loss = 0
        for vi in range(n_views):
            V, P = cameras[vi]
            rendered = model.render(V, P, img_size)
            target = targets[vi]

            loss = F.mse_loss(rendered, target)
            total_loss += loss

        avg_loss = total_loss / n_views
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        scheduler.step()

        mse_val = avg_loss.item()
        losses.append(mse_val)
        psnr_val = 10 * np.log10(1.0 / (mse_val + 1e-10))
        psnrs.append(psnr_val)

        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: loss={mse_val:.6f}, PSNR={psnr_val:.2f} dB")

    return model, losses, psnrs, cameras, targets


# ── Visualization ──

def visualize_results(model, losses, psnrs, cameras, targets, img_size=64,
                      save_dir=None):
    """Generate all visualization plots."""

    # 1. PSNR convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(psnrs, color='blue', alpha=0.3, label='raw')
    window = min(50, len(psnrs))
    if window > 1:
        smoothed = np.convolve(psnrs, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, window-1+len(smoothed)), smoothed,
                color='blue', linewidth=2, label='smoothed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('3D Gaussian Splatting: PSNR Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'psnr_convergence.png', dpi=150)
    plt.close()

    # 2. Rendered vs target views
    device = model.positions.device
    n_views = len(cameras)
    fig, axes = plt.subplots(2, n_views, figsize=(3 * n_views, 6))
    for vi in range(n_views):
        V, P = cameras[vi]
        with torch.no_grad():
            rendered = model.render(V, P, img_size).cpu().numpy()
        target = targets[vi].cpu().numpy()

        axes[0, vi].imshow(np.clip(rendered, 0, 1))
        axes[0, vi].set_title(f'View {vi+1}\n(Rendered)', fontsize=9)
        axes[0, vi].axis('off')

        axes[1, vi].imshow(np.clip(target, 0, 1))
        axes[1, vi].set_title(f'Target', fontsize=9)
        axes[1, vi].axis('off')

    plt.suptitle('3D Gaussian Splatting: Rendered vs Target', fontsize=13)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'rendered_vs_target.png', dpi=150)
    plt.close()

    # 3. Gaussian positions in 3D
    pos = model.positions.detach().cpu().numpy()
    opacities = torch.sigmoid(model.log_opacities).detach().cpu().numpy().squeeze()
    colors = torch.sigmoid(model.colors).detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Filter visible Gaussians (high opacity)
    visible = opacities > 0.3
    vis_pos = pos[visible]
    vis_colors = colors[visible]
    vis_opacities = opacities[visible]

    sc = ax.scatter(vis_pos[:, 0], vis_pos[:, 1], vis_pos[:, 2],
                    c=vis_colors, alpha=vis_opacities * 0.6, s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Gaussian Positions ({visible.sum()} visible / {len(pos)} total)')
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'gaussian_positions_3d.png', dpi=150)
    plt.close()

    # 4. Gaussian scale distribution
    scales = torch.exp(model.log_scales).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, name) in enumerate(zip(axes, ['Scale X', 'Scale Y', 'Scale Z'])):
        ax.hist(scales[:, i], bins=40, color=['red', 'green', 'blue'][i], alpha=0.7)
        ax.set_xlabel('Scale')
        ax.set_ylabel('Count')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    plt.suptitle('3D Gaussian Scale Distribution')
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'scale_distribution.png', dpi=150)
    plt.close()

    # 5. Opacity distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(opacities, bins=40, color='purple', alpha=0.7)
    ax.set_xlabel('Opacity')
    ax.set_ylabel('Count')
    ax.set_title('Gaussian Opacity Distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'opacity_distribution.png', dpi=150)
    plt.close()


# ── Main ──

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results_dir = Path(__file__).parent / "results" / "79-gaussian-splatting"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Training 3D Gaussian Splatting ===")
    model, losses, psnrs, cameras, targets = train_gaussian_splatting(
        n_gaussians=500, n_views=6, img_size=64, epochs=1500, device=device
    )

    print(f"\nFinal PSNR: {psnrs[-1]:.2f} dB")
    print(f"Peak PSNR: {max(psnrs):.2f} dB")

    print("\n=== Generating Visualizations ===")
    visualize_results(model, losses, psnrs, cameras, targets,
                      img_size=64, save_dir=results_dir)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
