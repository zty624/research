"""
Minimal Equivariant GNN Reproduction
=====================================
Reproduces core ideas from E(n) Equivariant GNNs (EGNN, 2102.09844, Satorras et al.):
1. E(n) equivariance: transformations commute with rotations/translations
2. Equivariant message passing: update both positions and features
3. Coordinate update: positions shifted by equivariant messages
4. Compare: EGNN vs plain GNN on invariant tasks
5. Show: equivariance verification test
6. Demonstrate: stability under rotation of inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Equivariant Layers ──

class EGNNLayer(nn.Module):
    """One layer of E(n) Equivariant Graph Neural Network.

    Updates both node features h and coordinates x equivariantly.
    """
    def __init__(self, node_dim=64, edge_dim=32, n_heads=1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Message MLP: takes (h_i, h_j, d_ij^2, a_ij) → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        # Coordinate MLP: takes message → coordinate update scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
        )
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm_h = nn.LayerNorm(node_dim)
        self.norm_x = nn.LayerNorm(node_dim)  # per-feature norm for coords

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        h: (N, node_dim) node features
        x: (N, 3) node coordinates
        edge_index: (2, E) source, target
        edge_attr: (E, edge_dim) edge features
        Returns: updated h, x
        """
        src, tgt = edge_index
        N = h.shape[0]

        # Pairwise distances squared
        diff = x[src] - x[tgt]  # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge attributes
        if edge_attr is None:
            edge_attr = torch.zeros(dist_sq.shape[0], self.edge_dim, device=h.device)

        # Message: (h_i, h_j, d_ij^2, a_ij)
        msg_input = torch.cat([h[src], h[tgt], dist_sq, edge_attr], dim=-1)
        msg = self.edge_mlp(msg_input)  # (E, node_dim)

        # Coordinate update: equivariant
        coord_weight = self.coord_mlp(msg)  # (E, 1)
        coord_update = coord_weight * diff  # (E, 3)
        # Aggregate: mean over edges targeting each node
        x_agg = torch.zeros_like(x)
        count = torch.zeros(N, 1, device=h.device)
        x_agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(coord_update), coord_update)
        count.scatter_add_(0, tgt.unsqueeze(-1), torch.ones(tgt.shape[0], 1, device=h.device))
        x = x + x_agg / (count + 1e-8)

        # Node feature update: invariant
        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
        count_h = torch.zeros(N, 1, device=h.device)
        count_h.scatter_add_(0, tgt.unsqueeze(-1), torch.ones(tgt.shape[0], 1, device=h.device))
        h_agg = h_agg / (count_h + 1e-8)

        h = h + self.node_mlp(torch.cat([h, h_agg], dim=-1))
        h = self.norm_h(h)

        return h, x


class PlainGNNLayer(nn.Module):
    """Standard GNN layer (non-equivariant) for comparison."""
    def __init__(self, node_dim=64, edge_dim=32):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, x, edge_index, edge_attr=None):
        src, tgt = edge_index
        dist_sq = ((x[src] - x[tgt]) ** 2).sum(dim=-1, keepdim=True)
        if edge_attr is None:
            edge_attr = torch.zeros(dist_sq.shape[0], 1, device=h.device)
        msg = self.msg_mlp(torch.cat([h[src], h[tgt], dist_sq, edge_attr], dim=-1))

        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
        count = torch.zeros(h.shape[0], 1, device=h.device)
        count.scatter_add_(0, tgt.unsqueeze(-1), torch.ones(tgt.shape[0], 1, device=h.device))
        h_agg = h_agg / (count + 1e-8)

        h = h + self.update_mlp(torch.cat([h, h_agg], dim=-1))
        h = self.norm(h)
        return h, x


# ── Full Models ──

class EGNNModel(nn.Module):
    """E(n) Equivariant GNN for property prediction."""
    def __init__(self, n_layers=4, node_dim=64, edge_dim=16, out_dim=1):
        super().__init__()
        self.node_emb = nn.Linear(1, node_dim)  # atom type → feature
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, edge_dim) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, out_dim)
        )

    def forward(self, atom_types, positions, edge_index, edge_attr=None):
        h = self.node_emb(atom_types.unsqueeze(-1))
        x = positions
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        # Global pooling (mean)
        h_global = h.mean(dim=0, keepdim=True)
        return self.head(h_global).squeeze()


class PlainGNNModel(nn.Module):
    """Non-equivariant GNN baseline."""
    def __init__(self, n_layers=4, node_dim=64, edge_dim=16, out_dim=1):
        super().__init__()
        self.node_emb = nn.Linear(1, node_dim)
        # Also embed positions as features (so it can learn invariance)
        self.pos_emb = nn.Linear(3, node_dim)
        self.layers = nn.ModuleList([
            PlainGNNLayer(node_dim, edge_dim) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, out_dim)
        )

    def forward(self, atom_types, positions, edge_index, edge_attr=None):
        h = self.node_emb(atom_types.unsqueeze(-1)) + self.pos_emb(positions)
        x = positions
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        h_global = h.mean(dim=0, keepdim=True)
        return self.head(h_global).squeeze()


# ── Synthetic Molecular Data ──

class MolecularDataset:
    """Synthetic molecule-like graphs with rotational properties."""
    def __init__(self, n_samples=500, n_atoms=8, device='cpu'):
        self.device = device
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        torch.manual_seed(42)

        self.data = []
        for _ in range(n_samples):
            # Random atom types
            atom_types = torch.randint(0, 5, (n_atoms,), device=device).float()

            # Random positions (molecule-like)
            positions = torch.randn(n_atoms, 3, device=device) * 2.0

            # Build edges: connect atoms within distance threshold
            dists = torch.cdist(positions, positions)
            src, tgt = torch.where((dists < 3.0) & (dists > 0.1))
            edge_index = torch.stack([src, tgt])

            # Edge attributes: distance bins
            d = dists[src, tgt]
            edge_attr = F.one_hot((d * 3).long().clamp(0, 15), 16).float()

            # Target: moment of inertia (rotationally invariant)
            com = positions.mean(dim=0)
            moi = ((positions - com) ** 2).sum()  # scalar

            self.data.append((atom_types, positions, edge_index, edge_attr, moi))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


def rotate_positions(positions, angle=None, axis=None):
    """Apply random rotation to positions."""
    if angle is None:
        angle = torch.rand(1).item() * 2 * np.pi
    if axis is None:
        axis = F.normalize(torch.randn(3), dim=0)

    # Rodrigues' rotation formula
    K = torch.zeros(3, 3)
    K[0, 1] = -axis[2]; K[0, 2] = axis[1]
    K[1, 0] = axis[2];  K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] = axis[0]

    R = torch.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    R = R.to(positions.device)
    return positions @ R.T


# ── Training ──

def train_model(model, dataset, n_steps=1500, lr=1e-3, augment_rotation=False, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, len(dataset), (1,)).item()
        atom_types, positions, edge_index, edge_attr, target = dataset[idx]

        if augment_rotation:
            positions = rotate_positions(positions)

        pred = model(atom_types, positions, edge_index, edge_attr)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return losses


# ── Equivariance Test ──

def test_equivariance(model, dataset, n_tests=50, device='cpu'):
    """Test that predictions are invariant to rotations of input."""
    model.eval()
    diffs = []
    with torch.no_grad():
        for _ in range(n_tests):
            idx = torch.randint(0, len(dataset), (1,)).item()
            atom_types, positions, edge_index, edge_attr, _ = dataset[idx]

            pred_orig = model(atom_types, positions, edge_index, edge_attr)

            # Random rotation
            pos_rot = rotate_positions(positions)
            pred_rot = model(atom_types, pos_rot, edge_index, edge_attr)

            rel_diff = (pred_orig - pred_rot).abs() / (pred_orig.abs() + 1e-8)
            diffs.append(rel_diff.item())

    return np.mean(diffs), np.std(diffs)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "118-equivariant-gnn"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Creating Dataset ===")
    dataset = MolecularDataset(n_samples=500, n_atoms=8, device=device)

    # ── Experiment 1: EGNN vs Plain GNN ──
    print("\n=== Training EGNN ===")
    egnn = EGNNModel(n_layers=4, node_dim=64, edge_dim=16, out_dim=1).to(device)
    egnn_losses = train_model(egnn, dataset, n_steps=1500, device=device)
    print(f"  Final loss: {np.mean(egnn_losses[-50:]):.4f}")

    print("\n=== Training Plain GNN ===")
    pgnn = PlainGNNModel(n_layers=4, node_dim=64, edge_dim=16, out_dim=1).to(device)
    pgnn_losses = train_model(pgnn, dataset, n_steps=1500, device=device)
    print(f"  Final loss: {np.mean(pgnn_losses[-50:]):.4f}")

    # ── Experiment 2: Equivariance verification ──
    print("\n=== Equivariance Test ===")
    egnn_mean, egnn_std = test_equivariance(egnn, dataset, device=device)
    pgnn_mean, pgnn_std = test_equivariance(pgnn, dataset, device=device)
    print(f"  EGNN: relative diff = {egnn_mean:.6f} ± {egnn_std:.6f}")
    print(f"  Plain GNN: relative diff = {pgnn_mean:.6f} ± {pgnn_std:.6f}")

    # ── Experiment 3: Data augmentation helps Plain GNN ──
    print("\n=== Training with Rotation Augmentation ===")
    pgnn_aug = PlainGNNModel(n_layers=4, node_dim=64, edge_dim=16, out_dim=1).to(device)
    pgnn_aug_losses = train_model(pgnn_aug, dataset, n_steps=1500, augment_rotation=True, device=device)
    pgnn_aug_mean, pgnn_aug_std = test_equivariance(pgnn_aug, dataset, device=device)
    print(f"  Plain GNN (aug): relative diff = {pgnn_aug_mean:.6f} ± {pgnn_aug_std:.6f}")

    # ── Experiment 4: Depth sweep ──
    print("\n=== Depth Sweep ===")
    depth_results = {}
    for n_layers in [1, 2, 3, 4, 6, 8]:
        model = EGNNModel(n_layers=n_layers, node_dim=64, edge_dim=16, out_dim=1).to(device)
        losses = train_model(model, dataset, n_steps=800, device=device)
        mean_diff, _ = test_equivariance(model, dataset, n_tests=20, device=device)
        depth_results[n_layers] = {
            'final_loss': np.mean(losses[-30:]),
            'equiv_diff': mean_diff,
        }
        print(f"  layers={n_layers}: loss={depth_results[n_layers]['final_loss']:.4f}, "
              f"equiv_diff={mean_diff:.6f}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 20
    for ax, losses, label, color in [
        (axes[0], egnn_losses, 'EGNN', 'blue'),
        (axes[0], pgnn_losses, 'Plain GNN', 'red'),
        (axes[0], pgnn_aug_losses, 'Plain GNN (aug)', 'orange'),
    ]:
        s = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax.plot(s, label=label, linewidth=2, color=color)

    axes[0].set_title("Training Loss Comparison")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Equivariance comparison bar
    methods = ['EGNN', 'Plain\nGNN', 'Plain GNN\n(augmented)']
    diffs = [egnn_mean, pgnn_mean, pgnn_aug_mean]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    axes[1].bar(methods, diffs, color=colors, alpha=0.7)
    axes[1].set_ylabel("Relative Diff (lower = more equivariant)")
    axes[1].set_title("Rotation Equivariance Test")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('EGNN vs Plain GNN (2102.09844)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png', dpi=150)
    plt.close()

    # 2. Depth sweep
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layers = sorted(depth_results.keys())
    losses_d = [depth_results[l]['final_loss'] for l in layers]
    equiv_d = [depth_results[l]['equiv_diff'] for l in layers]

    axes[0].plot(layers, losses_d, marker='o', color='blue', linewidth=2)
    axes[0].set_xlabel("Number of Layers")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Quality vs Depth")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, equiv_d, marker='s', color='red', linewidth=2)
    axes[1].set_xlabel("Number of Layers")
    axes[1].set_ylabel("Equivariance Error")
    axes[1].set_title("Equivariance vs Depth")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('EGNN Depth Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'depth_sweep.png', dpi=150)
    plt.close()

    # 3. Rotation invariance visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    angles = np.linspace(0, 2 * np.pi, 36)
    egnn_preds = []
    pgnn_preds = []

    idx = 0
    atom_types, positions, edge_index, edge_attr, target = dataset[idx]
    with torch.no_grad():
        for angle in angles:
            pos_rot = rotate_positions(positions, angle=angle)
            egnn_preds.append(egnn(atom_types, pos_rot, edge_index, edge_attr).item())
            pgnn_preds.append(pgnn(atom_types, pos_rot, edge_index, edge_attr).item())

    ax.plot(np.degrees(angles), egnn_preds, label='EGNN', color='blue', linewidth=2)
    ax.plot(np.degrees(angles), pgnn_preds, label='Plain GNN', color='red', linewidth=2)
    ax.axhline(target.item(), color='green', linestyle='--', alpha=0.7, label='True value')
    ax.set_xlabel("Rotation Angle (degrees)")
    ax.set_ylabel("Predicted Moment of Inertia")
    ax.set_title("Prediction Under Rotation (equivariant = flat line)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'rotation_invariance.png', dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    concept = (
        "E(n) Equivariant GNN (EGNN, 2102.09844)\n"
        "=" * 55 + "\n\n"
        "Key Insight:\n"
        "  Build equivariance into the architecture directly,\n"
        "  rather than learning it from data augmentation.\n\n"
        "EGNN Layer Update:\n"
        "  1. Compute messages: m_ij = φ_m(h_i, h_j, ||x_i-x_j||², a_ij)\n"
        "  2. Update coordinates: x_i' = x_i + Σ_j (φ_x(m_ij) · (x_i-x_j))\n"
        "     → Equivariant: rotation of inputs → same rotation of outputs\n"
        "  3. Update features: h_i' = h_i + φ_h(h_i, Σ_j m_ij)\n"
        "     → Invariant: rotation of inputs → same output features\n\n"
        "Why Equivariance Matters:\n"
        "  • Molecular properties are rotationally invariant\n"
        "  • EGNN guarantees this by construction\n"
        "  • Plain GNN must learn invariance from data\n"
        "  • Data augmentation helps but is imperfect\n\n"
        "E(n) = group of rotations, reflections, translations in n dimensions"
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
