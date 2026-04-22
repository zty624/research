"""
Minimal DARTS Reproduction
===========================
Reproduces core ideas from DARTS (1806.09055, Liu et al.):
1. Differentiable Architecture Search: relax discrete choices to continuous
2. Softmax over candidate operations (conv, pool, identity, zero)
3. Bi-level optimization: architecture params α vs model weights w
4. Compare: random architecture vs DARTS-discovered vs manual design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Operations ──

class ConvOp(nn.Module):
    """3x3 separable convolution."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)


class DilConvOp(nn.Module):
    """3x3 dilated separable convolution."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=stride, padding=2, dilation=2, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)


class PoolOp(nn.Module):
    """3x3 average pooling."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        return self.bn(self.pool(x))


class IdentityOp(nn.Module):
    """Identity (skip connection)."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        return self.bn(x)


class ZeroOp(nn.Module):
    """Zero operation (no connection)."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        return x.mul(0.0)


# ── Mixed Operation (DARTS) ──

class MixedOp(nn.Module):
    """Weighted combination of operations, parameterized by architecture weights α."""
    def __init__(self, C, stride=1):
        super().__init__()
        self.ops = nn.ModuleList([
            ConvOp(C, stride),
            DilConvOp(C, stride),
            PoolOp(C, stride),
            IdentityOp(C, stride),
            ZeroOp(C, stride),
        ])

    def forward(self, x, alpha):
        """alpha: (n_ops,) architecture weights for this edge."""
        weights = F.softmax(alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# ── DARTS Cell ──

class DARTSCell(nn.Module):
    """A DARTS cell with n_nodes intermediate nodes.
    Each edge between nodes has a MixedOp with learnable α.
    """
    def __init__(self, C_in, C_out, n_nodes=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_ops = 5  # number of candidate operations

        # Preprocessing
        self.preprocess = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out)
        )

        # Architecture parameters: α[i][j] for edge from node i to node j
        # Nodes: 0=input, 1..n_nodes=intermediate
        self.alpha = nn.ParameterList()
        n_edges = 0
        for j in range(1, n_nodes + 1):
            # Each intermediate node connects from all previous nodes
            for i in range(j):
                alpha_ij = nn.Parameter(torch.zeros(self.n_ops))
                self.alpha.append(alpha_ij)
                n_edges += 1

        # Mixed operations (all stride=1 within cell)
        self.edges = nn.ModuleList()
        for j in range(1, n_nodes + 1):
            for i in range(j):
                self.edges.append(MixedOp(C_out, stride=1))

        self.n_edges = n_edges

    def forward(self, x):
        x = self.preprocess(x)
        nodes = [x]

        alpha_idx = 0
        for j in range(1, self.n_nodes + 1):
            # Collect inputs from all previous nodes
            node_sum = torch.zeros_like(x)
            for i in range(j):
                node_sum = node_sum + self.edges[alpha_idx](nodes[i], self.alpha[alpha_idx])
                alpha_idx += 1
            nodes.append(node_sum)

        # Output is concatenation of all intermediate nodes
        return torch.cat(nodes[1:], dim=1)


# ── DARTS Network ──

class DARTSNet(nn.Module):
    """Small network with DARTS cells for MNIST."""
    def __init__(self, C=16, n_cells=2, n_nodes=3, n_classes=10):
        super().__init__()
        self.C = C
        self.n_nodes = n_nodes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        # Cells (with optional stride between cells)
        self.cells = nn.ModuleList()
        self.downsample = nn.ModuleList()
        c_in = C
        for i in range(n_cells):
            self.cells.append(DARTSCell(c_in, C, n_nodes))
            c_out = C * n_nodes  # cell output channels
            if i == 0:
                # Downsample after first cell
                self.downsample.append(nn.Sequential(
                    nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, groups=c_out, bias=False),
                    nn.BatchNorm2d(c_out)
                ))
            else:
                self.downsample.append(nn.Identity())
            c_in = c_out  # next cell takes output of previous

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Each cell outputs C * n_nodes channels (concatenation)
        self.classifier = nn.Linear(C * n_nodes * n_cells, n_classes)

    def forward(self, x):
        h = self.stem(x)
        cell_outputs = []
        for i, cell in enumerate(self.cells):
            h = cell(h)
            h = self.downsample[i](h)
            cell_outputs.append(h)

        # Concatenate all cell outputs and pool
        # Each cell output: (B, C*n_nodes, H, W)
        h = torch.cat(cell_outputs, dim=1)
        h = self.pool(h).flatten(1)
        return self.classifier(h)

    def get_alpha(self):
        """Get all architecture parameters."""
        alphas = []
        for cell in self.cells:
            for a in cell.alpha:
                alphas.append(F.softmax(a, dim=-1).detach())
        return alphas

    def get_genotype(self):
        """Get the discrete architecture (top-1 operation per edge)."""
        genotype = []
        for cell_idx, cell in enumerate(self.cells):
            cell_geno = []
            alpha_idx = 0
            for j in range(1, cell.n_nodes + 1):
                for i in range(j):
                    op_idx = cell.alpha[alpha_idx].argmax().item()
                    op_names = ['conv3x3', 'dil_conv3x3', 'avg_pool3x3', 'skip', 'zero']
                    cell_geno.append((i, j, op_names[op_idx]))
                    alpha_idx += 1
            genotype.append(cell_geno)
        return genotype


# ── Fixed Architecture (baseline) ──

class FixedNet(nn.Module):
    """Network with fixed architecture (conv + skip)."""
    def __init__(self, C=16, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(C, C*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C*2), nn.ReLU(),
            nn.Conv2d(C*2, C*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C*2), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C*2, n_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# ── Training ──

def train_darts(model, train_loader, val_loader, n_epochs=15, lr=1e-2, lr_alpha=3e-3, device='cpu'):
    """Bi-level optimization: update α on val, w on train."""
    optimizer_w = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_alpha = torch.optim.Adam(
        [a for cell in model.cells for a in cell.alpha],
        lr=lr_alpha, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, n_epochs)

    train_losses = []
    val_accs = []
    alpha_history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            # Step 1: Update weights w on training loss
            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer_w.zero_grad()
            loss.backward()
            optimizer_w.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Step 2: Update architecture α on validation loss
        model.eval()
        val_correct = 0
        val_total = 0
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            val_loss = F.cross_entropy(logits, by)

            optimizer_alpha.zero_grad()
            val_loss.backward()
            optimizer_alpha.step()

            val_correct += (logits.argmax(1) == by).sum().item()
            val_total += by.shape[0]

        val_accs.append(val_correct / val_total)

        # Record alpha
        alphas = model.get_alpha()
        alpha_history.append([a.cpu().numpy() for a in alphas])

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            geno = model.get_genotype()
            print(f"  Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")
            for i, cell_geno in enumerate(geno):
                print(f"    Cell {i}: {cell_geno}")

    return train_losses, val_accs, alpha_history


def train_fixed(model, train_loader, test_loader, n_epochs=15, lr=1e-2, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_losses = []
    test_accs = []

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
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        test_accs.append(correct / total)

    return train_losses, test_accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "34-darts"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Split train into train/val for DARTS
    train_subset = torch.utils.data.Subset(train_dataset, range(8000))
    val_subset = torch.utils.data.Subset(train_dataset, range(8000, 10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    # 1. DARTS
    print("=== Training DARTS ===")
    darts = DARTSNet(C=16, n_cells=2, n_nodes=3).to(device)
    darts_params = sum(p.numel() for p in darts.parameters())
    print(f"  DARTS params: {darts_params:,}")
    darts_losses, darts_val_accs, alpha_hist = train_darts(
        darts, train_loader, val_loader, n_epochs=15, device=device
    )

    # 2. Fixed architecture
    print("\n=== Training Fixed Architecture ===")
    fixed = FixedNet(C=16).to(device)
    fixed_params = sum(p.numel() for p in fixed.parameters())
    print(f"  Fixed params: {fixed_params:,}")
    fixed_losses, fixed_accs = train_fixed(
        fixed, train_loader, test_loader, n_epochs=15, device=device
    )

    # 3. Random architecture baseline
    print("\n=== Training Random Architecture ===")
    random_net = DARTSNet(C=16, n_cells=2, n_nodes=3).to(device)
    # Freeze alpha at random values
    with torch.no_grad():
        for cell in random_net.cells:
            for a in cell.alpha:
                a.copy_(torch.randn_like(a))
    random_losses, random_accs = train_fixed(
        random_net, train_loader, test_loader, n_epochs=15, device=device
    )

    # ── Final evaluation ──
    print("\n=== Final Evaluation ===")
    def evaluate(model):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        return correct / total

    darts_acc = evaluate(darts)
    fixed_acc = evaluate(fixed)
    random_acc = evaluate(random_net)

    print(f"  DARTS:   {darts_acc:.4f}")
    print(f"  Fixed:   {fixed_acc:.4f}")
    print(f"  Random:  {random_acc:.4f}")

    # Discovered architecture
    geno = darts.get_genotype()
    print(f"\n  Discovered genotype: {geno}")

    # ── Visualization ──

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(darts_losses, label='DARTS', color='blue')
    axes[0].plot(fixed_losses, label='Fixed', color='red')
    axes[0].plot(random_losses, label='Random', color='gray')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(darts_val_accs, label='DARTS (val)', color='blue')
    axes[1].plot(fixed_accs, label='Fixed (test)', color='red')
    axes[1].plot(random_accs, label='Random (test)', color='gray')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DARTS: Differentiable Architecture Search", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 2. Alpha evolution
    if alpha_hist:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot first cell's alpha weights over epochs
        n_cell0_alphas = len(alpha_hist[0]) // 2  # assume 2 cells
        op_names = ['conv3x3', 'dil_conv', 'avg_pool', 'skip', 'zero']

        for edge_idx in range(min(3, n_cell0_alphas)):
            ax = axes[0] if edge_idx < 2 else axes[1]
            alpha_over_time = np.array([alpha_hist[e][edge_idx] for e in range(len(alpha_hist))])
            for op_idx in range(5):
                ax.plot(alpha_over_time[:, op_idx], label=f'{op_names[op_idx]}' if edge_idx == 0 else "",
                       alpha=0.7)
            ax.set_title(f"Edge {edge_idx} α weights (Cell 0)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Softmax Weight")
            ax.grid(True, alpha=0.3)

        if n_cell0_alphas > 0:
            axes[0].legend(fontsize=7)
        plt.suptitle("DARTS: Architecture Parameter Evolution", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / "alpha_evolution.png", dpi=150)
        plt.close()

    # 3. Final architecture visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    for cell_idx, cell_geno in enumerate(geno):
        y_offset = cell_idx * 0.45
        ax.text(0.5, 0.9 - y_offset, f"Cell {cell_idx}", fontsize=14,
               fontweight='bold', ha='center', va='center')

        for src, dst, op in cell_geno:
            x_src = src * 0.2 + 0.1
            x_dst = dst * 0.2 + 0.1
            y_src = 0.8 - y_offset
            y_dst = 0.6 - y_offset

            color = {'conv3x3': 'blue', 'dil_conv3x3': 'cyan',
                    'avg_pool3x3': 'green', 'skip': 'orange', 'zero': 'gray'}[op]

            ax.annotate('', xy=(x_dst, y_dst), xytext=(x_src, y_src),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
            mx, my = (x_src + x_dst) / 2, (y_src + y_dst) / 2
            ax.text(mx, my - 0.02, op, fontsize=7, ha='center', color=color)

        # Draw nodes
        for n in range(max(src for src, _, _ in cell_geno) + 2):
            xn = n * 0.2 + 0.1
            circle = plt.Circle((xn, 0.8 - y_offset if n == 0 else 0.6 - y_offset),
                               0.02, color='lightblue', ec='blue')
            ax.add_patch(circle)

    ax.set_title("DARTS: Discovered Architecture", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(results_dir / "discovered_architecture.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("Discrete\nSearch", "Combinatorial\nexponential cost\nTrain each arch\nfrom scratch", 0.14, 'gray'),
        ("DARTS\n(Continuous)", "Relax with softmax\nα_i → weight of op_i\nGradient-based!\n→ Efficient search", 0.5, 'blue'),
        ("Bi-level\nOptimization", "w* = argmin_w L_train(w, α)\nα* = argmin_α L_val(w*, α)\n→ Search on val set", 0.86, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("DARTS: From Discrete to Differentiable Architecture Search", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "darts_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
