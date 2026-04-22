"""
Minimal Layer Normalization Reproduction
=========================================
Reproduces core ideas from Layer Normalization (1607.06450, Ba et al.):
1. Normalize across features (not batch dimension like BN)
2. Same computation at train and test time (no running stats)
3. Works with variable batch sizes and sequence lengths
4. Compare: BN vs LN vs GN vs No Norm on RNNs and Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Custom LayerNorm ──

class CustomLayerNorm(nn.Module):
    """Manual implementation of Layer Normalization."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GroupNorm2(nn.Module):
    """Group Normalization for sequence data (B, T, D).
    Treats D as channels, groups D into num_groups groups.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels)
        # Ensure num_channels is divisible by num_groups
        while num_channels % self.num_groups != 0:
            self.num_groups -= 1
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D) -> GroupNorm expects (N, C, *) format
        B, T, D = x.shape
        # Reshape: treat B*T as batch, D as channels
        x_reshaped = x.permute(0, 2, 1).contiguous()  # (B, D, T)
        x_reshaped = x_reshaped.reshape(B, D, -1)  # (B, D, T)
        x_normed = F.group_norm(x_reshaped, self.num_groups, self.gamma, self.beta, self.eps)
        x_normed = x_normed.reshape(B, D, T)
        return x_normed.permute(0, 2, 1).contiguous()  # (B, T, D)


# ── Models ──

class LSTMModel(nn.Module):
    """LSTM for sequence classification with different normalization."""
    def __init__(self, input_size=1, hidden_size=64, n_classes=10, norm_type='none'):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_type = norm_type

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True)

        # Normalization after LSTM
        if norm_type == 'layer':
            self.norm = CustomLayerNorm(hidden_size)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(hidden_size)
        else:
            self.norm = None

        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (B, T, input_size)
        h, _ = self.lstm(x)
        h = h[:, -1, :]  # last timestep

        if self.norm is not None:
            if self.norm_type == 'batch':
                h = self.norm(h)
            else:
                h = self.norm(h)

        return self.classifier(h)


class TransformerModel(nn.Module):
    """Small Transformer with different normalization strategies."""
    def __init__(self, vocab_size=16, d_model=64, n_heads=2, n_layers=2, n_classes=10, norm_type='layer'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)

        # Build layers manually to control norm placement
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(d_model, n_heads, norm_type))

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        for layer in self.layers:
            h = layer(h)
        h = h.mean(dim=1)  # pool
        return self.classifier(h)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, norm_type='layer'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        if norm_type == 'layer':
            self.norm1 = CustomLayerNorm(d_model)
            self.norm2 = CustomLayerNorm(d_model)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        elif norm_type == 'group':
            self.norm1 = GroupNorm2(8, d_model)
            self.norm2 = GroupNorm2(8, d_model)
        else:
            self.norm1 = self.norm2 = None

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        # Pre-norm
        if self.norm1 is not None:
            if isinstance(self.norm1, nn.BatchNorm1d):
                h = self.norm1(x.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.norm1(x)
        else:
            h = x

        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        if self.norm2 is not None:
            if isinstance(self.norm2, nn.BatchNorm1d):
                h2 = self.norm2(x.transpose(1, 2)).transpose(1, 2)
            else:
                h2 = self.norm2(x)
        else:
            h2 = x

        x = x + self.ff(h2)
        return x


# ── Data Generation ──

def generate_sequence_data(n_samples=5000, seq_len=20, vocab_size=16, n_classes=10):
    """Generate classification data: dominant token class."""
    sequences = torch.randint(0, vocab_size, (n_samples, seq_len))
    # Label = most common token in first 10 positions (easier task)
    labels = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        counts = torch.bincount(sequences[i, :10], minlength=vocab_size)
        labels[i] = counts.argmax() % n_classes
    return sequences, labels


# ── Training ──

def train_transformer(model, X, y, n_epochs=50, lr=3e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    accs = []

    X, y = X.to(device), y.to(device)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n_train)[:256]
        logits = model(X_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            correct = (model(X_test).argmax(1) == y_test).sum().item()
            accs.append(correct / len(y_test))

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {accs[-1]:.4f}")

    return losses, accs


def train_lstm(model, X, y, n_epochs=50, lr=3e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    accs = []

    X, y = X.to(device), y.to(device)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n_train)[:256]
        logits = model(X_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            correct = (model(X_test).argmax(1) == y_test).sum().item()
            accs.append(correct / len(y_test))

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {accs[-1]:.4f}")

    return losses, accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "37-layernorm"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    X, y = generate_sequence_data(n_samples=5000, seq_len=20, vocab_size=16)
    X_rnn = X.float().unsqueeze(-1)  # (N, T, 1) for LSTM

    # Experiment 1: LSTM with different norms
    print("=== LSTM Normalization Comparison ===")
    lstm_results = {}
    for norm_type in ['none', 'layer', 'batch']:
        print(f"\n  LSTM + {norm_type} norm:")
        model = LSTMModel(norm_type=norm_type).to(device)
        losses, accs = train_lstm(model, X_rnn, y, n_epochs=30, device=device)
        lstm_results[norm_type] = {'losses': losses, 'accs': accs, 'final_acc': accs[-1]}

    # Experiment 2: Transformer with different norms
    print("\n=== Transformer Normalization Comparison ===")
    transformer_results = {}
    for norm_type in ['none', 'layer', 'batch', 'group']:
        print(f"\n  Transformer + {norm_type} norm:")
        model = TransformerModel(norm_type=norm_type).to(device)
        losses, accs = train_transformer(model, X, y, n_epochs=30, device=device)
        transformer_results[norm_type] = {'losses': losses, 'accs': accs, 'final_acc': accs[-1]}

    # ── Results ──
    print("\n=== Final Results ===")
    print("LSTM:")
    for norm, r in lstm_results.items():
        print(f"  {norm}: {r['final_acc']:.4f}")
    print("Transformer:")
    for norm, r in transformer_results.items():
        print(f"  {norm}: {r['final_acc']:.4f}")

    # ── Visualization ──

    # 1. LSTM comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for norm, r in lstm_results.items():
        color = {'none': 'red', 'layer': 'blue', 'batch': 'green'}[norm]
        label = {'none': 'No Norm', 'layer': 'LayerNorm', 'batch': 'BatchNorm'}[norm]
        axes[0].plot(r['losses'], label=label, color=color, alpha=0.7)
    axes[0].set_title("LSTM: Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for norm, r in lstm_results.items():
        color = {'none': 'red', 'layer': 'blue', 'batch': 'green'}[norm]
        label = {'none': 'No Norm', 'layer': 'LayerNorm', 'batch': 'BatchNorm'}[norm]
        axes[1].plot(r['accs'], label=label, color=color, alpha=0.7)
    axes[1].set_title("LSTM: Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("LayerNorm vs BatchNorm in Recurrent Models", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "lstm_comparison.png", dpi=150)
    plt.close()

    # 2. Transformer comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for norm, r in transformer_results.items():
        color = {'none': 'red', 'layer': 'blue', 'batch': 'green', 'group': 'purple'}[norm]
        label = {'none': 'No Norm', 'layer': 'LayerNorm', 'batch': 'BatchNorm', 'group': 'GroupNorm'}[norm]
        axes[0].plot(r['losses'], label=label, color=color, alpha=0.7)
    axes[0].set_title("Transformer: Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for norm, r in transformer_results.items():
        color = {'none': 'red', 'layer': 'blue', 'batch': 'green', 'group': 'purple'}[norm]
        label = {'none': 'No Norm', 'layer': 'LayerNorm', 'batch': 'BatchNorm', 'group': 'GroupNorm'}[norm]
        axes[1].plot(r['accs'], label=label, color=color, alpha=0.7)
    axes[1].set_title("Transformer: Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Normalization in Transformers: LayerNorm Wins", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "transformer_comparison.png", dpi=150)
    plt.close()

    # 3. BN vs LN vs GN conceptual comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Show normalization dimension visually (matrix heatmap)
    B, D = 8, 16
    data = torch.randn(B, D)

    # BatchNorm: normalize along batch dim (columns)
    ax = axes[0]
    bn_normed = (data - data.mean(dim=0)) / (data.std(dim=0) + 1e-5)
    ax.imshow(bn_normed.numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title("BatchNorm\n(normalize per feature)", fontsize=10)
    ax.set_ylabel("Batch")
    ax.set_xlabel("Features")
    ax.axhline(y=-0.5, color='green', linewidth=3)  # highlight normalization direction

    # LayerNorm: normalize along feature dim (rows)
    ax = axes[1]
    ln_normed = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-5)
    ax.imshow(ln_normed.numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title("LayerNorm\n(normalize per sample)", fontsize=10)
    ax.set_ylabel("Batch")
    ax.set_xlabel("Features")
    # Highlight row normalization
    for i in range(B):
        ax.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)

    # GroupNorm: normalize within groups
    ax = axes[2]
    n_groups = 4
    # Manual GroupNorm on (B, D) data: treat as (B, D, 1) for F.group_norm
    data_gn = data.unsqueeze(2)  # (B, D, 1)
    gn_weight = torch.ones(D)
    gn_bias = torch.zeros(D)
    gn_normed = F.group_norm(data_gn, n_groups, gn_weight, gn_bias, 1e-5).squeeze(2).detach()
    ax.imshow(gn_normed.numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title("GroupNorm\n(normalize per group)", fontsize=10)
    ax.set_ylabel("Batch")
    ax.set_xlabel("Features")
    # Draw group boundaries
    for g in range(1, n_groups):
        ax.axvline(x=g * (D // n_groups) - 0.5, color='purple', linewidth=2)

    plt.suptitle("Normalization Strategies: Which Dimension to Normalize?", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "norm_dimensions.png", dpi=150)
    plt.close()

    # 4. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("BatchNorm\n(2015)", "Normalize across\nbatch dimension\nNeeds running stats\nFails with small B\n→ CNN standard", 0.14, 'green'),
        ("LayerNorm\n(2016)", "Normalize across\nfeature dimension\nSame train & test\nWorks with any B\n→ Transformer standard", 0.5, 'blue'),
        ("GroupNorm\n(2018)", "Normalize within\nfeature groups\nNo batch dependency\nMiddle ground\n→ Small-batch CNN", 0.86, 'purple'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Normalization: Batch → Layer → Group", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "layernorm_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
