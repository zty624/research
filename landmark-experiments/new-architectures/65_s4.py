"""
Minimal S4 (Structured State Space) Reproduction
==================================================
Reproduces core ideas from S4 (2111.00396, Gu et al.):
1. HiPPO matrix: special A initialization for long-range memory
2. Bilinear discretization: continuous -> discrete SSM
3. Convolution mode: O(L log L) training via FFT
4. Recurrent mode: O(N) per step for inference
5. S4D simplification: diagonal A approximation

Experiments:
1. Sequential MNIST (sMNIST): S4D vs LSTM vs Transformer
2. Long-range dependency: copy-first-token at various sequence lengths
3. HiPPO vs random A initialization ablation
4. Visualizations: training curves, accuracy vs length, kernel, concept diagram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import urllib.request
import gzip
import struct


# ── Data Loading ──

def load_mnist(data_dir):
    """Load MNIST from raw IDX format files, downloading if necessary."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    for fname in files.values():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)

    def parse_images(fname):
        with gzip.open(data_dir / fname, 'rb') as f:
            magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(n, rows * cols).astype(np.float32) / 255.0

    def parse_labels(fname):
        with gzip.open(data_dir / fname, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    train_x = torch.tensor(parse_images(files['train_images']))
    train_y = torch.tensor(parse_labels(files['train_labels']))
    test_x = torch.tensor(parse_images(files['test_images']))
    test_y = torch.tensor(parse_labels(files['test_labels']))

    return train_x, train_y, test_x, test_y


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x  # (N, 784)
        self.y = y  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(-1), self.y[idx]  # (784, 1), scalar


def generate_long_range_data(batch_size, seq_len, n_classes=10):
    """Long-range dependency task: predict first token's class.

    First token encodes the target class; remaining tokens are noise.
    The model must retain information from position 0 across the full sequence.
    """
    labels = torch.randint(0, n_classes, (batch_size,))
    x = torch.rand(batch_size, seq_len, 1) * 0.3  # noise in [0, 0.3]
    # Signal at position 0: distinguishable from noise
    x[:, 0, 0] = (labels.float() + 0.5) / n_classes  # in [0.05, 0.95]
    return x, labels


# ── HiPPO Matrix ──

def make_hippo_diagonal(N):
    """Diagonal of HiPPO-LegS matrix for S4D initialization.

    Full HiPPO-LegS matrix:
        A_{nk} = sqrt(2n+1)*sqrt(2k+1) if n>k, (n+1) if n=k, 0 if n<k
    Diagonal approximation: A_{nn} = -(n+1) (negative for stability).

    Just swapping random A for HiPPO A takes sMNIST from ~60% to ~98%.
    """
    return -(torch.arange(N, dtype=torch.float32) + 1)


def make_random_diagonal(N):
    """Random negative diagonal A (for ablation comparison with HiPPO)."""
    return -torch.rand(N) * 2 - 0.5


# ── S4D: Diagonal Structured State Space ──

class S4DCore(nn.Module):
    """S4D core: diagonal state space model with convolution mode via FFT.

    Continuous SSM:  x'(t) = Ax(t) + Bu(t),  y(t) = Cx(t) + Du(t)
    Bilinear discretization:
        A_bar = (I - dt/2 * A)^{-1} (I + dt/2 * A)
        B_bar = (I - dt/2 * A)^{-1} dt B
    Convolution mode: y = K * u  where  K[k] = C A_bar^k B_bar
    """

    def __init__(self, d_model, d_state=64, use_hippo=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A (parameterized as log(-A) to ensure A < 0)
        A_init = make_hippo_diagonal(d_state) if use_hippo else make_random_diagonal(d_state)
        self.log_A = nn.Parameter(torch.log(-A_init))  # A = -exp(log_A) < 0

        # B, C: (d_model, d_state) -- each feature dim has an independent SSM
        self.B = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / d_state**0.5))
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / d_state**0.5))

        # Step size dt (one per feature dim)
        self.log_dt = nn.Parameter(torch.log(torch.ones(d_model) * 0.01))

        # Skip connection D
        self.D = nn.Parameter(torch.ones(d_model))

    def _discretize(self):
        """Bilinear discretization of the diagonal SSM."""
        A = -torch.exp(self.log_A)  # (d_state,) negative
        dt = torch.exp(self.log_dt)  # (d_model,)

        # dtA[d, n] = dt[d] * A[n]
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0)  # (d_model, d_state)

        # Bilinear: A_bar = (1 + dtA/2) / (1 - dtA/2),  B_bar = dt*B / (1 - dtA/2)
        A_bar = (1 + dtA / 2) / (1 - dtA / 2)   # (d_model, d_state)
        B_bar = (dt.unsqueeze(-1) * self.B) / (1 - dtA / 2)  # (d_model, d_state)

        return A_bar, B_bar

    def _compute_kernel(self, L):
        """Compute convolution kernel K of length L.

        K[k, d] = sum_n C[d,n] * B_bar[d,n] * A_bar[d,n]^k   (Vandermonde)
        """
        A_bar, B_bar = self._discretize()

        # Clamp A_bar magnitudes for numerical stability
        A_bar = A_bar.clamp(-0.999, 0.999)

        CB = self.C * B_bar  # (d_model, d_state)

        # A_bar_power[d, n, k] = A_bar[d, n]^k
        exponents = torch.arange(L, device=A_bar.device, dtype=A_bar.dtype)
        A_bar_power = A_bar.unsqueeze(-1) ** exponents  # (d_model, d_state, L)

        # K[d, k] = sum_n CB[d,n] * A_bar_power[d,n,k]
        K = (CB.unsqueeze(-1) * A_bar_power).sum(dim=1)  # (d_model, L)

        return K

    def forward(self, u):
        """Forward pass in convolution mode via FFT.

        Args:
            u: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B_batch, L, _ = u.shape

        K = self._compute_kernel(L)  # (d_model, L)

        # Causal convolution y = K * u via FFT
        u_ = u.transpose(1, 2)  # (B, d_model, L)

        # Zero-pad to 2L for linear (non-circular) convolution
        u_pad = F.pad(u_, (0, L))
        K_pad = F.pad(K, (0, L))

        u_f = torch.fft.rfft(u_pad, dim=-1)
        K_f = torch.fft.rfft(K_pad, dim=-1)
        y_f = u_f * K_f.unsqueeze(0)
        y_ = torch.fft.irfft(y_f, n=2 * L, dim=-1)[:, :, :L]

        y = y_.transpose(1, 2)  # (B, L, d_model)

        # Skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def forward_recurrent(self, u, state=None):
        """Forward pass in recurrent mode (for inference / streaming).

        Args:
            u: (B, L, d_model)
            state: (B, d_model, d_state) or None
        Returns:
            y: (B, L, d_model)
            state: (B, d_model, d_state)
        """
        B_batch, L, _ = u.shape
        A_bar, B_bar = self._discretize()

        if state is None:
            state = torch.zeros(B_batch, self.d_model, self.d_state, device=u.device)

        outputs = []
        for t in range(L):
            # x_k = A_bar * x_{k-1} + B_bar * u_k
            state = state * A_bar.unsqueeze(0) + B_bar.unsqueeze(0) * u[:, t].unsqueeze(-1)
            # y_k = C * x_k + D * u_k
            y_t = (self.C.unsqueeze(0) * state).sum(dim=-1) + u[:, t] * self.D.unsqueeze(0)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, state


class S4DBlock(nn.Module):
    """S4D block: LayerNorm + S4DCore + GLU + Residual."""

    def __init__(self, d_model, d_state=64, use_hippo=True):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.s4d = S4DCore(d_model, d_state, use_hippo)
        self.glu_linear = nn.Linear(d_model, 2 * d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.s4d(x)
        # GLU activation
        x_glu = self.glu_linear(x)
        x = x_glu[:, :, :self.d_model] * torch.sigmoid(x_glu[:, :, self.d_model:])
        return x + residual


class S4DModel(nn.Module):
    """S4D model for sequence classification (sMNIST, long-range tasks)."""

    def __init__(self, d_input=1, d_model=64, n_blocks=4, d_state=64, n_classes=10, use_hippo=True):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            S4DBlock(d_model, d_state, use_hippo) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over sequence
        return self.head(x)


# ── Baseline Models ──

class LSTMClassifier(nn.Module):
    """LSTM baseline for sequence classification."""

    def __init__(self, d_input=1, d_model=64, n_layers=2, n_classes=10):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.proj_in(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.head(x)


class TransformerClassifier(nn.Module):
    """Transformer baseline for sequence classification."""

    def __init__(self, d_input=1, d_model=64, n_heads=4, n_layers=2, n_classes=10, max_len=1024):
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj_in(x) + self.pos_emb(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ── Training Functions ──

def train_classifier(model, train_loader, n_epochs, lr, device):
    """Train a sequence classifier, return per-epoch train accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    epoch_accs = []
    for epoch in range(n_epochs):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)

        scheduler.step()
        acc = correct / total
        epoch_accs.append(acc)
        print(f"    Epoch {epoch+1}/{n_epochs} | Acc: {acc:.4f}")

    return epoch_accs


def evaluate_classifier(model, test_loader, device):
    """Evaluate classifier accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)
    return correct / total


def train_long_range(model, seq_len, n_steps=1000, lr=1e-3, device='cpu',
                     n_classes=10, batch_size=64):
    """Train on long-range dependency task, return step-wise accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    accs = []
    for step in range(n_steps):
        x, y = generate_long_range_data(batch_size, seq_len, n_classes)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        acc = (logits.argmax(-1) == y).float().mean().item()
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}/{n_steps} | Acc: {acc:.4f}")

    return accs


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "65-s4"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Experiment 1: Sequential MNIST ──
    print("=== Experiment 1: Sequential MNIST (sMNIST) ===")

    data_dir = Path(__file__).parent / "data"
    try:
        train_x, train_y, test_x, test_y = load_mnist(data_dir)
        has_mnist = True
        print(f"  Loaded MNIST: {train_x.shape[0]} train, {test_x.shape[0]} test")
    except Exception as e:
        print(f"  Could not load MNIST ({e}), using synthetic data")
        has_mnist = False
        train_x = torch.rand(6000, 784)
        train_y = torch.randint(0, 10, (6000,))
        test_x = torch.rand(1000, 784)
        test_y = torch.randint(0, 10, (1000,))

    train_ds = MNISTDataset(train_x, train_y)
    test_ds = MNISTDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256)

    n_epochs = 5

    print("\n  S4D:")
    s4d = S4DModel(d_input=1, d_model=64, n_blocks=4, d_state=64,
                   n_classes=10, use_hippo=True).to(device)
    print(f"    Params: {sum(p.numel() for p in s4d.parameters()):,}")
    s4d_accs = train_classifier(s4d, train_loader, n_epochs, lr=1e-3, device=device)
    s4d_test = evaluate_classifier(s4d, test_loader, device)
    print(f"    Test Acc: {s4d_test:.4f}")

    print("\n  LSTM:")
    lstm = LSTMClassifier(d_input=1, d_model=64, n_layers=2, n_classes=10).to(device)
    print(f"    Params: {sum(p.numel() for p in lstm.parameters()):,}")
    lstm_accs = train_classifier(lstm, train_loader, n_epochs, lr=1e-3, device=device)
    lstm_test = evaluate_classifier(lstm, test_loader, device)
    print(f"    Test Acc: {lstm_test:.4f}")

    print("\n  Transformer:")
    tf = TransformerClassifier(d_input=1, d_model=64, n_heads=4, n_layers=2,
                               n_classes=10, max_len=1024).to(device)
    print(f"    Params: {sum(p.numel() for p in tf.parameters()):,}")
    tf_accs = train_classifier(tf, train_loader, n_epochs, lr=1e-3, device=device)
    tf_test = evaluate_classifier(tf, test_loader, device)
    print(f"    Test Acc: {tf_test:.4f}")

    # ── Experiment 2: Long-Range Dependency ──
    print("\n=== Experiment 2: Long-Range Dependency ===")

    seq_lengths = [64, 256, 1024]
    lr_results = {}

    for sl in seq_lengths:
        print(f"\n  SeqLen={sl}:")
        print("    S4D:")
        s4d_lr = S4DModel(d_input=1, d_model=32, n_blocks=2, d_state=32,
                          n_classes=10, use_hippo=True).to(device)
        s4d_lr_accs = train_long_range(s4d_lr, sl, n_steps=1000, lr=1e-3, device=device)

        print("    LSTM:")
        lstm_lr = LSTMClassifier(d_input=1, d_model=32, n_layers=2, n_classes=10).to(device)
        lstm_lr_accs = train_long_range(lstm_lr, sl, n_steps=1000, lr=1e-3, device=device)

        lr_results[sl] = {
            's4d': np.mean(s4d_lr_accs[-200:]),
            'lstm': np.mean(lstm_lr_accs[-200:]),
        }
        print(f"    S4D: {lr_results[sl]['s4d']:.4f}, LSTM: {lr_results[sl]['lstm']:.4f}")

    # ── Experiment 3: HiPPO vs Random A ──
    print("\n=== Experiment 3: HiPPO vs Random A ===")

    print("\n  HiPPO A:")
    s4d_hippo = S4DModel(d_input=1, d_model=64, n_blocks=4, d_state=64,
                         n_classes=10, use_hippo=True).to(device)
    hippo_accs = train_classifier(s4d_hippo, train_loader, n_epochs, lr=1e-3, device=device)
    hippo_test = evaluate_classifier(s4d_hippo, test_loader, device)
    print(f"    Test Acc: {hippo_test:.4f}")

    print("\n  Random A:")
    s4d_random = S4DModel(d_input=1, d_model=64, n_blocks=4, d_state=64,
                          n_classes=10, use_hippo=False).to(device)
    random_accs = train_classifier(s4d_random, train_loader, n_epochs, lr=1e-3, device=device)
    random_test = evaluate_classifier(s4d_random, test_loader, device)
    print(f"    Test Acc: {random_test:.4f}")

    # ── Conv vs Recurrent Equivalence Check ──
    print("\n=== Conv vs Recurrent Equivalence ===")
    s4d.eval()
    with torch.no_grad():
        core = s4d.blocks[0].s4d
        test_input = torch.randn(2, 50, 64).to(device)
        y_conv = core(test_input)
        y_rec, _ = core.forward_recurrent(test_input)
        max_diff = (y_conv - y_rec).abs().max().item()
        print(f"  Max |y_conv - y_recurrent| = {max_diff:.6f}")

    # ── Visualizations ──

    # 1. sMNIST training curves + test accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, n_epochs + 1)
    axes[0].plot(epochs, [a * 100 for a in s4d_accs], 'o-', label='S4D',
                 color='blue', linewidth=2)
    axes[0].plot(epochs, [a * 100 for a in lstm_accs], 's--', label='LSTM',
                 color='green', linewidth=2)
    axes[0].plot(epochs, [a * 100 for a in tf_accs], '^:', label='Transformer',
                 color='red', linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("sMNIST Training Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    models = ['S4D', 'LSTM', 'Transformer']
    test_accs = [s4d_test, lstm_test, tf_test]
    colors = ['blue', 'green', 'red']
    bars = axes[1].bar(models, [a * 100 for a in test_accs], color=colors, alpha=0.7)
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("sMNIST Final Test Accuracy")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, test_accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc*100:.1f}%', ha='center', fontsize=11)

    plt.suptitle("S4: Sequential MNIST Classification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "smnist_comparison.png", dpi=150)
    plt.close()

    # 2. Long-range task: accuracy vs sequence length
    fig, ax = plt.subplots(figsize=(8, 5))
    lens = sorted(lr_results.keys())
    ax.plot(lens, [lr_results[s]['s4d'] * 100 for s in lens], 'o-', label='S4D',
            color='blue', linewidth=2, markersize=8)
    ax.plot(lens, [lr_results[s]['lstm'] * 100 for s in lens], 's--', label='LSTM',
            color='green', linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Long-Range Dependency: Accuracy vs Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "long_range.png", dpi=150)
    plt.close()

    # 3. HiPPO vs Random A
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_range = range(1, n_epochs + 1)
    ax.plot(epochs_range, [a * 100 for a in hippo_accs], 'o-', label='HiPPO A',
            color='blue', linewidth=2)
    ax.plot(epochs_range, [a * 100 for a in random_accs], 's--', label='Random A',
            color='orange', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("HiPPO vs Random A Initialization (S4D on sMNIST)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate(f'HiPPO: {hippo_test*100:.1f}%\nRandom: {random_test*100:.1f}%',
                xy=(0.7, 0.3), xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(results_dir / "hippo_vs_random.png", dpi=150)
    plt.close()

    # 4. Learned SSM kernel visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    s4d.eval()
    with torch.no_grad():
        K = s4d.blocks[0].s4d._compute_kernel(200)  # (d_model, 200)

    for i, ax in enumerate(axes.flat):
        if i < K.shape[0]:
            ax.plot(K[i, :100].cpu().numpy(), linewidth=1.5)
            ax.set_title(f"Feature dim {i}")
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)

    plt.suptitle("Learned SSM Kernel K[k] = CA\u0304^k B\u0304  (first 6 channels)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "ssm_kernel.png", dpi=150)
    plt.close()

    # 5. Concept diagram: continuous SSM -> discrete -> conv/recurrent modes
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    concepts = [
        ("Continuous\nSSM",
         "x'(t) = Ax(t) + Bu(t)\ny(t) = Cx(t) + Du(t)\n\nA: HiPPO matrix\n(special init for\nlong-range memory)",
         0.12, 'purple'),
        ("Bilinear\nDiscretization",
         "\u0100 = (I-\u0394/2\u00b7A)\u207b\u00b9(I+\u0394/2\u00b7A)\n"
         "B\u0304 = (I-\u0394/2\u00b7A)\u207b\u00b9\u0394B\n\n"
         "Continuous \u2192 Discrete\nPreserves structure",
         0.38, 'blue'),
        ("Conv Mode\n(Training)",
         "K = (CB\u0304, C\u0100B\u0304, C\u0100\u00b2B\u0304,...)\ny = K * u  (via FFT)\n\nO(L log L) parallel\nFull sequence at once",
         0.64, 'green'),
        ("Recurrent Mode\n(Inference)",
         "x\u2096 = \u0100x\u2096\u208b\u2081 + B\u0304u\u2096\ny\u2096 = Cx\u2096 + Du\u2096\n\nO(N) per step\nStreaming / autoregressive",
         0.88, 'red'),
    ]

    for name, desc, x_pos, color in concepts:
        ax.text(x_pos, 0.8, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=9, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Arrows between concept boxes
    for x_start, x_end in [(0.22, 0.28), (0.48, 0.54), (0.74, 0.78)]:
        ax.annotate('', xy=(x_end, 0.55), xytext=(x_start, 0.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax.set_title("S4: From Continuous SSM to Efficient Computation",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "s4_concept.png", dpi=150)
    plt.close()

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
