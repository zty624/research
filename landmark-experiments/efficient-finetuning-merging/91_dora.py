"""
Minimal DoRA Reproduction
==========================
Reproduces core ideas from "DoRA: Weight-Decomposed Low-Rank Adaptation"
(2402.09353, Liu et al.):

1. Weight decomposition: W = m * (V / ||V||), where m is a magnitude vector
   (column-wise norms) and V/||V|| is the normalized direction matrix.
2. LoRA is applied only to the direction component V, while magnitude m is
   learned separately with its own parameters. This decouples the "how much"
   (magnitude) from "which direction" (direction) of weight updates.
3. Key insight: full finetuning adjusts both magnitude and direction
   independently per column, while vanilla LoRA couples them. DoRA recovers
   this decoupling with minimal extra parameters (one scalar per output dim
   per layer).

Comparison: Full finetuning vs LoRA vs DoRA on a synthetic classification task.
Visualizations: weight decomposition, magnitude/direction update patterns,
training curves, and parameter efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── LoRA Linear ──

class LoRALinear(nn.Module):
    """Standard LoRA: h = Wx + (alpha/r) * BAx."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.register_buffer('weight', torch.randn(out_features, in_features) * 0.02)
        self.register_buffer('bias', torch.zeros(out_features))

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


# ── DoRA Linear ──

class DoRALinear(nn.Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation.

    Decomposes the adapted weight W' = m * (V + BA) / ||V + BA||,
    where V is the frozen pretrained direction, BA is the LoRA update to
    direction, and m is a learnable magnitude vector (one scalar per output
    dimension). At initialization, m = ||W_pretrained||_col so W' = W.
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16.0,
                 pretrained_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Store pretrained weight as frozen buffer
        if pretrained_weight is not None:
            W = pretrained_weight.detach().float()
        else:
            W = torch.randn(out_features, in_features) * 0.02

        # Decompose: V = W (direction before normalization), m = ||W|| per column
        col_norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (out, 1)

        self.register_buffer('V', W.clone())          # frozen direction base
        self.register_buffer('bias', torch.zeros(out_features))
        self.m = nn.Parameter(col_norms.squeeze(1).clone())  # learnable magnitude (out,)

        # LoRA on direction only
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def _get_adapted_direction(self):
        """Compute adapted direction: (V + scaling * BA) / ||V + scaling * BA||."""
        delta = self.lora_B @ self.lora_A  # (out, in)
        V_adapted = self.V + delta * self.scaling  # adapted direction (before norm)
        col_norms = V_adapted.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return V_adapted / col_norms  # normalized direction

    def forward(self, x):
        direction = self._get_adapted_direction()          # (out, in), unit-norm rows
        weight = self.m.unsqueeze(1) * direction            # (out, in)
        return F.linear(x, weight, self.bias)


# ── Small Transformer ──

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        d = self.head_dim
        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SimpleSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=200, d_model=64, n_heads=4, d_ff=128,
                 n_layers=2, n_classes=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        h = h.mean(dim=1)  # pool over sequence
        return self.head(h)


# ── Apply LoRA / DoRA to model ──

def apply_lora(model, rank=8, alpha=16.0, target_names=None):
    """Replace Linear layers with LoRALinear (in-place)."""
    if target_names is None:
        target_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'head',
                        'ff.0', 'ff.2']

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            replacements.append((name, module))

    lora_params = 0
    for name, old in replacements:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer = LoRALinear(old.in_features, old.out_features, rank, alpha)
        layer.weight.copy_(old.weight.data)
        layer.bias.copy_(old.bias.data)
        lora_params += layer.lora_A.numel() + layer.lora_B.numel()
        setattr(parent, parts[-1], layer)

    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name

    return model, lora_params


def apply_dora(model, rank=8, alpha=16.0, target_names=None):
    """Replace Linear layers with DoRALinear (in-place)."""
    if target_names is None:
        target_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'head',
                        'ff.0', 'ff.2']

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            replacements.append((name, module))

    dora_params = 0
    for name, old in replacements:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer = DoRALinear(old.in_features, old.out_features, rank, alpha,
                           pretrained_weight=old.weight.data)
        layer.bias.copy_(old.bias.data)
        dora_params += layer.lora_A.numel() + layer.lora_B.numel() + layer.m.numel()
        setattr(parent, parts[-1], layer)

    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name or name.endswith('.m')

    return model, dora_params


# ── Synthetic Data ──

def generate_classification_data(batch_size, seq_len, vocab_size, n_classes):
    """Deterministic classification from token patterns."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Label based on sum of first-half tokens vs second-half
    first_half = x[:, :seq_len // 2].float().sum(dim=1)
    second_half = x[:, seq_len // 2:].float().sum(dim=1)
    ratio = first_half / (second_half + 1.0)
    y = (ratio * n_classes / (ratio.max() + 1)).long().clamp(0, n_classes - 1)
    return x, y


# ── Training ──

def train_model(model, vocab_size, seq_len, n_classes, n_steps=1000, lr=1e-3,
                device='cpu'):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses, accs = [], []
    for step in range(n_steps):
        x, y = generate_classification_data(64, seq_len, vocab_size, n_classes)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Weight Decomposition Analysis ──

def analyze_weight_decomposition(model, layer_name_filter='head'):
    """Extract magnitude and direction statistics from DoRA layers."""
    mag_ratios = []  # how much magnitude changed from init
    dir_changes = []  # how much direction changed (cosine distance)

    for name, module in model.named_modules():
        if not isinstance(module, DoRALinear):
            continue
        # Current adapted direction
        direction = module._get_adapted_direction().detach()  # (out, in)
        # Original direction
        V_orig = module.V
        V_orig_normed = V_orig / V_orig.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Cosine similarity per column
        cos_sim = F.cosine_similarity(direction, V_orig_normed, dim=1)
        dir_changes.append(1.0 - cos_sim)  # cosine distance

        # Magnitude change: current m vs original ||V||_col
        orig_mag = module.V.norm(dim=1)
        mag_ratios.append(module.m.detach() / orig_mag.clamp(min=1e-8))

    return mag_ratios, dir_changes


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "91-dora"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    vocab_size = 200
    seq_len = 32
    n_classes = 10
    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 2
    lora_rank = 8
    n_steps = 2000

    # Pre-train a base model
    print("=== Pre-training base model ===")
    base_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, n_classes
    ).to(device)
    pretrain_losses, _ = train_model(
        base_model, vocab_size, seq_len, n_classes,
        n_steps=500, lr=1e-3, device=device
    )
    print(f"  Pre-train final loss: {pretrain_losses[-1]:.4f}\n")

    # ── Full Finetuning ──
    print("=== Full Finetuning ===")
    torch.manual_seed(42)
    full_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, n_classes
    ).to(device)
    full_model.load_state_dict(base_model.state_dict())
    full_losses, full_accs = train_model(
        full_model, vocab_size, seq_len, n_classes,
        n_steps=n_steps, lr=5e-4, device=device
    )
    full_params = count_trainable(full_model)
    print(f"  Trainable: {full_params:,}\n")

    # ── LoRA ──
    print("=== LoRA ===")
    torch.manual_seed(42)
    lora_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, n_classes
    ).to(device)
    lora_model.load_state_dict(base_model.state_dict())
    lora_model, lora_p = apply_lora(lora_model, rank=lora_rank, alpha=16.0)
    lora_model = lora_model.to(device)
    lora_losses, lora_accs = train_model(
        lora_model, vocab_size, seq_len, n_classes,
        n_steps=n_steps, lr=1e-3, device=device
    )
    lora_params = count_trainable(lora_model)
    print(f"  Trainable: {lora_params:,} ({lora_params/full_params*100:.2f}%)\n")

    # ── DoRA ──
    print("=== DoRA ===")
    torch.manual_seed(42)
    dora_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, n_classes
    ).to(device)
    dora_model.load_state_dict(base_model.state_dict())
    dora_model, dora_p = apply_dora(dora_model, rank=lora_rank, alpha=16.0)
    dora_model = dora_model.to(device)
    dora_losses, dora_accs = train_model(
        dora_model, vocab_size, seq_len, n_classes,
        n_steps=n_steps, lr=1e-3, device=device
    )
    dora_params = count_trainable(dora_model)
    print(f"  Trainable: {dora_params:,} ({dora_params/full_params*100:.2f}%)\n")

    # ── Weight Decomposition Analysis ──
    print("=== Weight Decomposition Analysis ===")
    mag_ratios, dir_changes = analyze_weight_decomposition(dora_model)

    # Also analyze full FT: what would magnitude/direction changes look like?
    ft_mag_ratios = []
    ft_dir_changes = []
    base_sd = base_model.state_dict()
    full_sd = full_model.state_dict()
    for name in base_sd:
        if full_sd[name].dim() < 2:
            continue
        W_base = base_sd[name].float()
        W_ft = full_sd[name].float()
        delta = W_ft - W_base
        # Magnitude ratio
        base_norms = W_base.norm(dim=1)
        ft_norms = W_ft.norm(dim=1)
        ft_mag_ratios.append(ft_norms / base_norms.clamp(min=1e-8))
        # Direction change (cosine distance)
        base_dir = W_base / base_norms.unsqueeze(1).clamp(min=1e-8)
        ft_dir = W_ft / ft_norms.unsqueeze(1).clamp(min=1e-8)
        cos_sim = F.cosine_similarity(base_dir, ft_dir, dim=1)
        ft_dir_changes.append(1.0 - cos_sim)

    # LoRA delta analysis
    lora_mag_ratios = []
    lora_dir_changes = []
    for name, module in lora_model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        W_base = module.weight.detach()
        delta = (module.lora_B @ module.lora_A * module.scaling).detach()
        W_adapted = W_base + delta
        base_norms = W_base.norm(dim=1)
        adapted_norms = W_adapted.norm(dim=1)
        lora_mag_ratios.append(adapted_norms / base_norms.clamp(min=1e-8))
        base_dir = W_base / base_norms.unsqueeze(1).clamp(min=1e-8)
        adapted_dir = W_adapted / adapted_norms.unsqueeze(1).clamp(min=1e-8)
        cos_sim = F.cosine_similarity(base_dir, adapted_dir, dim=1)
        lora_dir_changes.append(1.0 - cos_sim)

    # Print summary
    for label, mags, dirs in [
        ("Full FT", ft_mag_ratios, ft_dir_changes),
        ("LoRA", lora_mag_ratios, lora_dir_changes),
        ("DoRA", mag_ratios, dir_changes),
    ]:
        all_mag = torch.cat(mags).cpu().numpy() if mags else np.array([])
        all_dir = torch.cat(dirs).cpu().numpy() if dirs else np.array([])
        if len(all_mag) > 0:
            print(f"  {label}: mag_ratio mean={all_mag.mean():.4f} std={all_mag.std():.4f} | "
                  f"dir_change mean={all_dir.mean():.6f} std={all_dir.std():.6f}")

    # ── Visualization ──

    # 1. Training loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    window = 50

    for losses, label, color in [
        (full_losses, 'Full FT', '#2196F3'),
        (lora_losses, 'LoRA', '#FF9800'),
        (dora_losses, 'DoRA', '#4CAF50'),
    ]:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=label, color=color, linewidth=2)

    for accs, label, color in [
        (full_accs, 'Full FT', '#2196F3'),
        (lora_accs, 'LoRA', '#FF9800'),
        (dora_accs, 'DoRA', '#4CAF50'),
    ]:
        smoothed = np.convolve(accs, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, label=label, color=color, linewidth=2)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DoRA: Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 2. Weight decomposition: magnitude ratio vs direction change scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, mags, dirs, label, color in [
        (axes[0], ft_mag_ratios, ft_dir_changes, 'Full FT', '#2196F3'),
        (axes[1], lora_mag_ratios, lora_dir_changes, 'LoRA', '#FF9800'),
        (axes[2], mag_ratios, dir_changes, 'DoRA', '#4CAF50'),
    ]:
        all_mag = torch.cat(mags).cpu().numpy() if mags else np.array([])
        all_dir = torch.cat(dirs).cpu().numpy() if dirs else np.array([])
        if len(all_mag) > 0:
            ax.scatter(all_mag, all_dir, alpha=0.3, s=8, color=color)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Magnitude Ratio (m_new / m_orig)')
        ax.set_ylabel('Direction Change (1 - cos similarity)')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.suptitle('DoRA: Magnitude-Direction Decomposition Analysis\n'
                 '(Full FT adjusts both independently; LoRA couples them; DoRA decouples)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "weight_decomposition.png", dpi=150)
    plt.close()

    # 3. Magnitude and direction change distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mags, label, color in [
        (ft_mag_ratios, 'Full FT', '#2196F3'),
        (lora_mag_ratios, 'LoRA', '#FF9800'),
        (mag_ratios, 'DoRA', '#4CAF50'),
    ]:
        all_mag = torch.cat(mags).cpu().numpy() if mags else np.array([])
        if len(all_mag) > 0:
            ax1.hist(all_mag, bins=50, alpha=0.5, label=label, color=color, density=True)

    for dirs, label, color in [
        (ft_dir_changes, 'Full FT', '#2196F3'),
        (lora_dir_changes, 'LoRA', '#FF9800'),
        (dir_changes, 'DoRA', '#4CAF50'),
    ]:
        all_dir = torch.cat(dirs).cpu().numpy() if dirs else np.array([])
        if len(all_dir) > 0:
            ax2.hist(all_dir, bins=50, alpha=0.5, label=label, color=color, density=True)

    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='No change')
    ax1.set_xlabel('Magnitude Ratio')
    ax1.set_ylabel('Density')
    ax1.set_title('Magnitude Change Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Direction Change (1 - cos sim)')
    ax2.set_ylabel('Density')
    ax2.set_title('Direction Change Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DoRA: Weight Update Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "update_distributions.png", dpi=150)
    plt.close()

    # 4. Parameter efficiency and final performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['Full FT', 'LoRA', 'DoRA']
    params = [full_params, lora_params, dora_params]
    final_accs = [full_accs[-1], lora_accs[-1], dora_accs[-1]]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    bars = ax1.bar(methods, params, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Trainable Parameters')
    ax1.set_title('Parameter Count')
    ax1.grid(True, alpha=0.3, axis='y')

    bars = ax2.bar(methods, final_accs, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, final_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Final Training Accuracy')
    ax2.set_title('Performance')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DoRA vs LoRA vs Full FT', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "performance_comparison.png", dpi=150)
    plt.close()

    # 5. DoRA magnitude trajectory for a sample layer
    # Re-train DoRA and track magnitude evolution
    print("\n=== Tracking DoRA Magnitude Evolution ===")
    torch.manual_seed(42)
    dora_track = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, n_classes
    ).to(device)
    dora_track.load_state_dict(base_model.state_dict())
    dora_track, _ = apply_dora(dora_track, rank=lora_rank, alpha=16.0)
    dora_track = dora_track.to(device)

    # Find the head layer for tracking
    head_layer = None
    init_mags = None
    for name, module in dora_track.named_modules():
        if isinstance(module, DoRALinear) and 'head' in name:
            head_layer = module
            init_mags = module.m.detach().clone()
            break

    mag_history = []
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, dora_track.parameters()),
        lr=1e-3, weight_decay=0.01
    )
    for step in range(500):
        x, y = generate_classification_data(64, seq_len, vocab_size, n_classes)
        x, y = x.to(device), y.to(device)
        logits = dora_track(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 and head_layer is not None:
            mag_history.append(head_layer.m.detach().cpu().clone())

    fig, ax = plt.subplots(figsize=(10, 5))
    if head_layer is not None and len(mag_history) > 0:
        n_dims = min(10, mag_history[0].shape[0])
        steps = list(range(0, 500, 50))
        for i in range(n_dims):
            values = [mh[i].item() for mh in mag_history]
            ax.plot(steps, values, alpha=0.7, label=f'dim {i}')
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='init ratio=1')
    ax.set_xlabel('Step')
    ax.set_ylabel('Magnitude m')
    ax.set_title('DoRA: Magnitude Parameter Evolution (head layer, first 10 dims)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "magnitude_evolution.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
