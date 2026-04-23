"""
Minimal Mixed Precision Training Reproduction
==============================================
Reproduces core ideas from mixed precision training literature:
1. Micikevicius et al. (1710.03740): FP16 training with loss scaling
2. Automatic Mixed Precision (AMP): dynamic loss scaling, FP32 master weights
3. Compare: FP32 vs FP16 vs BF16 training convergence
4. Show: gradient overflow frequency, memory usage, dynamic vs static loss scaling
5. Demonstrate: why BF16 is safer than FP16 (wider dynamic range)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── Model ──

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask)
        return self.head(h)


# ── Data ──

def generate_data(vocab_size=64, length=10000):
    """Structured sequence with patterns."""
    data = []
    for i in range(length):
        base = i % 20
        noise = np.random.randint(0, 5)
        data.append(min(base + noise, vocab_size - 1))
    return torch.tensor(data, dtype=torch.long)


# ── Loss Scalers (educational, for manual demonstration) ──

class DynamicLossScaler:
    """Dynamic loss scaling: adjust scale factor based on overflow history.

    - Start with a large scale factor
    - If gradients overflow (inf/nan), skip the step and halve the scale
    - If N consecutive steps succeed, double the scale
    """

    def __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_counter = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, params):
        """Unscale gradients and check for overflow."""
        overflow = False
        for p in params:
            if p.grad is not None:
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    overflow = True
                    break
                p.grad.data.div_(self.scale)
        if overflow:
            self._growth_counter = 0
            self.scale *= self.backoff_factor
            self.scale = max(self.scale, 1.0)
        else:
            self._growth_counter += 1
            if self._growth_counter >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_counter = 0
        return overflow


class StaticLossScaler:
    """Static loss scaling: fixed scale factor throughout training."""

    def __init__(self, scale=2**8):
        self.scale = scale

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, params):
        overflow = False
        for p in params:
            if p.grad is not None:
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    overflow = True
                    break
                p.grad.data.div_(self.scale)
        return overflow


# ── Training Functions ──

def train_with_autocast(model, data, n_steps=2000, seq_len=32, batch_size=32,
                        dtype=torch.float16, use_grad_scaler=True, lr=3e-4, device='cpu'):
    """Train using PyTorch's torch.autocast + GradScaler for correct mixed precision.

    This is the standard production approach:
    - autocast handles casting ops to lower precision automatically
    - GradScaler handles dynamic loss scaling + skipping overflow steps
    - Optimizer (AdamW) maintains FP32 master copy internally
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if use_grad_scaler:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    else:
        scaler = None

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    losses = []
    overflow_count = 0
    scale_history = []

    for step in range(n_steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        with torch.autocast(device_type, dtype=dtype, enabled=(device_type == 'cuda')):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Check for overflow after unscale
            overflow = False
            for p in model.parameters():
                if p.grad is not None and (torch.isinf(p.grad).any() or torch.isnan(p.grad).any()):
                    overflow = True
                    overflow_count += 1
                    break

            if not overflow:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU fallback: manual mixed precision
            loss.backward()

            # Check overflow
            overflow = False
            for p in model.parameters():
                if p.grad is not None and (torch.isinf(p.grad).any() or torch.isnan(p.grad).any()):
                    overflow = True
                    break

            if overflow:
                overflow_count += 1
                optimizer.zero_grad()
                losses.append(float('nan'))
                scale_history.append(0)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())
        if scaler is not None and device.type == 'cuda':
            scale_history.append(scaler.get_scale())
        else:
            scale_history.append(1.0)

        if (step + 1) % 500 == 0:
            valid_losses = [l for l in losses if not math.isnan(l)]
            avg = np.mean(valid_losses[-100:]) if valid_losses else float('nan')
            sc = scale_history[-1] if scale_history else 1.0
            print(f"  Step {step+1} | Loss: {avg:.4f} | Overflows: {overflow_count} | "
                  f"Scale: {sc:.0f}")

    return losses, overflow_count, scale_history


def train_manual_mixed_precision(model, data, n_steps=2000, seq_len=32, batch_size=32,
                                  dtype=torch.float16, scaler=None, lr=3e-4, device='cpu'):
    """Manual mixed precision training loop (educational).

    Demonstrates the core concepts without relying on torch.autocast:
    - Master weights in FP32
    - Forward pass in lower precision (FP16/BF16) via manual casting
    - Loss scaling to prevent gradient underflow
    - Gradient unscaling and overflow handling
    - Uses SGD for simplicity (master weight update is straightforward)
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # FP32 master copy of weights
    master_params = [p.clone().float().detach() for p in model.parameters()]

    losses = []
    overflow_count = 0
    scale_history = []

    for step in range(n_steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        # Copy master weights to model (cast to target dtype)
        for mp, p in zip(master_params, model.parameters()):
            p.data.copy_(mp.to(dtype))

        # Forward in lower precision
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        # Scale loss
        if scaler is not None:
            scaled_loss = scaler.scale_loss(loss)
        else:
            scaled_loss = loss

        # Backward
        optimizer.zero_grad()
        scaled_loss.backward()

        # Unscale gradients
        if scaler is not None:
            overflow = scaler.unscale_grads(model.parameters())
            if overflow:
                overflow_count += 1
                optimizer.zero_grad()
                losses.append(float('nan'))
                scale_history.append(scaler.scale)
                continue
        else:
            overflow = False
            for p in model.parameters():
                if p.grad is not None and (torch.isinf(p.grad).any() or torch.isnan(p.grad).any()):
                    overflow = True
                    break
            if overflow:
                overflow_count += 1
                optimizer.zero_grad()
                losses.append(float('nan'))
                scale_history.append(0)
                continue

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update master weights in FP32
        with torch.no_grad():
            for mp, p in zip(master_params, model.parameters()):
                if p.grad is not None:
                    mp.add_(p.grad.float(), alpha=-lr)

        # Copy updated master back to model
        for mp, p in zip(master_params, model.parameters()):
            p.data.copy_(mp.to(dtype))

        losses.append(loss.item())
        scale_history.append(scaler.scale if scaler is not None else 1.0)

        if (step + 1) % 500 == 0:
            valid_losses = [l for l in losses if not math.isnan(l)]
            avg = np.mean(valid_losses[-100:]) if valid_losses else float('nan')
            sc = scale_history[-1]
            print(f"  Step {step+1} | Loss: {avg:.4f} | Overflows: {overflow_count} | "
                  f"Scale: {sc:.0f}")

    return losses, overflow_count, scale_history


def train_fp32(model, data, n_steps=2000, seq_len=32, batch_size=32, lr=3e-4, device='cpu'):
    """Standard FP32 training baseline."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    for step in range(n_steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = np.mean(losses[-100:])
            print(f"  Step {step+1} | Loss: {avg:.4f}")

    return losses, 0, [1.0] * n_steps


# ── Precision Range Analysis ──

def analyze_precision_ranges():
    """Show dynamic range differences between FP32, FP16, BF16."""
    return {
        'FP32': {'max': torch.finfo(torch.float32).max, 'min': torch.finfo(torch.float32).tiny,
                 'bits': 32, 'exp_bits': 8, 'mant_bits': 23},
        'FP16': {'max': torch.finfo(torch.float16).max, 'min': torch.finfo(torch.float16).tiny,
                 'bits': 16, 'exp_bits': 5, 'mant_bits': 10},
        'BF16': {'max': torch.finfo(torch.bfloat16).max, 'min': torch.finfo(torch.bfloat16).tiny,
                 'bits': 16, 'exp_bits': 8, 'mant_bits': 7},
    }


def gradient_underflow_experiment(dtype=torch.float16, n_trials=1000):
    """Measure how often small gradients underflow to zero in lower precision."""
    underflow_count = 0
    grad_magnitudes = []

    for _ in range(n_trials):
        # Simulate small gradient (common in deep networks)
        grad_fp32 = torch.randn(256) * 1e-4
        grad_fp32_abs = grad_fp32.abs()
        grad_low = grad_fp32.to(dtype).float()
        grad_low_abs = grad_low.abs()

        # Check for underflow (non-zero in FP32 became zero in lower precision)
        underflow_mask = (grad_fp32_abs > 0) & (grad_low_abs == 0)
        underflow_count += underflow_mask.sum().item()

        # Track magnitude ratio
        ratio = grad_low_abs / (grad_fp32_abs + 1e-30)
        grad_magnitudes.append(ratio.mean().item())

    return underflow_count, grad_magnitudes


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "104-mixed-precision"
    results_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 64
    data = generate_data(vocab_size, length=10000)
    n_steps = 2000
    use_cuda = device.type == 'cuda'

    # ── Experiment 1: FP32 baseline ──
    print("=== Experiment 1: FP32 Baseline ===")
    torch.manual_seed(42)
    model_fp32 = SmallTransformer(vocab_size)
    fp32_params = sum(p.numel() for p in model_fp32.parameters())
    fp32_losses, _, _ = train_fp32(model_fp32, data, n_steps=n_steps, device=device)
    fp32_final = np.mean(fp32_losses[-100:])
    print(f"  Final loss: {fp32_final:.4f}, Params: {fp32_params:,}")

    # ── Experiment 2: FP16 with dynamic loss scaling (manual, educational) ──
    print("\n=== Experiment 2: FP16 with Dynamic Loss Scaling (Manual) ===")
    torch.manual_seed(42)
    model_fp16_dyn = SmallTransformer(vocab_size)
    dyn_scaler = DynamicLossScaler(init_scale=2**16, growth_interval=500)
    fp16_dyn_losses, fp16_dyn_overflow, fp16_dyn_scales = train_manual_mixed_precision(
        model_fp16_dyn, data, n_steps=n_steps, dtype=torch.float16,
        scaler=dyn_scaler, lr=1e-2, device=device
    )
    fp16_dyn_valid = [l for l in fp16_dyn_losses if not math.isnan(l)]
    fp16_dyn_final = np.mean(fp16_dyn_valid[-100:]) if fp16_dyn_valid else float('nan')
    print(f"  Final loss: {fp16_dyn_final:.4f}, Overflows: {fp16_dyn_overflow}")

    # ── Experiment 3: FP16 with static loss scaling ──
    print("\n=== Experiment 3: FP16 with Static Loss Scaling (Manual) ===")
    torch.manual_seed(42)
    model_fp16_static = SmallTransformer(vocab_size)
    static_scaler = StaticLossScaler(scale=2**8)
    fp16_static_losses, fp16_static_overflow, fp16_static_scales = train_manual_mixed_precision(
        model_fp16_static, data, n_steps=n_steps, dtype=torch.float16,
        scaler=static_scaler, lr=1e-2, device=device
    )
    fp16_static_valid = [l for l in fp16_static_losses if not math.isnan(l)]
    fp16_static_final = np.mean(fp16_static_valid[-100:]) if fp16_static_valid else float('nan')
    print(f"  Final loss: {fp16_static_final:.4f}, Overflows: {fp16_static_overflow}")

    # ── Experiment 4: BF16 with dynamic loss scaling (manual) ──
    print("\n=== Experiment 4: BF16 with Dynamic Loss Scaling (Manual) ===")
    torch.manual_seed(42)
    model_bf16_dyn = SmallTransformer(vocab_size)
    bf16_dyn_scaler = DynamicLossScaler(init_scale=2**16, growth_interval=500)
    bf16_dyn_losses, bf16_dyn_overflow, bf16_dyn_scales = train_manual_mixed_precision(
        model_bf16_dyn, data, n_steps=n_steps, dtype=torch.bfloat16,
        scaler=bf16_dyn_scaler, lr=1e-2, device=device
    )
    bf16_dyn_valid = [l for l in bf16_dyn_losses if not math.isnan(l)]
    bf16_dyn_final = np.mean(bf16_dyn_valid[-100:]) if bf16_dyn_valid else float('nan')
    print(f"  Final loss: {bf16_dyn_final:.4f}, Overflows: {bf16_dyn_overflow}")

    # ── Experiment 5: FP16 without loss scaling (to show the problem) ──
    print("\n=== Experiment 5: FP16 without Loss Scaling (Manual) ===")
    torch.manual_seed(42)
    model_fp16_noscale = SmallTransformer(vocab_size)
    fp16_noscale_losses, fp16_noscale_overflow, _ = train_manual_mixed_precision(
        model_fp16_noscale, data, n_steps=n_steps, dtype=torch.float16,
        scaler=None, lr=1e-2, device=device
    )
    fp16_noscale_valid = [l for l in fp16_noscale_losses if not math.isnan(l)]
    fp16_noscale_final = np.mean(fp16_noscale_valid[-100:]) if fp16_noscale_valid else float('nan')
    print(f"  Final loss: {fp16_noscale_final:.4f}, Overflows: {fp16_noscale_overflow}")

    # ── Experiment 5b: AMP on CUDA if available ──
    amp_losses = None
    amp_overflow = 0
    amp_scales = None
    amp_final = float('nan')
    if use_cuda:
        print("\n=== Experiment 5b: PyTorch AMP (autocast + GradScaler) ===")
        torch.manual_seed(42)
        model_amp = SmallTransformer(vocab_size)
        amp_losses, amp_overflow, amp_scales = train_with_autocast(
            model_amp, data, n_steps=n_steps, dtype=torch.float16,
            use_grad_scaler=True, device=device
        )
        amp_valid = [l for l in amp_losses if not math.isnan(l)]
        amp_final = np.mean(amp_valid[-100:]) if amp_valid else float('nan')
        print(f"  Final loss: {amp_final:.4f}, Overflows: {amp_overflow}")

    # ── Experiment 6: Gradient underflow analysis ──
    print("\n=== Experiment 6: Gradient Underflow Analysis ===")
    fp16_underflow, fp16_ratios = gradient_underflow_experiment(torch.float16)
    bf16_underflow, bf16_ratios = gradient_underflow_experiment(torch.bfloat16)
    print(f"  FP16 underflow count (out of 256000 values): {fp16_underflow}")
    print(f"  BF16 underflow count (out of 256000 values): {bf16_underflow}")
    print(f"  FP16 avg magnitude ratio: {np.mean(fp16_ratios):.4f}")
    print(f"  BF16 avg magnitude ratio: {np.mean(bf16_ratios):.4f}")

    # ── Experiment 7: Memory comparison ──
    print("\n=== Experiment 7: Memory Footprint ===")
    precision_info = analyze_precision_ranges()
    for name, info in precision_info.items():
        size_mb = fp32_params * info['bits'] / 8 / (1024 ** 2)
        print(f"  {name}: {size_mb:.2f} MB | Range: [{info['min']:.2e}, {info['max']:.2e}] | "
              f"Exp={info['exp_bits']}b, Mant={info['mant_bits']}b")

    # ── Visualization ──

    # 1. Training convergence comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 30
    configs = [
        ('FP32 (AdamW)', fp32_losses, 'blue', '-'),
        ('FP16 + Dynamic Scale', fp16_dyn_losses, 'green', '-'),
        ('FP16 + Static Scale', fp16_static_losses, 'orange', '--'),
        ('BF16 + Dynamic Scale', bf16_dyn_losses, 'purple', '-'),
        ('FP16 (no scaling)', fp16_noscale_losses, 'red', ':'),
    ]
    if amp_losses is not None:
        configs.append(('AMP (autocast)', amp_losses, 'cyan', '-.'))

    for name, losses, color, style in configs:
        smoothed = np.array([l for l in losses if not math.isnan(l)])
        if len(smoothed) > window:
            smoothed = np.convolve(smoothed, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, color=color, linestyle=style, label=name, alpha=0.8, linewidth=1.5)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].set_title("Training Convergence: FP32 vs Mixed Precision")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Final loss bar chart
    final_losses = {
        'FP32': fp32_final,
        'FP16\nDynamic': fp16_dyn_final,
        'FP16\nStatic': fp16_static_final,
        'BF16\nDynamic': bf16_dyn_final,
        'FP16\nNo Scale': fp16_noscale_final,
    }
    colors_bar = ['blue', 'green', 'orange', 'purple', 'red']
    if amp_losses is not None:
        final_losses['AMP'] = amp_final
        colors_bar.append('cyan')
    bars = axes[1].bar(final_losses.keys(), final_losses.values(), color=colors_bar, alpha=0.7)
    axes[1].set_ylabel("Final Loss (last 100 steps avg)")
    axes[1].set_title("Final Training Loss")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, final_losses.values()):
        if not math.isnan(v):
            axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                        f'{v:.3f}', ha='center', fontsize=8)

    plt.suptitle("Mixed Precision Training: Convergence Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "convergence_comparison.png", dpi=150)
    plt.close()

    # 2. Overflow analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    overflow_data = {
        'FP16\nDynamic': fp16_dyn_overflow,
        'FP16\nStatic': fp16_static_overflow,
        'BF16\nDynamic': bf16_dyn_overflow,
        'FP16\nNo Scale': fp16_noscale_overflow,
    }
    ocolors = ['green', 'orange', 'purple', 'red']
    bars = axes[0].bar(overflow_data.keys(), overflow_data.values(), color=ocolors, alpha=0.7)
    axes[0].set_ylabel("Gradient Overflow Count")
    axes[0].set_title(f"Gradient Overflows (out of {n_steps} steps)")
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, overflow_data.values()):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v), ha='center', fontsize=9)

    # Dynamic scale trajectory
    axes[1].plot(fp16_dyn_scales, color='green', label='FP16 Dynamic', alpha=0.7, linewidth=0.8)
    axes[1].axhline(y=2**8, color='orange', linestyle='--', alpha=0.5, label='FP16 Static (2^8)')
    axes[1].plot(bf16_dyn_scales, color='purple', label='BF16 Dynamic', alpha=0.7, linewidth=0.8)
    if amp_scales is not None:
        axes[1].plot(amp_scales, color='cyan', label='AMP GradScaler', alpha=0.7, linewidth=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss Scale Factor")
    axes[1].set_title("Dynamic vs Static Loss Scaling")
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Mixed Precision: Overflow & Loss Scaling", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "overflow_analysis.png", dpi=150)
    plt.close()

    # 3. Gradient underflow
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of magnitude ratios
    axes[0].hist(fp16_ratios, bins=50, alpha=0.6, color='red', label=f'FP16 (underflows={fp16_underflow})')
    axes[0].hist(bf16_ratios, bins=50, alpha=0.6, color='purple', label=f'BF16 (underflows={bf16_underflow})')
    axes[0].set_xlabel("Gradient Magnitude Ratio (low_prec / fp32)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Gradient Precision Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Dynamic range comparison
    precisions = analyze_precision_ranges()
    names = list(precisions.keys())
    max_vals = [precisions[n]['max'] for n in names]
    min_vals = [precisions[n]['min'] for n in names]
    bar_colors = ['blue', 'red', 'purple']

    x = np.arange(len(names))
    width = 0.3
    axes[1].bar(x - width/2, [math.log10(v) for v in max_vals], width,
                label='log10(max)', color=bar_colors, alpha=0.7)
    axes[1].bar(x + width/2, [math.log10(v) for v in min_vals], width,
                label='log10(min normal)', color=bar_colors, alpha=0.3, edgecolor=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("log10(value)")
    axes[1].set_title("Dynamic Range Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Annotate key difference
    bf16_max_val = precisions['BF16']['max']
    axes[1].annotate('BF16 has same\nexp range as FP32\nbut less mantissa',
                     xy=(2, math.log10(bf16_max_val)),
                     xytext=(1.3, math.log10(bf16_max_val) + 5),
                     fontsize=8, arrowprops=dict(arrowstyle='->', color='purple'),
                     color='purple')

    plt.suptitle("FP16 vs BF16: Gradient Underflow & Dynamic Range", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "gradient_underflow.png", dpi=150)
    plt.close()

    # 4. Memory footprint
    fig, ax = plt.subplots(figsize=(10, 5))

    sizes_mb = {name: fp32_params * precisions[name]['bits'] / 8 / (1024 ** 2) for name in names}
    bars = ax.bar(sizes_mb.keys(), sizes_mb.values(), color=bar_colors, alpha=0.7)
    ax.set_ylabel("Model Weight Memory (MB)")
    ax.set_title("Memory Footprint by Precision")
    ax.grid(True, alpha=0.3, axis='y')

    for bar, (name, v) in zip(bars, sizes_mb.items()):
        ratio = v / sizes_mb['FP32']
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f'{v:.2f}MB\n({ratio:.0%})', ha='center', fontsize=9)

    plt.suptitle("Mixed Precision: Memory Savings (2x Reduction)", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "memory_footprint.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Top: training pipeline
    pipeline = [
        ("1. Cast\nWeights", "FP32 master -> FP16\nfor forward pass", 0.10, 'blue'),
        ("2. Forward\n(FP16)", "Compute activations\nin FP16 (2x faster\non Tensor Cores)", 0.32, 'green'),
        ("3. Loss\nScale", "Multiply loss by S\n(e.g., 2^16) to prevent\ngradient underflow", 0.54, 'orange'),
        ("4. Backward\n(FP16)", "Gradients in FP16\nUnscale by 1/S\nCheck for overflow", 0.76, 'red'),
        ("5. Update\n(FP32)", "Copy grads to FP32\nUpdate master weights\nin full precision", 0.95, 'purple'),
    ]

    for name, desc, x_pos, color in pipeline:
        ax.text(x_pos, 0.78, name, fontsize=11, fontweight='bold',
                ha='center', va='center', color=color, transform=ax.transAxes)
        ax.text(x_pos, 0.42, desc, fontsize=8, ha='center', va='center',
                fontfamily='monospace', color=color, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    for x in [0.21, 0.43, 0.65, 0.86]:
        ax.annotate('->', xy=(x, 0.58), fontsize=20, ha='center', va='center',
                    color='gray', transform=ax.transAxes)

    # Bottom: key insights
    insights = [
        ("FP16 Problem", "Small gradients -> 0\n(underflow)\nFixed by loss scaling", 0.17, 'red'),
        ("BF16 Advantage", "Same exp range as FP32\nNo underflow risk\nLess mantissa precision", 0.50, 'purple'),
        ("Dynamic Scaling", "Auto-adjust scale:\n- Overflow -> halve\n- N steps OK -> double\nBest of both worlds", 0.83, 'green'),
    ]

    for name, desc, x_pos, color in insights:
        ax.text(x_pos, 0.18, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color=color, transform=ax.transAxes)
        ax.text(x_pos, 0.06, desc, fontsize=8, ha='center', va='center',
                fontfamily='monospace', color=color, transform=ax.transAxes)

    ax.set_title("Mixed Precision Training: FP16/BF16 with Loss Scaling (1710.03740)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "mixed_precision_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
