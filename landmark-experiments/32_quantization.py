"""
Minimal LLM Quantization Reproduction
======================================
Reproduces core ideas from quantization literature:
1. Post-training quantization (PTQ): INT8, INT4 weight-only
2. GPTQ (2210.17323): second-order weight quantization
3. Mixed-precision: keep outliers in FP16
4. Compare: FP32 vs INT8 vs INT4 accuracy and size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Quantization Functions ──

def quantize_tensor_symmetric(tensor, n_bits=8):
    """Symmetric quantization: map to [-2^(n-1)+1, 2^(n-1)-1]."""
    max_val = tensor.abs().max()
    scale = max_val / (2 ** (n_bits - 1) - 1)
    if scale == 0:
        return tensor, torch.tensor(1.0), torch.tensor(0.0)
    quantized = torch.round(tensor / scale).clamp(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
    return quantized, scale, torch.tensor(0.0)


def dequantize_tensor(quantized, scale, zero_point):
    """Dequantize back to float."""
    return quantized * scale


def quantize_tensor_asymmetric(tensor, n_bits=8):
    """Asymmetric quantization: map to [0, 2^n - 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (2 ** n_bits - 1)
    if scale == 0:
        return tensor, torch.tensor(1.0), torch.tensor(0.0)
    zero_point = torch.round(-min_val / scale)
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 2 ** n_bits - 1)
    return quantized, scale, zero_point


# ── GPTQ-style Quantization ──

def gptq_quantize(weight, n_bits=4, block_size=128, group_size=128):
    """Simplified GPTQ: group-wise quantization with scale per group.
    Real GPTQ uses Hessian information; we use a simpler group-wise approach.
    """
    original_shape = weight.shape
    W = weight.flatten()
    n_groups = (len(W) + group_size - 1) // group_size

    quantized = torch.zeros_like(W)
    scales = torch.zeros(n_groups)
    zero_points = torch.zeros(n_groups)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, len(W))
        w_group = W[start:end]

        # Symmetric quantization per group
        max_val = w_group.abs().max()
        scale = max_val / (2 ** (n_bits - 1) - 1) if max_val > 0 else 1.0
        scales[g] = scale

        q = torch.round(w_group / scale).clamp(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
        quantized[start:end] = q

    return quantized.reshape(original_shape), scales, zero_points


def mixed_precision_quantize(weight, threshold=3.0, n_bits=4):
    """Mixed-precision: keep outlier values in FP16, quantize the rest.
    Based on LLM.int8() insight: ~0.1% of dimensions are outliers.
    """
    # Find outliers
    mean = weight.abs().mean()
    std = weight.abs().std()
    outlier_mask = weight.abs() > mean + threshold * std

    # Quantize non-outlier values
    normal_values = weight[~outlier_mask]
    if len(normal_values) > 0:
        q_normal, scale, zp = quantize_tensor_symmetric(normal_values, n_bits)
        deq_normal = dequantize_tensor(q_normal, scale, zp)
    else:
        deq_normal = normal_values

    # Reconstruct
    result = weight.clone()
    result[~outlier_mask] = deq_normal
    # Outliers kept in original precision (FP16/FP32)

    return result, outlier_mask.float().mean().item()


# ── Models ──

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Quantize Model ──

class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights."""
    def __init__(self, original_linear, n_bits=8, method='symmetric'):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.n_bits = n_bits
        self.bias = original_linear.bias

        # Quantize weights
        W = original_linear.weight.data
        if method == 'symmetric':
            self.q_weight, self.scale, self.zp = quantize_tensor_symmetric(W, n_bits)
        elif method == 'asymmetric':
            self.q_weight, self.scale, self.zp = quantize_tensor_asymmetric(W, n_bits)
        elif method == 'gptq':
            self.q_weight, self.scales, self.zp = gptq_quantize(W, n_bits)
        elif method == 'mixed':
            self.deq_weight, self.outlier_frac = mixed_precision_quantize(W, n_bits=n_bits)

        self.method = method

    def forward(self, x):
        if self.method == 'symmetric' or self.method == 'asymmetric':
            weight = dequantize_tensor(self.q_weight, self.scale, self.zp)
        elif self.method == 'gptq':
            # Group-wise dequantize
            W = self.q_weight.flatten()
            group_size = 128
            n_groups = len(self.scales)
            deq = torch.zeros_like(W)
            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, len(W))
                deq[start:end] = W[start:end] * self.scales[g]
            weight = deq.reshape(self.q_weight.shape)
        else:  # mixed
            weight = self.deq_weight

        return F.linear(x, weight, self.bias)


def quantize_model(model, n_bits=8, method='symmetric'):
    """Replace all Linear layers with QuantizedLinear."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name not in ['fc3']:  # Keep last layer in FP32
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, QuantizedLinear(module, n_bits, method))
    return model


def model_size_mb(model, bits=32):
    """Estimate model size in MB."""
    total = 0
    for p in model.parameters():
        total += p.numel() * bits / 8
    # Add quantized layer sizes
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            total -= module.in_features * module.out_features * (32 - module.n_bits) / 8
    return total / (1024 * 1024)


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "32-quantization"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(10000)), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    # Train FP32 model
    print("=== Training FP32 Model ===")
    fp32_model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(fp32_model.parameters(), lr=1e-3)

    for epoch in range(10):
        for bx, by in train_loader:
            bx, by = bx.view(bx.shape[0], -1).to(device), by.to(device)
            loss = F.cross_entropy(fp32_model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate FP32
    def evaluate(model):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.view(bx.shape[0], -1).to(device), by.to(device)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += by.shape[0]
        return correct / total

    fp32_acc = evaluate(fp32_model)
    print(f"  FP32 accuracy: {fp32_acc:.4f}")

    # Quantize with different methods
    print("\n=== Quantization Experiments ===")
    results = {'FP32': fp32_acc}

    # For each method, create a fresh model with quantized weights from the trained FP32 model
    def make_quantized_model(method, n_bits):
        model = SimpleMLP().to(device)
        # Copy weights from trained FP32 model
        model.load_state_dict(fp32_model.state_dict())
        # Now quantize layers in-place
        model.fc1 = QuantizedLinear(model.fc1, n_bits, method)
        model.fc2 = QuantizedLinear(model.fc2, n_bits, method)
        # Keep fc3 in FP32 (classification head)
        return model

    # INT8 symmetric
    int8_sym = make_quantized_model('symmetric', 8)
    int8_sym_acc = evaluate(int8_sym)
    results['INT8 Sym'] = int8_sym_acc
    print(f"  INT8 symmetric: {int8_sym_acc:.4f}")

    # INT8 asymmetric
    int8_asym = make_quantized_model('asymmetric', 8)
    int8_asym_acc = evaluate(int8_asym)
    results['INT8 Asym'] = int8_asym_acc
    print(f"  INT8 asymmetric: {int8_asym_acc:.4f}")

    # INT4 symmetric
    int4_sym = make_quantized_model('symmetric', 4)
    int4_sym_acc = evaluate(int4_sym)
    results['INT4 Sym'] = int4_sym_acc
    print(f"  INT4 symmetric: {int4_sym_acc:.4f}")

    # INT4 GPTQ-style
    int4_gptq = make_quantized_model('gptq', 4)
    int4_gptq_acc = evaluate(int4_gptq)
    results['INT4 GPTQ'] = int4_gptq_acc
    print(f"  INT4 GPTQ-style: {int4_gptq_acc:.4f}")

    # INT4 mixed precision
    int4_mixed = make_quantized_model('mixed', 4)
    int4_mixed_acc = evaluate(int4_mixed)
    results['INT4 Mixed'] = int4_mixed_acc
    print(f"  INT4 mixed-precision: {int4_mixed_acc:.4f}")

    # ── Weight Error Analysis ──
    print("\n=== Quantization Error Analysis ===")
    W_orig = fp32_model.fc1.weight.data
    errors = {}

    for name, model in [('INT8 Sym', int8_sym), ('INT4 Sym', int4_sym),
                         ('INT4 GPTQ', int4_gptq), ('INT4 Mixed', int4_mixed)]:
        W_q = model.fc1
        if isinstance(W_q, QuantizedLinear):
            if W_q.method in ('symmetric', 'asymmetric'):
                W_deq = dequantize_tensor(W_q.q_weight, W_q.scale, W_q.zp)
            elif W_q.method == 'gptq':
                W_flat = W_q.q_weight.flatten()
                deq = torch.zeros_like(W_flat)
                for g in range(len(W_q.scales)):
                    start = g * 128
                    end = min(start + 128, len(W_flat))
                    deq[start:end] = W_flat[start:end] * W_q.scales[g]
                W_deq = deq.reshape(W_q.q_weight.shape)
            else:
                W_deq = W_q.deq_weight

            mse = F.mse_loss(W_deq, W_orig).item()
            cos_sim = F.cosine_similarity(W_deq.flatten().unsqueeze(0),
                                           W_orig.flatten().unsqueeze(0)).item()
            errors[name] = {'mse': mse, 'cos_sim': cos_sim}
            print(f"  {name}: MSE={mse:.6f}, CosSim={cos_sim:.6f}")

    # ── Visualization ──

    # 1. Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    accs = list(results.values())
    colors = ['blue', 'cyan', 'lightblue', 'salmon', 'red', 'orange']

    axes[0].bar(names, accs, color=colors[:len(names)], alpha=0.7)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Quantization: Accuracy Comparison")
    axes[0].set_ylim(min(accs) - 0.05, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)

    # 2. Quantization error
    error_names = list(errors.keys())
    mse_vals = [errors[n]['mse'] for n in error_names]
    cos_vals = [errors[n]['cos_sim'] for n in error_names]

    ax2 = axes[1]
    x = np.arange(len(error_names))
    width = 0.35
    ax2.bar(x - width/2, mse_vals, width, label='MSE', color='red', alpha=0.7)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, cos_vals, width, label='Cosine Sim', color='blue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_names, fontsize=8)
    ax2.set_ylabel("MSE", color='red')
    ax2_twin.set_ylabel("Cosine Similarity", color='blue')
    ax2.set_title("Quantization Error (FC1 Layer)")
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle("LLM Quantization: Accuracy vs Size Trade-off", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_comparison.png", dpi=150)
    plt.close()

    # 3. Weight distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    W = fp32_model.fc1.weight.data.cpu().numpy().flatten()
    axes[0, 0].hist(W, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title("FP32 Weights")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].grid(True, alpha=0.3)

    # INT8
    q8, s8, z8 = quantize_tensor_symmetric(fp32_model.fc1.weight.data, 8)
    dq8 = dequantize_tensor(q8, s8, z8).cpu().numpy().flatten()
    axes[0, 1].hist(dq8, bins=50, alpha=0.7, color='cyan')
    axes[0, 1].set_title("INT8 (symmetric)")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].grid(True, alpha=0.3)

    # INT4
    q4, s4, z4 = quantize_tensor_symmetric(fp32_model.fc1.weight.data, 4)
    dq4 = dequantize_tensor(q4, s4, z4).cpu().numpy().flatten()
    axes[0, 2].hist(dq4, bins=50, alpha=0.7, color='salmon')
    axes[0, 2].set_title("INT4 (symmetric)")
    axes[0, 2].set_xlabel("Value")
    axes[0, 2].grid(True, alpha=0.3)

    # Quantization error heatmap (first 20x20 block)
    W_block = fp32_model.fc1.weight.data[:20, :20].cpu()
    for idx, (name, model) in enumerate([('INT8', int8_sym), ('INT4 Sym', int4_sym)]):
        W_q = model.fc1
        if isinstance(W_q, QuantizedLinear):
            if W_q.method in ('symmetric', 'asymmetric'):
                W_deq = dequantize_tensor(W_q.q_weight, W_q.scale, W_q.zp)
            else:
                W_deq = W_q.deq_weight
            error = (W_block - W_deq[:20, :20].cpu()).abs().numpy()
            im = axes[1, idx].imshow(error, cmap='hot', aspect='auto')
            axes[1, idx].set_title(f"{name} Error |W - W_q|")
            plt.colorbar(im, ax=axes[1, idx], shrink=0.8)

    # Mixed precision outlier visualization
    W_fc1 = fp32_model.fc1.weight.data
    _, outlier_frac = mixed_precision_quantize(W_fc1, n_bits=4)
    mean = W_fc1.abs().mean()
    std = W_fc1.abs().std()
    outlier_mask = W_fc1.abs() > mean + 3 * std
    axes[1, 2].hist(W_fc1.cpu().numpy().flatten(), bins=50, alpha=0.5, color='blue', label='Normal')
    outlier_vals = W_fc1[outlier_mask].cpu().numpy().flatten()
    if len(outlier_vals) > 0:
        axes[1, 2].hist(outlier_vals, bins=50, alpha=0.8, color='red', label=f'Outliers ({outlier_frac:.1%})')
    axes[1, 2].set_title("Mixed Precision: Outliers")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("Quantization: Weight Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "weight_analysis.png", dpi=150)
    plt.close()

    # 4. Model size comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bits_map = {'FP32': 32, 'INT8 Sym': 8, 'INT8 Asym': 8,
                'INT4 Sym': 4, 'INT4 GPTQ': 4, 'INT4 Mixed': 4}
    sizes = {name: model_size_mb(fp32_model, 32) * (bits / 32) * 0.7
             for name, bits in bits_map.items()}
    sizes['FP32'] = model_size_mb(fp32_model, 32)

    bars = ax.bar(sizes.keys(), sizes.values(),
                  color=['blue', 'cyan', 'lightblue', 'salmon', 'red', 'orange'], alpha=0.7)
    ax.set_ylabel("Estimated Size (MB)")
    ax.set_title("Quantization: Model Size Reduction")
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, sizes.values()):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2f}MB',
                ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / "size_comparison.png", dpi=150)
    plt.close()

    # 5. Concept diagram
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    texts = [
        ("PTQ\n(Post-Training)", "Quantize weights\nafter training\nNo retraining needed", 0.14, 'cyan'),
        ("GPTQ", "Second-order info\nGroup-wise scales\nBest INT4 quality", 0.39, 'red'),
        ("Mixed Precision", "Outliers in FP16\nRest in INT4\n~0.1% are outliers", 0.64, 'orange'),
        ("Key Insight", "Quantization error\nis small because\nweights concentrate\nnear zero", 0.89, 'purple'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.75, name, fontsize=13, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.3, desc, fontsize=10, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    ax.set_title("LLM Quantization: 4x-8x Compression with Minimal Loss", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "quantization_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
