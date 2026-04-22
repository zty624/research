"""
Minimal LoRA Reproduction
==========================
Reproduces the core idea from "LoRA: Low-Rank Adaptation of Large Language Models" (2106.09685):
1. Low-rank decomposition: ΔW = BA
2. Comparison: Full fine-tuning vs LoRA vs BitFit
3. Demonstrates that LoRA achieves similar performance with far fewer parameters

Task: Classify MNIST digits using a small Transformer encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


# ── LoRA Layer ──

class LoRALinear(nn.Module):
    """LoRA-wrapped linear layer: h = Wx + BAx"""
    def __init__(self, original_linear, r=4, alpha=8.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        d_out = original_linear.out_features
        d_in = original_linear.in_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # B: (d_out, r), A: (r, d_in)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)

        # Initialize B=0, A=random → ΔW = BA = 0 at start

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    # Proxy attributes so nn.MultiheadAttention internals can access .weight etc.
    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    @property
    def lora_num_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


# ── Simple Transformer for MNIST ──

class SimpleViT(nn.Module):
    """Tiny ViT-like classifier for MNIST."""
    def __init__(self, d_model=64, n_heads=4, n_layers=2, n_classes=10):
        super().__init__()
        self.patch_embed = nn.Linear(28, d_model)  # treat each row as a "patch"
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 29, d_model) * 0.02)  # 28 rows + 1 cls
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 28, 28)  # (B, 28 rows, 28 pixels per row)
        x = self.patch_embed(x)  # (B, 28, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 29, d_model)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0])


# ── Apply LoRA to a model ──

def apply_lora(model, r=4, alpha=8.0, target_modules=None):
    """Replace Linear layers with LoRA-wrapped versions."""
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'head']

    total_params = 0
    lora_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should be LoRA'd
            should_lora = any(t in name for t in target_modules) or target_modules == ['all']
            if should_lora:
                # Find parent
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr = parts[-1]
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                setattr(parent, attr, lora_layer)
                lora_params += lora_layer.lora_num_params
            total_params += sum(p.numel() for p in module.parameters())

    return model, lora_params


def apply_bitfit(model):
    """Only train bias parameters."""
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training ──

def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('data', train=True, download=True, transform=transform)
    test = datasets.MNIST('data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_variant(name, model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"  [{name}] Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return train_accs, test_accs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data()
    epochs = 5

    results = {}

    # ── 1. Full Fine-Tuning ──
    print("=== Full Fine-Tuning ===")
    model_full = SimpleViT().to(device)
    total_params = count_trainable(model_full)
    train_accs, test_accs = train_variant("Full FT", model_full, train_loader, test_loader, device, epochs)
    results["Full FT"] = {"train": train_accs, "test": test_accs, "params": total_params}
    print(f"  Trainable params: {total_params:,}\n")

    # ── 2. LoRA (r=4) ──
    print("=== LoRA (r=4) ===")
    model_lora = deepcopy(model_full)  # Start from same pretrained weights
    model_lora, lora_params = apply_lora(model_lora, r=4, alpha=8.0, target_modules=['all'])
    model_lora = model_lora.to(device)
    trainable = count_trainable(model_lora)
    train_accs, test_accs = train_variant("LoRA r=4", model_lora, train_loader, test_loader, device, epochs)
    results["LoRA r=4"] = {"train": train_accs, "test": test_accs, "params": trainable}
    print(f"  Trainable params: {trainable:,} ({trainable/total_params*100:.2f}% of full)\n")

    # ── 3. LoRA (r=2) ──
    print("=== LoRA (r=2) ===")
    model_lora2 = deepcopy(model_full)
    model_lora2, lora_params2 = apply_lora(model_lora2, r=2, alpha=4.0, target_modules=['all'])
    model_lora2 = model_lora2.to(device)
    trainable2 = count_trainable(model_lora2)
    train_accs, test_accs = train_variant("LoRA r=2", model_lora2, train_loader, test_loader, device, epochs)
    results["LoRA r=2"] = {"train": train_accs, "test": test_accs, "params": trainable2}
    print(f"  Trainable params: {trainable2:,} ({trainable2/total_params*100:.2f}% of full)\n")

    # ── 4. BitFit ──
    print("=== BitFit ===")
    model_bitfit = deepcopy(model_full)
    model_bitfit, bitfit_params = apply_bitfit(model_bitfit)
    model_bitfit = model_bitfit.to(device)
    trainable_bf = count_trainable(model_bitfit)
    train_accs, test_accs = train_variant("BitFit", model_bitfit, train_loader, test_loader, device, epochs)
    results["BitFit"] = {"train": train_accs, "test": test_accs, "params": trainable_bf}
    print(f"  Trainable params: {trainable_bf:,} ({trainable_bf/total_params*100:.2f}% of full)\n")

    # ── Visualization ──
    results_dir = Path(__file__).parent / "results" / "02-lora"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = {'Full FT': 'blue', 'LoRA r=4': 'green', 'LoRA r=2': 'orange', 'BitFit': 'red'}
    for name, data in results.items():
        ax1.plot(data['train'], label=name, color=colors[name])
        ax2.plot(data['test'], label=name, color=colors[name])

    ax1.set_title("Train Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "lora_comparison.png", dpi=150)
    plt.close()

    # Param comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    params = [results[n]['params'] for n in names]
    final_accs = [results[n]['test'][-1] for n in names]
    x = range(len(names))
    bars = ax.bar(x, params, color=[colors[n] for n in names])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Trainable Parameters")
    ax.set_title(f"Trainable Params vs Final Test Acc\n" +
                 " | ".join(f"{n}: {a:.3f}" for n, a in zip(names, final_accs)))
    ax.set_yscale('log')
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{p:,}',
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(results_dir / "param_comparison.png", dpi=150)
    plt.close()

    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
