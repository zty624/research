"""
Minimal Mixture-of-Experts (MoE) Reproduction
===============================================
Reproduces core ideas from MoE literature:
1. Sparse MoE (1701.06538, Switch Transformer): top-k routing
2. Load balancing loss to prevent expert collapse
3. Expert capacity and routing mechanisms
4. Compare: dense model vs MoE with same FLOPs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Expert ──

class Expert(nn.Module):
    """A single FFN expert."""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ── Top-K Router ──

class TopKRouter(nn.Module):
    """Routes each token to top-k experts."""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        """x: (B, T, D) → returns weights and expert indices."""
        logits = self.gate(x)  # (B, T, n_experts)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices, logits


# ── MoE Layer ──

class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""
    def __init__(self, d_model, n_experts=8, top_k=2, capacity_factor=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])
        self.router = TopKRouter(d_model, n_experts, top_k)

    def forward(self, x):
        """x: (B, T, D)"""
        B, T, D = x.shape
        weights, indices, gate_logits = self.router(x)

        # Initialize output
        output = torch.zeros_like(x)

        # Route tokens to experts
        # Flatten for easier indexing
        x_flat = x.view(-1, D)  # (B*T, D)
        weights_flat = weights.view(-1, self.top_k)  # (B*T, top_k)
        indices_flat = indices.view(-1, self.top_k)  # (B*T, top_k)

        for k in range(self.top_k):
            for expert_idx in range(self.n_experts):
                # Find tokens routed to this expert at position k
                mask = (indices_flat[:, k] == expert_idx)
                if not mask.any():
                    continue

                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = x_flat[token_indices]
                expert_output = self.experts[expert_idx](expert_input)

                # Weighted combination
                output.view(-1, D)[token_indices] += (
                    weights_flat[token_indices, k].unsqueeze(-1) * expert_output
                )

        # Load balancing loss
        balance_loss = self._load_balance_loss(gate_logits)

        return output, balance_loss

    def _load_balance_loss(self, gate_logits):
        """Auxiliary loss to encourage balanced expert utilization.
        L_balance = n_experts * Σ_i (f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = fraction of router probability allocated to expert i
        """
        B, T, _ = gate_logits.shape
        n_tokens = B * T

        # Fraction of tokens routed to each expert (top-1)
        top1_indices = gate_logits.argmax(dim=-1)  # (B, T)
        f = torch.zeros(self.n_experts, device=gate_logits.device)
        for i in range(self.n_experts):
            f[i] = (top1_indices == i).float().mean()

        # Fraction of router probability
        P = F.softmax(gate_logits, dim=-1).mean(dim=(0, 1))  # (n_experts,)

        balance_loss = self.n_experts * (f * P).sum()
        return balance_loss


# ── Full Models ──

class DenseTransformerBlock(nn.Module):
    """Standard dense transformer block."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, torch.tensor(0.0, device=x.device)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN."""
    def __init__(self, d_model, n_heads, n_experts=8, top_k=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, n_experts, top_k)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        moe_out, balance_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, balance_loss


class DenseModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DenseTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        total_bl = 0
        for block in self.blocks:
            h, bl = block(h)
            total_bl += bl
        return self.head(self.norm(h)), total_bl


class MoEModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 n_experts=8, top_k=2, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, n_experts, top_k) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.n_experts = n_experts

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        total_bl = 0
        for block in self.blocks:
            h, bl = block(h)
            total_bl += bl
        return self.head(self.norm(h)), total_bl


# ── Training ──

def train_model(model, train_loader, n_epochs=5, lr=1e-3, balance_weight=0.01,
                preprocess_fn=None, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    balance_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_bl = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            if preprocess_fn is not None:
                bx, by = preprocess_fn(bx, by)
            logits, bl = model(bx)
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), by.reshape(-1))
            loss = ce_loss + balance_weight * bl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce_loss.item()
            epoch_bl += bl.item()

        losses.append(epoch_loss / len(train_loader))
        balance_losses.append(epoch_bl / len(train_loader))

    return losses, balance_losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_expert_utilization(model, test_loader, preprocess_fn, device='cpu'):
    """Measure how evenly experts are utilized."""
    model.eval()
    expert_counts = {}

    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            expert_counts[name] = torch.zeros(module.n_experts, device=device)

    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            if preprocess_fn is not None:
                bx, _ = preprocess_fn(bx, _)
            h = model.emb(bx) + model.pos_emb(torch.arange(bx.shape[1], device=device).unsqueeze(0))
            for block in model.blocks:
                # Run attention part manually
                h_norm = block.norm1(h)
                attn_out, _ = block.attn(h_norm, h_norm, h_norm)
                h = h + attn_out
                # Now h is what MoE sees after norm2
                moe_input = block.norm2(h)
                for name, module in model.named_modules():
                    if isinstance(module, MoELayer) and module is block.moe:
                        _, _, gate_logits = module.router(moe_input)
                        top1 = gate_logits.argmax(dim=-1)
                        for i in range(module.n_experts):
                            expert_counts[name][i] += (top1 == i).sum().item()
                # Complete the block forward
                moe_out, _ = block.moe(moe_input)
                h = h + moe_out
            # Only need one batch
            break

    return expert_counts


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "21-moe"
    results_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use subset for speed
    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    # Tokenize: flatten image to sequence of 4x4 patches
    vocab_size = 16  # 4-bit pixel values
    seq_len = 49     # 7x7 patches from 28x28

    def preprocess(batch_x, batch_y):
        B = batch_x.shape[0]
        # Downsample to 7x7 and quantize to 16 levels
        x_small = F.avg_pool2d(batch_x, 4)  # (B, 1, 7, 7)
        x_tokens = (x_small * (vocab_size - 1)).long().clamp(0, vocab_size-1)
        x_tokens = x_tokens.view(B, -1)  # (B, 49)
        # Next-token prediction: input is tokens[:-1], target is tokens[1:]
        return x_tokens[:, :-1], x_tokens[:, 1:]

    d_model = 64
    n_heads = 4
    n_layers = 3

    # Dense model
    print("=== Training Dense Model ===")
    dense = DenseModel(vocab_size, d_model, n_heads, n_layers).to(device)
    print(f"  Parameters: {count_parameters(dense):,}")
    dense_losses, _ = train_model(dense, train_loader, n_epochs=8, device=device,
                                   preprocess_fn=preprocess)

    # MoE model (8 experts, top-2)
    print("\n=== Training MoE Model (8 experts, top-2) ===")
    moe = MoEModel(vocab_size, d_model, n_heads, n_layers, n_experts=8, top_k=2).to(device)
    print(f"  Parameters: {count_parameters(moe):,}")
    moe_losses, moe_bl = train_model(moe, train_loader, n_epochs=8, device=device,
                                      preprocess_fn=preprocess)

    # MoE with more experts
    print("\n=== Training MoE Model (16 experts, top-2) ===")
    moe16 = MoEModel(vocab_size, d_model, n_heads, n_layers, n_experts=16, top_k=2).to(device)
    print(f"  Parameters: {count_parameters(moe16):,}")
    moe16_losses, moe16_bl = train_model(moe16, train_loader, n_epochs=8, device=device,
                                          preprocess_fn=preprocess)

    # Get expert utilization
    print("\n=== Expert Utilization ===")
    util_8 = get_expert_utilization(moe, test_loader, preprocess, device)
    util_16 = get_expert_utilization(moe16, test_loader, preprocess, device)

    for name, counts in util_8.items():
        total = counts.sum().item()
        fracs = counts / total if total > 0 else counts
        print(f"  {name} (8 experts): {fracs.cpu().numpy().round(3)}")

    # ── Visualization ──

    # 1. Training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(dense_losses, label='Dense', color='red')
    axes[0].plot(moe_losses, label='MoE-8', color='blue')
    axes[0].plot(moe16_losses, label='MoE-16', color='green')
    axes[0].set_title("Training Loss: Dense vs MoE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Balance loss
    axes[1].plot(moe_bl, label='MoE-8', color='blue')
    axes[1].plot(moe16_bl, label='MoE-16', color='green')
    axes[1].set_title("Load Balancing Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balance Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Mixture of Experts: Dense vs Sparse MoE", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "training_comparison.png", dpi=150)
    plt.close()

    # 3. Expert utilization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, (model_obj, util_dict)) in enumerate([
        ('8 Experts', (moe, util_8)),
        ('16 Experts', (moe16, util_16))
    ]):
        ax = axes[idx]
        for layer_name, counts in util_dict.items():
            total = counts.sum().item()
            fracs = (counts / total).cpu().numpy() if total > 0 else counts.cpu().numpy()
            ax.bar(range(len(fracs)), fracs, alpha=0.7, label=layer_name.replace('blocks.', 'L'))

        ax.axhline(y=1.0/len(fracs), color='red', linestyle='--', alpha=0.5, label='Uniform')
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Fraction of Tokens")
        ax.set_title(f"Expert Utilization ({name})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "expert_utilization.png", dpi=150)
    plt.close()

    # 4. Parameter comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Dense', 'MoE-8\ntop-2', 'MoE-16\ntop-2']
    params = [count_parameters(dense), count_parameters(moe), count_parameters(moe16)]
    colors = ['red', 'blue', 'green']
    ax.bar(names, [p/1e6 for p in params], color=colors, alpha=0.7)
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Size: Dense vs MoE")
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(params):
        ax.text(i, v/1e6 + 0.01, f'{v/1e6:.2f}M', ha='center')
    plt.tight_layout()
    plt.savefig(results_dir / "parameter_comparison.png", dpi=150)
    plt.close()

    # 5. MoE routing visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    texts = [
        ("Dense Model", "Every token uses\nall parameters\n→ O(N²d) FLOPs", 0.17, 'red'),
        ("Sparse MoE", "Each token routes to\nonly top-k experts\n→ Same FLOPs, more params", 0.5, 'blue'),
        ("Key Benefit", "Scale params 10-100×\nwithout increasing\ncompute per token", 0.83, 'green'),
    ]

    for name, desc, x_pos, color in texts:
        ax.text(x_pos, 0.7, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x_pos, 0.35, desc, fontsize=11, ha='center', va='center',
                fontfamily='monospace', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_title("Mixture of Experts: Scaling Parameters Without Scaling Compute", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "moe_concept.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
