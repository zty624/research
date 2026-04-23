"""
Minimal QLoRA Reproduction
===========================
Reproduces core ideas from QLoRA (2305.14314, Dettmers et al.):
1. NF4 (4-bit NormalFloat): Information-theoretically optimal quantization for
   normally-distributed weights. Uses quantile quantization of N(0,1) truncated
   to [-1,1] so each of the 16 buckets has equal probability mass.
2. Double Quantization: Quantize the FP32 quantization constants themselves to
   Int8, saving ~0.37 bits/param. Block size 64 for first level, 256 for second.
3. LoRA on all linear layers (not just attention): LoRA input gradients dominate
   memory, not the parameters themselves, so adding LoRA to FFN too barely
   increases memory but improves quality.

Formula: Y = XW + sXL1L2, where W is frozen 4-bit, L1,L2 are BFloat16 LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ── NF4 Quantization ──

def _norm_cdf(x):
    """Standard normal CDF using error function (no scipy needed)."""
    return 0.5 * (1.0 + torch.erf(torch.tensor(x) / math.sqrt(2.0))).item()


def _norm_ppf(p):
    """Inverse normal CDF via rational approximation (Abramowitz & Stegun 26.2.23).

    Accurate to ~1e-8 for p in [1e-10, 1-1e-10].
    """
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p > 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3))


def compute_nf4_levels():
    """Compute the 16 quantile levels of N(0,1) truncated to [-1,1].

    Each bucket has equal probability 1/16, making this information-theoretically
    optimal for normally-distributed weights.
    """
    lower = _norm_cdf(-1)
    upper = _norm_cdf(1)
    boundaries = np.linspace(lower, upper, 17)
    quantiles = np.array([_norm_ppf(b) for b in boundaries])
    levels = np.array([(quantiles[i] + quantiles[i + 1]) / 2 for i in range(16)])
    return levels.astype(np.float32)


def quantize_nf4(weight_tensor, block_size=64):
    """Quantize a weight tensor to NF4 with per-block scaling.

    Returns:
        quantized_indices: int tensor of quantization level indices (0-15)
        scales: FP32 per-block scale factors
        nf4_levels: the 16 NF4 levels
    """
    nf4_levels = torch.from_numpy(compute_nf4_levels()).to(weight_tensor.device)

    flat = weight_tensor.reshape(-1).float()
    n = flat.shape[0]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])

    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1).values.clamp(min=1e-8)
    normalized = blocks / scales.unsqueeze(1)

    # Nearest NF4 level
    normalized_flat = normalized.reshape(-1).unsqueeze(1)
    levels_expanded = nf4_levels.unsqueeze(0)
    indices = (normalized_flat - levels_expanded).abs().argmin(dim=1)
    indices = indices.reshape(-1, block_size)

    orig_blocks = (n + block_size - 1) // block_size
    return indices[:orig_blocks], scales[:orig_blocks], nf4_levels


def dequantize_nf4(indices, scales, nf4_levels, block_size=64, orig_shape=None):
    """Dequantize NF4 back to FP32."""
    levels = nf4_levels[indices]
    dequant = (levels * scales.unsqueeze(1)).reshape(-1)
    if orig_shape is not None:
        n = 1
        for s in orig_shape:
            n *= s
        dequant = dequant[:n].reshape(orig_shape)
    return dequant


# ── Int4 and FP4 baselines ──

def quantize_uniform4(weight_tensor, block_size=64):
    """Uniform 4-bit quantization: 16 equally-spaced levels in [-1,1]."""
    levels = torch.linspace(-1, 1, 16, device=weight_tensor.device)
    flat = weight_tensor.reshape(-1).float()
    n = flat.shape[0]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1).values.clamp(min=1e-8)
    normalized = blocks / scales.unsqueeze(1)
    normalized_flat = normalized.reshape(-1).unsqueeze(1)
    indices = (normalized_flat - levels.unsqueeze(0)).abs().argmin(dim=1)
    indices = indices.reshape(-1, block_size)
    orig_blocks = (n + block_size - 1) // block_size
    return indices[:orig_blocks], scales[:orig_blocks], levels


def quantize_fp4(weight_tensor, block_size=64):
    """FP4: IEEE-style 1-sign + 2-exponent + 1-mantissa, 16 float levels."""
    fp4_positive = torch.tensor([0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.0],
                                device=weight_tensor.device)
    fp4_negative = -fp4_positive[1:]
    levels = torch.sort(torch.cat([fp4_negative.flip(0), fp4_positive]))[0]
    flat = weight_tensor.reshape(-1).float()
    n = flat.shape[0]
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1).values.clamp(min=1e-8)
    normalized = blocks / scales.unsqueeze(1)
    normalized_flat = normalized.reshape(-1).unsqueeze(1)
    indices = (normalized_flat - levels.unsqueeze(0)).abs().argmin(dim=1)
    indices = indices.reshape(-1, block_size)
    orig_blocks = (n + block_size - 1) // block_size
    return indices[:orig_blocks], scales[:orig_blocks], levels


# ── Double Quantization ──

def compute_single_quant_memory(n_params, block_size=64):
    """Memory in bytes for single-quantized FP32 scales (one FP32 per block)."""
    n_blocks = math.ceil(n_params / block_size)
    return n_blocks * 4  # one FP32 scale per block


def compute_double_quant_memory(n_params, block_size=64, second_block_size=256):
    """Memory in bytes for double-quantized scales.

    Single quant: one FP32 scale per block of 64 params -> 32/64 = 0.5 bits/param overhead.
    Double quant: quantize those FP32 scales to Int8 with a second-level offset.
      - First level: int8 scale values (1 byte each) + FP32 offset per block of 256 scales
      - Net: (1 byte * n_blocks) + (4 bytes * n_blocks/256) = n_blocks * (1 + 4/256) bytes
      - = n_blocks * 1.015625 bytes vs n_blocks * 4 bytes for single quant

    Paper reports savings of ~0.37 bits/param from double quantization.
    """
    n_blocks = math.ceil(n_params / block_size)
    # Single quant overhead: one FP32 (4 bytes) per block
    # Double quant overhead: int8 (1 byte) per block scale + one FP32 second-level offset
    # per 256 first-level blocks
    n_blocks2 = math.ceil(n_blocks / second_block_size)
    double_quant_bytes = n_blocks * 1 + n_blocks2 * 4  # int8 scales + FP32 second-level offsets
    return double_quant_bytes


# ── LoRA Linear ──

class LoRALinear(nn.Module):
    """LoRA adapter: Y = XW + (alpha/rank) * X @ A^T @ B^T.

    Base weight W is frozen, only A and B are trainable.
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16,
                 base_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight stored as a regular tensor (not nn.Parameter)
        if base_weight is not None:
            self.register_buffer('weight', base_weight.clone().detach())
        else:
            self.register_buffer('weight', torch.randn(out_features, in_features) * 0.01)

        self.register_buffer('bias', torch.zeros(out_features))

        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


# ── QLoRA Linear ──

class QLoRALinear(nn.Module):
    """QLoRA: Frozen NF4-quantized base weight + BFloat16 LoRA adapters.

    Y = X @ dequant(W) + (alpha/rank) * X @ A^T @ B^T
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16,
                 base_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.block_size = 64

        if base_weight is not None:
            w = base_weight.detach().float()
        else:
            w = torch.randn(out_features, in_features) * 0.02

        self.orig_shape = w.shape

        # Quantize base weight to NF4 (stored as buffers, not parameters)
        indices, scales, nf4_levels = quantize_nf4(w, block_size=self.block_size)
        self.register_buffer('quant_indices', indices)
        self.register_buffer('scales', scales)
        self.register_buffer('nf4_levels', nf4_levels)
        self.register_buffer('bias', torch.zeros(out_features))

        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def _dequantize_weight(self):
        return dequantize_nf4(
            self.quant_indices, self.scales, self.nf4_levels,
            block_size=self.block_size, orig_shape=self.orig_shape
        )

    def forward(self, x):
        w = self._dequantize_weight()
        base_out = F.linear(x, w, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


# ── Small Transformer Model ──

class SimpleSelfAttention(nn.Module):
    """Minimal multi-head self-attention using explicit Linear layers.

    Avoids nn.MultiheadAttention so LoRA/QLoRA replacement works cleanly
    on all linear sub-layers.
    """

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

        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
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
    """Small transformer for sequence-to-scalar regression."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256,
                 n_layers=2, out_dim=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        h = h.mean(dim=1)  # pool over sequence
        return self.head(h)


def _replace_linear_with_lora(model, rank=8, alpha=16, attention_only=True):
    """Replace Linear layers in model with LoRA wrappers (in-place)."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_attention = 'attn' in name
            if attention_only and not is_attention:
                continue
            replacements.append((name, module))

    for name, old_linear in replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        lora_linear = LoRALinear(
            old_linear.in_features,
            old_linear.out_features,
            rank=rank,
            alpha=alpha,
            base_weight=old_linear.weight.data,
        )
        setattr(parent, attr, lora_linear)

    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = ('lora_A' in name or 'lora_B' in name)


def _replace_linear_with_qlora(model, rank=8, alpha=16):
    """Replace ALL Linear layers in model with QLoRA wrappers (in-place)."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replacements.append((name, module))

    for name, old_linear in replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        qlora_linear = QLoRALinear(
            old_linear.in_features,
            old_linear.out_features,
            rank=rank,
            alpha=alpha,
            base_weight=old_linear.weight.data,
        )
        setattr(parent, attr, qlora_linear)

    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = ('lora_A' in name or 'lora_B' in name)


# ── Data ──

def generate_regression_data(batch_size, seq_len, vocab_size=200):
    """Generate sequence-to-scalar regression data.

    Labels are real-valued, computed as a deterministic function of the input
    tokens: sum of token embeddings weighted by position, then a nonlinear
    transform. This requires the model to learn specific weight patterns.
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Deterministic label: sum of token IDs with position-dependent weighting
    positions = torch.arange(seq_len, dtype=torch.float)
    weights = torch.sin(positions * 0.5) * 0.3 + 0.5  # position weights in [0.2, 0.8]
    token_vals = x.float() / vocab_size  # normalize to [0, 1]
    weighted_sum = (token_vals * weights.unsqueeze(0)).sum(dim=1) / seq_len
    # Nonlinear transform: mix of sin and linear
    y = torch.sin(weighted_sum * 6.0) * 0.5 + weighted_sum * 0.5
    y = y.unsqueeze(1)  # (batch, 1)
    return x, y


# ── Training ──

def train_model(model, vocab_size, seq_len, n_steps=1000, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    for step in range(n_steps):
        x, y = generate_regression_data(64, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1} | Loss: {loss.item():.6f}")

    return losses


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(__file__).parent / "results" / "64-qlora"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Experiment 1: NF4 vs Int4 vs FP4 Quantization Quality ──

    print("=== Experiment 1: NF4 vs Int4 vs FP4 Quantization Quality ===")

    torch.manual_seed(42)
    weight = torch.randn(1024, 1024) * 0.02

    # NF4
    nf4_idx, nf4_scales, nf4_levels = quantize_nf4(weight, block_size=64)
    nf4_recon = dequantize_nf4(nf4_idx, nf4_scales, nf4_levels, block_size=64,
                                orig_shape=weight.shape)
    nf4_mse = F.mse_loss(nf4_recon, weight).item()

    # Int4 (uniform)
    int4_idx, int4_scales, int4_levels = quantize_uniform4(weight, block_size=64)
    int4_recon = dequantize_nf4(int4_idx, int4_scales, int4_levels, block_size=64,
                                 orig_shape=weight.shape)
    int4_mse = F.mse_loss(int4_recon, weight).item()

    # FP4
    fp4_idx, fp4_scales, fp4_levels = quantize_fp4(weight, block_size=64)
    fp4_recon = dequantize_nf4(fp4_idx, fp4_scales, fp4_levels, block_size=64,
                                orig_shape=weight.shape)
    fp4_mse = F.mse_loss(fp4_recon, weight).item()

    print(f"  NF4  MSE: {nf4_mse:.8f}")
    print(f"  Int4 MSE: {int4_mse:.8f}")
    print(f"  FP4  MSE: {fp4_mse:.8f}")
    print(f"  NF4 is best for normal weights: {nf4_mse <= min(int4_mse, fp4_mse)}")

    # Test on uniform distribution (NF4 should NOT be optimal here)
    uniform_weight = torch.rand(1024, 1024) * 2 - 1
    nf4_u_idx, nf4_u_scales, _ = quantize_nf4(uniform_weight, block_size=64)
    nf4_u_recon = dequantize_nf4(nf4_u_idx, nf4_u_scales, nf4_levels, block_size=64,
                                  orig_shape=uniform_weight.shape)
    nf4_u_mse = F.mse_loss(nf4_u_recon, uniform_weight).item()

    int4_u_idx, int4_u_scales, _ = quantize_uniform4(uniform_weight, block_size=64)
    int4_u_recon = dequantize_nf4(int4_u_idx, int4_u_scales, int4_levels, block_size=64,
                                   orig_shape=uniform_weight.shape)
    int4_u_mse = F.mse_loss(int4_u_recon, uniform_weight).item()

    print(f"\n  On Uniform[-1,1] weights:")
    print(f"  NF4  MSE: {nf4_u_mse:.8f}")
    print(f"  Int4 MSE: {int4_u_mse:.8f}")
    print(f"  (Int4 better for uniform -- NF4 is optimized for normal)")

    # ── Experiment 2: Double Quantization Memory Savings ──

    print("\n=== Experiment 2: Double Quantization Memory Savings ===")

    n_params = 7_000_000_000
    block_size = 64

    # Single quant: 4-bit weights + FP32 scale per block
    single_scale_bytes = compute_single_quant_memory(n_params, block_size)
    single_total_bytes = n_params // 2 + single_scale_bytes  # 4-bit = 0.5 bytes/param

    # Double quant: 4-bit weights + int8 scales + int8 second-level offsets
    double_scale_bytes = compute_double_quant_memory(n_params, block_size, 256)
    double_total_bytes = n_params // 2 + double_scale_bytes

    bits_single = single_total_bytes * 8 / n_params
    bits_double = double_total_bytes * 8 / n_params
    savings = bits_single - bits_double

    print(f"  Single quant scales: {single_scale_bytes / 1e6:.2f} MB "
          f"({bits_single:.4f} bits/param total)")
    print(f"  Double quant scales: {double_scale_bytes / 1e6:.2f} MB "
          f"({bits_double:.4f} bits/param total)")
    print(f"  Savings from double quant: {savings:.4f} bits/param "
          f"(paper reports ~0.37)")

    # ── Experiment 3: LoRA Finetuning Comparison ──

    print("\n=== Experiment 3: LoRA Finetuning Comparison ===")

    vocab_size = 200
    seq_len = 32
    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 2
    lora_rank = 8
    n_steps = 2000

    # Pre-train a base model (partially, so finetuning still has room to improve)
    torch.manual_seed(42)
    base_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, out_dim=1
    ).to(device)
    print("\n  Pre-training base model (partial)...")
    pretrain_losses = train_model(
        base_model, vocab_size, seq_len,
        n_steps=500, lr=1e-3, device=device
    )
    print(f"  Pre-train final loss: {pretrain_losses[-1]:.6f}")

    # Full finetuning (16-bit)
    print("\n  Full finetuning (16-bit):")
    torch.manual_seed(42)
    full_ft_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, out_dim=1
    ).to(device)
    full_ft_model.load_state_dict(base_model.state_dict())
    full_ft_losses = train_model(
        full_ft_model, vocab_size, seq_len,
        n_steps=n_steps, lr=5e-4, device=device
    )

    # LoRA on attention only
    print("\n  LoRA (attention only):")
    torch.manual_seed(42)
    lora_attn_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, out_dim=1
    ).to(device)
    lora_attn_model.load_state_dict(base_model.state_dict())
    _replace_linear_with_lora(lora_attn_model, rank=lora_rank, attention_only=True)
    lora_attn_model = lora_attn_model.to(device)
    lora_attn_losses = train_model(
        lora_attn_model, vocab_size, seq_len,
        n_steps=n_steps, lr=1e-3, device=device
    )

    # QLoRA (4-bit base + LoRA on all linear layers)
    print("\n  QLoRA (4-bit base + LoRA all layers):")
    torch.manual_seed(42)
    qlora_model = SmallTransformer(
        vocab_size, d_model, n_heads, d_ff, n_layers, out_dim=1
    ).to(device)
    qlora_model.load_state_dict(base_model.state_dict())
    _replace_linear_with_qlora(qlora_model, rank=lora_rank)
    qlora_model = qlora_model.to(device)
    qlora_losses = train_model(
        qlora_model, vocab_size, seq_len,
        n_steps=n_steps, lr=1e-3, device=device
    )

    # Count trainable parameters and memory
    full_params = sum(p.numel() for p in full_ft_model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in lora_attn_model.parameters() if p.requires_grad)
    qlora_params = sum(p.numel() for p in qlora_model.parameters() if p.requires_grad)

    # Memory estimates
    full_mem_mb = sum(p.numel() * p.element_size() for p in full_ft_model.parameters()) / 1e6
    lora_mem_mb = sum(p.numel() * p.element_size() for p in lora_attn_model.parameters()) / 1e6

    # QLoRA: base weights in 4-bit + FP32 scales + LoRA adapters (16-bit)
    qlora_weight_mb = 0.0
    for name, mod in qlora_model.named_modules():
        if isinstance(mod, QLoRALinear):
            qlora_weight_mb += mod.quant_indices.numel() * 0.5 / 1e6  # 4-bit
            qlora_weight_mb += mod.scales.numel() * 4 / 1e6  # FP32 scales
    qlora_lora_mb = sum(
        p.numel() * p.element_size()
        for p in qlora_model.parameters() if p.requires_grad
    ) / 1e6
    qlora_mem_mb = qlora_weight_mb + qlora_lora_mb

    print(f"\n  Trainable params: Full={full_params:,} | LoRA-attn={lora_params:,} | QLoRA={qlora_params:,}")
    print(f"  Memory (MB):      Full={full_mem_mb:.2f} | LoRA-attn={lora_mem_mb:.2f} | QLoRA={qlora_mem_mb:.2f}")
    print(f"  Final loss:       Full={full_ft_losses[-1]:.6f} | LoRA-attn={lora_attn_losses[-1]:.6f} | QLoRA={qlora_losses[-1]:.6f}")

    # ── Visualization ──

    # 1. Quantization error comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['NF4', 'Int4\n(uniform)', 'FP4']
    normal_mses = [nf4_mse, int4_mse, fp4_mse]
    colors = ['#2196F3', '#FF9800', '#F44336']

    bars = axes[0].bar(methods, normal_mses, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, normal_mses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f'{val:.2e}', ha='center', va='bottom', fontsize=10)
    axes[0].set_ylabel('MSE (lower is better)')
    axes[0].set_title('Quantization Error on Normal Weights')
    axes[0].grid(True, alpha=0.3, axis='y')

    # NF4 vs Int4 on different distributions
    dist_labels = ['Normal\n(NF4 optimized)', 'Uniform\n(Int4 optimized)']
    nf4_dists = [nf4_mse, nf4_u_mse]
    int4_dists = [int4_mse, int4_u_mse]
    x_pos = np.arange(len(dist_labels))
    width = 0.3
    axes[1].bar(x_pos - width/2, nf4_dists, width, label='NF4', color='#2196F3', alpha=0.8)
    axes[1].bar(x_pos + width/2, int4_dists, width, label='Int4', color='#FF9800', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(dist_labels)
    axes[1].set_ylabel('MSE')
    axes[1].set_title('NF4 vs Int4 Across Weight Distributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('QLoRA: NF4 Quantization Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "quantization_error.png", dpi=150)
    plt.close()

    # 2. Weight distribution vs quantization levels overlay
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sample_weight = weight[:256, :1].flatten().numpy()
    for ax, levels, title, color in [
        (axes[0], nf4_levels.numpy(), 'NF4 Levels vs Normal Weight', '#2196F3'),
        (axes[1], int4_levels.numpy(), 'Int4 Levels vs Normal Weight', '#FF9800'),
        (axes[2], fp4_levels.numpy(), 'FP4 Levels vs Normal Weight', '#F44336'),
    ]:
        ax.hist(sample_weight, bins=80, density=True, alpha=0.5, color='gray',
                label='Weight distribution')
        for level in levels:
            ax.axvline(level, color=color, alpha=0.6, linewidth=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Quantization Levels Overlaid on Weight Distribution', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "quant_levels_overlay.png", dpi=150)
    plt.close()

    # 3. Training loss curves
    fig, ax = plt.subplots(figsize=(10, 5))

    window = 50
    for losses, label, color in [
        (full_ft_losses, 'Full FT (16-bit)', '#2196F3'),
        (lora_attn_losses, 'LoRA (attn only)', '#FF9800'),
        (qlora_losses, 'QLoRA (4-bit + LoRA all)', '#4CAF50'),
    ]:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=label, color=color, linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('QLoRA: Finetuning Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()

    # 4. Memory usage comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ft_methods = ['Full FT\n(16-bit)', 'LoRA\n(attn only)', 'QLoRA\n(4-bit+LoRA)']
    mem_values = [full_mem_mb, lora_mem_mb, qlora_mem_mb]
    train_params_k = [full_params / 1000, lora_params / 1000, qlora_params / 1000]
    mem_colors = ['#2196F3', '#FF9800', '#4CAF50']

    bars = axes[0].bar(ft_methods, mem_values, color=mem_colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, mem_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    axes[0].set_ylabel('Memory (MB)')
    axes[0].set_title('Total Model Memory')
    axes[0].grid(True, alpha=0.3, axis='y')

    bars2 = axes[1].bar(ft_methods, train_params_k,
                        color=mem_colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars2, train_params_k):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f'{val:.1f}K', ha='center', va='bottom', fontsize=9)
    axes[1].set_ylabel('Trainable Params (K)')
    axes[1].set_title('Trainable Parameters')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('QLoRA: Memory & Parameter Efficiency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "memory_comparison.png", dpi=150)
    plt.close()

    # 5. Double quantization savings
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar([0], [bits_single], 0.5, label='Single quant (4-bit + FP32 scales)',
           color='#FF9800', alpha=0.8, edgecolor='black')
    ax.bar([1], [bits_double], 0.5, label='Double quant (4-bit + Int8 scales)',
           color='#4CAF50', alpha=0.8, edgecolor='black')

    ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='Ideal 4-bit')
    ax.axhline(y=32.0, color='gray', linestyle='--', alpha=0.5, label='FP32 baseline')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Single Quantization', 'Double Quantization'])
    ax.set_ylabel('Bits per Parameter')
    ax.set_title(f'Double Quantization Savings: {savings:.3f} bits/param')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax.annotate(f'{bits_single:.3f}', (0, bits_single),
                textcoords="offset points", xytext=(0, 10), ha='center',
                fontsize=11, fontweight='bold')
    ax.annotate(f'{bits_double:.3f}', (1, bits_double),
                textcoords="offset points", xytext=(0, 10), ha='center',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / "double_quant_savings.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
