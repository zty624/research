# VAE Experiments Reproduction Report

> Reproduced on 2026-04-22 | NVIDIA RTX 4060 Laptop (8GB VRAM) | PyTorch 2.11.0+cu128 | uv managed

---

## 1. Vanilla VAE — MNIST

**Paper**: Auto-Encoding Variational Bayes (Kingma & Welling, 2013) [1312.6114]

**Architecture**: FC 784→400→20→400→784, latent_dim=20

**Training**: Adam lr=1e-3, 20 epochs, batch_size=128

| Metric | Value |
|--------|-------|
| Train Loss | 166→104 |
| Val Loss | 128→104 |

**Outputs**: `results/01-vanilla-vae/`
- Reconstructions (original vs reconstructed)
- 2D latent space (PCA & t-SNE, colored by digit label)
- Prior samples (64 digits from N(0,I))
- Loss curve

**Findings**: ELBO converges normally. Latent space shows clear clustering by digit class. Samples from prior are recognizable digits, though slightly blurry — characteristic of VAE's mode-averaging behavior.

---

## 2. β-VAE — MNIST

**Paper**: β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (Higgins et al., 2017) [1804.03599]

**Architecture**: FC 784→512→256→10→256→512→784, latent_dim=10

**Training**: Adam lr=1e-3, 30 epochs per β, sweep β∈{1,2,4,10}

| β | Train Loss | Val Loss | Reconstruction Quality | Disentanglement |
|---|-----------|----------|----------------------|----------------|
| 1 | 104.0 | 104.1 | Best | Lowest |
| 2 | 116.9 | 118.2 | Good | Moderate |
| 4 | 141.8 | 142.6 | Degraded | Good |
| 10 | 180.1 | 180.0 | Poor | Highest |

**Outputs**: `results/02-beta-vae/`
- Per-β: reconstructions, latent traversal (10 dims × 11 values), loss curve

**Findings**: Higher β increases KL weight → worse reconstruction but more disentangled representations. Latent traversal shows: at β=1, each dimension controls mixed attributes; at β=10, individual dimensions control more independent factors (stroke width, rotation, etc.). This matches the original paper's core claim about the disentanglement-accuracy tradeoff.

---

## 3. VQ-VAE — MNIST + CIFAR-10

**Paper**: Neural Discrete Representation Learning (van den Oord et al., 2017) [1711.00937]

**Architecture**: Conv encoder (2 strided convs) → VectorQuantizer (K=512, dim=64, EMA update) → Conv decoder

**Training**: Adam lr=2e-4, commitment_cost=0.25

| Dataset | Epochs | Final Loss | Codebook Perplexity | Utilization |
|---------|--------|-----------|--------------------:|-------------|
| MNIST | 30 | 0.003 | 24/512 | 4.7% |
| CIFAR-10 | 50 | 0.005 | 411/512 | 80.3% |

**Outputs**: `results/03-vq-vae/`
- Per-dataset: reconstructions at epochs 1/10/20/30..., codebook perplexity curve, loss curve

**Findings**: MNIST's simple structure leads to low codebook utilization (only ~24 codes active out of 512). CIFAR-10's complexity drives utilization to 80%. EMA codebook updates remain stable throughout training. The commitment loss ensures encoder outputs stay close to their codebook vectors, while straight-through estimator enables gradient flow through the quantization bottleneck.

---

## 4. IWAE — MNIST

**Paper**: Importance Weighted Autoencoders (Burda et al., 2015) [1509.00519]

**Architecture**: Same as Vanilla VAE (FC 784→400→20→400→784), but samples K z's per forward pass

**Training**: Adam lr=1e-3, 20 epochs per K, sweep K∈{1,5,10,50}

**Evaluation**: Test log-likelihood computed with IWAE-50 bound (K=50 at eval time)

| K (training) | Test LL (IWAE-50) |
|:---:|:---:|
| 1 | -99.56 |
| 5 | -99.29 |
| 10 | **-98.87** |
| 50 | -99.02 |

**Outputs**: `results/04-iwae/`
- Per-K: reconstructions, prior samples, loss curve
- `ll_vs_k.png`: log-likelihood vs K plot

**Findings**: K=1 (vanilla VAE) gives the loosest bound. Increasing K tightens the IWAE bound, with K=10 yielding the best test log-likelihood (-98.87 nats). K=50 slightly underperforms, likely due to higher gradient variance during training with many importance samples. This aligns with the original paper: moderate K values offer the best tradeoff between bound tightness and optimization stability.

---

## 5. NVAE — CIFAR-10

**Paper**: NVAE: A Deep Hierarchical Variational Autoencoder (Vahdat & Kautz, 2020) [2007.03898]

**Architecture**: Simplified 2-level hierarchical VAE (adapted for 8GB VRAM):
- Encoder: Conv 32×32 → 8×8 + ResidualBlocks + GroupNorm + Swish
- Level 1: z1 (latent_dim=16, 8×8 spatial)
- Level 2: z2 (latent_dim=16, global, conditioned on z1 pooled)
- Decoder: concat(z1, z2_expanded) → ConvTranspose 8×8 → 32×32

**Parameters**: 1,290,179

**Training**: Adam lr=2e-4 + CosineAnnealing, batch_size=16, gradient clipping 1.0

| Epoch | Loss | Recon | KLD1 (spatial) | KLD2 (global) | LR |
|-------|------|-------|:---:|:---:|------|
| 1 | 103.65 | 76.30 | 26.83 | 0.52 | 2e-4 |
| 10 | 81.69 | 57.52 | 24.13 | 0.03 | 2e-4 |
| 30 | 78.92 | 55.80 | 23.09 | 0.03 | 1.6e-4 |
| 50 | 77.49 | 54.61 | 22.87 | 0.01 | 1e-4 |
| 70 | 75.54 | 52.69 | 22.84 | 0.00 | 4.1e-5 |

**Outputs**: `results/05-nvae/`
- Reconstructions at every 10 epochs
- Prior samples
- Loss curve

**Findings**: The hierarchical structure shows a clear asymmetry: z1 (spatial, 8×8) carries almost all information (KLD≈23), while z2 (global) quickly collapses to the prior (KLD→0). This means the simplified 2-level architecture doesn't effectively leverage the top-level latent — the spatial z1 is sufficient for CIFAR-10 reconstruction. The full NVAE paper uses 30+ residual groups with depthwise convolutions, which creates a much richer hierarchical decomposition. Nevertheless, reconstructions remain recognizable, and the model fits comfortably in 8GB VRAM.

---

## How to Run

```bash
cd vae-experiments

# Individual experiments
uv run experiments/01_vanilla_vae.py   # ~2 min
uv run experiments/02_beta_vae.py      # ~10 min
uv run experiments/03_vq_vae.py        # ~20 min
uv run experiments/04_iwae.py          # ~15 min
uv run experiments/05_nvae.py          # ~60 min

# All at once
./run_all.sh
```

All data (MNIST, CIFAR-10) downloads automatically to `data/`. Results saved to `results/`.
