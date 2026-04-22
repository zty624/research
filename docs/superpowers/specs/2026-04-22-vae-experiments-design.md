# VAE Experiments Reproduction — Design Spec

## Goal

Reproduce 5 classic VAE experiments on a local RTX 4060 Laptop (8GB VRAM), managed by uv, all data contained within the project directory.

## Project Structure

```
/mnt/data/Arch/workspace/research/vae-experiments/
├── pyproject.toml
├── data/                       # All datasets auto-downloaded here
├── results/                    # Training outputs
│   ├── 01-vanilla-vae/
│   ├── 02-beta-vae/
│   ├── 03-vq-vae/
│   ├── 04-iwae/
│   └── 05-nvae/
│       ├── checkpoints/
│       ├── samples/
│       └── metrics/
├── experiments/
│   ├── 01_vanilla_vae.py
│   ├── 02_beta_vae.py
│   ├── 03_vq_vae.py
│   ├── 04_iwae.py
│   └── 05_nvae.py
└── common/
    ├── data.py                 # Unified data loading
    └── viz.py                  # Visualization tools
```

## Environment

- uv for dependency management
- Python 3.13 + PyTorch (CUDA) + torchvision + matplotlib
- `TORCH_HOME` and `HF_HOME` set to `data/` to contain all downloads
- `/tmp` available for uv cache (default behavior)

## Experiment Details

### 1. Vanilla VAE (MNIST)

- **Architecture**: FC encoder (784→400→latent), FC decoder (latent→400→784), latent_dim=20
- **Loss**: ELBO = BCE reconstruction + KL divergence
- **Outputs**: reconstruction comparison, 2D latent space (PCA/t-SNE), random sampling grid, loss curve
- **Training**: ~20 epochs, <5 min on RTX 4060

### 2. β-VAE (dSprites)

- **Architecture**: Conv encoder/decoder (4 conv layers), latent_dim=10
- **Loss**: β × KL + BCE, sweep β ∈ {1, 2, 4, 10} to compare disentanglement
- **Data**: dSprites (~2.7GB download to data/), 64×64 grayscale
- **Outputs**: latent traversal plots, disentanglement comparison across β, traversal metric
- **Training**: ~30 epochs per β value

### 3. VQ-VAE (MNIST + CIFAR-10)

- **Architecture**: Conv encoder, codebook (K=512, dim=64), Conv decoder
- **Loss**: Reconstruction + codebook update (EMA) + commitment loss
- **Outputs**: reconstruction, codebook utilization histogram, samples (simple random or PixelCNN prior)
- **Training**: MNIST ~30 epochs, CIFAR-10 ~50 epochs

### 4. IWAE (MNIST)

- **Architecture**: Same as Vanilla VAE, but sample K z's per forward pass
- **Loss**: IWAE bound = log(1/K Σ exp(log p(x|z_k) + log p(z_k) - log q(z_k|x)))
- **Comparison**: K ∈ {1, 5, 10, 50} vs Vanilla VAE (K=1), compare test log-likelihood
- **Outputs**: log-likelihood curves across K, sample quality comparison
- **Training**: ~20 epochs

### 5. NVAE (CIFAR-10)

- **Architecture**: Simplified hierarchical VAE — 2 groups of latent variables, residual blocks with depthwise conv
- **Loss**: Hierarchical ELBO with spectral regularization
- **8GB adaptation**: batch_size=16, shallower residual blocks (~20 layers vs original 60+)
- **Outputs**: reconstruction, hierarchical sampling, loss curve
- **Training**: ~100 epochs

## Key Design Decisions

1. **Pure PyTorch, no frameworks** — each experiment hand-written for maximum understanding
2. **Self-contained scripts** — each experiment runs independently with `uv run experiments/0X_*.py`
3. **Common utilities** — `common/data.py` handles dataset download paths; `common/viz.py` provides shared visualization (latent plots, sample grids, loss curves)
4. **Data containment** — all datasets and model outputs stay within `vae-experiments/`; environment variables `TORCH_HOME`/`HF_HOME` redirect downloads
5. **Results structure** — each experiment gets its own subdirectory with checkpoints, samples, and metrics

## Success Criteria

- [ ] All 5 experiments train to convergence on RTX 4060 without OOM
- [ ] Vanilla VAE produces recognizable MNIST digits and coherent 2D latent space
- [ ] β-VAE shows visible disentanglement improvement as β increases
- [ ] VQ-VAE codebook utilization > 50% and reasonable reconstructions
- [ ] IWAE shows improved log-likelihood with increasing K
- [ ] NVAE produces recognizable CIFAR-10 reconstructions with hierarchical sampling working
