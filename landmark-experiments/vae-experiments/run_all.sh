#!/bin/bash
# Run all VAE experiments sequentially
set -e
cd "$(dirname "$0")"
echo "=== Running all VAE experiments ==="
echo "1/5: Vanilla VAE"
uv run experiments/01_vanilla_vae.py
echo "2/5: β-VAE"
uv run experiments/02_beta_vae.py
echo "3/5: VQ-VAE"
uv run experiments/03_vq_vae.py
echo "4/5: IWAE"
uv run experiments/04_iwae.py
echo "5/5: NVAE"
uv run experiments/05_nvae.py
echo "=== All experiments complete ==="
