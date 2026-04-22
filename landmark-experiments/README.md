# Landmark Paper Minimal Reproductions

Experiments reproducing the core ideas from landmark papers in the collection.

## How to Run

All experiments use the `vae-experiments` uv environment:

```bash
cd landmark-experiments
uv run --project ../vae-experiments python 01_transformer.py
uv run --project ../vae-experiments python 02_lora.py
uv run --project ../vae-experiments python 03_flow_matching.py
uv run --project ../vae-experiments python 04_mamba.py
uv run --project ../vae-experiments python 05_dit.py
uv run --project ../vae-experiments python 06_bert.py
uv run --project ../vae-experiments python 07_flash_attention.py
uv run --project ../vae-experiments python 08_cot.py
uv run --project ../vae-experiments python 09_grpo.py
uv run --project ../vae-experiments python 10_kv_cache.py
uv run --project ../vae-experiments python 11_dpo.py
uv run --project ../vae-experiments python 12_ddpm.py
uv run --project ../vae-experiments python 13_model_merging.py
uv run --project ../vae-experiments python 14_distillation.py
uv run --project ../vae-experiments python 15_clip.py
uv run --project ../vae-experiments python 16_ppo.py
uv run --project ../vae-experiments python 17_ssl.py
uv run --project ../vae-experiments python 18_rlhf.py
uv run --project ../vae-experiments python 19_consistency.py
uv run --project ../vae-experiments python 20_gan.py
uv run --project ../vae-experiments python 21_moe.py
uv run --project ../vae-experiments python 22_normalizing_flows.py
uv run --project ../vae-experiments python 23_rag.py
uv run --project ../vae-experiments python 24_kan.py
uv run --project ../vae-experiments python 25_vq_vae.py
uv run --project ../vae-experiments python 26_sac.py
uv run --project ../vae-experiments python 27_contrastive.py
uv run --project ../vae-experiments python 28_dropout.py
uv run --project ../vae-experiments python 29_vit.py
uv run --project ../vae-experiments python 30_optimizer.py
```

## Experiments

### 01. Transformer (1706.03762)
- **Task**: Sequence reversal on random digit sequences
- **Reproduces**: Scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder-decoder architecture, warmup LR schedule
- **Visualizes**: Training curves, self-attention patterns

### 02. LoRA (2106.09685)
- **Task**: MNIST classification with a tiny ViT
- **Reproduces**: Low-rank adaptation (ΔW = BA), zero initialization, scaling factor α/r
- **Compares**: Full fine-tuning vs LoRA (r=4, r=2) vs BitFit
- **Key finding**: LoRA matches or exceeds full fine-tuning with fewer parameters (confirmed!)

### 03. Flow Matching (2210.02747)
- **Task**: 2D point cloud generation
- **Reproduces**: Conditional Flow Matching (CFM), OT path vs VP (diffusion) path
- **Visualizes**: Trajectories (OT=straight, VP=curved), generated samples, sampling efficiency with fewer steps
- **Key finding**: OT paths are straighter and converge faster

### 04. Mamba / Selective SSM (2312.00752)
- **Task**: Selective copying on synthetic sequences
- **Reproduces**: S4 (time-invariant SSM) vs Selective SSM (input-dependent B, C, Δ)
- **Visualizes**: Input-dependent step size Δ, comparison of training dynamics
- **Key finding**: Selective SSM learns to modulate Δ based on input importance

### 05. DiT (2212.09748)
- **Task**: 2D point cloud diffusion
- **Reproduces**: Patchify, Transformer backbone, adaLN-Zero vs in-context conditioning
- **Visualizes**: Loss comparison, generated samples
- **Key finding**: adaLN-Zero condition injection is effective

### 06. BERT (1810.04805)
- **Task**: Synthetic sentence pair classification + masked LM
- **Reproduces**: Masked Language Model (MLM), Next Sentence Prediction (NSP), bidirectional attention, [CLS] token
- **Compares**: BERT (bidirectional) vs GPT-style (causal) attention
- **Visualizes**: MLM/NSP training curves, attention pattern comparison, bidirectional vs causal loss

### 07. FlashAttention (2205.14135)
- **Task**: IO-aware tiled attention computation
- **Reproduces**: Tiled attention with online softmax, SRAM vs HBM memory analysis
- **Compares**: Standard attention (O(N²) memory) vs FlashAttention (O(N) memory)
- **Visualizes**: Memory comparison, online softmax step-by-step, HBM I/O analysis
- **Key finding**: Tiled attention matches standard attention output (diff < 1e-6) while using far less memory

### 08. Chain-of-Thought (2201.11903)
- **Task**: Arithmetic addition with direct vs step-by-step prompting
- **Reproduces**: CoT prompting format, emergent reasoning at scale
- **Compares**: Direct prompting vs CoT at 4 model sizes (1L to 4L)
- **Visualizes**: Training curves at each scale, accuracy vs model size emergence plot
- **Key finding**: CoT provides more benefit as model size increases (emergent reasoning)

### 09. GRPO / DeepSeek-R1 (2501.12948)
- **Task**: Learning arithmetic via RL without a critic network
- **Reproduces**: Group Relative Policy Optimization (GRPO), rule-based reward model
- **Compares**: GRPO vs REINFORCE (GRPO uses group mean as baseline, no critic needed)
- **Visualizes**: Reward/loss curves, GRPO advantage normalization, algorithm comparison diagram
- **Key finding**: GRPO's group-relative advantage reduces variance vs REINFORCE

### 10. KV Cache & GQA (2301.08243, 2305.14314)
- **Task**: LLM inference optimization — KV cache, GQA, quantization
- **Reproduces**: KV cache incremental generation, Grouped-Query Attention (GQA/MQA), INT8/INT4 KV cache quantization
- **Compares**: MHA vs GQA vs MQA memory usage and training quality, FP16 vs INT8 vs INT4 quantization error
- **Visualizes**: KV cache memory at various seq lengths/batch sizes, quantization error distribution, generation cost with/without cache
- **Key finding**: KV cache matches full recomputation (diff < 1e-7), GQA drastically reduces memory with minimal quality loss

### 11. DPO (2305.18290)
- **Task**: Learning from preference pairs without a reward model
- **Reproduces**: Direct Preference Optimization loss, implicit reward via log-ratio, β sensitivity
- **Compares**: DPO vs RLHF (PPO) — DPO needs only policy + reference, no reward model
- **Visualizes**: Loss/reward margin comparison, RLHF vs DPO pipeline diagram, β sensitivity
- **Key finding**: DPO achieves comparable alignment with simpler pipeline (no reward model needed)

### 12. DDPM (2006.11239)
- **Task**: 2D point cloud generation with diffusion
- **Reproduces**: Forward/reverse diffusion process, linear vs cosine noise schedule, simplified noise prediction objective, DDPM vs DDIM sampling
- **Compares**: Linear vs cosine schedule, DDPM (200 steps) vs DDIM (20 steps) sampling efficiency
- **Visualizes**: Noise schedule curves, forward diffusion animation, generated samples, sampling efficiency
- **Key finding**: Cosine schedule provides more gradual noising; DDIM achieves similar quality with 10x fewer steps

### 13. Model Merging (2212.04089, 2306.01708)
- **Task**: Multi-task learning on permuted MNIST (3 tasks)
- **Reproduces**: Task Arithmetic (adding task vectors), TIES merging (trim + elect + sign), simple averaging
- **Compares**: Individual finetuning vs averaging vs Task Arithmetic vs TIES
- **Visualizes**: Per-task and average accuracy, Task Arithmetic scaling factor sweep, task vector magnitude distributions
- **Key finding**: Task Arithmetic significantly outperforms simple averaging; optimal scaling λ ≈ 0.8-1.0

### 14. Knowledge Distillation (1503.02531)
- **Task**: MNIST digit classification with teacher-student
- **Reproduces**: Soft target distillation with temperature, α balancing hard/soft loss, dark knowledge in soft targets
- **Compares**: Hard labels vs KD at different temperatures (T=2,4,8,16), Student(64h) vs Tiny(16h)
- **Visualizes**: Temperature and α sensitivity, soft target distributions at different T
- **Key finding**: Soft targets transfer "dark knowledge" — student trained with KD outperforms hard-label training

### 15. CLIP / Contrastive Learning (2103.00020)
- **Task**: MNIST digit classification via image-text alignment
- **Reproduces**: Contrastive learning (InfoNCE loss), learned temperature, zero-shot classification via text prompts
- **Compares**: Zero-shot CLIP vs linear probe on CLIP embeddings vs supervised baseline
- **Visualizes**: Embedding space (PCA), cross-modal similarity matrix, temperature evolution, accuracy comparison
- **Key finding**: Zero-shot classification works by aligning image and text embeddings — 98% zero-shot accuracy on MNIST

### 16. PPO (1707.06347)
- **Task**: 1D cart-pole balance control
- **Reproduces**: Clipped surrogate objective, GAE advantage estimation, actor-critic architecture
- **Compares**: PPO vs REINFORCE+baseline vs Vanilla PG
- **Visualizes**: Reward/loss curves, PPO clipping mechanism (positive/negative advantage), clip ε sensitivity
- **Key finding**: PPO's clipping provides stable policy updates — converges faster and more reliably than REINFORCE

### 17. Self-Supervised Learning (2002.05709, 2006.07733, 2111.06377)
- **Task**: MNIST representation learning without labels
- **Reproduces**: SimCLR (NT-Xent loss + augmentation), BYOL (EMA target + no negatives), MAE (mask + reconstruct)
- **Compares**: SimCLR vs BYOL vs MAE vs random vs supervised via linear probe
- **Visualizes**: Training losses, linear probe accuracy, MAE reconstruction, PCA feature spaces
- **Key finding**: MAE works best on MNIST (72% linear probe); SimCLR learns structured features; BYOL needs careful tuning

### 18. RLHF Full Pipeline (2203.02155)
- **Task**: Sequence generation with human preference alignment
- **Reproduces**: Complete 3-step pipeline — SFT → Reward Model → PPO, Bradley-Terry preference loss, KL penalty
- **Compares**: SFT-only vs full RLHF (SFT+RM+PPO)
- **Visualizes**: SFT loss, RM loss/accuracy, PPO reward and KL divergence, pipeline diagram
- **Key finding**: RLHF significantly improves over SFT — reward model accuracy reaches 100%, PPO reward increases 2.4×

### 19. Consistency Models (2303.01469)
- **Task**: 2D point cloud generation with single-step denoising
- **Reproduces**: Consistency training (CT), consistency property (f(x_t,t) = x_0 for all t), Karras noise schedule, EMA target
- **Compares**: DDPM (50 steps) vs DDIM (10 steps) vs Consistency (1 step!)
- **Visualizes**: Training loss, sample comparison, sampling efficiency, consistency property at different noise levels
- **Key finding**: Consistency models achieve single-step generation — 200× fewer evaluations than DDPM

### 20. GANs (1406.2661, 1701.07875, 1802.05983)
- **Task**: 2D mixture of Gaussians generation
- **Reproduces**: Vanilla GAN (BCE min-max), WGAN (Wasserstein distance + weight clipping), SN-GAN (spectral normalization + hinge loss)
- **Compares**: Vanilla vs WGAN vs SN-GAN on mode coverage and training stability
- **Visualizes**: Generated samples, training dynamics, Wasserstein distance, GAN evolution diagram
- **Key finding**: WGAN and SN-GAN provide more stable training; Wasserstein distance is a meaningful training metric

### 21. Mixture of Experts (1701.06538, Switch Transformer)
- **Task**: Next-token prediction on MNIST patch sequences
- **Reproduces**: Top-k routing, load balancing auxiliary loss, expert capacity
- **Compares**: Dense model vs MoE-8 (top-2) vs MoE-16 (top-2) — same FLOPs per token, more parameters
- **Visualizes**: Training loss comparison, load balancing loss, expert utilization, parameter comparison
- **Key finding**: Load balancing loss keeps expert utilization near-uniform (~12.5% per expert); MoE scales parameters without increasing per-token compute

### 22. Normalizing Flows (1505.05770, 1605.08803)
- **Task**: 2D density estimation on moons distribution
- **Reproduces**: Planar Flow (u·tanh(w^Tz+b)), Radial Flow, RealNVP (affine coupling layers with exact inverse)
- **Compares**: Planar vs Radial vs RealNVP — expressiveness and training stability
- **Visualizes**: Sample comparison, training loss, RealNVP learned density, flow transformation trajectories
- **Key finding**: RealNVP (affine coupling) provides exact log-likelihood and best density estimation; Planar/Radial flows are limited in expressiveness

### 23. RAG / Retrieval-Augmented Generation (2005.11401)
- **Task**: QA on synthetic knowledge base (number facts)
- **Reproduces**: Dense document indexing, similarity-based retrieval, context-conditioned generation
- **Compares**: Baseline (no retrieval) vs RAG (retrieve + augment + generate)
- **Visualizes**: Training loss, retrieval accuracy, QA accuracy comparison, document embedding space (PCA), pipeline diagram
- **Key finding**: RAG achieves 84.3% token accuracy vs 74.7% baseline — retrieval provides external knowledge that improves generation

### 24. KAN / Kolmogorov-Arnold Networks (2404.19756)
- **Task**: Function approximation on 5 mathematical functions (sincos, gaussian, x²+y², product, swiss)
- **Reproduces**: Learnable spline activations on edges, B-spline basis functions, Kolmogorov-Arnold representation
- **Compares**: KAN [2,5,1] (90 params) vs MLP [2,16,16,1] (337 params)
- **Visualizes**: Training curves per function, final loss, parameter efficiency, learned spline activations, architecture diagram
- **Key finding**: KAN achieves comparable approximation with ~4x fewer parameters; Gaussian function shows KAN advantage (lower MSE with fewer params)

### 25. VQ-VAE / Vector Quantized VAE (1711.00937)
- **Task**: MNIST reconstruction with discrete latent codes
- **Reproduces**: Vector quantization (nearest-neighbor codebook), straight-through estimator, commitment loss, EMA codebook updates
- **Compares**: VAE (continuous) vs VQ-VAE (standard) vs VQ-VAE (EMA)
- **Visualizes**: Training curves, reconstruction samples, codebook usage histogram, latent space (PCA), concept diagram
- **Key finding**: VQ-VAE uses 45% of codebook (29/64 codes) with perplexity 34; EMA variant can suffer codebook collapse without careful tuning; VQ-VAE recon approaches VAE quality with discrete bottleneck

### 26. SAC / Soft Actor-Critic (1801.01290)
- **Task**: Pendulum continuous control (torque in [-2,2])
- **Reproduces**: Maximum entropy RL (reward + α·H(π)), twin Q-networks, automatic temperature tuning, reparameterization trick
- **Compares**: SAC (auto α) vs SAC (fixed α) vs DDPG vs REINFORCE
- **Visualizes**: Reward curves, α evolution over training, learned policy visualization, concept diagram
- **Key finding**: SAC with auto α learns initially well but can over-reduce exploration (α→0); DDPG more stable on this simple task; entropy regularization is critical for exploration

### 27. Contrastive Learning (2002.05709, 1911.05722, 2006.07733)
- **Task**: MNIST self-supervised representation learning
- **Reproduces**: SimCLR (NT-Xent loss), MoCo (momentum encoder + queue), BYOL (EMA target + predictor, no negatives)
- **Compares**: Random vs SimCLR vs MoCo vs BYOL vs Supervised (via linear probe accuracy)
- **Visualizes**: Training loss, linear probe accuracy, feature space PCA, concept diagram
- **Key finding**: MoCo (44.1%) > SimCLR (35.0%) on small-batch MNIST; BYOL needs careful tuning to avoid representation collapse; momentum encoder + queue is more data-efficient than large-batch SimCLR

### 28. Dropout & Regularization (1207.0580, 1906.02629)
- **Task**: MNIST classification with various regularization methods
- **Reproduces**: Dropout (random unit dropout as implicit ensemble), Weight Decay (L2 penalty), Label Smoothing (soft targets), MC Dropout (uncertainty estimation)
- **Compares**: No reg vs Dropout vs Weight Decay vs Label Smoothing vs Combined
- **Visualizes**: Training dynamics, train vs test accuracy, generalization gap, MC Dropout uncertainty, label smoothing effect
- **Key finding**: Combined regularization gives best generalization; MC Dropout provides meaningful uncertainty estimates; label smoothing prevents overconfidence

### 29. ViT / Vision Transformer (2010.11929)
- **Task**: MNIST image classification
- **Reproduces**: Patch embedding (Conv2d projection), [CLS] token, learned positional embedding, Transformer encoder for vision
- **Compares**: ViT vs CNN, patch size effect (2/4/7), data scaling (5%-100% training data)
- **Visualizes**: Training curves, patch size comparison, data scaling, [CLS] feature PCA, positional embedding visualization, architecture diagram
- **Key finding**: CNN outperforms ViT on small-scale MNIST (98.4% vs 95.3%); ViT needs more data — at 5% data, ViT gets 76.6% vs CNN's 96.8%; at 100% data, ViT approaches CNN (98.9% vs 99.3%)

### 30. Optimizer Comparison (1412.6980, 1711.05101)
- **Task**: MNIST classification with various optimizers and LR schedules
- **Reproduces**: SGD, SGD+Momentum, Adam (adaptive LR), AdamW (decoupled weight decay), Cosine/LinearWarmup schedules
- **Compares**: 7 optimizer+schedule combinations on CNN training
- **Visualizes**: Training loss, test accuracy, LR schedules, weight norm comparison (Adam vs AdamW), optimizer concept diagram
- **Key finding**: AdamW keeps weight norms smaller than Adam (decoupled WD); LR scheduling significantly impacts convergence; SGD+momentum+cosine matches adaptive methods with proper tuning

## Prior Experiments

### VAE Series (`../vae-experiments/`)
- Vanilla VAE, β-VAE, VQ-VAE, IWAE, NVAE — all completed with results
