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
uv run --project ../vae-experiments python 31_speculative.py
uv run --project ../vae-experiments python 32_quantization.py
uv run --project ../vae-experiments python 33_diffusion_policy.py
uv run --project ../vae-experiments python 34_darts.py
uv run --project ../vae-experiments python 35_resnet.py
uv run --project ../vae-experiments python 36_batchnorm.py
uv run --project ../vae-experiments python 37_layernorm.py
uv run --project ../vae-experiments python 38_rmsnorm.py
uv run --project ../vae-experiments python 39_rope.py
uv run --project ../vae-experiments python 40_swiglu.py
uv run --project ../vae-experiments python 41_bpe_tokenization.py
uv run --project ../vae-experiments python 42_attention_sinks.py
uv run --project ../vae-experiments python 43_scaling_laws.py
uv run --project ../vae-experiments python 44_convnext.py
uv run --project ../vae-experiments python 45_nucleus_sampling.py
uv run --project ../vae-experiments python 46_grokking.py
uv run --project ../vae-experiments python 47_sparse_attention.py
uv run --project ../vae-experiments python 48_lottery_ticket.py
uv run --project ../vae-experiments python 49_ema.py
uv run --project ../vae-experiments python 50_transformer_xl.py
uv run --project ../vae-experiments python 51_alibi.py
uv run --project ../vae-experiments python 52_self_instruct.py
uv run --project ../vae-experiments python 53_rwkv.py
uv run --project ../vae-experiments python 54_gradcam.py
uv run --project ../vae-experiments python 55_cfg.py
uv run --project ../vae-experiments python 56_rectified_flow.py
uv run --project ../vae-experiments python 57_xlstm.py
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

### 31. Speculative Decoding (2211.17192)
- **Task**: Language model generation with draft-then-verify
- **Reproduces**: Draft model proposes γ tokens, target model verifies in one forward pass, modified rejection sampling preserves exact distribution
- **Compares**: Autoregressive vs Speculative (γ=2,4,6), accept rates, target model calls, output distribution
- **Visualizes**: Speed comparison, accept rates, target forward passes, output distribution comparison, concept diagram
- **Key finding**: Speculative decoding preserves output distribution (low KL divergence); accept rate ~50% at γ=2; real speedup requires memory-bandwidth-bound target model (not visible in small models)

### 32. LLM Quantization (2208.07339, 2210.17323)
- **Task**: MNIST classification with quantized MLP weights
- **Reproduces**: Symmetric/asymmetric quantization, GPTQ-style group-wise quantization, mixed-precision (outliers in FP16)
- **Compares**: FP32 vs INT8 Sym vs INT4 Sym vs INT4 GPTQ vs INT4 Mixed-Precision
- **Visualizes**: Accuracy comparison, quantization error (MSE, cosine similarity), weight distributions, error heatmaps, model size, concept diagram
- **Key finding**: INT8 preserves accuracy (94.73% vs 94.71% FP32); INT4 with GPTQ/mixed-precision achieves cosine similarity 0.99+; group-wise scales and outlier handling are critical for INT4 quality

### 33. Diffusion Policy (2303.04137)
- **Task**: 1D target tracking with expert demonstrations
- **Reproduces**: Diffusion-based action generation, DDPM denoising conditioned on state, multi-modal action distribution
- **Compares**: Deterministic policy (MSE) vs Diffusion policy (noise prediction), action distributions, trajectory tracking
- **Visualizes**: Training loss, trajectory comparison with uncertainty bands, action distributions, denoising process, concept diagram
- **Key finding**: Diffusion policy produces stochastic action distributions (std=0.33) capturing uncertainty; deterministic policy gives single point; diffusion policy's strength is in multi-modal action spaces

### 34. DARTS / Differentiable Architecture Search (1806.09055)
- **Task**: MNIST classification with architecture search
- **Reproduces**: Differentiable relaxation of discrete architecture choices, softmax over candidate operations (conv, dilated conv, pool, skip, zero), bi-level optimization (α on val, w on train)
- **Compares**: DARTS-discovered architecture vs fixed (CNN) vs random architecture
- **Visualizes**: Training curves, architecture parameter (α) evolution, discovered architecture graph, concept diagram
- **Key finding**: DARTS discovers dilated conv-dominated architecture with selective zero connections; discovered architecture outperforms random baseline

### 35. ResNet / Skip Connections (1512.03385)
- **Task**: MNIST classification at various depths
- **Reproduces**: Residual learning (F(x) + x), skip connections enabling deeper networks, gradient flow through shortcuts
- **Compares**: Plain network vs ResNet at 2/4/8 blocks per stage
- **Visualizes**: Training curves at each depth, accuracy vs depth, gradient norm vs depth, concept diagram
- **Key finding**: At depth 8, plain network degrades (96.95%) while ResNet improves (99.11%); gradient norms explode in plain deep networks

### 36. Batch Normalization (1502.03167)
- **Task**: MNIST classification with deep MLP at various learning rates
- **Reproduces**: Batch normalization (custom implementation), internal covariate shift, learnable γ and β
- **Compares**: No BN vs BN at 4 learning rates (1e-4 to 5e-2), custom BN vs PyTorch BN
- **Visualizes**: LR sensitivity, activation distribution tracking, custom vs PyTorch BN, concept diagram
- **Key finding**: At LR=0.05, no-BN model fails completely (11.35% random), BN achieves 95.92%; BN enables 7-8x higher learning rates

### 37. Layer Normalization (1607.06450)
- **Task**: Sequence classification with LSTM and Transformer
- **Reproduces**: Custom LayerNorm implementation, BN vs LN vs GroupNorm normalization dimensions
- **Compares**: LSTM with no/LN/BN norm, Transformer with no/LN/BN/GN norm
- **Visualizes**: Training curves, normalization dimension heatmap (BN vs LN vs GN), concept diagram
- **Key finding**: LayerNorm is standard for Transformers (works per-sample); BatchNorm fails on sequences; GroupNorm is a middle ground

### 38. RMSNorm (2019.03210)
- **Task**: Language modeling with Transformer
- **Reproduces**: Custom RMSNorm (x/RMS(x) * γ, no mean subtraction, no β), LayerNorm vs RMSNorm comparison
- **Compares**: LayerNorm vs RMSNorm vs No Norm, scaling at 2/4/8 layers
- **Visualizes**: Training curves, scaling comparison, normalization effect on distributions, concept diagram
- **Key finding**: RMSNorm is ~20% faster than LayerNorm with comparable quality; re-centering is redundant (β cancels -μ)

### 39. RoPE / Rotary Position Embedding (2104.09864)
- **Task**: Language modeling with different position encodings
- **Reproduces**: Rotary position embedding via 2D rotation, q_m·k_n = f(m-n) relative position property
- **Compares**: Sinusoidal vs learned vs RoPE position encoding, length extrapolation
- **Visualizes**: Training curves, length extrapolation, relative position dot product, rotation visualization, concept diagram
- **Key finding**: RoPE achieves best training loss (0.94 vs 1.11) and best length extrapolation (42.5% at 2x length); q·k variance is 0 at same relative distance

### 40. SwiGLU / GLU Activations (2002.05202, 1705.03122)
- **Task**: Language modeling with Transformer FFN variants
- **Reproduces**: ReLU, GELU, Swish, GLU, SwiGLU, GeGLU activation functions; gated linear unit mechanism (gate ⊗ value)
- **Compares**: 6 FFN variants in Transformer LM
- **Visualizes**: Training curves, final loss comparison, activation shapes, parameter counts, concept diagram
- **Key finding**: Gated variants significantly outperform plain activations (GeGLU 0.67, SwiGLU 0.82 vs ReLU 1.05); SwiGLU is LLaMA standard

### 41. BPE Tokenization (1508.07909)
- **Task**: Subword tokenization for language modeling
- **Reproduces**: Byte-Pair Encoding (iterative most-frequent pair merging), character/word/BPE tokenizers, OOV handling
- **Compares**: Character-level vs BPE-50/100/200 tokenization for LM training
- **Visualizes**: Training curves, compression ratio, merge operations, vocabulary growth, tokenization spectrum, concept diagram
- **Key finding**: BPE handles OOV without <unk> tokens; BPE-100 achieves 4.6x compression vs char-level with better loss (0.022 vs 0.049)

### 42. Attention Sinks / StreamingLLM (2309.17453)
- **Task**: KV cache management for infinite-length generation
- **Reproduces**: Attention sink phenomenon, full KV cache vs naive eviction vs StreamingLLM (keep sinks + window)
- **Compares**: KL divergence from full cache for naive eviction vs StreamingLLM
- **Visualizes**: Attention heatmaps, attention to position 0 over sequence, KL divergence over generation steps, concept diagram
- **Key finding**: Naive eviction causes catastrophic collapse (KL=6.12); StreamingLLM preserves distribution (KL=0.60); first tokens act as attention sinks

### 43. Scaling Laws / Chinchilla (2001.08361, 2203.15556)
- **Task**: Understanding compute-optimal LLM training
- **Reproduces**: Power-law scaling of loss with model size and data size; Chinchilla optimal compute allocation
- **Compares**: 6 model sizes (18K-817K params), 6 data sizes (500-20K tokens), compute-optimal frontier
- **Visualizes**: Loss vs parameters (power law), loss vs data, Chinchilla compute trade-off, concept diagram
- **Key finding**: Loss follows power-law with model size and data; for fixed compute, smaller models trained longer can match larger models

### 44. ConvNeXt (2201.03545)
- **Task**: MNIST classification with modernized CNN
- **Reproduces**: Step-by-step modernization of ResNet (depthwise conv → large kernel → GELU → inverted bottleneck), ConvNeXt block design
- **Compares**: ResNet vs ConvNeXt, incremental modernization steps
- **Visualizes**: Training curves, modernization step accuracy, concept diagram
- **Key finding**: Each modernization step contributes: depthwise conv drops acc but reduces params, large kernel recovers, GELU adds ~1%, inverted bottleneck adds ~0.7%

### 45. Nucleus Sampling (1904.09751)
- **Task**: Language model generation with different sampling strategies
- **Reproduces**: Temperature scaling, top-k sampling, top-p (nucleus) sampling, greedy decoding
- **Compares**: 7 sampling methods on diversity and quality
- **Visualizes**: Temperature effect on distributions, cumulative probability (nucleus), diversity comparison, concept diagram
- **Key finding**: Temperature controls distribution sharpness (entropy 0.68→3.06); nucleus sampling adapts vocabulary size dynamically

### 46. Grokking (2201.02177)
- **Task**: Modular arithmetic (a+b)%p with delayed generalization
- **Reproduces**: Grokking phenomenon — memorization then sudden generalization, weight decay necessity, training fraction effect
- **Compares**: With vs without weight decay, different training data fractions
- **Visualizes**: Grokking curves, weight decay comparison, training fraction effect, concept diagram
- **Key finding**: Train acc hits 100% immediately (memorization), test acc rises slowly (grokking); weight decay drives the phase transition

### 47. Sparse Attention / Longformer (2004.05150)
- **Task**: Document classification with efficient attention
- **Reproduces**: Full O(N²) attention, sliding window attention O(N×w), Longformer (sliding window + global tokens)
- **Compares**: Full vs Sliding Window vs Longformer attention on classification quality and speed
- **Visualizes**: Attention pattern heatmaps, training curves, time scaling, concept diagram
- **Key finding**: Sliding window and Longformer achieve comparable accuracy with much less compute; Longformer's global tokens add task-specific information flow

### 48. Lottery Ticket Hypothesis (1803.03635)
- **Task**: Finding sparse trainable subnetworks at initialization
- **Reproduces**: Iterative Magnitude Pruning (IMP), winning ticket (prune + reset to init + retrain), random pruning baseline
- **Compares**: IMP (winning ticket) vs random pruning at multiple sparsity levels (0-95%)
- **Visualizes**: Accuracy vs sparsity, training curves at 80% sparsity, pruned weight distributions, concept diagram
- **Key finding**: IMP at 90% sparsity: 96.00% vs random 94.92%; at 95%: IMP 95.58% vs random 93.90% — winning tickets exist and outperform random pruning

### 49. EMA / Exponential Moving Average (various — DDPM, BYOL)
- **Task**: Stable model training via weight averaging
- **Reproduces**: EMA (θ_ema = β·θ_ema + (1-β)·θ), EMA for diffusion model generation quality, EMA for classifier accuracy
- **Compares**: Raw model vs EMA model, β decay sensitivity (0.9, 0.99, 0.999, 0.9999)
- **Visualizes**: Raw vs EMA generated samples, accuracy comparison, weight trajectory smoothing, concept diagram
- **Key finding**: β=0.99 gives slight improvement over raw; β=0.9999 too slow for 15 epochs; EMA smooths weight oscillations for more stable outputs

### 50. Transformer-XL (1802.04799)
- **Task**: Copy task with segment-level recurrence
- **Reproduces**: Segment-level recurrence (cache hidden states from previous segment), relative positional encoding, memory mechanism
- **Compares**: Transformer-XL (with memory) vs Transformer-XL (no memory) vs Vanilla Transformer
- **Visualizes**: Training curves, sequence length scaling, relative vs absolute position encoding, concept diagram
- **Key finding**: Segment-level recurrence enables cross-segment information flow; relative position encoding generalizes across segments

### 51. ALiBi (1910.03193)
- **Task**: Periodic sequence prediction with length extrapolation
- **Reproduces**: Attention with Linear Biases (no positional embeddings), geometric slope sequence 2^(-8h/H), length extrapolation
- **Compares**: ALiBi vs learned position vs sinusoidal position encoding
- **Visualizes**: Training curves, length extrapolation accuracy, per-head bias patterns, slope distribution, concept diagram
- **Key finding**: ALiBi extrapolates to 2.5x training length (94.5% acc) while learned position drops to 74.3%; slopes are NOT learned

### 52. Self-Instruct (2302.04761)
- **Task**: Bootstrapping instruction-following data from seed instructions
- **Reproduces**: Instruction generation from seed examples, deduplication filtering, temperature-controlled creativity, self-training loop
- **Compares**: Different temperatures on pool diversity and quality
- **Visualizes**: Pool growth, instruction length distribution, temperature effects, self-training quality improvement, concept diagram
- **Key finding**: Model quality improves iteratively through self-training (0.30 → 0.51 over 5 iterations); temperature controls diversity-redundancy trade-off

### 53. RWKV (2305.13048)
- **Task**: Periodic sequence prediction with linear attention
- **Reproduces**: WKV attention with channel-wise decay, time-shift mixing, squared ReLU FFN, RNN-mode inference
- **Compares**: RWKV vs Transformer on training loss and sequence length scaling
- **Visualizes**: Training curves, loss vs sequence length, per-head decay patterns, memory complexity, concept diagram
- **Key finding**: RWKV achieves competitive loss with O(N) memory; learned decay rates differ per head; competitive with Transformer at longer sequences

### 54. GradCAM (1610.02391)
- **Task**: Visual explanations of CNN predictions on MNIST
- **Reproduces**: Gradient-weighted Class Activation Mapping, α_k = GAP(∂y^c/∂A_k), L_GradCAM = ReLU(Σ α_k · A_k)
- **Compares**: GradCAM vs vanilla gradient saliency maps, class-specific explanations
- **Visualizes**: Saliency maps vs GradCAM overlays, per-class activation maps, top-5 class explanations
- **Key finding**: GradCAM provides cleaner, more localized explanations than saliency maps; different classes highlight different image regions

### 55. Classifier-Free Guidance (2207.12598)
- **Task**: Conditional 2D point cloud generation with class guidance
- **Reproduces**: Condition dropout during training, classifier-free guidance formula ê = (1+w)·ε(x,c) - w·ε(x,∅)
- **Compares**: Guidance scales w=0 (unconditional) to w=5 (strong guidance)
- **Visualizes**: Guidance scale effect on samples, class-conditional generation, accuracy vs diversity trade-off
- **Key finding**: w=2-3 gives best accuracy-diversity trade-off; w>1 amplifies condition effect; distance from target drops from 0.33 (w=0) to 0.13 (w=2)

### 56. Rectified Flow (2209.03003)
- **Task**: 2D point cloud generation via straight ODE paths
- **Reproduces**: Linear interpolation z_t = t·x_1 + (1-t)·x_0, velocity prediction v = x_1 - x_0, Euler ODE solving
- **Compares**: Rectified Flow vs DDPM diffusion on sampling steps needed
- **Visualizes**: Training curves, trajectory comparison (straight vs curved), few-step generation quality, concept diagram
- **Key finding**: Rectified Flow converges with 5-10 steps (std ~0.8-0.95) vs diffusion needing 10+; straight paths enable efficient sampling

### 57. xLSTM (2405.04517)
- **Task**: Periodic sequence prediction with extended LSTM
- **Reproduces**: sLSTM (exponential gating + scalar memory), mLSTM (matrix memory with key-value store), alternating sLSTM/mLSTM blocks
- **Compares**: xLSTM vs standard LSTM vs Transformer, sLSTM vs mLSTM ablation
- **Visualizes**: Training curves, sequence length scaling, architecture analysis, concept diagram
- **Key finding**: xLSTM with exponential gating outperforms standard LSTM at short sequences (1.30 vs 1.73); mLSTM matrix memory needs careful stabilization for longer sequences

## Prior Experiments

### VAE Series (`../vae-experiments/`)
- Vanilla VAE, β-VAE, VQ-VAE, IWAE, NVAE — all completed with results
