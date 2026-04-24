# AI/ML Research Timeline — Key Landmarks

> The intellectual lineage from foundational ideas to the 2026 frontier.
> Each entry: year, paper, why it mattered, what it enabled.

---

## 2013–2017: Deep Generative Models Era

| Year | Paper | Impact |
|------|-------|--------|
| 2013 | [VAE: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | Unified probabilistic modeling + neural networks; latent spaces become differentiable |
| 2014 | [GAN: Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | Implicit generative models via adversarial training |
| 2015 | [IWAE](https://arxiv.org/abs/1509.00519) | Tighter variational bounds → better generative quality |
| 2017 | [VQ-VAE](https://arxiv.org/abs/1711.00937) | Discrete latent codes → bridge between continuous and discrete generation |
| 2017 | [Transformer: Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Parallelizable sequence modeling; replaces RNNs |

## 2018–2020: Transformers Rise + Generative Explosion

| Year | Paper | Impact |
|------|-------|--------|
| 2018 | [BERT](https://arxiv.org/abs/1810.04805) | Bidirectional pretraining; NLP paradigm shift |
| 2018 | [Glow](https://arxiv.org/abs/1807.03039) | Invertible generative models; exact likelihood |
| 2018 | [Neural ODE](https://arxiv.org/abs/1806.07366) | Continuous-depth networks; bridges ODEs and deep learning |
| 2019 | GPT-2 (no arXiv) | Language models as unsupervised multitask learners |
| 2020 | [DDPM](https://arxiv.org/abs/2006.11239) | Diffusion for generation — begins the diffusion era |
| 2020 | [Score SDE](https://arxiv.org/abs/2011.13456) | Unifies score matching + diffusion via SDEs |
| 2020 | [SimCLR](https://arxiv.org/abs/2002.05709) | Simple contrastive learning framework — SSL goes mainstream |
| 2020 | [MoCo](https://arxiv.org/abs/1911.05722) | Momentum contrast — SSL without large batches |
| 2020 | [VQGAN](https://arxiv.org/abs/2012.09841) | VQ-VAE + adversarial training → high-fidelity generation |
| 2020 | [RAG](https://arxiv.org/abs/2005.11401) | Retrieval-augmented generation (not in collection — non-arxiv ID) |

## 2021: Scaling + Alignment

| Year | Paper | Impact |
|------|-------|--------|
| 2021 | [S4: Structured State Spaces](https://arxiv.org/abs/2111.00396) | Linear-time sequence modeling → foundation for Mamba |
| 2021 | [LoRA](https://arxiv.org/abs/2106.09685) | Low-rank adaptation → efficient fine-tuning paradigm |
| 2021 | [CLIP](https://arxiv.org/abs/2103.00020) | Vision-language alignment via contrastive pretraining |
| 2021 | [DINO](https://arxiv.org/abs/2104.14294) | Self-supervised ViTs with emerging properties |
| 2021 | [MAE](https://arxiv.org/abs/2111.06377) | Masked autoencoding scales ViT pretraining |
| 2021 | [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155) | Aligning LLMs with human feedback → ChatGPT paradigm |

## 2022: Diffusion + Alignment

| Year | Paper | Impact |
|------|-------|--------|
| 2022 | [LDM / Stable Diffusion](https://arxiv.org/abs/2112.10752) | Latent diffusion → open-source image generation revolution |
| 2022 | [CFG: Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) | Conditional generation without classifiers |
| 2022 | [Constitutional AI](https://arxiv.org/abs/2212.08073) | AI self-alignment via principles |
| 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware exact attention → enables long context |
| 2022 | [Whisper](https://arxiv.org/abs/2212.04356) | Robust speech recognition at scale |
| 2022 | [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) | Data quality > model size → reshape training strategies |
| 2022 | [DiT](https://arxiv.org/abs/2212.09748) | Diffusion Transformer — replaces U-Net for diffusion |
| 2022 | [RT-1](https://arxiv.org/abs/2212.06817) | Transformer-based robot control at scale |

## 2023: Agents + Reasoning

| Year | Paper | Impact |
|------|-------|--------|
| 2023 | [Mamba](https://arxiv.org/abs/2312.00752) | Selective state spaces → linear-time Transformer alternative |
| 2023 | [DPO](https://arxiv.org/abs/2305.18290) | Direct preference optimization — RLHF without RL |
| 2023 | [LLaVA](https://arxiv.org/abs/2304.08485) | Visual instruction tuning → open VLM paradigm |
| 2023 | [HuggingGPT](https://arxiv.org/abs/2303.17580) | LLM as orchestrator → agent paradigm begins |
| 2023 | [Toolformer](https://arxiv.org/abs/2302.04761) | Self-supervised tool use → models learn to call APIs |
| 2023 | [RT-2 / VLA](https://arxiv.org/abs/2307.15818) | Vision-language-action models → web knowledge → robot control |
| 2023 | [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Diffusion for robot action generation |
| 2023 | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | Memory-efficient LLM serving at scale |
| 2023 | [LeanDojo](https://arxiv.org/abs/2306.15626) | Theorem proving with retrieval → neural theorem proving toolkit |
| 2023 | [I-JEPA](https://arxiv.org/abs/2301.08243) | Joint-embedding prediction → LeCun's vision of self-supervised learning |
| 2023 | [SEDD](https://arxiv.org/abs/2310.16834) | Discrete diffusion language models → non-autoregressive text generation paradigm |
| 2023 | [CoT / Reasoning](https://arxiv.org/abs/2201.11903) | Chain-of-thought prompting → reasoning in LLMs |

## 2024: Reasoning Revolution + Architecture Diversification

| Year | Paper | Impact |
|------|-------|--------|
| 2025 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | RL-induced reasoning → open-source reasoning model |
| 2024 | [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) | Group relative policy optimization → simpler RL for reasoning |
| 2024 | [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) | Structured state space duality → faster Mamba |
| 2024 | [TTT](https://arxiv.org/abs/2407.04620) | Learning at test time → expressive hidden states |
| 2024 | [KAN](https://arxiv.org/abs/2404.19756) | Kolmogorov-Arnold Networks → alternative to MLPs |
| 2024 | [π0](https://arxiv.org/abs/2410.24164) | VLA flow matching → general robot control |
| 2024 | [OpenVLA](https://arxiv.org/abs/2406.09246) | Open-source 7B VLA → democratized robot learning |
| 2023 | [LeanDojo](https://arxiv.org/abs/2306.15626) | Theorem proving with retrieval → neural theorem proving toolkit |
| 2024 | [SD3 / Rectified Flow](https://arxiv.org/abs/2209.03003) | Flow matching beats DDPM for image generation |
| 2024 | [HybridFlow / veRL](https://arxiv.org/abs/2409.19256) | Flexible RLHF training framework → standard for RL infra |

## 2025–2026: Agentic AI + Convergence

| Year | Paper | Impact |
|------|-------|--------|
| 2025 | [GR00T N1](https://arxiv.org/abs/2503.14734) | Open humanoid robot foundation model |
| 2026 | [Mamba-3](https://arxiv.org/abs/2603.15569) | Complex-valued states, half the state size (ICLR 2026) |
| 2026 | [LLMs Gaming Verifiers](https://arxiv.org/abs/2604.15149) | Reveals reward hacking in RLVR — fundamental flaw exposed |
| 2026 | [LeWorldModel](https://arxiv.org/abs/2603.19312) | First stable end-to-end JEPA from pixels (LeCun group) |
| 2026 | [ClawSafety](https://arxiv.org/abs/2604.01438) | Safety alignment fails in agentic settings |
| 2026 | [MSA: 100M Token Context](https://arxiv.org/abs/2603.23516) | Orders-of-magnitude context length breakthrough |
| 2026 | [Helium: Agent-Aware Serving](https://arxiv.org/abs/2603.16104) | Serving systems designed for agents, not chat |

---

## The Grand Arc

```
2013  VAE                                          "Latent spaces are differentiable"
  │
2018  Transformer                                  "Attention replaces recurrence"
  │
2020  DDPM + GPT-3                                 "Generation + scale"
  │
2021  CLIP + LoRA + DiT                            "Alignment + efficiency + architecture"
  │
2022  Stable Diffusion + RLHF + FlashAttention      "Open generation + alignment + efficiency"
  │
2023  Mamba + DPO + LLaVA + Agents                  "Alt architectures + simper alignment + VLMs + autonomy"
  │
2024  DeepSeek-R1 + KAN + π0 + TTT                  "Reasoning + new primitives + robotics + test-time"
  │
2025-2026  World Models + Agent Safety + MCP         "Convergence: VLM→VLA→world model; agents get guardrails"
```

**The 2026 thesis:** Every major thread — generation, reasoning, alignment, architecture, robotics — is converging on **agentic AI that perceives, reasons, and acts** in the world, with **safety as a first-class constraint**.
