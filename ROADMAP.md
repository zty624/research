# AI/ML Research Reading Roadmap

> Suggested reading order for researchers entering each topic area.
> Landmark (开山) papers form the backbone — read these first, then explore recent work.

---

## How to Use This Roadmap

Each topic has a **3-tier reading path**:
1. **Foundations** (开山 papers + key surveys) — understand what and why
2. **Core Advances** (2023-2024 influential work) — understand how it evolved
3. **Frontier** (2025-2026 latest work) — understand where it's going

---

## 1. Diffusion Models

**Foundations:**
- DDPM (2020) → Score SDE (2020) → LDM/Stable Diffusion (2022) → DiT (2022)
- Classifier-Free Guidance (2022) → Consistency Models (2023)
- Understanding Diffusion Models: A Unified Perspective (2022) — best tutorial

**Core Advances:**
- SD3 / Rectified Flow (2024) → Flow Matching → SANA (2024)
- PixArt series → video diffusion (Sora review)

**Frontier:**
- 1.x-Distill, FlashAttention-4, FP4 training, agentic video generation
- Diffusion Language Models: SEDD → LangFlow, LogicDiff, Dynin-Omni (emerging paradigm)

**Cross-topic prerequisites:** VAE (understands latent space), Transformer architecture

---

## 2. VAE & Representation Learning

**Foundations:**
- VAE (Kingma & Welling 2013) → IWAE (2015) → VQ-VAE (2017) → VQGAN (2020)
- SimCLR (2020) → DINO (2021) → MAE (2021)
- Glow / Normalizing Flows (2018)

**Core Advances:**
- DINOv2 (2023), I-JEPA (2023), Matryoshka representations (2022)
- Video VAEs (CV-VAE, LeanVAE) — critical for video diffusion

**Frontier:**
- Representation autoencoders for DiT, FSQ, VP-VAE

**Cross-topic bridges:** Diffusion (LDM uses VAE encoder), new architectures (SSM-based VAEs)

---

## 3. New Architectures (SSM/RWKV/xLSTM/KAN/MoE/TTT)

**Foundations:**
- S4 (2021) → Mamba (2023) → Mamba-2 (2024)
- RWKV → xLSTM → KAN → TTT (2024)
- Mixtral / MoE (2024)

**Core Advances:**
- Falcon Mamba (7B attention-free), hybrid Mamba-Transformer architectures
- Vision Mamba surveys

**Frontier:**
- Mamba-3 (2026), cross-architecture distillation, compiler-first SSM

**Prerequisites:** Transformer fundamentals, linear attention

---

## 4. LLM Reasoning, RL & Alignment

**Foundations:**
- InstructGPT / RLHF (2022) → Constitutional AI (2022) → DPO (2023)
- CoT (2022) → DeepSeek-R1 (2024) → GRPO (2024)

**Core Advances:**
- PRM / step-level supervision → test-time compute scaling
- Reward hacking analysis (2026)

**Frontier:**
- Token-level policy optimization, value gradient flow, SUPERNOVA
- Multi-modal reasoning RL

**Cross-topic bridges:** Agent systems (reasoning in agents), safety (alignment)

---

## 5. AI Agent, Tool Use, RAG & VLA

**Foundations:**
- HuggingGPT (2023) → Toolformer (2023) → RAG (2020)
- RT-2 / VLA (2023)

**Core Advances:**
- MCP protocol, agentic workflows, code agents
- RAG + reasoning integration

**Frontier:**
- Autonomous agent protocols, context management (MEMENTO)
- GUI/mobile agents, security (MCPThreatHive)

**Cross-topic bridges:** Multimodal (VLA), safety (agent risks), infra (agent serving)

---

## 6. Multimodal VLM

**Foundations:**
- CLIP (2021) → LLaVA (2023)
- Qwen-VL, InternVL series

**Core Advances:**
- VILA, Cambrian-1, MiniCPM-V (mobile)
- Qwen2-VL

**Frontier:**
- Perception-grounded policy optimization, omni-MLLM orchestration
- OpenVLThinkerV2

**Prerequisites:** Vision Transformer, LLM fundamentals, CLIP

---

## 7. AI Infra & Operator Optimization

**Foundations:**
- vLLM / PagedAttention (2023) → FlashAttention (2022-23)
- Speculative decoding

**Core Advances:**
- Disaggregated serving, multi-LoRA, heterogeneous serving
- KV cache optimization
- RL training frameworks (veRL, OpenRLHF)

**Frontier:**
- CPU-free inference (Blink), NPU kernel generation, AI-native OS
- RL training infrastructure (SortedRL, rollout serving, rollout-training co-design)

**Prerequisites:** GPU architecture basics, CUDA programming

---

## 8. Efficient Fine-tuning, LoRA & Model Merging

**Foundations:**
- LoRA (2021)

**Core Advances:**
- LoRA variants (QLoRA, DoRA, rsLoRA)
- Model merging (TIES, DARE, task arithmetic)
- Knowledge distillation surveys

**Frontier:**
- Adaptive rank allocation, Fourier-based expansion, RL-guided merging
- Communication-aware distributed LoRA

**Cross-topic bridges:** New architectures (LoRA for Mamba/SSM), infra (LoRA serving)

---

## 9. Long Context & Data Curation

**Foundations:**
- FlashAttention (2022) → FlashAttention-2 (2023)
- RoPE position encoding

**Core Advances:**
- Sparse attention, KV cache compression
- Data curation for pre-training

**Frontier:**
- 100M token context, hybrid sparse+linear attention
- FlashAttention-4, attention editing

**Cross-topic bridges:** New architectures (SSM for long context), infra (memory optimization)

---

## 10. AI Safety, Alignment & Interpretability

**Foundations:**
- Red teaming (2023) → jailbreak attacks → safety alignment

**Core Advances:**
- Multi-modal jailbreaks, LLM unlearning
- Sparse autoencoders for mechanistic interpretability

**Frontier:**
- Agentic AI attack surfaces, self-play safety alignment
- Watermarking, unlearning in MoE

**Cross-topic bridges:** Agent (agent safety), reasoning (reasoning-based attacks)

---

## 11. AI for Science

**Foundations:**
- DeepONet / FNO / PINN (2020-22)

**Core Advances:**
- AlphaFold 3, protein design, drug docking
- Weather/climate foundation models, PDE solvers

**Frontier:**
- AutoBinder agent (MCP-based), biomolecular modeling at scale
- AI for materials science

**Prerequisites:** Domain knowledge in relevant science, equivariant networks

---

## 12. Generative Models (Image/Video/3D/Audio)

**Foundations:** (covered in Diffusion Models section)

**Core focus areas:**
- T2I controllability, video generation, 3D (Gaussian Splatting)
- Audio/music generation, voice cloning

**Frontier:**
- DiT serving optimization, RL for visual generation
- Real-time streaming video, multi-track music

---

## 13. AI Compiler, Hardware & Co-Design

**Foundations:**
- TVM (2018) → Ansor (2020) → Relax (2023)

**Core Advances:**
- NPU kernel generation, Triton-based optimization
- Processing-in-memory, neuromorphic computing

**Frontier:**
- LLM-based kernel synthesis, 3D-stacked accelerators
- Ascend NPU ecosystem

**Prerequisites:** Computer architecture, compiler design, CUDA

---

## 14. Speech & Audio Language Models

**Foundations:**
- Whisper (2022) → Encodec (2022) → VALL-E (2023)

**Core Advances:**
- Full-duplex speech models, real-time dialogue
- End-to-end speech LLMs

**Frontier:**
- Multi-dialect speech, domain adaptation, cross-modal distillation

**Cross-topic bridges:** Generative models (TTS), multimodal (audio-visual)

---

## 15. Robotics Foundation Models

**Foundations:**
- RT-1 (2022) → RT-2 (2023) → Diffusion Policy (2023)

**Core Advances:**
- OpenVLA, Octo, π0, RDT-1B (2024)
- 3D Diffusion Policy, GR-1

**Frontier:**
- GR00T N1 (humanoid), RL fine-tuning at scale, visual trace prompting

**Prerequisites:** VLM foundations, diffusion models, basic robotics

---

## 16. AI for Math & Theorem Proving

**Foundations:**
- LeanDojo (2023)

**Core Advances:**
- DeepSeek-Prover V1/V1.5/V2 (2024-25)
- Kimina-Prover, PutnamBench

**Frontier:**
- Hilbert (99.2% miniF2F), Seed-Prover, agent-based proving
- Formal verification for math reasoning

**Prerequisites:** Lean/Coq basics, LLM reasoning

---

## 17. AI Ethics, Fairness & Governance

**Foundations:** (surveys from 2024 serve as foundations)
- Fairness in LLMs, bias in LLMs, AI guardrails surveys (2024)

**Core areas:**
- LLM fairness & bias mitigation
- Computer vision fairness, RL fairness
- EU AI Act governance, frontier AI safety cases

**Frontier:**
- Agentic AI governance: runtime guardrails, scheming detection
- Safety alignment vs. agentic deployment gap
- Mental-health AI vulnerabilities, reasoning model jailbreaks

---

## Topic Dependency Graph

```
                        VAE/Representation
                       /        |        \
                      /         |         \
              Diffusion ---- Multimodal --- New Architectures
                |    \         |    \           |      \
                |     \        |     \          |       \
         Generative   \       VLM    \     Long Context  Infra
          Models       \              \         |          |
             |        Robotics     AI Agent     |      Compiler
             |          |         /    \        |       Hardware
          Speech ----  VLA     RAG   Safety  Data      Co-Design
          Audio                 |      |     Curation
                                |   Ethics &
                             Reasoning   Fairness
                               RL
                                |
                          Math/Theorem
                            Proving
                                |
                           AI for Science
```

**Recommended entry points for different profiles:**

| Profile | Start Here | Then Explore |
|---------|-----------|-------------|
| ML generalist | Diffusion → VAE → New Architectures | Generative, Long Context |
| NLP/LLM focus | Reasoning/RL → Agents → Long Context | Safety, Multimodal |
| Systems engineer | Infra → Compiler/HW → Efficient FT | Long Context, New Arch |
| CV/multimodal | VAE → Diffusion → Multimodal VLM | Generative, Robotics |
| AI safety | Safety → Ethics → Reasoning/RL | Agent, Multimodal |
| Domain scientist | AI for Science → Math Proving | VAE, Diffusion |
| Robotics | Robotics Foundations → Multimodal VLM | Diffusion, Agent |
