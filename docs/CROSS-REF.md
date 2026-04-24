# Cross-Topic Reference Index

> Papers that appear in multiple topic files, organized by theme.
> Use this to trace ideas across disciplinary boundaries.

---

## Agent Safety & Governance (3 files)


| Paper                                                 | Files                 | Why Cross-Topic                             |
| ----------------------------------------------------- | --------------------- | ------------------------------------------- |
| SoK: Attack Surface of Agentic AI (2603.22928)        | agent, safety, ethics | Defines agentic security landscape          |
| Governance Norms to Enforceable Controls (2604.05229) | agent, safety, ethics | Runtime guardrails for agents               |
| ClawSafety: "Safe" LLMs, Unsafe Agents (2604.01438)   | safety, ethics        | Safety alignment doesn't transfer to agents |
| Constitutional Black-Box Monitoring (2603.00829)      | safety, ethics        | Detecting scheming agent behavior           |


## World Models & VLA (4 files)


| Paper                                               | Files                 | Why Cross-Topic             |
| --------------------------------------------------- | --------------------- | --------------------------- |
| Learning VLA World Models for Driving (2604.09059)  | agent, multimodal     | VLA + world modeling        |
| Agentic Video Generation (2604.10383)               | agent, generative     | Video gen as agent planning |
| Diffusion Models for Joint Audio-Video (2603.16093) | diffusion, generative | Cross-modal generation      |
| OmniSonic: Audio from Video+Text (2604.04348)       | diffusion, generative | Video-conditioned audio     |


## Compiler ↔ Infra (6+ papers)


| Paper                                                  | Files           | Why Cross-Topic                   |
| ------------------------------------------------------ | --------------- | --------------------------------- |
| Nautilus: Auto-Scheduling Tensor Compiler (2604.14825) | compiler, infra | Kernel optimization for serving   |
| TCL: Cross-Hardware Tensor Optimization (2604.12891)   | compiler, infra | Portable tensor programs          |
| Blink: CPU-Free Inference (2604.07609)                 | compiler, infra | SmartNIC offloading               |
| Foundry: CUDA Graph Materialization (2604.06664)       | compiler, infra | Graph-level serving optimization  |
| HiFloat4 for Ascend NPUs (2604.08826)                  | compiler, infra | NPU number format                 |
| ENEC: Ascend NPU Compression (2604.03298)              | compiler, infra | NPU model compression             |
| DeepStack: 3D-Stacked Accelerators (2604.04750)        | compiler, infra | Hardware design space exploration |


## Diffusion ↔ VAE Foundations (10+ papers)


| Paper                               | Files          | Why Cross-Topic              |
| ----------------------------------- | -------------- | ---------------------------- |
| DDPM (2006.11239)                   | diffusion, vae | Shared foundation            |
| LDM / Stable Diffusion (2112.10752) | diffusion, vae | VAE encoder in diffusion     |
| Score SDE (2011.13456)              | diffusion, vae | Unified generative framework |
| CV-VAE (2405.20279)                 | diffusion, vae | Video VAE for latent video   |
| LeanVAE (2503.14325)                | diffusion, vae | Efficient video tokenizer    |
| Deep Compression AE (2410.10733)    | diffusion, vae | High-res VAE compression     |


## RL Infrastructure ↔ Reasoning (2+ papers)


| Paper                                             | Files                     | Why Cross-Topic                         |
| ------------------------------------------------- | ------------------------- | --------------------------------------- |
| ProRL Agent: Rollout-as-a-Service (2603.18815)    | infra, reasoning          | RL rollout serving for LLM agents       |
| FlexMARL: Rollout-Training Co-Design (2602.09578) | infra, (reasoning)        | Multi-agent RL orchestration            |
| LongRLVR: Long-Context RL (2603.02146)            | reasoning, (long-context) | Verifiable rewards need context scaling |


## Reasoning ↔ Safety (3+ papers)


| Paper                                              | Files                          | Why Cross-Topic           |
| -------------------------------------------------- | ------------------------------ | ------------------------- |
| LLMs Gaming Verifiers (2604.15149)                 | reasoning, safety              | Reward hacking in RLVR    |
| Reasoning Models Are Jailbreak Agents (2508.04039) | safety, ethics                 | Reasoning enables attacks |
| MEMENTO: Context Management (2604.09852)           | agent, reasoning, long-context | Reasoning + memory        |


## LoRA ↔ Infra (6+ papers)


| Paper                                             | Files                 | Why Cross-Topic                |
| ------------------------------------------------- | --------------------- | ------------------------------ |
| InfiniLoRA: Disaggregated Multi-LoRA (2604.07173) | infra, efficient-ft   | Serving many LoRA adapters     |
| EdgeFlow: Cold Starts on Mobile (2604.09083)      | infra, efficient-ft   | On-device LoRA deployment      |
| TalkLoRA: Communication-Aware (2604.06291)        | efficient-ft, (infra) | Distributed LoRA communication |
| APreQEL: Adaptive Mixed Precision (2603.23575)    | infra, efficient-ft   | Quantization for edge LoRA     |


## New Architecture ↔ Others (5+ papers)


| Paper                                        | Files                             | Why Cross-Topic                       |
| -------------------------------------------- | --------------------------------- | ------------------------------------- |
| Attention to Mamba Distillation (2604.14191) | new-arch, efficient-ft, reasoning | Cross-architecture knowledge transfer |
| COREY: Operator Fusion for SSMs (2604.10597) | new-arch, compiler                | SSM kernel optimization               |
| FourierMoE (2604.01762)                      | new-arch, efficient-ft            | MoE as PEFT method                    |
| YOCO++ KV Residual (2604.13556)              | new-arch, infra                   | Architecture for efficient serving    |
| Dynamic Upcycling of Experts (2603.29765)    | new-arch, efficient-ft            | Converting dense to MoE               |


## MCP Ecosystem (2+ files)


| Paper                                | Files               | Why Cross-Topic         |
| ------------------------------------ | ------------------- | ----------------------- |
| Semantic Tool Discovery (2603.20313) | agent, long-context | Retrieval for MCP tools |
| MCPThreatHive (2604.13849)           | agent, safety       | MCP security threats    |


## Multimodal ↔ Speech (3+ papers)


| Paper                                     | Files                         | Why Cross-Topic                  |
| ----------------------------------------- | ----------------------------- | -------------------------------- |
| Audio-Omni (2604.10708)                   | diffusion, generative, speech | Unified audio generation         |
| Adversarial Attacks on MLLMs (2603.27918) | multimodal, speech            | Cross-modal attacks              |
| OmniSonic (2604.04348)                    | diffusion, generative         | Video+text→audio                 |
| PersonaPlex (2602.06053)                  | generative, speech            | Voice cloning in dialogue models |
| FlashLabs Chroma 1.0 (2601.11141)         | generative, speech            | End-to-end speech with cloning   |


## Diffusion ↔ Generative (11 papers)


| Paper                                 | Files                 | Why Cross-Topic                    |
| ------------------------------------- | --------------------- | ---------------------------------- |
| ACE-Step (2506.00045)                 | diffusion, generative | Music generation foundation model  |
| SegmentDreamer (2507.05256)           | diffusion, generative | 3D text-to-3D via distillation     |
| Audio Palette (2510.12175)            | diffusion, generative | DiT for Foley synthesis            |
| Image Generation History (2603.07455) | diffusion, generative | Technical history survey           |
| SafeCtrl (2604.03941)                 | diffusion, generative | Safety control for T2I             |
| 1.x-Distill (2604.04018)              | diffusion, generative | Distribution matching distillation |
| LPM 1.0 (2604.07823)                  | diffusion, generative | Video character performance        |
| Prompt Relay (2604.10030)             | diffusion, generative | Temporal control for video         |


## Diffusion ↔ VAE Foundations (6 papers)


| Paper                                          | Files          | Why Cross-Topic                           |
| ---------------------------------------------- | -------------- | ----------------------------------------- |
| Deep Generative Modelling Review (2103.04922)  | diffusion, vae | Comparative review of generative families |
| Understanding Diffusion Models (2208.11970)    | diffusion, vae | Unifies VAE/Diffusion/Flow perspective    |
| Diffusion & Representation Survey (2407.00783) | diffusion, vae | Diffusion models for representation       |
| OD-VAE (2409.01199)                            | diffusion, vae | Video VAE for latent diffusion            |


## Long Context ↔ New Architectures (2 papers)


| Paper                         | Files                  | Why Cross-Topic                            |
| ----------------------------- | ---------------------- | ------------------------------------------ |
| FlashAttention (2205.14135)   | long-context, new-arch | Attention efficiency → architecture choice |
| FlashAttention-2 (2307.08691) | long-context, new-arch | Faster attention for all architectures     |


## Agent ↔ Robotics (2 papers)


| Paper                   | Files           | Why Cross-Topic                   |
| ----------------------- | --------------- | --------------------------------- |
| Voyager (2305.16291)    | agent, robotics | Open-ended embodied agent         |
| RT-2 / VLA (2307.15818) | agent, robotics | Vision-language-action for robots |


## AI for Science ↔ Math Proving (7 papers)


| Paper                                      | Files         | Why Cross-Topic                      |
| ------------------------------------------ | ------------- | ------------------------------------ |
| DeepSeek-Prover (2405.14333)               | science, math | Theorem proving via synthetic data   |
| Safe step-aware verification (2506.04592)  | science, math | Formal verification for reasoning    |
| Decomposition for formal math (2507.15225) | science, math | Iterative reflection in proving      |
| Seed-Prover (2507.23726)                   | science, math | Deep/broad reasoning for ATP         |
| Hilbert (2509.22819)                       | science, math | 99.2% miniF2F via recursive proving  |
| Neural TP for Verification (2601.18944)    | science, math | Real-world formal verification       |
| VeriSoftBench (2602.18307)                 | science, math | Repository-scale formal verification |


## Ethics ↔ Safety (2 papers)


| Paper                                       | Files          | Why Cross-Topic             |
| ------------------------------------------- | -------------- | --------------------------- |
| Strategic Dishonesty (2509.18058)           | ethics, safety | Safety evaluation integrity |
| Vulnerability-Amplifying Loops (2602.01347) | ethics, safety | AI mental-health failures   |


## Efficient FT ↔ Others (6 papers)


| Paper                                        | Files                    | Why Cross-Topic                       |
| -------------------------------------------- | ------------------------ | ------------------------------------- |
| MiniCPM-V (2408.01800)                       | efficient-ft, multimodal | Mobile MLLM via efficient methods     |
| SAE-Constructed Low-Rank (2512.23260)        | efficient-ft, safety     | Interpretable safety via SAE+LoRA     |
| tLoRA (2602.07263)                           | efficient-ft, infra      | Elastic multi-LoRA training           |
| Distillation for LLMs Survey (2603.13765)    | efficient-ft, infra      | Knowledge distillation methods        |
| Effective Distillation to xLSTM (2603.15590) | efficient-ft, new-arch   | Distillation for hybrid architectures |
| Small VLMs as Compressors (2604.08120)       | efficient-ft, multimodal | Efficient video understanding         |
| REAM (2604.04356)                            | efficient-ft, infra      | Merging improves expert pruning       |
| Teacher-Student Cooperation (2604.14164)     | efficient-ft, reasoning  | Fine-tuning reasoning models          |


## Reasoning ↔ Others (5 papers)


| Paper                                    | Files                   | Why Cross-Topic                     |
| ---------------------------------------- | ----------------------- | ----------------------------------- |
| OpenVLThinkerV2 (2604.08539)             | reasoning, multimodal   | Multimodal reasoning model          |
| Chain of Modality (2604.14520)           | reasoning, multimodal   | Dynamic orchestration in omni-MLLMs |
| CoTEvol (2604.14768)                     | reasoning, long-context | Self-evolving CoT data synthesis    |
| SWE-TRACE (2604.14820)                   | reasoning, agent        | Process reward for SWE agents       |
| Task-Capability Coevolution (2604.14969) | reasoning, agent        | Novel expert discovery              |
| SPEED-Bench (2604.09557)                 | reasoning, infra        | Speculative decoding benchmark      |
| RationalRewards (2604.11626)             | reasoning, generative   | Reasoning rewards for visual gen    |
| SOAR (2604.12617)                        | reasoning, generative   | Self-correction in diffusion        |


## Infra ↔ Others (8 papers)


| Paper                                  | Files               | Why Cross-Topic               |
| -------------------------------------- | ------------------- | ----------------------------- |
| Mixtral of Experts (2401.04088)        | infra, new-arch     | MoE serving optimization      |
| Distributed Training Bugs (2506.10426) | infra, reasoning    | Framework reliability         |
| MSA (2603.23516)                       | infra, long-context | 100M token attention          |
| MemBoost (2603.26557)                  | infra, multimodal   | Cost-aware inference          |
| Goose (2603.00510)                     | infra, multimodal   | Speculative decoding for VLMs |
| CUDA Agent (2512.21473)                | infra, agent        | RL for kernel generation      |
| Bugs in PyTorch Compiler (2604.08720)  | infra, compiler     | Compiler correctness          |


## Compiler ↔ Others (9 papers)


| Paper                                  | Files             | Why Cross-Topic                   |
| -------------------------------------- | ----------------- | --------------------------------- |
| TritonForge (2512.09196)               | compiler, infra   | Automated Triton optimization     |
| FlashFuser (2512.12949)                | compiler, infra   | Inter-core kernel fusion          |
| HW Acceleration Survey (2512.23914)    | compiler, infra   | Hardware acceleration overview    |
| AscendKernelGen (2601.07160)           | compiler, infra   | NPU kernel generation             |
| End-to-End PiM (2601.14260)            | compiler, infra   | Processing-in-memory              |
| ENEC (2601.15127)                      | compiler, infra   | Ascend NPU compression            |
| Axe (2601.19092)                       | compiler, infra   | Unified ML compiler layout        |
| AscendCraft (2601.22760)               | compiler, infra   | NPU kernel via DSL                |
| CuTe Layout (2603.02298)               | compiler, infra   | Tensor algebra for compilers      |
| RedFuser (2603.10026)                  | compiler, infra   | Cascaded reduction fusion         |
| AE-LLM (2603.20492)                    | compiler, infra   | Adaptive efficiency for LLMs      |
| Energy Efficient CoDesign (2603.23668) | compiler, infra   | TinyML to LLMs                    |
| Event Tensor (2604.13327)              | compiler, infra   | Dynamic megakernel compilation    |
| Physics-Informed SNN (2511.21784)      | compiler, science | Spiking networks for PDEs         |
| SPINONet (2603.21674)                  | compiler, science | Spiking physics-informed operator |


## New Architecture ↔ Others (4 papers)


| Paper                                  | Files                | Why Cross-Topic              |
| -------------------------------------- | -------------------- | ---------------------------- |
| KAN for PDEs (2512.22283)              | new-arch, science    | KAN for scientific computing |
| MoE-FM (2604.15009)                    | new-arch, diffusion  | MoE flow matching for LLMs   |
| Linear Attention in MLLMs (2604.10064) | new-arch, multimodal | Efficient attention for VLMs |
| Nucleus-Image (2604.12163)             | new-arch, generative | Sparse MoE for image gen     |


## Misc Cross-Topic (3 papers)


| Paper                                   | Files                    | Why Cross-Topic                   |
| --------------------------------------- | ------------------------ | --------------------------------- |
| Context Engineering Survey (2507.13334) | agent, long-context      | Context as engineering discipline |
| AI Hippocampus (2601.09113)             | generative, long-context | Memory systems across modalities  |
| Multi-Robot MLLM Survey (2604.00061)    | agent, multimodal        | Multi-robot via MLLM              |


## Agent ↔ Multimodal VLM (3 papers)


| Paper                               | Files             | Why Cross-Topic                   |
| ----------------------------------- | ----------------- | --------------------------------- |
| VLA for Driving Survey (2506.24044) | agent, multimodal | VLA models for autonomous driving |
| MLLM Tools Survey (2508.10955)      | agent, multimodal | External tools for MLLMs          |
| Agentic MLLM Survey (2510.10991)    | agent, multimodal | Agentic multimodal LLMs           |


## Agent ↔ Long Context (1 paper)


| Paper                            | Files               | Why Cross-Topic              |
| -------------------------------- | ------------------- | ---------------------------- |
| Memory in AI Agents (2512.13564) | agent, long-context | Memory management for agents |


## Robotics ↔ VAE (1 paper)


| Paper                     | Files         | Why Cross-Topic             |
| ------------------------- | ------------- | --------------------------- |
| LeWorldModel (2603.19312) | robotics, vae | JEPA from pixels — JEPA 里程碑 |


## Diffusion Language Models (cross-file, 3 files)


| Paper                                | Files               | Why Cross-Topic                        |
| ------------------------------------ | ------------------- | -------------------------------------- |
| MoE-FM (2604.15009)                  | diffusion, new-arch | MoE + flow matching for language       |
| Expert-Choice Routing (2604.01622)   | diffusion, new-arch | Adaptive compute in DLMs               |
| DLM-Scope (2602.05859)               | diffusion, safety   | SAEs for diffusion LM interpretability |
| Diffusion LM for Speech (2604.14001) | diffusion, speech   | DLMs applied to ASR                    |


