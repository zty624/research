# 2026 Must-Read Papers

> ~30 papers that define the 2026 research frontier. Curated for maximum signal.
> Last updated: 2026-04-19

---

## Reasoning & RL for LLMs


| Paper                                                                                                     | Why It Matters                                                                                         |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149)                | Reveals fundamental flaw in dominant RLVR paradigm — models enumerate labels instead of learning rules |
| [Re2: Reinforcement Learning with Re-solving](https://arxiv.org/abs/2603.07197)                           | Teaches reasoning models to abandon dead-end paths (ICLR 2026)                                         |
| [Delta-Reasoner: Test-Time Gradient Descent in Latent Space](https://arxiv.org/abs/2603.04948)            | Proves inference-time GD is dual to KL-regularized RL; 20%+ accuracy gains (ICLR 2026)                 |
| [SUPERNOVA: Eliciting General Reasoning via RL on Natural Instructions](https://arxiv.org/abs/2604.08477) | General reasoning without task-specific RL training                                                    |
| [MEMENTO: Teaching LLMs to Manage Their Own Context](https://arxiv.org/abs/2604.09852)                    | Self-managed context — critical for long reasoning chains                                              |


## New Architectures


| Paper                                                                                              | Why It Matters                                                                                        |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [Mamba-3: Improved Sequence Modeling via State Space Principles](https://arxiv.org/abs/2603.15569) | Next-gen Mamba with complex-valued states, half the state size (ICLR 2026)                            |
| [Functional Component Ablation in Hybrid Language Models](https://arxiv.org/abs/2603.22473)        | Shows removing SSM/linear attention causes >35,000x perplexity degradation — proves hybrids need both |
| [Attention to Mamba: Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191)            | Transfer knowledge from Transformers to Mamba — enables architecture switching                        |


## Agent Safety & MCP


| Paper                                                                                              | Why It Matters                                                                                   |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [ClawSafety: "Safe" LLMs, Unsafe Agents](https://arxiv.org/abs/2604.01438)                         | Proves safety alignment does NOT transfer to agentic settings                                    |
| [Constitutional Black-Box Monitoring for Scheming in LLM Agents](https://arxiv.org/abs/2603.00829) | Detecting deceptive/scheming agent behavior at deployment time                                   |
| [SoK: Security and Safety in the MCP Ecosystem](https://arxiv.org/abs/2512.08290)                  | First comprehensive MCP security systematization                                                 |
| [How Are AI Agents Used? Evidence from 177,000 MCP Tools](https://arxiv.org/abs/2603.23802)        | Empirical: action tools rose 27%→65% over 16 months — agents are shifting to modify environments |


## Infrastructure & Serving


| Paper                                                                                   | Why It Matters                                                                    |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [Helium: Efficient LLM Serving for Agentic Workflows](https://arxiv.org/abs/2603.16104) | First serving system designed for agentic workloads, not just chat                |
| [ProRL Agent: Rollout-as-a-Service](https://arxiv.org/abs/2603.18815)                   | Solves core scalability bottleneck for RL training of multi-turn agents           |
| [SortedRL: Length-Aware Scheduling for RL Training](https://arxiv.org/abs/2603.23414)   | Rollout phase is the bottleneck; length-aware scheduling eliminates padding waste |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451)                                    | Algorithm-kernel pipelining co-design for next-gen hardware                       |


## Robotics & Embodied AI


| Paper                                                                        | Why It Matters                                                                 |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [DFM-VLA: Discrete Flow Matching VLA](https://arxiv.org/abs/2603.26320)      | 95.7% LIBERO success — flow matching enters VLA paradigm                       |
| [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312)                      | First JEPA training stably end-to-end from raw pixels (LeCun group) — JEPA 里程碑 |
| [Memento-Skills: Let Agents Design Agents](https://arxiv.org/abs/2603.18743) | Self-evolving agent architectures — paradigm shift from hand-designed agents   |


## Diffusion & Generative


| Paper                                                                                                    | Why It Matters                                          |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [1.x-Distill: Breaking the Distribution Matching Distillation Barrier](https://arxiv.org/abs/2604.04018) | Near-single-step generation with quality preservation   |
| [FP4 Explore, BF16 Train: Diffusion RL via Efficient Rollout Scaling](https://arxiv.org/abs/2604.06916)  | 4-bit training for diffusion — massive efficiency gains |


## VLM & Multimodal


| Paper                                                                                              | Why It Matters                                                       |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [Chain of Modality: From Static Fusion to Dynamic Orchestration](https://arxiv.org/abs/2604.14520) | Dynamic modality routing in omni-MLLMs                               |
| [Do VLMs Need to Process Image Tokens?](https://arxiv.org/abs/2604.09425)                          | CVPR 2026 Oral — challenges fundamental VLM architecture assumptions |
| [OpenVLThinkerV2: Generalist Multimodal Reasoning](https://arxiv.org/abs/2604.08539)               | Generalist multimodal reasoning model                                |


## Long Context


| Paper                                                                                   | Why It Matters                                                                |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [MSA: Memory Sparse Attention Scaling to 100M Tokens](https://arxiv.org/abs/2603.23516) | 3,170+ community votes — shatters long-context barrier by orders of magnitude |


## Interpretability


| Paper                                                                                                   | Why It Matters                                                          |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [Not All Tokens See Equally: Perception-Grounded Policy Optimization](https://arxiv.org/abs/2604.01840) | Token-level importance for VLMs — connects interpretability to training |


---

## Emerging Signals (Watch Closely)

These 2026 trends are still early but could reshape their fields:


| Trend                     | Key Papers                                            | Potential                                            |
| ------------------------- | ----------------------------------------------------- | ---------------------------------------------------- |
| Diffusion Language Models | LangFlow (2604.11748), MoE-FM (2604.15009)            | Could rival autoregressive LLMs                      |
| Self-evolving agents      | Memento-Skills (2603.18743), Autogenesis (2604.15034) | Agents that design agents                            |
| Sub-4-bit training        | FP4 DiT, HiFloat4, BitNet 1.58                        | Training, not just inference, at ultra-low precision |
| World models convergence  | LeWorldModel, VLA world models, video world models    | VLA + video gen + planning unified                   |
| MCP as infrastructure     | MCP-Atlas, MCP security ecosystem                     | Becoming the HTTP of AI tool use                     |


