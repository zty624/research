# AI/ML Research Paper Collection — Master Index

> Curated on 2026-04-19. **1,429 unique papers** (1,544 total references) across 17 topic files.
> Focus: 2023–2026 recent work + key landmarks.
>
> **New:** [Reading Roadmap](docs/ROADMAP.md) — suggested reading order by topic with dependency graph
> **New:** [Cross-Topic Reference](docs/CROSS-REF.md) — papers appearing in multiple files, organized by theme
> **New:** `./search-papers.sh <query>` — CLI search by keyword, arxiv ID, or regex
> **New:** [2026 Must-Read Papers](docs/MUST-READ-2026.md) — ~30 papers that define the 2026 frontier
> **New:** [Research Timeline](docs/TIMELINE.md) — intellectual lineage from VAE (2013) to agentic AI (2026)

---

## Topic Files

| # | Topic | Papers | Sections | 2025-26 % | Key Landmarks |
|---|-------|--------|----------|-----------|---------------|
| 1 | [AI Infra & Operator Optimization](ai-infra-operator-optimization-arxiv-urls.md) | 151 | 9 | 89% | vLLM, Speculative Decoding, veRL |
| 2 | [AI for Science](ai-for-science-arxiv-urls.md) | 143 | 8 | 82% | DeepONet, FNO, PINN |
| 3 | [Diffusion Models](diffusion-arxiv-urls.md) | 154 | 11 | 60% | DiT, Flow Matching, Consistency Models |
| 4 | [VAE & Representation Learning](vae-representation-learning-arxiv-urls.md) | 117 | 13 | 22% | VAE, VQ-VAE, VQGAN, SimCLR, DINO, MAE, Glow, Neural ODE |
| 5 | [New Architectures (SSM/RWKV/xLSTM/KAN/MoE/TTT)](new-architectures-arxiv-urls.md) | 108 | 8 | 66% | S4, Mamba, TTT, RWKV, xLSTM, KAN, Mixtral |
| 6 | [AI Safety, Alignment & Interpretability](ai-safety-interpretability-arxiv-urls.md) | 121 | 8 | 78% | Red Teaming, Constitutional AI |
| 7 | [Generative Models (Image/Video/3D/Audio)](generative-models-arxiv-urls.md) | 84 | 8 | 93% | — |
| 8 | [AI Agent, Tool Use, RAG & VLA](ai-agent-rag-vla-arxiv-urls.md) | 105 | 9 | 92% | HuggingGPT, Toolformer, RAG, RT-2, MCP |
| 9 | [Multimodal VLM](multimodal-vlm-arxiv-urls.md) | 92 | 8 | 89% | CLIP, LLaVA |
| 10 | [Long Context, Data Curation & Retrieval](long-context-data-curation-arxiv-urls.md) | 85 | 7 | 97% | FlashAttention |
| 11 | [Efficient Fine-tuning, LoRA, Merging & Distillation](efficient-finetuning-merging-arxiv-urls.md) | 81 | 7 | 98% | LoRA |
| 12 | [AI Compiler, Hardware & Co-Design](ai-compiler-hardware-codesign-arxiv-urls.md) | 77 | 7 | 83% | TVM, Ansor, Relax |
| 13 | [LLM Reasoning, RL & Alignment](llm-reasoning-rl-alignment-arxiv-urls.md) | 87 | 10 | 87% | RLHF/DPO, DeepSeek-R1, GRPO, CoT |
| 14 | [Speech & Audio Language Models](speech-audio-language-models-arxiv-urls.md) | 73 | 7 | 85% | Whisper, Encodec, VALL-E |
| 15 | [Robotics Foundation Models & Embodied AI](robotics-foundation-models-arxiv-urls.md) | 22 | 5 | 56% | RT-1/2, Diffusion Policy, π0 |
| 16 | [AI for Math & Theorem Proving](ai-math-theorem-proving-arxiv-urls.md) | 25 | 4 | 48% | LeanDojo, DeepSeek-Prover |
| 17 | [AI Ethics, Fairness & Governance](ai-ethics-fairness-governance-arxiv-urls.md) | 19 | 5 | 37% | — |

---

## Year Distribution

| Period | Count | % |
|--------|-------|---|
| <=2020 | 46 | 3.2% |
| 2021–22 | 56 | 4.0% |
| 2023 | 69 | 4.9% |
| 2024 | 159 | 11.2% |
| 2025 | 434 | 30.6% |
| 2026 | 652 | 46.0% |

---

## Cross-Topic Map

Papers often span multiple topics. Major overlaps:

```
AI Compiler ◄──20──► AI Infra          (kernel/NPU/serving overlap)
Diffusion    ◄──11──► Generative       (video/audio/3D generation)
Diffusion    ◄──10──► VAE              (foundational models: DDPM, LDM, VQ-VAE)
AI Science   ◄──7──►  Math Proving     (theorem proving + scientific reasoning)
AI Ethics    ◄──7──►  Safety           (governance + agentic safety)
AI Infra     ◄──6──► Efficient FT      (LoRA serving, quantization, pruning)
Agent/RAG    ◄──5──► Multimodal VLM    (VLA, embodied AI)
Efficient FT ◄──4──► New Architectures (PEFT for SSM/MoE)
Agent/RAG    ◄──4──► Long Context      (RAG + memory management)
Agent/RAG    ◄──3──► LLM Reasoning     (agent reasoning/RL)
Generative   ◄──3──► Speech            (TTS/audio generation)
```

115 papers appear in 2+ files (8.1% overlap rate).

---

## Notable Gaps & Suggestions

1. ~~**AI Ethics & Fairness**~~ — **ADDED**
2. ~~**Robotics & Embodied AI**~~ — **ADDED** (now includes 2026 VLA papers)
3. ~~**AI for Math / Theorem Proving**~~ — **ADDED**
4. ~~**VAE file lacks 开山 markers**~~ — **FIXED**
5. **Dataset & Benchmark Methodology** — 48 benchmark papers exist but methodology-focused papers (not just new benchmarks) are still sparse
6. **Generative vs Diffusion boundary** — 11 shared papers; could sharpen delineation
7. ~~**Compiler/Hardware lacks 开山**~~ — **FIXED**

## 2026 Hot Topic Clusters

| Cluster | Papers | Files | Trend |
|---------|--------|-------|-------|
| LoRA Variants | ~34 | efficient-ft+infra | Fragmenting into specialized methods |
| MoE Optimization | ~25 | new-arch+infra+generative | Shifted from architecture to serving |
| World Models (VLA+Video+Planning) | ~21 | 4 files | Strongest cross-cutting trend |
| SAEs for Interpretability | ~21 | safety+efficient-ft | Default tool for mechanistic interpretability |
| KV Cache Compression | ~19 | infra+long-context | Densest cluster; driven by long-context reasoning |
| Speculative Decoding | ~20 | infra+multimodal | Primary LLM inference speedup strategy |
| GRPO / DeepSeek Reasoning RL | ~14 | reasoning+math | Direct descendants of DeepSeek-R1 |
| MCP Ecosystem | ~15 | agent+safety+long-context | Becoming the HTTP of AI tool use |
| Diffusion Distillation | ~12 | diffusion | Consistency models → step distillation |
| Agent Guardrails / Agentic Safety | ~11 | 3 files | Safety community pivoting from chatbot→agent |
| Gaussian Splatting | ~7 | generative | 3D representation standard for generation |
| Diffusion Language Models | ~16 | diffusion+new-arch+safety | Emerging paradigm; could rival autoregressive |
| Test-Time Compute | ~6 | reasoning+long-context | Scaling inference instead of training |
| KAN Variants | ~18 | new-arch+science | Already has quantization & hardware papers |
| Sub-4-bit Quantization (FP4/INT2) | ~7 | 4 files | Active across LLMs, diffusion, KANs |
| RL Training Infrastructure | ~7 | infra+reasoning | veRL/OpenRLHF ecosystem; rollout serving |

## Still Underrepresented

- **MCP Ecosystem** — ~~only 4 papers~~ **ADDED** (15 papers: security, agents, benchmarks, domain apps)
- **RL Infrastructure for Reasoning** — ~~only ProRL Agent~~ **ADDED** (8 papers: veRL, OpenRLHF, SortedRL, SparrowRL, FlexMARL, SUPO, VerlTool, NExt)
- **Data Curation for Reasoning Training** — **ADDED** (2 papers in new Data Synthesis section + VersaPRM; 17 synthetic data papers total across collection)
- ~~**AI Ethics 2026**~~ — **ADDED** (5 papers in section 5: agentic governance, ClawSafety, scheming monitoring, SoK, vulnerability loops)

**Current gaps:**
- **On-Device / Edge AI** — scattered across files; no dedicated section
- **Multilingual / Low-Resource NLP** — sparse representation outside speech file

---

## Quick Reference: All 58 Landmark (开山) Papers

| Paper | Year | Topic File |
|-------|------|------------|
| GAN | 2014 | vae-representation-learning |
| Transformer | 2017 | new-architectures |
| BERT | 2018 | llm-reasoning-rl-alignment |
| S4: Structured State Spaces | 2021 | new-architectures |
| Mamba: Selective State Spaces | 2023 | new-architectures |
| TTT: Learning at Test Time | 2024 | new-architectures |
| RWKV | 2023 | new-architectures |
| xLSTM | 2024 | new-architectures |
| KAN: Kolmogorov-Arnold Networks | 2024 | new-architectures |
| Mixtral of Experts | 2024 | new-architectures |
| InstructGPT (RLHF) | 2022 | llm-reasoning-rl-alignment |
| DPO | 2023 | llm-reasoning-rl-alignment |
| DeepSeek-R1 | 2024 | llm-reasoning-rl-alignment |
| DeepSeekMath / GRPO | 2024 | llm-reasoning-rl-alignment |
| PRM: Process Reward Models | — | llm-reasoning-rl-alignment |
| Chain-of-Thought | 2022 | llm-reasoning-rl-alignment |
| DeepONet / FNO / PINN | 2020-22 | ai-for-science |
| FNO: Fourier Neural Operator | 2020 | ai-for-science |
| Molecular Generation | — | ai-for-science |
| HuggingGPT | 2023 | ai-agent-rag-vla |
| Toolformer | 2023 | ai-agent-rag-vla |
| RAG | 2020 | ai-agent-rag-vla |
| RT-2 (VLA) | 2023 | ai-agent-rag-vla |
| DiT: Diffusion Transformer | 2022 | diffusion |
| Flow Matching | 2022 | diffusion |
| Rectified Flow | 2022 | diffusion |
| Consistency Models | 2023 | diffusion |
| SEDD: Discrete Diffusion Language Models | 2023 | diffusion |
| Whisper | 2022 | speech-audio-language-models |
| EnCodec | 2022 | speech-audio-language-models |
| VALL-E | 2023 | speech-audio-language-models |
| LoRA | 2021 | efficient-finetuning-merging |
| CLIP | 2021 | multimodal-vlm |
| LLaVA | 2023 | multimodal-vlm |
| vLLM / PagedAttention | 2023 | ai-infra-operator-optimization |
| Speculative Decoding | — | ai-infra-operator-optimization |
| HybridFlow / veRL (RLHF Framework) | 2024 | ai-infra-operator-optimization |
| Causal Abstraction (MI) | — | ai-safety-interpretability |
| Classifier-Free Diffusion Guidance | 2022 | diffusion |
| RT-1 / RT-2 | 2022-23 | robotics-foundation-models |
| Diffusion Policy | 2023 | robotics-foundation-models |
| LeanDojo | 2023 | ai-math-theorem-proving |
| DeepSeek-Prover | 2024 | ai-math-theorem-proving |
| VAE (Kingma & Welling) | 2013 | vae-representation-learning |
| VQ-VAE | 2017 | vae-representation-learning |
| VQGAN | 2020 | vae-representation-learning |
| SimCLR | 2020 | vae-representation-learning |
| DINO | 2021 | vae-representation-learning |
| MAE | 2021 | vae-representation-learning |
| Glow (Normalizing Flow) | 2018 | vae-representation-learning |
| Neural ODE | 2018 | vae-representation-learning |
| TVM | 2018 | ai-compiler-hardware-codesign |
| Ansor | 2020 | ai-compiler-hardware-codesign |
| Relax | 2023 | ai-compiler-hardware-codesign |
| FlashAttention | 2022 | long-context-data-curation |
| Chinchilla (Scaling Laws) | 2022 | long-context-data-curation |
| I-JEPA | 2023 | vae-representation-learning |
