# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A curated collection of ~1,420 AI/ML research paper references organized by topic, focused on 2023-2026 work with key landmark papers. The primary content is 17 topic-specific markdown files containing arxiv URLs with brief descriptions in markdown tables.

## Repository Structure

- **`*-arxiv-urls.md`** — The 17 topic files. Each has a header with paper count/date, then sections of `| URL | Description |` tables. Landmark papers are marked with 开山.
- **`README-index.md`** — Master index with paper counts, cross-topic map, gap analysis, hot topic clusters.
- **`ROADMAP.md`** — 3-tier reading path per topic (Foundations → Core Advances → Frontier) with dependency graph.
- **`CROSS-REF.md`** — Papers appearing in multiple topic files, organized by cross-cutting theme.
- **`TIMELINE.md`** — Intellectual lineage from 2013 to 2026.
- **`MUST-READ-2026.md`** — ~30 papers defining the 2026 frontier.
- **`search-papers.sh`** — CLI search tool: `./search-papers.sh <keyword|arxiv-id|regex>`
- **`papers/`** — Downloaded PDFs (named by arxiv ID).
- **`vae/`** — Downloaded PDFs for VAE/representation learning papers.
- **`notes/<topic>/`** — Paper reading notes in Chinese, organized by topic. Named by arxiv ID. Topics match the 17 `*-arxiv-urls.md` names. Orphaned notes go to `notes/_unsorted/`.
- **`landmark-experiments/<topic>/`** — Minimal experiment reproductions, organized by topic. Each topic has `NN_name.py` scripts and `results/NN-name/` output dirs. Shared `data/` and `pyproject.toml` at the `landmark-experiments/` level.
- **`landmark-experiments/vae-representation-learning/vae-experiments/`** — Separate VAE-focused experiment suite with its own `pyproject.toml`.

## Key Conventions

- Arxiv IDs must be verified with `curl -sI https://arxiv.org/abs/XXXX.XXXXX` — do NOT use WebFetch or subagents for verification, as they produce false positives.
- Paper descriptions in tables follow the format: `Title (Authors, Year) — optional notes`. Landmark papers get 开山 marker.
- When adding papers, check for cross-topic overlap and add to all relevant files plus CROSS-REF.md if a paper spans 2+ topics.
- Notes are written in Chinese (中文).

## Search

```bash
./search-papers.sh mamba          # keyword
./search-papers.sh 2604.14191     # exact arxiv ID
./search-papers.sh 开山           # all landmark papers
./search-papers.sh "2026.*quant"  # regex
```

Or directly: `grep -rn "abs/ID" *-arxiv-urls.md`

## Topic Files (17)

1. ai-infra-operator-optimization — vLLM, speculative decoding, RL training infra
2. ai-for-science — PDE solvers, AlphaFold, molecular design
3. diffusion — DDPM, DiT, flow matching, consistency models
4. vae-representation-learning — VAE, VQ-VAE, SSL (SimCLR/DINO/MAE), normalizing flows
5. new-architectures — SSM/Mamba, RWKV, xLSTM, KAN, MoE, TTT
6. ai-safety-interpretability — red teaming, SAEs, watermarking
7. generative-models — image/video/3D/audio generation
8. ai-agent-rag-vla — agents, tool use, RAG, VLA, MCP
9. multimodal-vlm — CLIP, LLaVA, vision-language models
10. long-context-data-curation — FlashAttention, KV cache, data curation
11. efficient-finetuning-merging — LoRA variants, model merging, distillation
12. ai-compiler-hardware-codesign — TVM, NPU kernels, Triton
13. llm-reasoning-rl-alignment — RLHF/DPO, CoT, DeepSeek-R1, GRPO
14. speech-audio-language-models — Whisper, EnCodec, speech LLMs
15. robotics-foundation-models — RT-1/2, Diffusion Policy, π0
16. ai-math-theorem-proving — LeanDojo, DeepSeek-Prover
17. ai-ethics-fairness-governance — fairness, bias, agentic governance
