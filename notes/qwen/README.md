# Qwen 系列论文读书笔记

从 4090 集群 `claude-code-rev-1/doc/notes/` 同步，按类别整理。

## 目录结构

```
notes/
├── base-llm/        Qwen / Qwen2 / Qwen2.5 / Qwen3 技术报告
├── vlm/             Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL
├── audio/           Qwen-Audio / Qwen2-Audio / Qwen3-ASR / Qwen3-TTS
├── math/            Qwen2.5-Math
├── coder/           Qwen2.5-Coder / Qwen3-Coder-Next
├── omni/            Qwen2.5-Omni / Qwen3-Omni / Qwen3.5-Omni
├── long-context/    Qwen2.5-1M
├── image/           Qwen-Image
└── embedding/       Qwen3 Embedding / Qwen3-VL-Embedding-Reranker
```

## 基础大语言模型

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen Technical Report | 2309.16609 | [Qwen_Technical_Report.md](base-llm/Qwen_Technical_Report.md) | 2023-09 |
| Qwen2 Technical Report | 2407.10671 | [Qwen2_Technical_Report.md](base-llm/Qwen2_Technical_Report.md) | 2024-07 |
| Qwen2.5 Technical Report | 2412.15115 | [Qwen2.5_Technical_Report.md](base-llm/Qwen2.5_Technical_Report.md) | 2024-12 |
| Qwen3 Technical Report | 2505.09388 | [Qwen3_Technical_Report.md](base-llm/Qwen3_Technical_Report.md) | 2025-05 |

> 注：Qwen1.5 没有独立论文，通过博客发布。

## 视觉-语言模型

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen-VL | 2308.12966 | [Qwen-VL.md](vlm/Qwen-VL.md) | 2023-08 |
| Qwen2-VL | 2409.12191 | [Qwen2-VL.md](vlm/Qwen2-VL.md) | 2024-09 |
| Qwen2.5-VL | 2502.13923 | [Qwen2.5-VL.md](vlm/Qwen2.5-VL.md) | 2025-02 |
| Qwen3-VL | 2511.21631 | [Qwen3-VL.md](vlm/Qwen3-VL.md) | 2025-11 |

## 音频模型

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen-Audio | 2311.07919 | [Qwen-Audio.md](audio/Qwen-Audio.md) | 2023-11 |
| Qwen2-Audio | 2407.10759 | [Qwen2-Audio.md](audio/Qwen2-Audio.md) | 2024-07 |
| Qwen3-ASR | 2601.21337 | [Qwen3-ASR.md](audio/Qwen3-ASR.md) | 2026-01 |
| Qwen3-TTS | 2601.15621 | [Qwen3-TTS.md](audio/Qwen3-TTS.md) | 2026-01 |

## 数学模型

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen2.5-Math | 2409.12122 | [Qwen2.5-Math.md](math/Qwen2.5-Math.md) | 2024-09 |

## 代码模型

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen2.5-Coder | 2409.12186 | [Qwen2.5-Coder.md](coder/Qwen2.5-Coder.md) | 2024-09 |
| Qwen3-Coder-Next | 2603.00729 | [Qwen3-Coder-Next.md](coder/Qwen3-Coder-Next.md) | 2026-02 |

## 全模态（Omni）模型

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen2.5-Omni | 2503.20215 | [Qwen2.5-Omni.md](omni/Qwen2.5-Omni.md) | 2025-03 |
| Qwen3-Omni | 2509.17765 | [Qwen3-Omni.md](omni/Qwen3-Omni.md) | 2025-09 |
| Qwen3.5-Omni | 2604.15804 | [Qwen3.5-Omni.md](omni/Qwen3.5-Omni.md) | 2026-04 |

## 长上下文

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen2.5-1M | 2501.15383 | [Qwen2.5-1M.md](long-context/Qwen2.5-1M.md) | 2025-01 |

## 图像生成/编辑

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen-Image | 2508.02324 | [Qwen-Image.md](image/Qwen-Image.md) | 2025-08 |

## 嵌入与检索

| 论文 | arXiv ID | 笔记 | 发表时间 |
|------|----------|------|----------|
| Qwen3 Embedding | 2506.05176 | [Qwen3_Embedding.md](embedding/Qwen3_Embedding.md) | 2025-06 |
| Qwen3-VL-Embedding & Reranker | 2601.04720 | [Qwen3-VL-Embedding-Reranker.md](embedding/Qwen3-VL-Embedding-Reranker.md) | 2026-01 |

---

## Qwen 系列发展时间线

```
2023-08  Qwen-VL          --- 首个视觉语言模型
2023-09  Qwen              --- 首个基础语言模型（7B/14B）
2023-11  Qwen-Audio        --- 首个音频理解模型
2024-07  Qwen2             --- 第二代基础模型（0.5B~72B），GQA+SwiGLU
2024-07  Qwen2-Audio       --- 第二代音频模型
2024-09  Qwen2-VL          --- 任意分辨率视觉理解，MRoPE
2024-09  Qwen2.5-Math      --- 数学专用模型，自改进训练
2024-09  Qwen2.5-Coder     --- 代码专用模型
2024-12  Qwen2.5           --- 第三代基础模型，Dense+MoE全系列
2025-01  Qwen2.5-1M        --- 百万token长上下文
2025-02  Qwen2.5-VL        --- 增强视觉语言，文档/视频理解
2025-03  Qwen2.5-Omni      --- 全模态统一模型（文本+视觉+音频）
2025-05  Qwen3             --- 思考/非思考模式统一，8个模型全开源
2025-06  Qwen3 Embedding   --- 文本嵌入与重排序
2025-08  Qwen-Image        --- 图像生成与编辑
2025-09  Qwen3-Omni        --- 第三代全模态
2025-11  Qwen3-VL          --- 第三代视觉语言
2026-01  Qwen3-ASR         --- 语音识别
2026-01  Qwen3-TTS         --- 语音合成
2026-01  Qwen3-VL-Embed    --- 多模态检索与排序
2026-02  Qwen3-Coder-Next  --- 下一代代码模型
2026-04  Qwen3.5-Omni      --- 最新全模态模型
```

## 没有独立论文的 Qwen 变体

- **Qwen1.5** — Qwen2 技术报告的前代，博客发布
- **Qwen2-Math** — 被 Qwen2.5-Math 取代，博客发布
- **Qwen3-Math** — 数学能力在 Qwen3 技术报告中涵盖
- **Qwen-Agent** — 开源工具仓库，无独立论文
