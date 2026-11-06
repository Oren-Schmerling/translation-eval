# Pragmatic MT Evaluation Report

## Evaluation Framework & Metric Glossary

Before reviewing the results below, refer to this glossary for metric definitions and evaluation tiers:

### Core Metrics

| Metric | Full Name | Definition | Scale / Goal |
|---|---|---|---|
| **chrF** | Character F-score | Reference-based n-gram lexical overlap against gold translations and acceptable synonyms. | 0–100 (Higher is better) |
| **Judge** | LLM Ensemble Score | Average score across independent judges for formality, gender, dialect, and naturalness. | 1.0–5.0 (Higher is better) |
| **ACR %** | Anchor Compliance Rate | Percentage of translations strictly honoring required lexical anchors without forbidden markers. | 0–100% (Higher is better) |
| **POR %** | Parameter Override Rate | Percentage of adversarial conflicts where explicit API parameters override misleading source cues. | 0–100% (Higher is better) |
| **TTFT** | Time To First Token | Latency until first streamed token reaches client. | Milliseconds (Lower is better) |
| **Total Latency** | End-to-End Latency | Total request duration until stream completion or response return. | Milliseconds (Lower is better) |
| **Cost / 1k** | Projected Cost | Projected API expenditure per 1,000 requests under uncached and cached token pricing. | USD ($) |

### Evaluation Tiers

- **Tier 1 (Default Zero-Shot)**: Free translation with no pragmatic instructions; reveals default bias and unguided quality.
- **Tier 2A (Coarse Parameters)**: Explicit formality, speaker/listener gender, and dialect parameters.
- **Tier 2B (Granular Parameters)**: Fine-grained pronouns, kinship pairs, and interpersonal relationship constraints.
- **Tier 2C (Adversarial Conflict)**: Tests whether parameter overrides intentionally contradictory source cues.
- **Tier 3A / 3B / 3C (Few-Shot Adaptation)**: In-context prompts with 5 positive, 20 positive, or 5 contrastive examples.

---

## Table 1: Tier 1 (Default Zero-Shot) Quality & Bias Distribution
*Reveals default register, gender bias, and dialect bias when no pragmatic constraints are supplied.*

| Provider | Lang | n | Mean chrF | Mean Judge | % Male | % Female | % L1 Intimate* | % L2 Polite* | % L3 Honorific* | % Northern† | % Southern† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| google-ai-studio:gemma-4-26b-a4b-it | gu | 1 | 9.557 | 3.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| google-ai-studio:gemma-4-26b-a4b-it | he | 1 | 4.535 | 3.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| google-ai-studio:gemma-4-26b-a4b-it | vi | 1 | 7.122 | 3.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-haiku-5.5:batch | gu | 1 | 58.390 | 3.88 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-haiku-5.5:batch | he | 1 | 65.171 | 4.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-haiku-5.5:batch | vi | 1 | 47.685 | 4.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | gu | 1 | 58.390 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | he | 1 | 71.064 | 4.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | vi | 1 | 47.685 | 4.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:cohere/north-mini-code:free | gu | 1 | 20.897 | 1.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:cohere/north-mini-code:free | he | 1 | 48.061 | 1.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:cohere/north-mini-code:free | vi | 1 | 45.823 | 3.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | gu | 1 | 58.390 | 3.88 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | he | 1 | 65.171 | 4.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | vi | 1 | 59.791 | 4.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:google/gemma-4-31b-it | gu | 1 | 53.794 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:google/gemma-4-31b-it | he | 1 | 71.064 | 4.88 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:google/gemma-4-31b-it | vi | 1 | 47.685 | 4.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | gu | 1 | 40.665 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | he | 1 | 71.064 | 4.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | vi | 1 | 47.547 | 4.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | gu | 1 | 52.020 | 3.62 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | he | 1 | 65.171 | 4.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | vi | 1 | 47.685 | 4.12 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3.5-lightning:free | gu | 1 | 10.394 | 2.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3.5-lightning:free | he | 1 | 47.446 | 3.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3.5-lightning:free | vi | 1 | 47.685 | 3.88 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b | gu | 1 | 28.034 | 3.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b | he | 1 | 46.619 | 4.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b | vi | 1 | 26.548 | 4.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | gu | 1 | 40.697 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | he | 1 | 62.056 | 4.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | vi | 1 | 47.685 | 4.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | gu | 1 | 9.194 | 2.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | he | 1 | 9.469 | 2.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | vi | 1 | 53.586 | 4.38 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | gu | 1 | 40.665 | 3.88 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | he | 1 | 46.619 | 4.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | vi | 1 | 47.685 | 4.00 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | gu | 1 | 37.785 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | he | 1 | 71.064 | 4.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | vi | 1 | 34.695 | 4.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy3:free | gu | 1 | 40.665 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy3:free | he | 1 | 65.171 | 4.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy3:free | vi | 1 | 47.685 | 3.75 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | gu | 1 | 58.390 | 3.62 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | he | 1 | 71.064 | 4.88 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | vi | 1 | 59.791 | 4.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

*\* Gujarati formality registers: Level 1 (intimate), Level 2 (polite), Level 3 (honorific).*  
*† Vietnamese regional dialect lexical markers: Northern vs. Southern.*

## Table 2: Tier 2 (Coarse & Granular Parameters) Compliance (ACR %)
*Compliance rate against explicit parameter instructions, including unsupported or partially supported provider axes.*

| Provider | Tier | n | Compliance % | UnsupportedByProvider | PartiallySupported missing axes |
|---|---|---:|---:|---|---|
| google-ai-studio:gemma-4-26b-a4b-it | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:anthropic/claude-haiku-5.5:batch | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:anthropic/claude-sonnet-5.5:batch | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:cohere/north-mini-code:free | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:deepseek/deepseek-v4.1-flash:batch | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:google/gemma-4-31b-it | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:minimax/minimax-m3.1-flash | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:nvidia/nemotron-3.5-lightning:free | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:qwen/qwen3.8-27b | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:qwen/qwen3.8-flash | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:tencent/hy-mt2-1.8b | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:tencent/hy-mt2-30b-a3b | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:tencent/hy-mt2-7b | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:tencent/hy3:free | Tier 2A: Coarse Parameters | 3 | 0.0% | 0 | 0 |
| openrouter:z-ai/glm-5.3-flash:batch | Tier 2A: Coarse Parameters | 3 | 33.3% | 0 | 0 |

*Note: Missing axes indicate parameters the provider received but did not honor (e.g. dialect).*

## Table 3: Tier 2C (Adversarial Conflict) Parameter Override Rate (POR %)
*Frequency with which explicit parameters take precedence over contradictory cues embedded in source text.*

| Provider | Conflict Requests (n) | Override Hits | POR % |
|---|---:|---:|---:|
| n/a | n/a | n/a | n/a |

## Table 4: Tier 3 (Few-Shot Adaptation) Delta
*Quality improvement gained by supplying in-context demonstration examples (positive and contrastive pairs).*

| Provider | Tier 1 (Default) chrF | Tier 1 (Default) Judge | Tier 3A (5 Pos) chrF | Tier 3A (5 Pos) Judge | Tier 3B (20 Pos) chrF | Tier 3B (20 Pos) Judge | Tier 3C (5 Contrastive) chrF | Tier 3C (5 Contrastive) Judge | Δ chrF 1→3A | Δ Judge 1→3A | Δ chrF 3A→3B | Δ Judge 3A→3B | Δ chrF 3A→3C | Δ Judge 3A→3C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| google-ai-studio:gemma-4-26b-a4b-it | 7.071 | 3.29 | 5.239 | 2.25 | n/a | n/a | n/a | n/a | -1.83 | -1.04 | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-haiku-5.5:batch | 57.082 | 4.21 | 60.195 | 4.54 | n/a | n/a | n/a | n/a | +3.11 | +0.33 | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | 59.046 | 4.29 | 67.261 | 4.62 | n/a | n/a | n/a | n/a | +8.21 | +0.33 | n/a | n/a | n/a | n/a |
| openrouter:cohere/north-mini-code:free | 38.260 | 2.12 | 16.155 | 1.38 | n/a | n/a | n/a | n/a | -22.10 | -0.75 | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | 61.118 | 4.12 | 65.457 | 4.25 | n/a | n/a | n/a | n/a | +4.34 | +0.12 | n/a | n/a | n/a | n/a |
| openrouter:google/gemma-4-31b-it | 57.514 | 4.21 | 54.566 | 4.42 | n/a | n/a | n/a | n/a | -2.95 | +0.21 | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | 53.092 | 4.33 | 56.797 | 4.33 | n/a | n/a | n/a | n/a | +3.71 | +0.00 | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | 54.959 | 4.08 | 60.225 | 4.17 | n/a | n/a | n/a | n/a | +5.27 | +0.08 | n/a | n/a | n/a | n/a |
| openrouter:nvidia/nemotron-3.5-lightning:free | 35.175 | 3.08 | 47.399 | 4.29 | n/a | n/a | n/a | n/a | +12.22 | +1.21 | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b | 33.734 | 4.17 | 63.585 | 3.88 | n/a | n/a | n/a | n/a | +29.85 | -0.29 | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | 50.146 | 4.08 | 71.300 | 4.33 | n/a | n/a | n/a | n/a | +21.15 | +0.25 | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | 24.083 | 3.21 | 16.883 | 2.83 | n/a | n/a | n/a | n/a | -7.20 | -0.38 | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | 44.990 | 4.21 | 64.902 | 4.21 | n/a | n/a | n/a | n/a | +19.91 | +0.00 | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | 47.848 | 4.21 | 50.826 | 4.08 | n/a | n/a | n/a | n/a | +2.98 | -0.12 | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy3:free | 51.174 | 4.04 | 59.726 | 4.33 | n/a | n/a | n/a | n/a | +8.55 | +0.29 | n/a | n/a | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | 63.082 | 4.33 | 53.812 | 4.12 | n/a | n/a | n/a | n/a | -9.27 | -0.21 | n/a | n/a | n/a | n/a |

*Note: Δ columns measure relative improvement; positive delta indicates benefit from prompt demonstrations.*

## Table 5: Operational Telemetry
*Latency distributions and operational expenditure projections per 1,000 translation queries.*

| Provider | n | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | p95 Total (ms) | Mean Uncached Cost | Mean Cached Cost | Uncached / 1k | Cached / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| google-ai-studio:gemma-4-26b-a4b-it | 9 | 1014.03 | 1110.44 | 19128.91 | 22238.45 | $0.000161 | $0.000155 | $0.161067 | $0.154804 |
| openrouter:anthropic/claude-haiku-5.5:batch | 9 | 601.99 | 681.45 | 788.00 | 916.76 | $0.000020 | $0.000016 | $0.019617 | $0.015522 |
| openrouter:anthropic/claude-sonnet-5.5:batch | 9 | 665.13 | 2973.34 | 1172.79 | 3209.04 | $0.000651 | $0.000569 | $0.651111 | $0.569011 |
| openrouter:cohere/north-mini-code:free | 9 | 2486.88 | 7071.21 | 2486.88 | 7071.21 | $0.000000 | $0.000000 | $0.000000 | $0.000000 |
| openrouter:deepseek/deepseek-v4.1-flash:batch | 9 | 544.30 | 930.41 | 689.53 | 1175.33 | $0.000027 | $0.000021 | $0.027166 | $0.020771 |
| openrouter:google/gemma-4-31b-it | 9 | 321.86 | 516.00 | 508.93 | 780.21 | $0.000040 | $0.000030 | $0.040311 | $0.029651 |
| openrouter:minimax/minimax-m3.1-flash | 9 | 2956.74 | 4777.50 | 2960.92 | 4791.49 | $0.000231 | $0.000215 | $0.230683 | $0.215308 |
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | 9 | 1274.64 | 2315.79 | 1274.64 | 2315.79 | $0.000000 | $0.000000 | $0.000000 | $0.000000 |
| openrouter:nvidia/nemotron-3.5-lightning:free | 9 | 537.56 | 630.39 | 537.56 | 630.39 | $0.000000 | $0.000000 | $0.000000 | $0.000000 |
| openrouter:qwen/qwen3.8-27b | 9 | 528.17 | 1334.92 | 1051.79 | 31576.87 | $0.000058 | $0.000049 | $0.058042 | $0.049267 |
| openrouter:qwen/qwen3.8-flash | 9 | 913.56 | 6682.73 | 975.36 | 6887.07 | $0.000035 | $0.000026 | $0.034849 | $0.026269 |
| openrouter:tencent/hy-mt2-1.8b | 9 | 993.47 | 1082.97 | 1332.82 | 3150.20 | $0.000050 | $0.000047 | $0.049810 | $0.047060 |
| openrouter:tencent/hy-mt2-30b-a3b | 9 | 932.63 | 1201.99 | 1123.16 | 1523.21 | $0.000023 | $0.000018 | $0.022565 | $0.017829 |
| openrouter:tencent/hy-mt2-7b | 9 | 906.29 | 1030.25 | 1205.38 | 1356.34 | $0.000025 | $0.000020 | $0.025133 | $0.019590 |
| openrouter:tencent/hy3:free | 9 | 2202.40 | 3068.28 | 2202.40 | 3068.28 | $0.000000 | $0.000000 | $0.000000 | $0.000000 |
| openrouter:z-ai/glm-5.3-flash:batch | 9 | 3286.29 | 14838.03 | 3608.82 | 15115.95 | $0.000087 | $0.000084 | $0.087191 | $0.083531 |

*Note: TTFT applies to streaming LLMs; Cost/1k shows projected spend under standard vs cached token rates.*

