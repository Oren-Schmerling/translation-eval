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
| openrouter:anthropic/claude-haiku-5.5:batch | gu | 60 | 39.420 | 4.40 | 20.0% | 80.0% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:anthropic/claude-haiku-5.5:batch | he | 60 | 45.909 | 4.37 | 76.9% | 23.1% | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-haiku-5.5:batch | vi | 60 | 49.833 | 4.23 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | gu | 60 | 37.988 | 4.42 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | he | 60 | 47.174 | 4.41 | 100.0% | 0.0% | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | vi | 60 | 49.696 | 4.21 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | gu | 60 | 39.578 | 4.44 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | he | 60 | 48.669 | 4.38 | 84.6% | 15.4% | n/a | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | vi | 60 | 50.351 | 4.26 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | gu | 60 | 35.455 | 4.17 | 16.7% | 83.3% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | he | 60 | 44.663 | 4.16 | 100.0% | 0.0% | n/a | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | vi | 60 | 47.488 | 4.25 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b:batch | gu | 60 | 32.333 | 3.79 | 20.0% | 80.0% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:qwen/qwen3.8-27b:batch | he | 60 | 40.652 | 4.13 | 83.3% | 16.7% | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b:batch | vi | 60 | 47.573 | 4.19 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | gu | 60 | 36.333 | 4.29 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | he | 60 | 45.987 | 4.32 | 100.0% | 0.0% | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | vi | 60 | 49.318 | 4.20 | n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| openrouter:tencent/hy-mt2-1.8b | gu | 60 | 9.576 | 1.82 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | he | 60 | 19.689 | 2.21 | 100.0% | 0.0% | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | vi | 60 | 40.294 | 3.68 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | gu | 60 | 39.490 | 4.35 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | he | 60 | 44.428 | 4.43 | 56.2% | 43.8% | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | vi | 60 | 44.104 | 4.19 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | gu | 60 | 41.818 | 4.39 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | he | 60 | 47.858 | 4.39 | 31.2% | 68.8% | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | vi | 60 | 44.699 | 4.12 | n/a | n/a | n/a | n/a | n/a | 75.0% | 25.0% |
| openrouter:z-ai/glm-5.3-flash:batch | gu | 60 | 24.106 | 3.36 | 20.0% | 80.0% | 0.0% | 100.0% | 0.0% | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | he | 60 | 45.508 | 4.32 | 92.3% | 7.7% | n/a | n/a | n/a | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | vi | 60 | 51.050 | 4.30 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

*\* Gujarati formality registers: Level 1 (intimate), Level 2 (polite), Level 3 (honorific).*  
*† Vietnamese regional dialect lexical markers: Northern vs. Southern.*

## Table 2: Tier 2 (Coarse & Granular Parameters) Compliance (ACR %)
*Compliance rate against explicit parameter instructions, including unsupported or partially supported provider axes.*

| Provider | Tier | n | Compliance % | UnsupportedByProvider | PartiallySupported missing axes |
|---|---|---:|---:|---|---|
| n/a | n/a | n/a | n/a | n/a | n/a |

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
| openrouter:anthropic/claude-haiku-5.5:batch | 45.054 | 4.33 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:anthropic/claude-sonnet-5.5:batch | 44.953 | 4.34 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:deepseek/deepseek-v4.1-flash:batch | 46.200 | 4.36 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:minimax/minimax-m3.1-flash | 42.535 | 4.19 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-27b:batch | 40.186 | 4.04 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:qwen/qwen3.8-flash | 43.879 | 4.27 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-1.8b | 23.186 | 2.57 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-30b-a3b | 42.674 | 4.33 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:tencent/hy-mt2-7b | 44.792 | 4.30 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| openrouter:z-ai/glm-5.3-flash:batch | 40.221 | 3.99 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

*Note: Δ columns measure relative improvement; positive delta indicates benefit from prompt demonstrations.*

## Table 5: Operational Telemetry
*Latency distributions and operational expenditure projections per 1,000 translation queries.*

| Provider | n | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | p95 Total (ms) | Mean Uncached Cost | Mean Cached Cost | Uncached / 1k | Cached / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openrouter:anthropic/claude-haiku-5.5:batch | 180 | 447.27 | 557.66 | 628.16 | 753.95 | $0.000013 | $0.000013 | $0.013185 | $0.013185 |
| openrouter:anthropic/claude-sonnet-5.5:batch | 180 | 629.77 | 1385.86 | 947.09 | 1669.98 | $0.000290 | $0.000290 | $0.289500 | $0.289500 |
| openrouter:deepseek/deepseek-v4.1-flash:batch | 180 | 612.70 | 2053.00 | 869.53 | 2396.00 | $0.000016 | $0.000016 | $0.015818 | $0.015818 |
| openrouter:minimax/minimax-m3.1-flash | 180 | 1244.87 | 3334.44 | 1252.79 | 3337.56 | $0.000161 | $0.000161 | $0.160572 | $0.160572 |
| openrouter:qwen/qwen3.8-27b:batch | 180 | 693.64 | 2382.80 | 1059.69 | 2640.26 | $0.000050 | $0.000050 | $0.050137 | $0.050137 |
| openrouter:qwen/qwen3.8-flash | 180 | 740.42 | 5707.06 | 993.78 | 5811.77 | $0.000021 | $0.000021 | $0.021465 | $0.021465 |
| openrouter:tencent/hy-mt2-1.8b | 180 | 852.14 | 1026.82 | 1033.64 | 1363.05 | $0.000013 | $0.000013 | $0.012530 | $0.012530 |
| openrouter:tencent/hy-mt2-30b-a3b | 180 | 1333.42 | 1986.59 | 1591.84 | 2570.45 | $0.000015 | $0.000015 | $0.014702 | $0.014702 |
| openrouter:tencent/hy-mt2-7b | 180 | 1321.67 | 4641.44 | 1648.99 | 4980.27 | $0.000017 | $0.000017 | $0.016823 | $0.016823 |
| openrouter:z-ai/glm-5.3-flash:batch | 180 | 3159.01 | 6374.26 | 3205.89 | 6407.24 | $0.000099 | $0.000099 | $0.099129 | $0.099129 |

*Note: TTFT applies to streaming LLMs; Cost/1k shows projected spend under standard vs cached token rates.*

