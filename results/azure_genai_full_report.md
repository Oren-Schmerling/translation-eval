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
| azure-genai-translator | gu | 288 | 36.132 | n/a | 20.4% | 79.6% | 52.8% | 47.2% | 0.0% | n/a | n/a |
| azure-genai-translator | he | 192 | 47.741 | n/a | 83.3% | 16.7% | n/a | n/a | n/a | n/a | n/a |
| azure-genai-translator | vi | 240 | 40.803 | n/a | 52.6% | 47.4% | n/a | n/a | n/a | 80.6% | 19.4% |

*\* Gujarati formality registers: Level 1 (intimate), Level 2 (polite), Level 3 (honorific).*  
*† Vietnamese regional dialect lexical markers: Northern vs. Southern.*

## Table 2: Tier 2 (Coarse & Granular Parameters) Compliance (ACR %)
*Compliance rate against explicit parameter instructions, including unsupported or partially supported provider axes.*

| Provider | Tier | n | Compliance % | UnsupportedByProvider | PartiallySupported missing axes |
|---|---|---:|---:|---|---|
| azure-genai-translator | Tier 2A: Coarse Parameters | 720 | 24.3% | 0 | 240 (dialect) |

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
| azure-genai-translator | 40.784 | n/a | 47.291 | n/a | n/a | n/a | n/a | n/a | +6.51 | n/a | n/a | n/a | n/a | n/a |

*Note: Δ columns measure relative improvement; positive delta indicates benefit from prompt demonstrations.*

## Table 5: Operational Telemetry
*Latency distributions and operational expenditure projections per 1,000 translation queries.*

| Provider | n | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | p95 Total (ms) | Mean Uncached Cost | Mean Cached Cost | Uncached / 1k | Cached / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| azure-genai-translator | 2160 | n/a | n/a | 1179.17 | 1878.72 | $0.000367 | $0.000367 | $0.367384 | $0.367384 |

*Note: TTFT applies to streaming LLMs; Cost/1k shows projected spend under standard vs cached token rates.*

