# Pragmatic MT Evaluation Report

## Table 1: Tier 1 (Default Zero-Shot) Quality & Bias Distribution

| Provider | Lang | n | Mean chrF | Mean Judge | % Male | % Female | % L1 Intimate | % L2 Polite | % L3 Honorific | % Northern | % Southern |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| azure-genai-translator | gu | 288 | 36.132 | n/a | 20.4% | 79.6% | 52.8% | 47.2% | 0.0% | n/a | n/a |
| azure-genai-translator | he | 192 | 47.741 | n/a | 83.3% | 16.7% | n/a | n/a | n/a | n/a | n/a |
| azure-genai-translator | vi | 240 | 40.803 | n/a | 52.6% | 47.4% | n/a | n/a | n/a | 80.6% | 19.4% |

## Table 2: Tier 2 (Coarse & Granular Parameters) Compliance (ACR %)

| Provider | Tier | n | Compliance % | UnsupportedByProvider | PartiallySupported missing axes |
|---|---|---:|---:|---|---|
| azure-genai-translator | Tier 2A: Coarse Parameters | 720 | 24.3% | 0 | 240 (dialect) |

## Table 3: Tier 2C (Adversarial Conflict) Parameter Override Rate (POR %)

| Provider | Conflict Requests (n) | Override Hits | POR % |
|---|---:|---:|---:|
| n/a | n/a | n/a | n/a |

## Table 4: Tier 3 (Few-Shot Adaptation) Delta

| Provider | Tier 1 (Default) chrF | Tier 1 (Default) Judge | Tier 3A (5 Pos) chrF | Tier 3A (5 Pos) Judge | Tier 3B (20 Pos) chrF | Tier 3B (20 Pos) Judge | Tier 3C (5 Contrastive) chrF | Tier 3C (5 Contrastive) Judge | Δ chrF 1→3A | Δ Judge 1→3A | Δ chrF 3A→3B | Δ Judge 3A→3B | Δ chrF 3A→3C | Δ Judge 3A→3C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| azure-genai-translator | 40.784 | n/a | 47.291 | n/a | n/a | n/a | n/a | n/a | +6.51 | n/a | n/a | n/a | n/a | n/a |

## Table 5: Operational Telemetry

| Provider | n | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | p95 Total (ms) | Mean Uncached Cost | Mean Cached Cost | Uncached / 1k | Cached / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| azure-genai-translator | 2160 | n/a | n/a | 1179.17 | 1878.72 | $0.000367 | $0.000367 | $0.367384 | $0.367384 |

