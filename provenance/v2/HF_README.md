---
language:
- he
- gu
- vi
license: mit
task_categories:
- translation
- text-generation
pretty_name: Pragmatic MT Benchmark v2 (Hebrew, Gujarati, Vietnamese)
size_categories:
- n<1K
tags:
- machine-translation
- pragmatic-meaning
- casual-conversational
- formality
- dialect
- anti-translationese
configs:
- config_name: default
  data_files:
    - split: train
      path: dataset_v2.jsonl
- config_name: v2
  data_files:
    - split: train
      path: dataset_v2.jsonl
- config_name: v1
  data_files:
    - split: train
      path: variants.jsonl
---

# Pragmatic MT Benchmark v2: Hebrew, Gujarati, and Vietnamese

## 1. Design Thesis

Standard machine translation benchmarks—such as `IWSLT` (rehearsed TED monologues), `Tatoeba` (isolated textbook grammar sentences), and `WinoMT` (occupational gender probes)—suffer from two structural blindspots:
1. **Translationese and Formal Skew**: Upstream parallel corpora are dominated by scripted speeches, UN proceedings, parliamentary records, and software localizations. They severely under-represent authentic colloquial dialogue, regional vocabulary, and spontaneous spoken speech.
2. **Context Blindness across Pragmatic Dimensions**: Languages like Hebrew (`he`), Gujarati (`gu`), and Vietnamese (`vi`) encode critical interpersonal dimensions into grammatical morphology, pronoun choice, and regional dialect. Standard MT benchmarks evaluate propositional accuracy without measuring whether translations respect register, kinship, speaker/listener gender, or regional dialect.

**Pragmatic MT Benchmark v2** addresses these flaws directly. It evaluates single-sentence translation (`en -> he`, `en -> gu`, `en -> vi`) across authentic conversational speech with zero synthetic target golds.

---

## 2. Track Quarantine Architecture

To ensure diagnostic tests never pollute conversational scores, the dataset enforces **3 strictly quarantined evaluation tracks** (`track` field):

| Track | Target Count | Description | Primary Upstream Corpora |
|---|---:|---|---|
| `casual_conversational` | 600 (200/lang) | Primary benchmark track measuring everyday colloquial speech, interpersonal register, and regional dialects. | OpenSubtitles, Ivrit-AI Spoken Hebrew, Biu3010/ViDia2Std, ViMD, OPUS-100 |
| `register_contrastive` | Optional / Diagnostic | Paired contrastive tests isolating formal vs. informal register shifts. | CoCoA-MT (probed dynamically) |
| `diagnostic_morphology` | 150 (50/lang) | Quarantined textbook unit tests used as calibration baselines; never mixed into conversational metrics. | Helsinki-NLP/Tatoeba baselines |

---

## 3. Anti-Translationese & Target-First Methodology

Existing MT evaluation benchmarks frequently rely on AI-generated target paraphrases, introducing subtle synthetic artifacts ("model accent"). In v2:
- **Zero Synthetic Target Golds**: AI models are NEVER permitted to generate, paraphrase, or rewrite target-language (`he`, `gu`, `vi`) references.
- **Target-First Sourcing (`origin_direction: "target_first"`)**: Monolingual native recordings, spontaneous transcripts, and authentic regional dialogue are collected first. An LLM (`deepseek/deepseek-v4.1-flash` at temperature 0.0) is used **strictly for target-first back-translation** (`native target -> natural casual English source`). Every such row is permanently tagged `origin_direction: "target_first"`.

---

## 4. Backward Label Harvesting

All pragmatic labels are harvested **deterministically backward from human-written native target text** or verified dataset metadata. Labels are NEVER inferred from English source strings.

### The 5 Standardized Pragmatic Axes:
- `listener_gender`: `"male" | "female" | "neutral" | null`
- `speaker_gender`: `"male" | "female" | null`
- `register_tier`: `"intimate" | "polite" | "honorific" | null`
- `regional_dialect`: `"vi_north" | "vi_south" | null`
- `social_relationship`: Kinship or interpersonal pair (e.g. `"anh_em"`, `"chi_em"`, `"chu_chau"`, `"may_tao"`, `"ban_minh"`, `"con_bo"`, `"con_me"`) or `null`

### Language-Specific Harvesting Rules:
- **Hebrew (`he`)**:
  - `listener_gender`: Extracted ONLY from orthographically unambiguous unvocalized markers: explicit 2nd-person pronouns (`אתה` -> male, `את` -> female) or unambiguous feminine 2nd-person morphology (`תעשי`, `תבואי`, `תגידי`, `הולכת`, `יודעת`, `מוכנה`, `בטוחה`). Ambiguous unvocalized past verbs (e.g. bare `עשית`) remain `null`.
- **Gujarati (`gu`)**:
  - `register_tier`: Extracted from closed-class pronouns:
    - `intimate`: L1 intimate pronouns (`તું`, `તને`, `તારું`, `તારી`, `તારા`).
    - `polite`: L2 polite pronouns (`તમે`, `તમને`, `તમારું`, `તમારી`, `તમારા`).
    - `honorific`: L3 honorific pronouns (`આપ`, `આપને`, `આપનું`, `આપની`, `આપના`).
- **Vietnamese (`vi`)**:
  - `social_relationship`: Harvested ONLY when BOTH pronouns of a kinship pair appear in the native sentence (e.g., `anh` + `em` -> `"anh_em"`, `mày` + `tao` -> `"may_tao"`). Single pronouns or pro-drop sentences remain `null`.
  - `regional_dialect`: Sourced from `ViDia2Std`/`ViMD` region tags and verified dialect markers (`vi_south`: `muỗng`, `vô`, `hổng`, `nè`, `nghen`, `má`, `tui`, `bông`; `vi_north`: `thìa`, `vào`, `không`, `nhé`, `nhỉ`, `bố`, `tôi`).

---

## 5. Dataset Statistics (Auto-Generated from Manifest)

*Generated directly from `provenance/v2/manifest.json`.*

### Overall Item Counts:
- **Total Benchmark Rows**: 750
- **Casual Conversational Track**: 600 (200 `he`, 200 `gu`, 200 `vi`)
- **Diagnostic Morphology Track**: 150 (50 `he`, 50 `gu`, 50 `vi`)

### Casual Track Origin Breakdown:
| Language | Target-First (Native Utterances) | Source-First (Parallel Casual) | Target-First Share |
|---|---:|---:|---:|
| Hebrew (`he`) | 72 | 128 | 36.0% (>= 30% quota met) |
| Gujarati (`gu`) | 0 | 200 | 0.0% |
| Vietnamese (`vi`) | 103 | 97 | 51.5% (>= 30% quota met) |

### Gujarati Register Balance (`casual_conversational`):
| Register Tier | Pronoun Class | Row Count | Percentage |
|---|---|---:|---:|
| `intimate` | L1 (`તું`, `તને`, `તારું`) | 62 | 31.0% (>= 25% quota met) |
| `honorific` | L3 (`આપ`, `આપને`, `આપનું`) | 45 | 22.5% (>= 15% quota met) |
| `polite` | L2 (`તમે`, `તમને`, `તમારું`) | 43 | 21.5% (<= 60% quota met) |
| *(unmarked)* | No 2nd-person pronoun (IN22-Conv turns) | 50 | 25.0% |

### Vietnamese Dialect Balance (`casual_conversational`):
| Regional Dialect | Row Count | Percentage |
|---|---:|---:|
| `vi_south` (Southern) | 64 | 58.7% (within 55%–65% target) |
| `vi_north` (Northern) | 45 | 41.3% (within 35%–45% target) |

---

## 6. Upstream Corpora & Licenses

| Corpus | Target Language | Split Used | License | Role in Benchmark |
|---|---|---|---|---|
| `sentence-transformers/parallel-sentences-opensubtitles` | `he`, `vi` | `train` | Open / CC | Parallel casual dialogue |
| `ivrit-ai/crowd-transcribe-v5` | `he` | `train` | Open Spoken (CC) | Spontaneous Israeli spoken Hebrew transcripts |
| `ai4bharat/IN22-Conv` | `gu` | `test` | CC-BY-4.0 | Everyday interpersonal conversation turns |
| `Helsinki-NLP/opus-100` | `gu`, `vi` | `train` | CC-BY-SA-4.0 | Natural parallel Gujarati & Vietnamese dialogue |
| `Biu3010/ViDia2Std` | `vi` | `train`, `dev`, `test` | MIT / Academic | Native regional dialect target references |
| `nguyendv02/ViMD_Dataset` | `vi` | `test` | Open Academic | Regional speech metadata & transcripts |
| `Helsinki-NLP/tatoeba` | `he`, `gu`, `vi` | `train` | CC-BY-2.0 | Isolated textbook baseline unit tests |

---

## 7. Schema Definition

```json
{
  "item_id": "v2_vi_cas_0001",
  "dataset_version": "v2",
  "track": "casual_conversational",
  "lang": "vi",
  "source_en": "Are you heading over there later?",
  "reference_target": "Lát nữa em có qua bển hổng?",
  "source_corpus": "Biu3010/ViDia2Std",
  "source_split": "test",
  "origin_direction": "target_first",
  "pragmatic_profile": {
    "listener_gender": null,
    "speaker_gender": null,
    "register_tier": "intimate",
    "regional_dialect": "vi_south",
    "social_relationship": "anh_em"
  },
  "required_anchors": ["hổng", "em"],
  "forbidden_markers": ["không", "nhé", "thìa"],
  "harvest_confidence": "deterministic_exact"
}
```

---

## 8. Reproducibility

To re-curate the benchmark from scratch:
```bash
# 1. Run dry-run verification
python provenance/v2/build_dataset.py --dry-run --limit-per-source 20

# 2. Run full curation pipeline
python provenance/v2/build_dataset.py

# 3. Verify quality test suite
pytest provenance/v2/tests/
```
