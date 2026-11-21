# Pragmatic MT Evaluation

A reproducible harness for measuring whether machine translation systems honor
**pragmatic meaning** — formality, speaker and listener gender, regional
dialect, pronouns, and kinship — rather than only lexical adequacy.

The project ships a curated Hebrew / Gujarati / Vietnamese benchmark, typed
provider adapters, a deterministic three-layer grader, and Markdown/CSV
reporting. Everything needed to fetch the dataset and run the full test suite
works offline with no API keys.

---

## Dataset (v2 Benchmark)

- **Published Benchmark**: [https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval](https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval)
- **Total Benchmark Rows**: 950 (600 casual conversational + 100 register contrastive + 250 diagnostic morphology calibration items).
- **Languages**: Hebrew (`he`), Gujarati (`gu`), Vietnamese (`vi`).
- **Legacy v1 Access**: Preserved under Git tag `v1.0` and accessible via `--dataset-version=v1`.

### Fetch the published benchmark:
```bash
# Fetch full v2 benchmark (default, verifies SHA-256 against manifest)
python3 scripts/fetch_dataset.py

# Fetch specific track
python3 scripts/fetch_dataset.py --track=casual_conversational
python3 scripts/fetch_dataset.py --track=register_contrastive
python3 scripts/fetch_dataset.py --track=diagnostic_morphology

# Fetch legacy v1 dataset (720 variants)
python3 scripts/fetch_dataset.py --dataset-version=v1
```

### Dataset Statistics (Auto-Generated from Manifest)

*Generated directly from `provenance/v2/manifest.json`.*

#### Track Breakdown:
| Track | Hebrew (`he`) | Gujarati (`gu`) | Vietnamese (`vi`) | Total Rows |
|---|---:|---:|---:|---:|
| `casual_conversational` | 200 | 200 | 200 | **600** |
| `register_contrastive` | 0 | 0 | 100 | **100** |
| `diagnostic_morphology` | 150 | 50 | 50 | **250** |
| **Total** | **350** | **250** | **350** | **950** |

#### Casual Track Sourcing & Anti-Translationese Blend:
| Language | Target-First Native Utterances | Parallel Casual Turns | Target-First Share | Primary Authentic Sources |
|---|---:|---:|---:|---|
| Hebrew (`he`) | 72 | 128 | 36.0% (>= 30% quota met) | `ivrit-ai/crowd-transcribe-v5` (36.0%) |
| Gujarati (`gu`) | 20 | 180 | 10.0% | `snorbyte` (turns) + `IN22-Conv` (35.0%) |
| Vietnamese (`vi`) | 110 | 90 | 55.0% (>= 50% quota met) | `ViDia2Std` (91) + `ViMD` (19) (55.0%) |

#### Gujarati Register Balance (`casual_conversational`):
| Register Tier | Pronoun Class | Row Count | Percentage |
|---|---|---:|---:|
| `intimate` | L1 (`તું`, `તને`, `તારું`) | 63 | 31.5% (>= 25% quota met) |
| `honorific` | L3 (`આપ`, `આપને`, `આપનું`) | 45 | 22.5% (>= 15% quota met) |
| `polite` | L2 (`તમે`, `તમને`, `તમારું`) | 23 | 11.5% (<= 60% quota met) |
| *(unmarked)* | No 2nd-person pronoun (IN22-Conv & snorbyte turns) | 69 | 34.5% |

#### Vietnamese Dialect Balance (`casual_conversational`):
| Regional Dialect | Row Count | Percentage |
|---|---:|---:|
| `vi_south` (Southern) | 68 | 59.6% (within 55%–65% target) |
| `vi_north` (Northern) | 46 | 40.4% (within 35%–45% target) |

---

## Research questions

| ID | Question | Primary tiers |
|----|----------|---------------|
| **RQ1** | What pragmatic choices does a system make by default, and how accurate and fluent is the output? | Tier 1 |
| **RQ2** | When pragmatic constraints are supplied explicitly, does the system comply — including coarse parameters, fine-grained parameters, and adversarial conflicts? | Tiers 2A, 2B, 2C |
| **RQ3** | Does few-shot prompting improve pragmatic compliance, and do contrastive examples help beyond positive-only examples? | Tiers 3A, 3B, 3C |

## The seven evaluation tiers

| Tier | Name | Shorthand | What it tests |
|------|------|-----------|---------------|
| 1 | Default Zero-Shot | `Tier1DefaultZeroShot` | Free translation with no pragmatic instruction. Reveals default register, gender bias, and dialect bias. |
| 2A | Coarse Parameters | `Tier2ACoarseParams` | Explicit formality, speaker gender, listener gender, and dialect parameters. |
| 2B | Granular Parameters | `Tier2BGranularParams` | Fine-grained pronoun tier, kinship pair, and relationship constraints. |
| 2C | Adversarial Conflict | `Tier2CAdversarialConflict` | Source cues conflict with API parameters; the parameters must override the cues. |
| 3A | Few-Shot (5 Positive) | `Tier3AFewShotK5Pos` | Five positive in-language examples in the prompt. |
| 3B | Few-Shot (20 Positive) | `Tier3BFewShotK20Pos` | Twenty positive examples for stronger in-context adaptation. |
| 3C | Few-Shot (5 Contrastive) | `Tier3CFewShotK5Contrastive` | Five contrastive example pairs (correct target plus an explicitly rejected alternative). |

---

## Architecture

```
.
├── Cargo.toml
├── README.md
├── data/
│   ├── resources/closed_class.json     # per-language closed-class + synset resources
│   ├── templates/                      # he_curation.csv, gu_curation.csv, vi_curation.csv
│   └── compiled/eval_dataset.json      # fetched from Hugging Face (or built via provenance/)
├── provenance/                 # Construction scripts, templates, and card (see provenance/README.md)
│   ├── generate_sheets_template.py
│   ├── harvest_hf_datasets.py
│   ├── prepare_hf_dataset.py
│   ├── hf_dataset_card.md
│   └── README.md
├── scripts/
│   ├── fetch_dataset.py        # End-user dataset fetch and validation script
│   ├── sidecar.py              # COMET scoring + validation quiz export
│   └── requirements.txt
└── src/
    ├── lib.rs                          # crate docs + public re-exports
    ├── main.rs                         # clap CLI: compile / run / grade / report
    ├── schema.rs                       # dataset, request, telemetry, run-record types
    ├── compiler/
    │   ├── mod.rs                      # CSV -> EvalDataset
    │   └── autodiff.rs                 # normalization + anchor inference
    ├── providers/
    │   ├── mod.rs                      # TranslationProvider trait, costs, latency
    │   ├── gcloud.rs                   # Google basic NMT + adaptive translation LLM
    │   ├── azure.rs                    # Azure standard NMT + generative translator
    │   ├── deepl.rs                    # DeepL REST adapter
    │   ├── amazon.rs                   # Amazon Translate adapter
    │   └── llm_router.rs               # OpenAI / Anthropic / Vertex / Cohere / OpenRouter streaming
    └── grader/
        ├── mod.rs
        ├── layer1_anchors.rs           # deterministic lexical compliance
        ├── layer2_chrf.rs              # chrF++ (character + word n-grams)
        ├── layer3_ensemble.rs          # two-judge agreement diagnostics
        └── report.rs                   # Markdown + CSV reports
```

### Data flow

1. **Fetch** the published evaluation dataset from Hugging Face (`python3 scripts/fetch_dataset.py`).
2. **Run** each selected variant and tier through a provider, writing JSONL.
3. **Grade** with anchors, chrF++, default-bias classification, and optional judges.
4. **Report** as Markdown and CSV; optionally score with COMET and export a blinded quiz.
(For dataset construction and reconstruction instructions, see `provenance/README.md`).

---

## Data schema

### Compiled dataset

End users fetch this compiled JSON file from Hugging Face (or reconstruct it locally via `compile`). It contains a pretty-printed `{"variants": [...]}` object. Each variant
carries fields plus inferred `anchors`:

```json
{
  "must_contain_any": ["..."],
  "must_not_contain": ["..."],
  "synset_equivalents": {"token": ["equivalent"]},
  "acceptable_synonyms": []
}
```

Anchors are inferred from sibling gold variants within the same item and
language, using closed-class/resource membership, synsets, and conservative
morphological relatedness. They are deterministic and NFC-normalized.

### Provenance

For dataset construction templates, the 720-row breakdown, curation CSV columns, and reconstruction pipeline, see [provenance/README.md](provenance/README.md).

---

## Quickstart

Prerequisites: a recent stable Rust toolchain and Python 3.10+.

```bash
# 1. Fetch the published dataset (720 variants).
python3 scripts/fetch_dataset.py

# 2. Run the offline test suite (no API keys required).
cargo test
```

The compiled dataset contains **720 variants**: 192 Hebrew, 288 Gujarati, and
240 Vietnamese.

---

## Running providers

Choose a provider with `--provider`:

`google-nmt`, `google-adaptive`, `azure-nmt`, `azure-genai`, `deepl`, `amazon`, `llm`, `openrouter`.

Common options:

```bash
cargo run -- run \
  --dataset data/compiled/eval_dataset.json \
  --provider google-nmt \
  --tiers 1,2a,3a \
  --language he,gu \
  --limit 20 \
  --output results/results.jsonl
```

- `--tiers` accepts `all` (default), tier names, or short aliases such as
  `1`, `2a`, `2b`, `2c`, `3a`, `3b`, `3c`.
- `--language` accepts a comma-separated filter (`he`, `gu`, `vi`).
- `--limit` caps variants per language and tier.
- `--base-url` overrides the production endpoint.
- Requests that a provider cannot serve are written with an empty translation
  and an `UnsupportedByProvider` capability; no network call is made.
- Successful and partial-capability requests are translated and appended to the
  output JSONL. Per-request failures print a warning and continue.

### Google Cloud

```bash
export GOOGLE_API_KEY="..."
export GCP_PROJECT_ID="my-gcp-project"

# Basic NMT supports Tier 1 only.
cargo run -- run --provider google-nmt --tiers 1 --output results/google-nmt.jsonl

# Adaptive translation LLM supports Tier 1 and Tier 3A.
cargo run -- run --provider google-adaptive --tiers 1,3a --output results/google-adaptive.jsonl
```

### Azure AI Translator

Standard NMT uses the v3.0 API and a Translator resource key:

```bash
export AZURE_TRANSLATOR_KEY="..."
export AZURE_TRANSLATOR_REGION="eastus"

cargo run -- run --provider azure-nmt --tiers 1 --output results/azure-nmt.jsonl
```

Generative translation uses the `2026-06-06` API and requires a Microsoft
Foundry (`AIServices`) resource with a deployed model. Point `--base-url` at the
resource endpoint (or set `AZURE_TRANSLATOR_ENDPOINT`) and pass the deployment
name with `--deployment`:

```bash
export AZURE_TRANSLATOR_KEY="..."          # Foundry resource key
export AZURE_TRANSLATOR_REGION="westus"
export AZURE_TRANSLATOR_ENDPOINT="https://<resource>.cognitiveservices.azure.com"

cargo run -- run --provider azure-genai \
  --deployment gpt-5.4-mini \
  --tiers 1,2a,3a \
  --output results/azure-genai.jsonl
```

Deployments observed to work with the `2026-06-06` API: `gpt-5.1`, `gpt-5.2`,
`gpt-5.4`, and `gpt-5.4-mini`. `gpt-5.4-mini` is the cheapest supported option.
`gpt-4o-mini` can no longer be newly deployed, and `gpt-5.4-nano`, `gpt-5-mini`,
`gpt-5-nano`, and the Luna models are rejected by Translator.

### DeepL

```bash
export DEEPL_API_KEY="..."
cargo run -- run --provider deepl --tiers 1,2a --output results/deepl.jsonl
```

`--base-url` defaults to `https://api-free.deepl.com`; pass the Pro endpoint if
your key requires it.

### Amazon Translate

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
cargo run -- run --provider amazon --tiers 1,2a --output results/amazon.jsonl
```

The adapter deliberately leaves request signing to the surrounding transport or
a compatible proxy; the supplied access key is forwarded for that purpose.

### Streaming LLMs

```bash
export OPENAI_API_KEY="..."
cargo run -- run --provider llm \
  --model-family openai \
  --model gpt-4o-mini \
  --output results/openai.jsonl
```

Other wire formats:

```bash
export ANTHROPIC_API_KEY="..."
cargo run -- run --provider llm --model-family anthropic \
  --model claude-3-5-sonnet-latest \
  --base-url https://api.anthropic.com \
  --output results/anthropic.jsonl

export GOOGLE_API_KEY="..."
cargo run -- run --provider llm --model-family vertex \
  --model gemini-1.5-flash \
  --base-url https://generativelanguage.googleapis.com \
  --output results/vertex.jsonl

export COHERE_API_KEY="..."
cargo run -- run --provider llm --model-family cohere \
  --model command-r-plus \
  --base-url https://api.cohere.ai/compatibility/v1 \
  --output results/cohere.jsonl
```

#### OpenRouter

```bash
export OPENROUTER_API_KEY="..."
cargo run -- run --provider openrouter \
  --model google/gemini-2.0-flash-001 \
  --output results/openrouter-gemini.jsonl
```

`--model` accepts any OpenRouter model id, whether translation-specific or a
general LLM. Examples include `openai/gpt-4o-mini`,
`anthropic/claude-3.5-sonnet`, `google/gemini-2.0-flash-001`, and
`meta-llama/llama-3.3-70b-instruct`; check the OpenRouter model catalogue for
the current list. `--provider llm --model-family openrouter` is equivalent, and
`--base-url` defaults to `https://openrouter.ai/api/v1`. When `--api-key` is
omitted, the key is read from `OPENROUTER_API_KEY`.

Production defaults when `--base-url` is omitted:

| Provider | Default endpoint |
|----------|------------------|
| Google | `https://translation.googleapis.com` |
| Azure | `https://api.cognitive.microsofttranslator.com` |
| DeepL | `https://api-free.deepl.com` |
| Amazon | `https://translate.us-east-1.amazonaws.com` |
| LLM | `https://api.openai.com/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

API keys can be passed with `--api-key`; otherwise the standard environment
variables above are used.

---

## Grading

```bash
cargo run -- grade \
  --input results/results.jsonl \
  --output results/graded.jsonl \
  --dataset data/compiled/eval_dataset.json
```

The grader adds:

- **Layer 1 — anchors:** boundary-aware lexical compliance (`must_contain_any`,
  `must_not_contain`, synset equivalents, acceptable synonyms).
- **Layer 2 — chrF++:** best score over the gold reference and acceptable
  synonyms.
- **Default-bias classification:** evaluates Tier 1 output against same-language,
  same-item siblings to infer register, gender, and dialect.
- **Layer 3 — judge ensemble (optional):** two concurrent OpenAI-compatible
  judges scoring formality, gender, dialect, and fluency; disagreement is
  flagged per axis.

Judge configuration (`--judges-config judges.json`):

```json
{
  "judge_a": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4o-mini"
  },
  "judge_b": {
    "base_url": "https://api.example.com/v1",
    "api_key": "...",
    "model": "another-model"
  }
}
```

Judge failures are reported as warnings and do not abort grading.

## Reporting

```bash
cargo run -- report \
  --input results/graded.jsonl \
  --markdown results/report.md \
  --csv results/summary.csv
```

The Markdown report contains five tables: default quality and bias distribution,
Tier 2 parameter compliance, Tier 2C override rate, Tier 3 few-shot deltas, and
operational telemetry (latency percentiles and cost projections).

---

## Python sidecars

Install the optional dependencies:

```bash
python3 -m pip install -r scripts/requirements.txt
```

COMET scoring (skips gracefully if `unbabel-comet` is unavailable):

```bash
python3 scripts/sidecar.py comet \
  --input results/results.jsonl \
  --output results/results.comet.jsonl
```

Blinded human-validation quiz export (all high-disagreement records plus a
deterministic stratified sample):

```bash
python3 scripts/sidecar.py export-validation-quiz \
  --input results/graded.jsonl \
  --csv results/validation_quiz.csv \
  --forms-json results/validation_quiz_forms.json \
  --sample-rate 0.2 \
  --seed 42
```

---

## Dataset provenance & v2 Curation Methodology

### 1. Design Thesis
Standard machine translation benchmarks (`IWSLT`, `Tatoeba`, `WinoMT`) suffer from formal/translationese skew and fail to evaluate interpersonal pragmatic fidelity across `he`, `gu`, and `vi`. The v2 benchmark curates authentic colloquial dialogue, regional dialect variants, and spontaneous speech without synthetic target golds.

### 2. Track Quarantine Architecture
- **`casual_conversational`**: Primary benchmark track (600 items; 200 items each for `he`, `gu`, `vi`). Evaluates everyday spoken turns, intimate/polite/honorific registers, and regional dialects.
- **`register_contrastive`**: Optional contrastive track probing formal vs. informal register shifts.
- **`diagnostic_morphology`**: Quarantined textbook calibration baselines (150 items; 50/lang) from Tatoeba; never mixed into casual conversational benchmark scores.

### 3. Anti-Translationese Methodology
- **Zero Synthetic Target Golds**: AI models are NEVER used to generate or paraphrase target-language references.
- **Target-First Sourcing**: Spontaneous speech transcripts (`ivrit-ai`) and native regional dialect corpora (`ViDia2Std`, `ViMD`) are ingested first. Casual English sources are back-translated via `deepseek/deepseek-v4.1-flash` at temperature 0.0 with deterministic JSONL caching. Every such item is tagged `origin_direction: "target_first"`.

### 4. Backward Label Harvesting
Labels are harvested deterministically backward from native target gold strings (never inferred from English):
- **Hebrew (`he`)**: `listener_gender` extracted exclusively from unvocalized orthographic markers (`אתה` vs. `את`, unambiguous feminine imperatives/participles like `תעשי`, `תבואי`, `הולכת`, `מוכנה`).
- **Gujarati (`gu`)**: `register_tier` harvested from closed-class pronouns: L1 intimate (`તું`, `તને`), L2 polite (`તમે`, `તમને`), L3 honorific (`આપ`, `આપને`).
- **Vietnamese (`vi`)**: `social_relationship` harvested only when both pronouns of a kinship pair appear (`anh_em`, `chi_em`, `chu_chau`, `may_tao`, `con_bo`, `con_me`). `regional_dialect` harvested from source metadata and anchors (`vi_south`: `muỗng`, `vô`, `hổng`, `nè`, `nghen`, `má`, `tui`; `vi_north`: `thìa`, `vào`, `không`, `nhé`, `nhỉ`, `bố`, `tôi`).

### 5. Upstream Corpora & Licenses
| Corpus | Language | Split | License | Benchmark Role |
|---|---|---|---|---|
| `sentence-transformers/parallel-sentences-opensubtitles` | `he`, `vi` | `train` | Open / CC | Parallel casual dialogue |
| `ivrit-ai/crowd-transcribe-v5` | `he` | `train` | Open Spoken (CC) | Spontaneous Israeli spoken Hebrew transcripts |
| `ai4bharat/IN22-Conv` | `gu` | `test` | CC-BY-4.0 | Everyday interpersonal conversation turns |
| `snorbyte/indic-audio-natural-conversations-sample` | `gu` | `samples` | Research | Spoken conversational turns with speaker gender |
| `Helsinki-NLP/opus-100` | `gu`, `vi` | `train` | CC-BY-SA-4.0 | Natural parallel Gujarati & Vietnamese dialogue |
| `Biu3010/ViDia2Std` | `vi` | `train`, `dev`, `test` | MIT / Academic | Native regional dialect target references |
| `nguyendv02/ViMD_Dataset` | `vi` | `test` | Open Academic | Regional speech metadata & transcripts |
| `amazon-science/contrastive-controlled-mt` | `vi` | `test` | Apache-2.0 | Formality control contrastive pairs with annotations |
| `gabrielStanovsky/mt_gender` | `he` | `aggregates` | MIT / Academic | WinoMT coreference gender diagnostic suite |
| `Helsinki-NLP/tatoeba` | `he`, `gu`, `vi` | `train` | CC-BY-2.0 | Isolated textbook baseline unit tests |

### 6. Reproducibility
All curation and verification scripts live under `provenance/v2/`:
```bash
# Dry-run verification
python provenance/v2/build_dataset.py --dry-run --limit-per-source 20

# Full build
python provenance/v2/build_dataset.py

# Quality test assertions
pytest provenance/v2/tests/

# Publish to Hugging Face Hub (dry-run or live push)
python provenance/v2/publish_hf.py --dry-run
python provenance/v2/publish_hf.py --push
```

Published dataset: [https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval](https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval)

---

## Capabilities and limitations

- **Capabilities are declared, not assumed.** Each provider reports `Supported`,
  `PartiallySupported` (with missing axes), or `UnsupportedByProvider` before a
  request is sent. This makes cross-provider comparisons explicit.
- **Anchors are conservative.** They capture closed-class and resource-backed
  contrasts plus cautious morphology; they are not a substitute for human
  judgment.
- **chrF++ is reference-based.** A semantically valid paraphrase can score
  lower than a literal translation. Use acceptable synonyms and the judge layer
  alongside it.
- **LLM judging is noisy.** The ensemble reports disagreement rather than
  hiding it; high-disagreement rows are prioritized by the validation quiz.
- **Amazon signing is external.** The adapter does not implement AWS SigV4; a
  signed transport or compatible proxy is expected.
- **Provider limits vary.** Basic NMT endpoints usually support only Tier 1;
  unsupported tiers are recorded without a network call.
- **The curated benchmark is small and deliberate.** It targets pragmatic
  contrasts, not broad-domain coverage.

---

## Offline mocked tests

```bash
cargo test
```

All tests run offline. Provider tests use `wiremock` to validate request
bodies, headers, query parameters, streaming SSE parsing, and error handling.
No live credentials or network access are required.

## License

MIT. See the `license` field in `Cargo.toml`.
