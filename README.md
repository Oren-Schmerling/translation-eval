# Pragmatic MT Evaluation

A reproducible harness for measuring whether machine translation systems honor
**pragmatic meaning** — formality, speaker and listener gender, regional
dialect, pronouns, and kinship — rather than only lexical adequacy.

The project ships a curated Hebrew / Gujarati / Vietnamese benchmark, typed
provider adapters, a deterministic three-layer grader, and Markdown/CSV
reporting. Everything needed to fetch the dataset and run the full test suite
works offline with no API keys.

---

## Dataset

- **Published Dataset**: [https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval](https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval)
- **Variants**: 720 total (192 Hebrew, 288 Gujarati, 240 Vietnamese across 36 base items).

Fetch the published dataset:
```bash
python3 scripts/fetch_dataset.py
```

---

## Research questions

| ID | Question | Primary tiers |
|----|----------|---------------|
| **RQ1** | What pragmatic choices does a system make by default, and how accurate and fluent is the output? | Tier 1 |
| **RQ2** | When pragmatic constraints are supplied explicitly, does the system comply — including coarse parameters, fine-grained parameters, and adversarial conflicts? | Tiers 2A, 2B, 2C |
| **RQ3** | Does few-shot prompting improve pragmatic compliance, and do contrastive examples help beyond positive-only examples? | Tiers 3A, 3B, 3C |

## The seven evaluation tiers

| Tier | Name | What it tests |
|------|------|---------------|
| 1 | `Tier1DefaultZeroShot` | Free translation with no pragmatic instruction. Reveals default register, gender bias, and dialect bias. |
| 2A | `Tier2ACoarseParams` | Explicit formality, speaker gender, listener gender, and dialect parameters. |
| 2B | `Tier2BGranularParams` | Fine-grained pronoun tier, kinship pair, and relationship constraints. |
| 2C | `Tier2CAdversarialConflict` | Source cues conflict with API parameters; the parameters must override the cues. |
| 3A | `Tier3AFewShotK5Pos` | Five positive in-language examples in the prompt. |
| 3B | `Tier3BFewShotK20Pos` | Twenty positive examples for stronger in-context adaptation. |
| 3C | `Tier3CFewShotK5Contrastive` | Five contrastive example pairs (correct target plus an explicitly rejected alternative). |

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

## Dataset provenance

For dataset construction templates, reconstruction instructions, and the 720-row breakdown, see [provenance/README.md](provenance/README.md).

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
