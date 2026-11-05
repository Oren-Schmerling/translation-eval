---
language:
- he
- gu
- vi
license: mit
task_categories:
- translation
- text-generation
pretty_name: Pragmatic MT Benchmark for Hebrew, Gujarati, and Vietnamese
size_categories:
- n<1K
source_datasets: []
tags:
- machine-translation
- pragmatic-meaning
- evaluation
- benchmark
- formality
- gender
- dialect
configs:
- config_name: variants
  data_files:
    - split: train
      path: variants.jsonl
- config_name: curation
  data_files:
    - split: he
      path: he_curation.csv
    - split: gu
      path: gu_curation.csv
    - split: vi
      path: vi_curation.csv
---

# Pragmatic MT Benchmark (Hebrew, Gujarati, Vietnamese)

## Dataset Summary

This is a **reference-based, pragmatics-focused evaluation benchmark** for Hebrew (he), Gujarati (gu), and Vietnamese (vi), with English source sentences. The benchmark evaluates pragmatic meaning in machine translation across seven dimensions: formality, speaker gender, listener gender, dialect, pronoun tier, kinship pair, and relationship.

Each base item appears in multiple language-specific variants that differ along one or more pragmatic axes. Bucket C items include conflict scenarios where the cued source text conflicts with an override condition, testing the model's ability to resolve pragmatic tensions.

**This is not a general MT benchmark.** It is specifically designed to probe pragmatic fidelity—whether translations preserve pragmatic nuance rather than just propositional content.

## Languages

| Language | Language Code | Variants |
|----------|---------------|----------|
| Hebrew   | he            | 192      |
| Gujarati | gu            | 288      |
| Vietnamese | vi          | 240      |
| **Total**|               | **720**  |

All source sentences are in English. Each language has variants in tracks `family_and_relatives` and `culture_and_daily_life`, distributed across buckets A, B, and C.

## Dataset Structure

### Files

| File | Description |
|------|-------------|
| `variants.jsonl` | Flattened, viewer-friendly export: one JSON object per variant (720 lines). Contains all pragmatic fields at top level (formality, speaker_gender, listener_gender, dialect, pronoun_tier, kinship_pair, relationship, conflict_override, anchors fields, gold_reference). |
| `eval_dataset.json` | Canonical compiled dataset: a single JSON object `{"variants": [...]}` with nested coarse/granular/anchors/conflict_override objects. Byte-identical to `data/compiled/eval_dataset.json`. |
| `he_curation.csv` | Hebrew curation sheet (192 rows) |
| `gu_curation.csv` | Gujarati curation sheet (288 rows) |
| `vi_curation.csv` | Vietnamese curation sheet (240 rows) |
| `README.md` | This dataset card |

### Variant Fields

| Field | Type | Description |
|-------|------|-------------|
| `variant_id` | string | Unique identifier (e.g., `cul_a_01-gu-level_1_intimate-female`) |
| `item_id` | string | Base item identifier |
| `source_en` | string | English source sentence |
| `lang` | string | Target language code (`he`, `gu`, `vi`) |
| `track` | string | Topic track (`family_and_relatives`, `culture_and_daily_life`) |
| `bucket` | string | Difficulty bucket (`A`, `B`, `C`) |
| `is_cued_source` | boolean | Whether the source includes a pragmatic cue |
| `formality` | string | Formality level (`default`, `formal`, `level_1_intimate`, `level_2_polite`, `level_3_honorific`) |
| `speaker_gender` | string | Source speaker gender (`male`, `female`) |
| `listener_gender` | string | Target listener gender (`male`, `female`, `null`) |
| `dialect` | string | Regional dialect (`northern`, `southern`, `null`) |
| `pronoun_tier` | string | Second-person pronoun/register (`tu`, `tame`, `aap`, `anh_em`, `chi_em`, `con_ong`, `chu_chau`, `co_chau`, `null`) |
| `kinship_pair` | string | Kinship pair (`null`, or specific pair like `mother-father`, etc.) |
| `relationship` | string | Social relationship (`null`, or description like `parent-child`, etc.) |
| `conflict_override` | object\|null | Conflict condition for bucket C cued items; null otherwise |
| `must_contain_any` | array[string] | Terms that must appear in any valid translation |
| `must_not_contain` | array[string] | Terms that must not appear |
| `synset_equivalents` | object | Synset-equivalent term mappings |
| `acceptable_synonyms` | array[string] | Additional acceptable synonym terms |
| `gold_reference` | string | Gold reference translation |

### Nested canonical schema (`eval_dataset.json`)

The `eval_dataset.json` file is a byte-for-byte copy of `data/compiled/eval_dataset.json`. Its structure is a single JSON object `{"variants": [...]}` where each variant carries nested objects under `coarse` (containing `formality`, `speaker_gender`, `listener_gender`, `dialect`), `granular` (containing `pronoun_tier`, `kinship_pair`, `relationship`), `conflict_override` (nested object or null), and `anchors` (containing `must_contain_any`, `must_not_contain`, `synset_equivalents`, `acceptable_synonyms`). The `variants.jsonl` file, by contrast, flattens all of these top-level fields for viewer-friendly per-variant JSON objects.

## Evaluation Tiers

| Tier | Name | Shorthand | Description |
|------|------|-----------|-------------|
| 1 | Default Zero-Shot | `Tier1DefaultZeroShot` | Free translation with no pragmatic instruction. Reveals default register, gender bias, and dialect bias. |
| 2A | Coarse Parameters | `Tier2ACoarseParams` | Explicit formality, speaker gender, listener gender, and dialect parameters. |
| 2B | Granular Parameters | `Tier2BGranularParams` | Fine-grained pronoun tier, kinship pair, and relationship constraints. |
| 2C | Adversarial Conflict | `Tier2CAdversarialConflict` | Source cues conflict with API parameters; the parameters must override the cues. |
| 3A | Few-Shot (5 Positive) | `Tier3AFewShotK5Pos` | Five positive in-language examples in the prompt. |
| 3B | Few-Shot (20 Positive) | `Tier3BFewShotK20Pos` | Twenty positive examples for stronger in-context adaptation. |
| 3C | Few-Shot (5 Contrastive) | `Tier3CFewShotK5Contrastive` | Five contrastive example pairs (correct target plus an explicitly rejected alternative). |

## Usage

### Loading with HuggingFace `datasets`

```python
from datasets import load_dataset

# Load variants
ds = load_dataset("<org>/<dataset-name>", name="variants", split="train")

# Load curation sheets
he_ds = load_dataset("<org>/<dataset-name>", name="curation", split="he")
gu_ds = load_dataset("<org>/<dataset-name>", name="curation", split="gu")
vi_ds = load_dataset("<org>/<dataset-name>", name="curation", split="vi")
```

### Evaluation Mapping

To use this benchmark with an evaluation harness:

1. Load the dataset with `load_dataset(...)`.
2. For each variant, generate a translation from `source_en` into the target language.
3. Check the translation against `anchors.must_contain_any`, `anchors.must_not_contain`, etc.
4. Report per-tier and per-language accuracy.

## Provenance & Construction

- **Base items**: 36 base source sentences
- **Script**: `provenance/generate_sheets_template.py` generates the curation templates
- **Compiler**: Compiles templates into `data/compiled/eval_dataset.json`
- **Preparation**: `provenance/prepare_hf_dataset.py` exports this dataset card and JSONL/parquet artifacts

## Intended Use & Limitations

### Intended Use

- Evaluating pragmatic meaning preservation in MT for he/gu/vi
- Testing formality, gender, dialect, and pronoun register handling
- Conflict resolution in bucket C cued items

### Limitations

- Only three target languages (Hebrew, Gujarati, Vietnamese)
- Source language is English only
- 36 base items (720 variants total)—small-scale benchmark
- Pragmatic tiers are limited to those annotated in the data

## Licensing

MIT License

## Citation

```bibtex
@misc{pragmatic-mt-benchmark,
  title = {Pragmatic MT Benchmark for Hebrew, Gujarati, and Vietnamese},
  author = {Pragmatic MT Eval contributors},
  year = {2026},
  url = {https://github.com/pragmatic-mt-eval/pragmatic-mt-eval},
  note = {Reference-based, pragmatics-focused evaluation benchmark}
}
```
