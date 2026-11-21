# Dataset Construction and Provenance

This directory contains construction scripts, dataset templates, and provenance metadata for the published evaluation benchmark.

- **Published Benchmark (v2)**: [https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval](https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval)
- **v2 Pipeline Directory**: [`provenance/v2/`](v2/)

---

## v2 Pipeline (Active Primary Benchmark)

All code for the v2 benchmark curation, filtering, harvesting, and publication is housed under [`provenance/v2/`](v2/):

- `provenance/v2/build_dataset.py`: Main curation pipeline across Tracks A, B, and C.
- `provenance/v2/filters.py`: Self-containment, length (6-22 words), subtitle artifact, and script hygiene filters.
- `provenance/v2/label_harvester.py`: Backward label harvesters for Hebrew, Gujarati, and Vietnamese.
- `provenance/v2/backtranslate.py`: Target-first backtranslation via OpenRouter with JSONL caching.
- `provenance/v2/publish_hf.py`: Hugging Face Hub publication client with v1 tag preservation.
- `provenance/v2/manifest.json`: Full provenance audit trail with exact counts, hashes, and distributions.
- `provenance/v2/tests/`: Quality verification test suite.

```bash
# Run v2 curation pipeline
python provenance/v2/build_dataset.py

# Verify all v2 quality assertions
pytest provenance/v2/tests/
```

---

## v1 Legacy Pipeline (Historical / Tagged v1.0)

The benchmark expands 36 English base sentences (18 family and relatives, 18 culture and daily life) through a deterministic pragmatic matrix into **720 total evaluation variants**:

| File | Rows | Dimensions |
|------|-----:|------------|
| `data/templates/he_curation.csv` | 192 | default/formal × speaker gender × listener gender (bucket C) |
| `data/templates/gu_curation.csv` | 288 | three formality levels × speaker gender × listener gender (bucket C) |
| `data/templates/vi_curation.csv` | 240 | northern/southern × kinship pair × speaker/listener gender (bucket C) |

### Curation CSV Columns

Each row represents one target-language variant of an English source item:

```
item_id, variant_id, source_en, lang, track, bucket, is_cued_source,
formality, speaker_gender, listener_gender, dialect,
pronoun_tier, kinship_pair, relationship,
conflict_formality, conflict_speaker_gender, conflict_listener_gender, conflict_dialect,
gold_reference
```

- `lang`: `he`, `gu`, or `vi`.
- `track`: `family_and_relatives` or `culture_and_daily_life`.
- `bucket`: `A` (listener focus), `B` (speaker focus), or `C` (cued dialogic).
- `is_cued_source`: `true` only for bucket C.
- `conflict_*`: Populated only for bucket C to drive Tier 2C adversarial overrides.
- `gold_reference`: Curated target translation used for chrF++ scoring and anchor verification.

---

## Artifacts and Git Ignore Policy

- `scripts/upload_to_hf.py` is intentionally kept local and gitignored (`scripts/upload_to_hf.py`), as it is not part of the published repository.
- Generated artifacts (`data/templates/*.csv`, `data/compiled/*.json`, `data/hf_export/`) are gitignored and generated locally during build/construction steps.
