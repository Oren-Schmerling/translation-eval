# Dataset Construction and Provenance

This directory contains construction scripts, dataset templates, and provenance metadata for the published evaluation benchmark.

- **Published Dataset**: [https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval](https://huggingface.co/datasets/OrenSchmerling/pragmatic-mt-eval)

---

## Important Notice for End Users

This directory is **construction/provenance only**. End users evaluating translation systems do not need to generate or compile the dataset from scratch. Instead, simply fetch the published benchmark directly:

```bash
python3 scripts/fetch_dataset.py
```

---

## Construction Pipeline

If you are reconstructing the dataset or modifying the source templates, run the pipeline in this exact sequence:

1. **Generate and validate the curation templates**:
   ```bash
   python3 provenance/generate_sheets_template.py
   ```
2. **Compile the dataset and infer anchors**:
   ```bash
   cargo run -- compile --input-dir data/templates --output data/compiled/eval_dataset.json
   ```
3. **Prepare the Hugging Face export package**:
   ```bash
   python3 provenance/prepare_hf_dataset.py
   ```

---

## Dataset Breakdown (720 Variants / 36 Base Items)

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
