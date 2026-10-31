#!/usr/bin/env python3
"""
prepare_hf_dataset.py

Reads data/compiled/eval_dataset.json and the three data/templates/*_curation.csv
files, validates the dataset against expected invariants, and writes a canonical
export directory ready for HuggingFace Hub upload.

Stdlib only; pyarrow is optional (for parquet).
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
import sys
from pathlib import Path

# --- Constants -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_JSON = REPO_ROOT / "data" / "compiled" / "eval_dataset.json"
TEMPLATE_DIR = REPO_ROOT / "data" / "templates"
HF_CARD = REPO_ROOT / "provenance" / "hf_dataset_card.md"

EXPECTED_TOTAL = 720
EXPECTED_LANG_COUNTS = {"he": 192, "gu": 288, "vi": 240}
VALID_LANG = {"he", "gu", "vi"}
VALID_TRACK = {"family_and_relatives", "culture_and_daily_life"}
VALID_BUCKET = {"A", "B", "C"}

# Fields that must be present and non-empty in every variant
REQUIRED_NON_EMPTY_FIELDS = [
    "variant_id",
    "item_id",
    "source_en",
    "lang",
    "track",
    "bucket",
    "is_cued_source",
    "coarse",
    "granular",
    "conflict_override",  # can be null (not empty string)
    "gold_reference",
    "anchors",
]

# Required keys inside anchors
ANCHOR_KEYS = ["must_contain_any", "must_not_contain", "synset_equivalents", "acceptable_synonyms"]

# Required keys inside coarse
COARSE_KEYS = ["formality", "speaker_gender", "listener_gender", "dialect"]

# Required keys inside granular
GRANULAR_KEYS = ["pronoun_tier", "kinship_pair", "relationship"]

# Required keys inside conflict_override (when present)
CONFLICT_OVERRIDEs_KEYS = ["formality", "speaker_gender", "listener_gender", "dialect"]


# --- Validation helpers --------------------------------------------------------

def validate(variants: list[dict]) -> list[str]:
    """Run all validation checks. Returns list of error messages (empty = OK)."""
    errors: list[str] = []

    # 1. Total variant count
    if len(variants) != EXPECTED_TOTAL:
        errors.append(f"Expected {EXPECTED_TOTAL} variants, got {len(variants)}")

    # 2. Per-language counts
    lang_counts: dict[str, int] = {}
    for v in variants:
        lang_counts[v.get("lang", "")] = lang_counts.get(v.get("lang", ""), 0) + 1
    for lang, expected_count in EXPECTED_LANG_COUNTS.items():
        actual = lang_counts.get(lang, 0)
        if actual != expected_count:
            errors.append(f"Language '{lang}': expected {expected_count} variants, got {actual}")

    # 3. Unique variant_id
    variant_ids = [v.get("variant_id") for v in variants]
    seen: set[str] = set()
    for vid in variant_ids:
        if vid in seen:
            errors.append(f"Duplicate variant_id: {vid}")
        seen.add(vid)
    if len(seen) != len(variants):
        errors.append(
            f"Unique variant_ids ({len(seen)}) != total variants ({len(variants)})"
        )
    # 4. Required non-empty top-level fields
    for v in variants:
        for field in REQUIRED_NON_EMPTY_FIELDS:
            val = v.get(field)
            if val is None:
                # conflict_override can legitimately be null — allow it
                if field == "conflict_override":
                    continue
                errors.append(
                    f"Variant '{v.get('variant_id', '?')}': field '{field}' is null"
                )
            elif isinstance(val, str) and val.strip() == "":
                errors.append(
                    f"Variant '{v.get('variant_id', '?')}': field '{field}' is empty string"
                )
            elif isinstance(val, (dict, list)) and len(val) == 0 and field not in (
                "conflict_override",
            ):
                # anchors and coarse/granular can have empty dicts/lists; check below
                pass

    # 5. Lang / track / bucket enums
    for v in variants:
        vid = v.get("variant_id", "?")
        if v.get("lang") not in VALID_LANG:
            errors.append(f"Variant '{vid}': invalid lang '{v.get('lang')}'")
        if v.get("track") not in VALID_TRACK:
            errors.append(f"Variant '{vid}': invalid track '{v.get('track')}'")
        if v.get("bucket") not in VALID_BUCKET:
            errors.append(f"Variant '{vid}': invalid bucket '{v.get('bucket')}'")

    # 6. Anchors shape
    for v in variants:
        a = v.get("anchors")
        if not isinstance(a, dict):
            errors.append(f"Variant '{v.get('variant_id', '?')}': anchors is not a dict")
            continue
        for key in ANCHOR_KEYS:
            if key not in a:
                errors.append(
                    f"Variant '{v.get('variant_id', '?')}': anchors missing key '{key}'"
                )

    # 7. Coarse shape
    for v in variants:
        c = v.get("coarse")
        if not isinstance(c, dict):
            errors.append(f"Variant '{v.get('variant_id', '?')}': coarse is not a dict")
            continue
        for key in COARSE_KEYS:
            if key not in c:
                errors.append(
                    f"Variant '{v.get('variant_id', '?')}': coarse missing key '{key}'"
                )

    # 8. Granular shape
    for v in variants:
        g = v.get("granular")
        if not isinstance(g, dict):
            errors.append(f"Variant '{v.get('variant_id', '?')}': granular is not a dict")
            continue
        for key in GRANULAR_KEYS:
            if key not in g:
                errors.append(
                    f"Variant '{v.get('variant_id', '?')}': granular missing key '{key}'"
                )

    # 9. is_cued_source <-> conflict_override consistency
    for v in variants:
        vid = v.get("variant_id", "?")
        is_cued = v.get("is_cued_source", False)
        co = v.get("conflict_override")
        if is_cued and co is None:
            errors.append(
                f"Variant '{vid}': is_cued_source=True but conflict_override is null"
            )
        if not is_cued and co is not None:
            errors.append(
                f"Variant '{vid}': is_cued_source=False but conflict_override is not null "
                f"(value: {co})"
            )

    return errors


# --- Flattening ----------------------------------------------------------------

# Ordered list of flattened column names
FLATTEN_COLUMNS = [
    "variant_id",
    "item_id",
    "source_en",
    "lang",
    "track",
    "bucket",
    "is_cued_source",
    "formality",
    "speaker_gender",
    "listener_gender",
    "dialect",
    "pronoun_tier",
    "kinship_pair",
    "relationship",
    "conflict_override",
    "must_contain_any",
    "must_not_contain",
    "synset_equivalents",
    "acceptable_synonyms",
    "gold_reference",
]


def flatten_variant(v: dict) -> dict:
    """Flatten coarse/granular/anchors into a single flat JSON object."""
    coarse = v.get("coarse", {}) or {}
    granular = v.get("granular", {}) or {}
    anchors = v.get("anchors", {}) or {}

    return {
        "variant_id": v["variant_id"],
        "item_id": v["item_id"],
        "source_en": v["source_en"],
        "lang": v["lang"],
        "track": v["track"],
        "bucket": v["bucket"],
        "is_cued_source": v["is_cued_source"],
        "formality": coarse.get("formality"),
        "speaker_gender": coarse.get("speaker_gender"),
        "listener_gender": coarse.get("listener_gender"),
        "dialect": coarse.get("dialect"),
        "pronoun_tier": granular.get("pronoun_tier"),
        "kinship_pair": granular.get("kinship_pair"),
        "relationship": granular.get("relationship"),
        "conflict_override": v.get("conflict_override"),  # nested object or null
        "must_contain_any": anchors.get("must_contain_any", []),
        "must_not_contain": anchors.get("must_not_contain", []),
        "synset_equivalents": anchors.get("synset_equivalents", {}),
        "acceptable_synonyms": anchors.get("acceptable_synonyms", []),
        "gold_reference": v["gold_reference"],
    }


def sort_key(v_flat: dict) -> tuple:
    """Deterministic sort key: lang, then item_id, then variant_id."""
    return (v_flat["lang"], v_flat["item_id"], v_flat["variant_id"])


# --- Export functions ----------------------------------------------------------

def write_variants_jsonl(flat_variants: list[dict], out_path: Path) -> None:
    """Write flat variants as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for v in flat_variants:
            f.write(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n")


def write_variants_parquet(flat_variants: list[dict], out_path: Path) -> bool:
    """Write flat variants as parquet if pyarrow is available. Returns success."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("[NOTICE] pyarrow not available — skipping variants.parquet")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(flat_variants)
    pq.write_table(table, str(out_path))
    return True


def copy_curation_csvs(out_dir: Path) -> list[str]:
    """Copy the three curation CSVs verbatim."""
    copied = []
    for lang in ("he", "gu", "vi"):
        src = TEMPLATE_DIR / f"{lang}_curation.csv"
        dst = out_dir / f"{lang}_curation.csv"
        if not src.exists():
            raise FileNotFoundError(f"Curation CSV not found: {src}")
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def copy_readme(out_dir: Path) -> str:
    """Copy README.md from provenance/hf_dataset_card.md."""
    dst = out_dir / "README.md"
    shutil.copy2(HF_CARD, dst)
    return str(dst)


# --- Main -------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare curated pragmatic-MT benchmark for HuggingFace Hub upload"
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "hf_export"),
        help="Output directory for exported files (default: data/hf_export)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run validation only; do not write any files",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()

    # --- Load source data ---
    if not SRC_JSON.exists():
        print(f"ERROR: Compiled dataset not found: {SRC_JSON}", file=sys.stderr)
        return 1

    with open(SRC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants = data["variants"]
    print(f"Loaded {len(variants)} variants from {SRC_JSON}")

    # --- Validate ---
    errors = validate(variants)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors", file=sys.stderr)
        return 1

    print("Validation passed!")
    print(f"  - Total variants: {len(variants)}")
    lang_counts = {}
    for v in variants:
        lang_counts[v["lang"]] = lang_counts.get(v["lang"], 0) + 1
    for lang in sorted(lang_counts):
        print(f"  - {lang}: {lang_counts[lang]}")

    if args.check_only:
        print("\n--check-only: skipping file writes")
        return 0

    # --- Flatten + sort variants ---
    flat_variants = [flatten_variant(v) for v in variants]
    flat_variants.sort(key=sort_key)

    # --- Write files ---
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    canonical_path = out_dir / "eval_dataset.json"
    shutil.copy2(SRC_JSON, canonical_path)
    written.append(str(canonical_path))

    jsonl_path = out_dir / "variants.jsonl"
    write_variants_jsonl(flat_variants, jsonl_path)
    written.append(str(jsonl_path))
    print(f"  Wrote {len(flat_variants)} flat records to {jsonl_path}")

    parquet_path = out_dir / "variants.parquet"
    if write_variants_parquet(flat_variants, parquet_path):
        written.append(str(parquet_path))

    copied_csvs = copy_curation_csvs(out_dir)
    written.extend(copied_csvs)

    readme_path = copy_readme(out_dir)
    written.append(str(readme_path))

    # --- Post-export verification ---
    # 1. variants.jsonl line count must equal 720
    with open(jsonl_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    if line_count != EXPECTED_TOTAL:
        print(f"ERROR: variants.jsonl has {line_count} lines, expected {EXPECTED_TOTAL}", file=sys.stderr)
        return 1
    print(f"  Verified: variants.jsonl has {line_count} lines")

    # 2. eval_dataset.json must be byte-identical to the source compiled dataset
    if not filecmp.cmp(SRC_JSON, canonical_path, shallow=False):
        print(f"ERROR: eval_dataset.json is not byte-identical to {SRC_JSON}", file=sys.stderr)
        return 1
    print(f"  Verified: eval_dataset.json is byte-identical to {SRC_JSON}")

    # --- Summary ---
    print("\n=== Export Summary ===")
    print(f"Output directory: {out_dir}")
    print(f"Total variants: {len(variants)}")
    print(f"Flat records: {len(flat_variants)}")
    print(f"Files written ({len(written)}):")
    for f in written:
        size = Path(f).stat().st_size
        print(f"  {Path(f).name}  ({size:,} bytes)  -- {f}")
    print()
    print("Export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
