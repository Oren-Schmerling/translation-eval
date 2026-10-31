#!/usr/bin/env python3
"""Fetch the published pragmatic-MT evaluation dataset from Hugging Face Hub.

Uses huggingface_hub if available, falling back to urllib.request. Validates
the downloaded dataset against structural invariants (720 variants, per-language
counts, unique variant IDs, required fields, and anchors).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "OrenSchmerling/pragmatic-mt-eval"
DEFAULT_FILENAME = "eval_dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "compiled" / "eval_dataset.json"
DEFAULT_REVISION = "main"

EXPECTED_TOTAL = 720
EXPECTED_LANG_COUNTS = {"he": 192, "gu": 288, "vi": 240}
REQUIRED_FIELDS = [
    "variant_id",
    "item_id",
    "source_en",
    "lang",
    "track",
    "bucket",
    "gold_reference",
]


def validate_dataset(data: dict) -> list[str]:
    errors = []
    if not isinstance(data, dict):
        return ["Top-level JSON is not an object"]
    variants = data.get("variants")
    if not isinstance(variants, list):
        return ["Missing 'variants' list at top level"]

    if len(variants) != EXPECTED_TOTAL:
        errors.append(f"Expected {EXPECTED_TOTAL} variants, got {len(variants)}")

    lang_counts: dict[str, int] = {}
    variant_ids: set[str] = set()

    for i, v in enumerate(variants):
        if not isinstance(v, dict):
            errors.append(f"Variant at index {i} is not a dict")
            continue
        vid = v.get("variant_id")
        if not vid or not str(vid).strip():
            errors.append(f"Variant at index {i} has empty or missing variant_id")
        elif vid in variant_ids:
            errors.append(f"Duplicate variant_id: {vid}")
        else:
            variant_ids.add(vid)

        lang = v.get("lang")
        if lang in EXPECTED_LANG_COUNTS:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        for field in REQUIRED_FIELDS:
            val = v.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(f"Variant '{vid or i}': missing or empty required field '{field}'")

        anchors = v.get("anchors")
        if not isinstance(anchors, dict):
            errors.append(f"Variant '{vid or i}': missing or invalid 'anchors' object")

    for lang, expected_count in EXPECTED_LANG_COUNTS.items():
        actual = lang_counts.get(lang, 0)
        if actual != expected_count:
            errors.append(f"Language '{lang}': expected {expected_count} variants, got {actual}")

    return errors


def download_file(repo_id: str, filename: str, revision: str, token: str | None, dest: Path) -> None:
    used_hub = False
    try:
        from huggingface_hub import hf_hub_download
        used_hub = True
        print(f"Downloading {filename} from {repo_id} (revision={revision}) using huggingface_hub...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            token=token,
        )
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_path, dest)
    except (ImportError, Exception) as exc:
        if used_hub:
            print(f"Notice: huggingface_hub download failed ({exc}), falling back to urllib...", file=sys.stderr)
        import urllib.request
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
        print(f"Downloading {filename} from {url} using urllib.request...")
        req_headers = {}
        if token:
            req_headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=req_headers)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
                out_file.write(response.read())
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from {url}: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and validate published evaluation dataset from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help=f"Filename in repository (default: {DEFAULT_FILENAME})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Local output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help=f"Repository revision/branch (default: {DEFAULT_REVISION})")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face access token (or set HF_TOKEN env var)")

    args = parser.parse_args()
    out_path = Path(args.output).resolve()

    if out_path.exists():
        print(f"Notice: Output file {out_path} already exists and will be overwritten.")

    try:
        download_file(args.repo_id, args.filename, args.revision, args.token, out_path)
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        return 1

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"ERROR: Failed to parse downloaded JSON: {exc}", file=sys.stderr)
        return 1

    errors = validate_dataset(data)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors", file=sys.stderr)
        return 1

    dataset_url = f"https://huggingface.co/datasets/{args.repo_id}"
    print("\n=== Dataset Fetch & Validation Success ===")
    print(f"Dataset URL: {dataset_url}")
    print(f"Saved to:    {out_path}")
    print("Summary:")
    print(f"  - Total variants: {len(data['variants'])}")
    for lang, count in EXPECTED_LANG_COUNTS.items():
        print(f"  - Language '{lang}': {count} variants")
    print("  - All required fields, unique IDs, and anchors present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
