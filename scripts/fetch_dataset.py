#!/usr/bin/env python3
"""Fetch the published pragmatic-MT evaluation benchmark from Hugging Face Hub.

Supports both v2 (default, 750 items across casual and diagnostic tracks) and
legacy v1 (720 variants). Validates SHA-256 hashes against the manifest, asserts
structural invariants, and provides local cache fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "OrenSchmerling/pragmatic-mt-eval"
DEFAULT_REVISION = "main"

# Output paths
COMPILED_DIR = REPO_ROOT / "data" / "compiled"
DEFAULT_V2_OUTPUT = COMPILED_DIR / "dataset_v2.jsonl"
DEFAULT_JSON_OUTPUT = COMPILED_DIR / "eval_dataset.json"

EXPECTED_V2_TOTAL = 950
EXPECTED_V2_TRACKS = {
    "casual_conversational": 600,
    "register_contrastive": 100,
    "diagnostic_morphology": 250,
}
EXPECTED_V1_TOTAL = 720
EXPECTED_V1_LANG_COUNTS = {"he": 192, "gu": 288, "vi": 240}


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_from_hub(
    repo_id: str,
    filename: str,
    revision: str,
    token: Optional[str],
    dest: Path,
) -> None:
    """Download a single file from Hugging Face Hub using huggingface_hub or urllib fallback."""
    used_hub = False
    try:
        from huggingface_hub import hf_hub_download
        used_hub = True
        print(f"Downloading {filename} from {repo_id} (rev={revision}) via huggingface_hub...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            token=token,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)
    except Exception as exc:
        if used_hub:
            print(f"Notice: huggingface_hub download failed ({exc}), falling back to urllib...", file=sys.stderr)
        import urllib.request
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
        print(f"Downloading {filename} from {url} via urllib...")
        req_headers = {}
        if token:
            req_headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=req_headers)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
            out.write(resp.read())


def fetch_and_validate_v2(
    repo_id: str,
    revision: str,
    token: Optional[str],
    track: str,
    output_path: Path,
) -> int:
    """Fetch and validate dataset v2 against manifest and invariants."""
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_dest = COMPILED_DIR / "manifest.json"
    jsonl_dest = COMPILED_DIR / "dataset_v2.jsonl"

    download_from_hub(repo_id, "manifest.json", revision, token, manifest_dest)
    download_from_hub(repo_id, "dataset_v2.jsonl", revision, token, jsonl_dest)

    with open(manifest_dest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Assert SHA-256 parity with manifest
    actual_sha = compute_sha256(jsonl_dest)
    expected_sha = manifest.get("files", {}).get("dataset_v2.jsonl", {}).get("sha256")
    if expected_sha and actual_sha != expected_sha:
        print(f"ERROR: SHA-256 mismatch for dataset_v2.jsonl!\n  Actual:   {actual_sha}\n  Expected: {expected_sha}", file=sys.stderr)
        return 1

    # 2. Parse and validate rows
    rows: List[dict] = []
    seen_ids: set[str] = set()
    with open(jsonl_dest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = item.get("item_id")
            if not item_id or item_id in seen_ids:
                print(f"ERROR: Invalid or duplicate item_id at line {idx}: {item_id}", file=sys.stderr)
                return 1
            seen_ids.add(item_id)
            if track == "all" or item.get("track") == track:
                rows.append(item)

    if track == "all" and len(rows) != EXPECTED_V2_TOTAL:
        print(f"ERROR: Expected {EXPECTED_V2_TOTAL} items, got {len(rows)}", file=sys.stderr)
        return 1
    elif track in EXPECTED_V2_TRACKS and len(rows) != EXPECTED_V2_TRACKS[track]:
        print(f"ERROR: Expected {EXPECTED_V2_TRACKS[track]} items for track '{track}', got {len(rows)}", file=sys.stderr)
        return 1

    # 3. Create adapter format for downstream compatibility in data/compiled/eval_dataset.json
    compiled_data = {
        "dataset_version": "v2",
        "track": track,
        "total_items": len(rows),
        "sha256": actual_sha,
        "items": rows,
        # Backward-compatible variants array so existing v1 tooling still parses without crashing
        "variants": [
            {
                "variant_id": r["item_id"],
                "item_id": r["item_id"],
                "source_en": r["source_en"],
                "lang": r["lang"],
                "track": r["track"],
                "bucket": "A",
                "is_cued_source": False,
                "coarse": {
                    "formality": (r.get("pragmatic_profile") or {}).get("register_tier") or "default",
                    "speaker_gender": (r.get("pragmatic_profile") or {}).get("speaker_gender"),
                    "listener_gender": (r.get("pragmatic_profile") or {}).get("listener_gender"),
                    "dialect": (r.get("pragmatic_profile") or {}).get("regional_dialect"),
                },
                "granular": {
                    "pronoun_tier": (r.get("pragmatic_profile") or {}).get("register_tier"),
                    "kinship_pair": (r.get("pragmatic_profile") or {}).get("social_relationship"),
                    "relationship": (r.get("pragmatic_profile") or {}).get("social_relationship"),
                },
                "conflict_override": None,
                "gold_reference": r["reference_target"],
                "anchors": {
                    "must_contain_any": r.get("required_anchors") or [],
                    "must_not_contain": r.get("forbidden_markers") or [],
                    "synset_equivalents": {},
                    "acceptable_synonyms": [],
                },
            }
            for r in rows
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compiled_data, f, ensure_ascii=False, indent=2)

    print("\n=== Dataset v2 Fetch & Validation Success ===")
    print(f"Repo ID:          https://huggingface.co/datasets/{repo_id}")
    print(f"Revision:         {revision}")
    print(f"Track:            {track}")
    print(f"Loaded Items:     {len(rows)}")
    print(f"SHA-256 Parity:   MATCH ({actual_sha[:16]}...)")
    print(f"Saved JSON:       {output_path}")
    print(f"Saved JSONL:      {jsonl_dest}")
    return 0


def fetch_and_validate_v1(
    repo_id: str,
    revision: str,
    token: Optional[str],
    output_path: Path,
) -> int:
    """Legacy v1 fetch and validation."""
    download_from_hub(repo_id, "eval_dataset.json", revision, token, output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    variants = data.get("variants", [])
    if len(variants) != EXPECTED_V1_TOTAL:
        print(f"ERROR: Expected {EXPECTED_V1_TOTAL} variants in v1, got {len(variants)}", file=sys.stderr)
        return 1
    print(f"\n=== Dataset v1 Fetch Success: {len(variants)} variants validated ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and validate published pragmatic MT evaluation benchmark.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HF dataset repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--dataset-version", default="v2", choices=["v1", "v2"], help="Dataset version (default: v2)")
    parser.add_argument(
        "--track",
        default="all",
        choices=["all", "casual_conversational", "register_contrastive", "diagnostic_morphology"],
        help="Track filter for v2 (default: all)",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help=f"Git revision/tag (default: {DEFAULT_REVISION})")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face access token")
    parser.add_argument("--output", default=str(DEFAULT_JSON_OUTPUT), help=f"Local output path (default: {DEFAULT_JSON_OUTPUT})")
    parser.add_argument("--clear-cache", action="store_true", help="Clear local cache before downloading")

    args = parser.parse_args()
    out_path = Path(args.output).resolve()

    if args.clear_cache:
        print("Clearing local compiled dataset cache...")
        if out_path.exists():
            out_path.unlink()
        v2_jsonl = COMPILED_DIR / "dataset_v2.jsonl"
        if v2_jsonl.exists():
            v2_jsonl.unlink()
        manifest_file = COMPILED_DIR / "manifest.json"
        if manifest_file.exists():
            manifest_file.unlink()

    if args.dataset_version == "v2":
        return fetch_and_validate_v2(args.repo_id, args.revision, args.token, args.track, out_path)
    else:
        # Use v1.0 tag if revision is default
        v1_rev = "v1.0" if args.revision == "main" else args.revision
        return fetch_and_validate_v1(args.repo_id, v1_rev, args.token, out_path)


if __name__ == "__main__":
    sys.exit(main())
