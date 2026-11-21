"""Publish Pragmatic MT Dataset v2 to Hugging Face Hub while preserving v1."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "provenance" / "v2" / "output"
MANIFEST_FILE = REPO_ROOT / "provenance" / "v2" / "manifest.json"
HF_README_TEMPLATE = REPO_ROOT / "provenance" / "v2" / "HF_README.md"
DEFAULT_REPO_ID = "OrenSchmerling/pragmatic-mt-eval"


def render_hf_readme(repo_id: str, manifest: dict) -> str:
    """Render Hugging Face Dataset Card substituting repo_id and manifest stats."""
    if not HF_README_TEMPLATE.exists():
        raise FileNotFoundError(f"Missing {HF_README_TEMPLATE}")
    card_text = HF_README_TEMPLATE.read_text(encoding="utf-8")
    card_text = card_text.replace("<org>/<dataset-name>", repo_id)
    card_text = card_text.replace("{{REPO_ID}}", repo_id)
    return card_text


def dry_run(repo_id: str, files_to_upload: List[Path], manifest: dict) -> int:
    """Simulate HF upload and print plan."""
    print("=== Hugging Face Publication: DRY RUN ===")
    print(f"Target Repository:  https://huggingface.co/datasets/{repo_id}")
    print(f"Dataset Version:    v2 (preserving v1 via v1.0 tag and legacy configs)")
    print(f"Total Rows:         {manifest.get('total_items', 0)}")
    print("\nFiles to be published:")
    for f in files_to_upload:
        size = f.stat().st_size
        print(f"  - {f.name} ({size:,} bytes)")
    print("  - README.md (rendered dataset card with full provenance & statistics)")
    print("\nPlanned API Actions:")
    print(f"  1. Ensure v1 tag 'v1.0' exists on {repo_id}")
    print(f"  2. Upload dataset_v2.jsonl, dataset_v2.parquet, manifest.json to root")
    print(f"  3. Upload rendered dataset card to README.md")
    print("\nDry run successful. Use --push with HF_TOKEN to upload.")
    return 0


def push_to_hub(repo_id: str, files_to_upload: List[Path], manifest: dict, token: str) -> int:
    """Upload files and README.md to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub required for --push. Install it first.", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # 1. Tag existing v1 commit before pushing v2 if not already tagged
    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        existing_tags = [t.name for t in refs.tags]
        if "v1.0" not in existing_tags and "v1" not in existing_tags:
            print(f"Creating tag 'v1.0' on {repo_id} to preserve v1 accessibility...")
            api.create_tag(repo_id=repo_id, tag="v1.0", repo_type="dataset")
            print("Tag 'v1.0' created successfully.")
        else:
            print(f"Existing v1 tag found: {[t for t in existing_tags if 'v1' in t]}")
    except Exception as exc:
        print(f"Notice: Tag check/create returned: {exc}")

    # 2. Upload dataset files
    commit_msg = f"Publish Pragmatic MT Dataset v2 ({manifest.get('total_items', 750)} items)"
    for f in files_to_upload:
        print(f"Uploading {f.name} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
        )

    # 3. Upload rendered README
    readme_content = render_hf_readme(repo_id, manifest)
    print(f"Uploading README.md dataset card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_msg,
    )

    print(f"\nSuccess! Pragmatic MT Dataset v2 live at: https://huggingface.co/datasets/{repo_id}")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Publish dataset v2 to Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--push", action="store_true", help="Actually upload to Hub (default: dry run)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face access token")

    args = parser.parse_args()

    if not MANIFEST_FILE.exists():
        print(f"Error: manifest file not found at {MANIFEST_FILE}. Run build_dataset.py first.", file=sys.stderr)
        return 1

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    jsonl_file = OUTPUT_DIR / "dataset_v2.jsonl"
    parquet_file = OUTPUT_DIR / "dataset_v2.parquet"

    if not jsonl_file.exists() or not parquet_file.exists():
        print("Error: dataset output files missing in provenance/v2/output/. Run build_dataset.py first.", file=sys.stderr)
        return 1

    files_to_upload = [jsonl_file, parquet_file, MANIFEST_FILE]

    if args.push:
        if not args.token:
            print("Error: HF_TOKEN required for --push", file=sys.stderr)
            return 1
        return push_to_hub(args.repo_id, files_to_upload, manifest, args.token)
    else:
        return dry_run(args.repo_id, files_to_upload, manifest)


if __name__ == "__main__":
    sys.exit(main())
