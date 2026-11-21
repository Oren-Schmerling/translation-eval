"""Main dataset builder for Pragmatic MT benchmark v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from dotenv import load_dotenv

from provenance.v2.backtranslate import Backtranslator
from provenance.v2.filters import Deduplicator, normalize_source_en, validate_candidate
from provenance.v2.label_harvester import harvest_pragmatic_labels
from provenance.v2.loaders.base import CandidateRow
from provenance.v2.loaders.diagnostic_sources import (
    load_tatoeba_baseline,
    probe_cocoa_mt,
    probe_wino_mt,
)
from provenance.v2.loaders.gu_sources import (
    load_gu_in22,
    load_gu_opus100,
    probe_gu_baatcheet,
    probe_gu_snorbyte,
)
from provenance.v2.loaders.he_sources import (
    load_he_opensubtitles,
    load_he_spontaneous,
)
from provenance.v2.loaders.vi_sources import (
    load_vi_opensubtitles,
    load_vi_vidia2std,
    load_vi_vimd,
)
from provenance.v2.schema import V2Item

DEFAULT_OUT_DIR = REPO_ROOT / "provenance" / "v2" / "output"


def process_candidates(
    candidates: List[CandidateRow],
    lang: str,
    track: str,
    backtranslator: Backtranslator,
    deduplicator: Deduplicator,
    target_quota: int,
    stats_stage: dict,
) -> List[V2Item]:
    """Filter, backtranslate, harvest labels, and convert candidate rows into validated V2Items."""
    items: List[V2Item] = []
    track_code = "cas" if track == "casual_conversational" else "reg" if track == "register_contrastive" else "dia"

    for cand in candidates:
        stats_stage["candidates_seen"] += 1

        # 1. Backtranslate target-first sentences if needed
        if cand.origin_direction == "target_first":
            hint = cand.metadata.get("standard_vietnamese")
            source_en = backtranslator.translate(cand.reference_target, lang, hint_en=hint)
        else:
            source_en = cand.source_en or ""

        clean_en = normalize_source_en(source_en)

        # 2. Validate against self-containment & hygiene filters
        ok, reason = validate_candidate(clean_en, cand.reference_target, lang, deduplicator)
        if not ok:
            stats_stage["rejected"][reason] = stats_stage["rejected"].get(reason, 0) + 1
            continue

        deduplicator.add(clean_en)
        stats_stage["accepted"] += 1

        # 3. Deterministic backward label harvesting
        profile, req_anchors, forb_markers = harvest_pragmatic_labels(
            cand.reference_target, lang, metadata=cand.metadata
        )

        item_idx = len(items) + 1
        item_id = f"v2_{lang}_{track_code}_{item_idx:04d}"

        v2_item = V2Item(
            item_id=item_id,
            dataset_version="v2",
            track=track,  # type: ignore[arg-type]
            lang=lang,  # type: ignore[arg-type]
            source_en=clean_en,
            reference_target=cand.reference_target.strip(),
            source_corpus=cand.source_corpus,
            source_split=cand.source_split,
            origin_direction=cand.origin_direction,
            pragmatic_profile=profile,
            required_anchors=req_anchors,
            forbidden_markers=forb_markers,
            harvest_confidence="deterministic_exact",
        )
        items.append(v2_item)

        if len(items) >= target_quota:
            break

    return items


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of an exported file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build(
    output_dir: Path = DEFAULT_OUT_DIR,
    dry_run: bool = False,
    limit_per_source: Optional[int] = None,
) -> int:
    """Assemble tracks, apply filters and quotas, write outputs and manifest."""
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)
    backtranslator = Backtranslator(dry_run=dry_run)
    deduplicator = Deduplicator()

    manifest: Dict[str, object] = {
        "dataset_version": "v2",
        "dry_run": dry_run,
        "sources": {},
        "stages": {},
        "distributions": {},
        "files": {},
    }

    stages_stats: Dict[str, dict] = {}
    all_items: List[V2Item] = []

    # Quotas
    if dry_run:
        q_cas_lang = limit_per_source or 20
        q_dia_lang = min(limit_per_source or 20, 10)
    else:
        q_cas_lang = 200
        q_dia_lang = 50

    print(f"=== Assembling Pragmatic MT Dataset v2 (dry_run={dry_run}) ===")

    # -----------------------------------------------------------------------
    # Track A: Casual Conversational - Hebrew (he)
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Hebrew Casual ---")
    st_he = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
    stages_stats["he_casual"] = st_he

    he_cand_spont, he_spont_log = load_he_spontaneous(limit=300 if not dry_run else q_cas_lang)
    he_cand_open = load_he_opensubtitles(limit=500 if not dry_run else q_cas_lang)
    manifest["sources"]["he_spontaneous"] = he_spont_log
    manifest["sources"]["he_opensubtitles"] = "sentence-transformers/parallel-sentences-opensubtitles en-he"

    # Prefetch Hebrew backtranslations in parallel
    if not dry_run:
        backtranslator.prefetch([(c.reference_target, "he", None) for c in he_cand_spont])

    target_first_quota_he = int(q_cas_lang * 0.36) if not dry_run else int(q_cas_lang * 0.30)
    he_items_spont = process_candidates(
        he_cand_spont, "he", "casual_conversational", backtranslator, deduplicator, target_first_quota_he, st_he
    )
    he_items_open = process_candidates(
        he_cand_open, "he", "casual_conversational", backtranslator, deduplicator, q_cas_lang - len(he_items_spont), st_he
    )
    he_items = he_items_spont + he_items_open
    for i, it in enumerate(he_items, 1):
        it.item_id = f"v2_he_cas_{i:04d}"
    all_items.extend(he_items)
    print(f"Hebrew casual items built: {len(he_items)} (target-first: {len(he_items_spont)})")

    # -----------------------------------------------------------------------
    # Track A: Casual Conversational - Gujarati (gu)
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Gujarati Casual ---")
    st_gu = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
    stages_stats["gu_casual"] = st_gu

    gu_cands_in22, in22_log = load_gu_in22(limit=150 if not dry_run else q_cas_lang)
    _, baat_log = probe_gu_baatcheet()
    _, snor_log = probe_gu_snorbyte()
    manifest["sources"]["gu_in22"] = in22_log
    manifest["sources"]["gu_baatcheet"] = baat_log
    manifest["sources"]["gu_snorbyte"] = snor_log

    # Target blend: ~35% IN22-Conv (~70) + OPUS-100 L1/L3/L2 (~130)
    target_in22 = int(q_cas_lang * 0.35) if not dry_run else max(1, int(q_cas_lang * 0.35))
    gu_items_in22 = process_candidates(
        gu_cands_in22, "gu", "casual_conversational", backtranslator, deduplicator, target_in22, st_gu
    )

    # In OPUS-100: enforce balanced register with >=25% intimate, >=15% honorific
    target_l1 = int(q_cas_lang * 0.28) if not dry_run else max(1, int(q_cas_lang * 0.25))
    target_l3 = int(q_cas_lang * 0.18) if not dry_run else max(1, int(q_cas_lang * 0.15))
    gu_cands_opus, gu_opus_log = load_gu_opus100(limit=q_cas_lang * 3, min_l1=target_l1 + 10, min_l3=target_l3 + 10)
    manifest["sources"]["gu_opus100"] = gu_opus_log

    cands_l1 = [c for c in gu_cands_opus if c.metadata.get("tier") == "l1"]
    cands_l3 = [c for c in gu_cands_opus if c.metadata.get("tier") == "l3"]
    cands_l2 = [c for c in gu_cands_opus if c.metadata.get("tier") not in ("l1", "l3")]

    gu_items_l1 = process_candidates(cands_l1, "gu", "casual_conversational", backtranslator, deduplicator, target_l1, st_gu)
    gu_items_l3 = process_candidates(cands_l3, "gu", "casual_conversational", backtranslator, deduplicator, target_l3, st_gu)

    rem_gu = q_cas_lang - len(gu_items_in22) - len(gu_items_l1) - len(gu_items_l3)
    gu_items_l2 = process_candidates(cands_l2, "gu", "casual_conversational", backtranslator, deduplicator, rem_gu, st_gu)

    gu_items = gu_items_in22 + gu_items_l1 + gu_items_l3 + gu_items_l2
    for i, it in enumerate(gu_items, 1):
        it.item_id = f"v2_gu_cas_{i:04d}"
    all_items.extend(gu_items)
    print(f"Gujarati casual items built: {len(gu_items)} (IN22: {len(gu_items_in22)}, L1: {len(gu_items_l1)}, L3: {len(gu_items_l3)}, L2: {len(gu_items_l2)})")

    # -----------------------------------------------------------------------
    # Track A: Casual Conversational - Vietnamese (vi)
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Vietnamese Casual ---")
    st_vi = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
    stages_stats["vi_casual"] = st_vi

    # Target 60% South / 40% North for dialect items
    target_dialect_total = int(q_cas_lang * 0.50) if not dry_run else 10
    target_south = int(target_dialect_total * 0.60)
    target_north = target_dialect_total - target_south

    vi_cands_vidia, vidia_log = load_vi_vidia2std(target_south=target_south * 2, target_north=target_north * 2)
    vi_cands_vimd, vimd_log = load_vi_vimd(limit=30 if not dry_run else 10)
    vi_cands_open = load_vi_opensubtitles(limit=400 if not dry_run else q_cas_lang)

    manifest["sources"]["vi_vidia2std"] = vidia_log
    manifest["sources"]["vi_vimd"] = vimd_log
    manifest["sources"]["vi_opensubtitles"] = "sentence-transformers/parallel-sentences-opensubtitles en-vi"

    # Prefetch Vietnamese backtranslations in parallel
    if not dry_run:
        backtranslator.prefetch(
            [(c.reference_target, "vi", c.metadata.get("standard_vietnamese")) for c in vi_cands_vimd]
        )

    # Balance South vs North
    cands_south = [c for c in vi_cands_vidia if c.metadata.get("region") == "vi_south"]
    cands_north = [c for c in vi_cands_vidia if c.metadata.get("region") == "vi_north"]

    vi_items_south = process_candidates(cands_south, "vi", "casual_conversational", backtranslator, deduplicator, target_south, st_vi)
    vi_items_north = process_candidates(cands_north, "vi", "casual_conversational", backtranslator, deduplicator, target_north, st_vi)
    vi_items_vimd = process_candidates(vi_cands_vimd, "vi", "casual_conversational", backtranslator, deduplicator, 15 if not dry_run else 2, st_vi)

    vi_tf_items = vi_items_south + vi_items_north + vi_items_vimd
    rem_vi = q_cas_lang - len(vi_tf_items)
    vi_items_open = process_candidates(vi_cands_open, "vi", "casual_conversational", backtranslator, deduplicator, rem_vi, st_vi)

    vi_items = vi_tf_items + vi_items_open
    for i, it in enumerate(vi_items, 1):
        it.item_id = f"v2_vi_cas_{i:04d}"
    all_items.extend(vi_items)
    print(f"Vietnamese casual items built: {len(vi_items)} (target-first: {len(vi_tf_items)}, South: {len(vi_items_south)}, North: {len(vi_items_north)})")

    # -----------------------------------------------------------------------
    # Track B: Register Contrastive Probe
    # -----------------------------------------------------------------------
    _, cocoa_log = probe_cocoa_mt()
    manifest["sources"]["facebook_cocoa_mt"] = cocoa_log

    # -----------------------------------------------------------------------
    # Track C: Diagnostic Morphology Baselines
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Diagnostic Morphology Track ---")
    _, wino_log = probe_wino_mt()
    manifest["sources"]["stanovsky_wino_mt"] = wino_log

    for lang in ["he", "gu", "vi"]:
        tat_cands, tat_log = load_tatoeba_baseline(lang, limit=q_dia_lang * 4)
        manifest["sources"][f"tatoeba_{lang}"] = tat_log
        st_dia = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
        stages_stats[f"tatoeba_{lang}"] = st_dia
        dia_items = process_candidates(
            tat_cands, lang, "diagnostic_morphology", backtranslator, deduplicator, q_dia_lang, st_dia
        )
        for i, it in enumerate(dia_items, 1):
            it.item_id = f"v2_{lang}_dia_{i:04d}"
        all_items.extend(dia_items)
        print(f"Diagnostic Tatoeba {lang} built: {len(dia_items)}")

    # -----------------------------------------------------------------------
    # Serialization & Distribution Analytics
    # -----------------------------------------------------------------------
    jsonl_path = output_dir / "dataset_v2.jsonl"
    parquet_path = output_dir / "dataset_v2.parquet"

    rows_dict = [it.model_dump() for it in all_items]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows_dict:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows_dict)
    # Flatten pragmatic_profile for parquet serialization
    df_parquet = df.copy()
    for col in ["listener_gender", "speaker_gender", "register_tier", "regional_dialect", "social_relationship"]:
        df_parquet[col] = df_parquet["pragmatic_profile"].apply(lambda p: p.get(col) if isinstance(p, dict) else None)
    df_parquet = df_parquet.drop(columns=["pragmatic_profile"])
    df_parquet.to_parquet(parquet_path, index=False)

    manifest["total_items"] = len(all_items)
    manifest["stages"] = stages_stats
    manifest["files"]["dataset_v2.jsonl"] = {
        "path": str(jsonl_path),
        "sha256": compute_file_sha256(jsonl_path),
        "count": len(all_items),
    }
    manifest["files"]["dataset_v2.parquet"] = {
        "path": str(parquet_path),
        "sha256": compute_file_sha256(parquet_path),
        "count": len(all_items),
    }

    # Distributions
    dist: Dict[str, dict] = {
        "by_track": df["track"].value_counts().to_dict(),
        "by_lang": df["lang"].value_counts().to_dict(),
        "by_origin_direction": df["origin_direction"].value_counts().to_dict(),
    }
    casual_df = df[df["track"] == "casual_conversational"]
    dist["casual_by_lang_origin"] = (
        casual_df.groupby(["lang", "origin_direction"]).size().unstack(fill_value=0).to_dict()
    )

    # Register tier distribution for Gujarati casual
    gu_cas = casual_df[casual_df["lang"] == "gu"]
    gu_regs = gu_cas["pragmatic_profile"].apply(lambda p: p.get("register_tier")).value_counts().to_dict()
    dist["gu_casual_registers"] = gu_regs

    # Dialect distribution for Vietnamese casual
    vi_cas = casual_df[casual_df["lang"] == "vi"]
    vi_dias = vi_cas["pragmatic_profile"].apply(lambda p: p.get("regional_dialect")).dropna().value_counts().to_dict()
    dist["vi_casual_dialects"] = vi_dias

    manifest["distributions"] = dist

    manifest_path = REPO_ROOT / "provenance" / "v2" / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Dataset v2 Build Complete ===")
    print(f"Total items written: {len(all_items)}")
    print(f"JSONL: {jsonl_path} ({manifest['files']['dataset_v2.jsonl']['sha256'][:16]}...)")
    print(f"Parquet: {parquet_path}")
    print(f"Manifest: {manifest_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Pragmatic MT Dataset v2.")
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry-run with low limits")
    parser.add_argument("--limit-per-source", type=int, default=None, help="Limit items per source")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")

    args = parser.parse_args()
    return build(
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        limit_per_source=args.limit_per_source,
    )


if __name__ == "__main__":
    sys.exit(main())
