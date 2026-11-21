"""Main assembly pipeline for Pragmatic MT Dataset v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

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
    load_cocoa_mt_github,
    load_tatoeba_baseline,
    load_wino_mt_github,
)
from provenance.v2.loaders.gu_sources import (
    load_gu_in22,
    load_gu_opus100,
    load_gu_snorbyte,
    probe_gu_baatcheet,
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
    deduplicator: Optional[Deduplicator],
    target_quota: int,
    stats_stage: dict,
    allow_null_target: bool = False,
    allow_duplicate_en: bool = False,
) -> List[V2Item]:
    """Filter, backtranslate, harvest labels, and convert candidate rows into validated V2Items."""
    items: List[V2Item] = []
    track_code = "cas" if track == "casual_conversational" else "reg" if track == "register_contrastive" else "dia"

    for cand in candidates:
        stats_stage["candidates_seen"] += 1

        # 1. Backtranslate target-first sentences if needed
        if cand.origin_direction == "target_first" and cand.reference_target is not None:
            hint = cand.metadata.get("standard_vietnamese")
            source_en = backtranslator.translate(cand.reference_target, lang, hint_en=hint)
        else:
            source_en = cand.source_en or ""

        clean_en = normalize_source_en(source_en)

        # 2. Validate against self-containment & hygiene filters
        ok, reason = validate_candidate(
            clean_en,
            cand.reference_target,
            lang,
            deduplicator,
            allow_null_target=allow_null_target,
            allow_duplicate_en=allow_duplicate_en,
        )
        if not ok:
            stats_stage["rejected"][reason] = stats_stage["rejected"].get(reason, 0) + 1
            continue

        if not allow_duplicate_en and deduplicator is not None:
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
            reference_target=cand.reference_target.strip() if cand.reference_target is not None else None,
            source_corpus=cand.source_corpus,
            source_split=cand.source_split,
            origin_direction=cand.origin_direction,
            pragmatic_profile=profile,
            required_anchors=req_anchors,
            forbidden_markers=forb_markers,
            harvest_confidence="deterministic_exact" if cand.reference_target is not None else "metadata_verified",
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
        q_reg_pairs = min(limit_per_source or 20, 5)
        q_wino = min(limit_per_source or 20, 10)
    else:
        q_cas_lang = 200
        q_dia_lang = 50
        q_reg_pairs = 50
        q_wino = 100

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

    gu_cands_snor, snor_log = load_gu_snorbyte(limit=50 if not dry_run else 5)
    gu_cands_in22, in22_log = load_gu_in22(limit=150 if not dry_run else 10)
    _, baat_log = probe_gu_baatcheet()
    manifest["sources"]["gu_snorbyte"] = snor_log
    manifest["sources"]["gu_in22"] = in22_log
    manifest["sources"]["gu_baatcheet"] = baat_log

    # Prefetch snorbyte backtranslations in parallel
    if not dry_run and gu_cands_snor:
        backtranslator.prefetch([(c.reference_target, "gu", None) for c in gu_cands_snor])

    target_snor = min(len(gu_cands_snor), 20 if not dry_run else 2)
    gu_items_snor = process_candidates(
        gu_cands_snor, "gu", "casual_conversational", backtranslator, deduplicator, target_snor, st_gu
    )

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

    rem_gu = q_cas_lang - len(gu_items_snor) - len(gu_items_in22) - len(gu_items_l1) - len(gu_items_l3)
    gu_items_l2 = process_candidates(cands_l2, "gu", "casual_conversational", backtranslator, deduplicator, rem_gu, st_gu)

    gu_items = gu_items_snor + gu_items_in22 + gu_items_l1 + gu_items_l3 + gu_items_l2
    for i, it in enumerate(gu_items, 1):
        it.item_id = f"v2_gu_cas_{i:04d}"
    all_items.extend(gu_items)
    print(f"Gujarati casual items built: {len(gu_items)} (snorbyte: {len(gu_items_snor)}, IN22: {len(gu_items_in22)}, L1: {len(gu_items_l1)}, L3: {len(gu_items_l3)})")

    # -----------------------------------------------------------------------
    # Track A: Casual Conversational - Vietnamese (vi)
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Vietnamese Casual ---")
    st_vi = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
    stages_stats["vi_casual"] = st_vi

    # Enforce ViDia2Std + ViMD >= 50% of casual vi (target 110 items)
    target_dialect_total = int(q_cas_lang * 0.55) if not dry_run else 10
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

    vi_items_vimd = process_candidates(
        vi_cands_vimd, "vi", "casual_conversational", backtranslator, deduplicator, 20 if not dry_run else 2, st_vi
    )
    vimd_south = sum(1 for it in vi_items_vimd if it.pragmatic_profile.regional_dialect == "vi_south")
    vimd_north = len(vi_items_vimd) - vimd_south

    vi_items_south = process_candidates(
        cands_south, "vi", "casual_conversational", backtranslator, deduplicator, max(0, target_south - vimd_south), st_vi
    )
    vi_items_north = process_candidates(
        cands_north, "vi", "casual_conversational", backtranslator, deduplicator, max(0, target_north - vimd_north), st_vi
    )

    vi_tf_items = vi_items_south + vi_items_north + vi_items_vimd
    rem_vi = q_cas_lang - len(vi_tf_items)
    vi_items_open = process_candidates(vi_cands_open, "vi", "casual_conversational", backtranslator, deduplicator, rem_vi, st_vi)

    vi_items = vi_tf_items + vi_items_open
    for i, it in enumerate(vi_items, 1):
        it.item_id = f"v2_vi_cas_{i:04d}"
    all_items.extend(vi_items)
    print(f"Vietnamese casual items built: {len(vi_items)} (target-first: {len(vi_tf_items)}, ViDia2Std+ViMD: {len(vi_tf_items)})")

    # -----------------------------------------------------------------------
    # Track B: Register Contrastive (en-vi)
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Register Contrastive Track (vi) ---")
    st_reg = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
    stages_stats["register_contrastive_vi"] = st_reg

    cocoa_cands, cocoa_log = load_cocoa_mt_github(limit_pairs=q_reg_pairs)
    manifest["sources"]["amazon_science_cocoa_mt"] = cocoa_log

    reg_items = process_candidates(
        cocoa_cands,
        "vi",
        "register_contrastive",
        backtranslator,
        None,  # No deduplicator since formal/informal share source_en
        q_reg_pairs * 2,
        st_reg,
        allow_duplicate_en=True,
    )
    for i, it in enumerate(reg_items, 1):
        it.item_id = f"v2_vi_reg_{i:04d}"
    all_items.extend(reg_items)
    print(f"Register contrastive vi items built: {len(reg_items)} ({len(reg_items)//2} pairs)")

    # -----------------------------------------------------------------------
    # Track C: Diagnostic Morphology Baselines (WinoMT + Tatoeba)
    # -----------------------------------------------------------------------
    print("\n--- Harvesting Diagnostic Morphology Track ---")
    st_wino = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
    stages_stats["wino_mt_he"] = st_wino

    wino_cands, wino_log = load_wino_mt_github(limit_pro=q_wino // 2, limit_anti=q_wino // 2)
    manifest["sources"]["wino_mt"] = wino_log

    wino_items = process_candidates(
        wino_cands,
        "he",
        "diagnostic_morphology",
        backtranslator,
        deduplicator,
        q_wino,
        st_wino,
        allow_null_target=True,
    )
    for i, it in enumerate(wino_items, 1):
        it.item_id = f"v2_he_dia_wino_{i:04d}"
    all_items.extend(wino_items)
    print(f"Diagnostic WinoMT he items built: {len(wino_items)}")

    for lang in ["he", "gu", "vi"]:
        tat_cands, tat_log = load_tatoeba_baseline(lang, limit=q_dia_lang * 4)
        manifest["sources"][f"tatoeba_{lang}"] = tat_log
        st_dia = {"candidates_seen": 0, "accepted": 0, "rejected": {}}
        stages_stats[f"tatoeba_{lang}"] = st_dia
        dia_items = process_candidates(
            tat_cands, lang, "diagnostic_morphology", backtranslator, deduplicator, q_dia_lang, st_dia
        )
        for i, it in enumerate(dia_items, 1):
            it.item_id = f"v2_{lang}_dia_tat_{i:04d}"
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
    dist_track: Dict[str, int] = {}
    dist_lang: Dict[str, int] = {}
    dist_origin: Dict[str, int] = {}
    dist_origin_cas: Dict[str, Dict[str, int]] = {"source_first": {}, "target_first": {}}
    for it in all_items:
        dist_track[it.track] = dist_track.get(it.track, 0) + 1
        dist_lang[it.lang] = dist_lang.get(it.lang, 0) + 1
        dist_origin[it.origin_direction] = dist_origin.get(it.origin_direction, 0) + 1
        if it.track == "casual_conversational":
            sub = dist_origin_cas[it.origin_direction]
            sub[it.lang] = sub.get(it.lang, 0) + 1

    manifest["distributions"]["by_track"] = dist_track
    manifest["distributions"]["by_lang"] = dist_lang
    manifest["distributions"]["by_origin_direction"] = dist_origin
    manifest["distributions"]["casual_by_lang_origin"] = dist_origin_cas

    # Gujarati register counts
    gu_cas = [it for it in all_items if it.track == "casual_conversational" and it.lang == "gu"]
    gu_reg_counts: Dict[str, int] = {}
    for it in gu_cas:
        reg = it.pragmatic_profile.register_tier or "unmarked"
        gu_reg_counts[reg] = gu_reg_counts.get(reg, 0) + 1
    manifest["distributions"]["gu_casual_registers"] = gu_reg_counts

    # Vietnamese dialect counts
    vi_cas = [it for it in all_items if it.track == "casual_conversational" and it.lang == "vi"]
    vi_dia_counts: Dict[str, int] = {}
    for it in vi_cas:
        d = it.pragmatic_profile.regional_dialect
        if d:
            vi_dia_counts[d] = vi_dia_counts.get(d, 0) + 1
    manifest["distributions"]["vi_casual_dialects"] = vi_dia_counts

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Also sync manifest to provenance/v2/manifest.json
    root_manifest = REPO_ROOT / "provenance" / "v2" / "manifest.json"
    with open(root_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Dataset v2 Build Complete ===")
    print(f"Total items written: {len(all_items)}")
    print(f"JSONL: {jsonl_path} ({manifest['files']['dataset_v2.jsonl']['sha256'][:16]}...)")
    print(f"Parquet: {parquet_path}")
    print(f"Manifest: {root_manifest}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble Pragmatic MT Dataset v2.")
    parser.add_argument("--dry-run", action="store_true", help="Harvest small sample without full backtranslations")
    parser.add_argument("--limit-per-source", type=int, default=None, help="Debug limit on items per source")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Destination directory")
    args = parser.parse_args()

    return build(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        limit_per_source=args.limit_per_source,
    )


if __name__ == "__main__":
    sys.exit(main())
