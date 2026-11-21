"""Gujarati source loaders for conversational track."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from provenance.v2.filters import check_word_count, has_subtitle_artifacts
from provenance.v2.loaders.base import CandidateRow

load_dotenv()

GU_L1_MARKERS = ["તું", "તને", "તારું", "તારી", "તારા", "તારાથી"]
GU_L2_MARKERS = ["તમે", "તમને", "તમારું", "તમારી", "તમારા"]
GU_L3_MARKERS = ["આપ", "આપને", "આપનું", "આપની", "આપના"]


def load_gu_opus100(
    limit: int = 200,
    min_l1: int = 60,
    min_l3: int = 40,
) -> Tuple[List[CandidateRow], dict]:
    """Load Gujarati sentences from OPUS-100 enforcing clean L1 intimate and L3 honorific representation."""
    token = os.environ.get("HF_TOKEN")
    rows: List[CandidateRow] = []
    l1_rows: List[CandidateRow] = []
    l2_rows: List[CandidateRow] = []
    l3_rows: List[CandidateRow] = []
    manifest_log = {"source": "Helsinki-NLP/opus-100", "split": "train", "status": "pending"}

    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-gu", split="train", streaming=True, token=token)
        for i, item in enumerate(ds):
            trans = item.get("translation") or {}
            en = (trans.get("en") or "").strip()
            gu = (trans.get("gu") or "").strip()
            if not en or not gu:
                continue

            # Pre-filter for word count and artifacts to ensure high yield
            if not check_word_count(en, 6, 22) or has_subtitle_artifacts(en) or has_subtitle_artifacts(gu):
                continue

            # Classify register
            has_l1 = any(m in gu for m in GU_L1_MARKERS)
            has_l2 = any(m in gu for m in GU_L2_MARKERS)
            has_l3 = any(m in gu for m in GU_L3_MARKERS)

            row = CandidateRow(
                source_en=en,
                reference_target=gu,
                source_corpus="Helsinki-NLP/opus-100",
                source_split="train",
                origin_direction="source_first",
                metadata={"tier": "l1" if has_l1 else "l3" if has_l3 else "l2" if has_l2 else "other"},
            )

            if has_l1 and not has_l2 and not has_l3:
                if len(l1_rows) < min_l1:
                    l1_rows.append(row)
            elif has_l3 and not has_l1 and not has_l2:
                if len(l3_rows) < min_l3:
                    l3_rows.append(row)
            elif has_l2 and not has_l1 and not has_l3:
                if len(l2_rows) < (limit - min_l1 - min_l3):
                    l2_rows.append(row)

            if len(l1_rows) >= min_l1 and len(l3_rows) >= min_l3 and (len(l1_rows) + len(l2_rows) + len(l3_rows)) >= limit:
                break
            if i > 50000:
                break

        rows.extend(l1_rows)
        rows.extend(l3_rows)
        rows.extend(l2_rows)
        manifest_log["status"] = f"loaded {len(rows)} items (L1={len(l1_rows)}, L3={len(l3_rows)}, L2={len(l2_rows)})"
    except Exception as exc:
        manifest_log["error"] = str(exc)

    return rows, manifest_log


def probe_gu_baatcheet() -> Tuple[List[CandidateRow], dict]:
    """Inspect HimangY/BaatCheet-Corpus dynamically; ingest only if gu split exists."""
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    manifest_log = {"source": "HimangY/BaatCheet-Corpus", "status": "pending"}
    try:
        files = api.list_repo_files("HimangY/BaatCheet-Corpus", repo_type="dataset")
        gu_files = [f for f in files if "gu" in f.lower() or "guj" in f.lower()]
        if not gu_files:
            manifest_log["status"] = "skipped: no gu config"
            return [], manifest_log
        manifest_log["status"] = f"found gu files: {gu_files[:3]}"
        return [], manifest_log
    except Exception as exc:
        manifest_log["status"] = f"skipped: {exc}"
        return [], manifest_log


def probe_gu_snorbyte() -> Tuple[List[CandidateRow], dict]:
    """Attempt access to snorbyte dataset; log 401/403/gated cleanly."""
    token = os.environ.get("HF_TOKEN")
    manifest_log = {"source": "snorbyte/indic-audio-natural-conversations-sample", "status": "pending"}
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            "snorbyte/indic-audio-natural-conversations-sample",
            "data_shard_000_zstd.parquet",
            repo_type="dataset",
            token=token,
        )
        manifest_log["status"] = "accessible"
        return [], manifest_log
    except Exception as exc:
        manifest_log["status"] = f"skipped: gated ({type(exc).__name__})"
        return [], manifest_log


def load_gu_in22(limit: int = 70) -> Tuple[List[CandidateRow], dict]:
    """Load casual interpersonal dialogue turns from ai4bharat/IN22-Conv."""
    token = os.environ.get("HF_TOKEN")
    rows: List[CandidateRow] = []
    manifest_log = {"source": "ai4bharat/IN22-Conv", "split": "test", "status": "pending"}
    try:
        ds = load_dataset("ai4bharat/IN22-Conv", "default", split="test", streaming=True, token=token)
        for item in ds:
            en = (item.get("eng_Latn") or "").strip()
            gu = (item.get("guj_Gujr") or "").strip()
            if not en or not gu:
                continue
            rows.append(
                CandidateRow(
                    source_en=en,
                    reference_target=gu,
                    source_corpus="ai4bharat/IN22-Conv",
                    source_split="test",
                    origin_direction="source_first",
                    metadata={"scenario": item.get("scenario"), "speaker": item.get("speaker")},
                )
            )
            if len(rows) >= limit:
                break
        manifest_log["status"] = f"loaded {len(rows)} items"
    except Exception as exc:
        manifest_log["status"] = f"skipped: {exc}"
    return rows, manifest_log
