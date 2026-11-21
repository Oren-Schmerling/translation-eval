"""Gujarati source loaders for conversational track."""

from __future__ import annotations

import ast
import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

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

            if not check_word_count(en, 6, 22) or has_subtitle_artifacts(en) or has_subtitle_artifacts(gu):
                continue

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


def load_gu_snorbyte(limit: int = 50) -> Tuple[List[CandidateRow], dict]:
    """Load text dialogue turns from snorbyte without audio decoding."""
    token = os.environ.get("HF_TOKEN")
    manifest_log = {
        "source": "snorbyte/indic-audio-natural-conversations-sample",
        "method": "pyarrow_metadata_only",
        "status": "pending",
    }
    rows: List[CandidateRow] = []

    try:
        fs = HfFileSystem(token=token)
        parquet_files = sorted(fs.glob("datasets/snorbyte/indic-audio-natural-conversations-sample/*.parquet"))
        if not parquet_files:
            manifest_log["status"] = "skipped: no parquet files found"
            return [], manifest_log

        for fpath in parquet_files:
            with fs.open(fpath, "rb") as f:
                pf = pq.ParquetFile(f)
                table = pf.read(
                    columns=["id", "language", "dialogue", "turns", "user_gender_side_0", "user_gender_side_1"]
                )
                df = table.to_pandas()

            gu_df = df[df["language"].astype(str).str.lower().isin(["gu", "gujarati"])]
            for _, r in gu_df.iterrows():
                try:
                    dialogue_turns = ast.literal_eval(r["dialogue"]) if isinstance(r["dialogue"], str) else r["dialogue"]
                    turn_indices = ast.literal_eval(r["turns"]) if isinstance(r["turns"], str) else r["turns"]
                except Exception:
                    continue

                g0 = "female" if "wom" in str(r.get("user_gender_side_0", "")).lower() else "male"
                g1 = "female" if "wom" in str(r.get("user_gender_side_1", "")).lower() else "male"

                for turn_idx, text in zip(turn_indices, dialogue_turns):
                    clean_text = str(text).strip()
                    if not clean_text or not check_word_count(clean_text, 4, 25):
                        continue
                    if has_subtitle_artifacts(clean_text):
                        continue

                    gender = g0 if (int(turn_idx) % 2 == 0) else g1
                    rows.append(
                        CandidateRow(
                            source_en=None,
                            reference_target=clean_text,
                            source_corpus="snorbyte/indic-audio-natural-conversations-sample",
                            source_split="samples",
                            origin_direction="target_first",
                            metadata={
                                "gender": gender,
                                "dialogue_id": str(r.get("id")),
                                "turn_idx": int(turn_idx),
                            },
                        )
                    )
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break

        manifest_log["status"] = f"loaded {len(rows)} natural conversational turns"
    except Exception as exc:
        manifest_log["status"] = f"skipped: {exc}"

    return rows, manifest_log


def load_gu_in22(limit: int = 70) -> Tuple[List[CandidateRow], dict]:
    """Load casual interpersonal dialogue turns from ai4bharat/IN22-Conv."""
    token = os.environ.get("HF_TOKEN")
    rows: List[CandidateRow] = []
    manifest_log = {"source": "ai4bharat/IN22-Conv", "split": "test", "status": "pending"}

    # Attempt eng_Latn-guj_Gujr config first; fall back to default if unavailable
    ds = None
    config_used = "eng_Latn-guj_Gujr"
    try:
        ds = load_dataset("ai4bharat/IN22-Conv", "eng_Latn-guj_Gujr", split="test", streaming=True, token=token)
    except Exception:
        config_used = "default"
        try:
            ds = load_dataset("ai4bharat/IN22-Conv", "default", split="test", streaming=True, token=token)
        except Exception as exc:
            manifest_log["status"] = f"failed to load: {exc}"
            return [], manifest_log

    manifest_log["config"] = config_used
    try:
        for item in ds:
            en = (item.get("sentence_eng_Latn") or item.get("eng_Latn") or "").strip()
            gu = (item.get("sentence_guj_Gujr") or item.get("guj_Gujr") or "").strip()
            if not en or not gu:
                continue
            rows.append(
                CandidateRow(
                    source_en=en,
                    reference_target=gu,
                    source_corpus="ai4bharat/IN22-Conv",
                    source_split="test",
                    origin_direction="source_first",
                    metadata={
                        "domain": item.get("domain"),
                        "topic": item.get("topic"),
                        "scenario": item.get("scenario"),
                        "turn": item.get("turn"),
                        "speaker": item.get("speaker"),
                    },
                )
            )
            if len(rows) >= limit:
                break
        manifest_log["status"] = f"loaded {len(rows)} items"
    except Exception as exc:
        manifest_log["status"] = f"error: {exc}"

    return rows, manifest_log
