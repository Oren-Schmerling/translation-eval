"""Vietnamese source loaders for casual conversational track."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import pandas as pd
from provenance.v2.loaders.base import CandidateRow

load_dotenv()


def load_vi_opensubtitles(limit: int = 100) -> List[CandidateRow]:
    """Load parallel casual dialogue from opensubtitles / opus-100 en-vi."""
    token = os.environ.get("HF_TOKEN")
    rows: List[CandidateRow] = []
    # 1. Try opensubtitles
    try:
        ds = load_dataset(
            "sentence-transformers/parallel-sentences-opensubtitles",
            "en-vi",
            split="train",
            streaming=True,
            token=token,
        )
        for item in ds:
            en = (item.get("english") or "").strip()
            vi = (item.get("non_english") or "").strip()
            if not en or not vi:
                continue
            rows.append(
                CandidateRow(
                    source_en=en,
                    reference_target=vi,
                    source_corpus="sentence-transformers/parallel-sentences-opensubtitles",
                    source_split="train",
                    origin_direction="source_first",
                    metadata={},
                )
            )
            if len(rows) >= limit:
                break
    except Exception as exc:
        print(f"Notice: Failed to stream opensubtitles en-vi: {exc}")

    # 2. Fallback to opus-100 if needed
    if len(rows) < limit:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train", streaming=True, token=token)
            for item in ds:
                trans = item.get("translation") or {}
                en = (trans.get("en") or "").strip()
                vi = (trans.get("vi") or "").strip()
                if not en or not vi:
                    continue
                rows.append(
                    CandidateRow(
                        source_en=en,
                        reference_target=vi,
                        source_corpus="Helsinki-NLP/opus-100",
                        source_split="train",
                        origin_direction="source_first",
                        metadata={},
                    )
                )
                if len(rows) >= limit:
                    break
        except Exception as exc:
            print(f"Notice: Failed to stream opus-100 en-vi: {exc}")

    return rows


def load_vi_vidia2std(
    target_south: int = 75,
    target_north: int = 50,
) -> Tuple[List[CandidateRow], dict]:
    """Direct CSV download from Biu3010/ViDia2Std preserving dialect as native target gold."""
    token = os.environ.get("HF_TOKEN")
    manifest_log = {"source": "Biu3010/ViDia2Std", "method": "direct_csv_download", "status": "pending"}
    rows: List[CandidateRow] = []
    south_rows: List[CandidateRow] = []
    north_rows: List[CandidateRow] = []

    splits = ["test.csv", "dev.csv", "train.csv"]
    for split_fname in splits:
        try:
            fpath = hf_hub_download(
                "Biu3010/ViDia2Std",
                split_fname,
                repo_type="dataset",
                token=token,
            )
            df = pd.read_csv(fpath)
            for _, r in df.iterrows():
                region = str(r.get("region") or "").strip().lower()
                dialect_text = str(r.get("dialect") or "").strip()
                std_text = str(r.get("standard") or "").strip()

                if not dialect_text or region == "central":
                    continue

                row = CandidateRow(
                    source_en=None,
                    reference_target=dialect_text,
                    source_corpus="Biu3010/ViDia2Std",
                    source_split=split_fname.replace(".csv", ""),
                    origin_direction="target_first",
                    metadata={
                        "standard_vietnamese": std_text,
                        "region": "vi_south" if region == "southern" else "vi_north",
                    },
                )

                if region == "southern" and len(south_rows) < target_south:
                    south_rows.append(row)
                elif region == "northern" and len(north_rows) < target_north:
                    north_rows.append(row)

                if len(south_rows) >= target_south and len(north_rows) >= target_north:
                    break
            if len(south_rows) >= target_south and len(north_rows) >= target_north:
                break
        except Exception as exc:
            print(f"Notice: Failed to load ViDia2Std split {split_fname}: {exc}")

    rows.extend(south_rows)
    rows.extend(north_rows)
    manifest_log["status"] = f"loaded {len(rows)} items (South={len(south_rows)}, North={len(north_rows)})"
    return rows, manifest_log


def load_vi_vimd(limit: int = 30) -> Tuple[List[CandidateRow], dict]:
    """Load metadata columns only from nguyendv02/ViMD_Dataset with zero audio decoding."""
    token = os.environ.get("HF_TOKEN")
    manifest_log = {"source": "nguyendv02/ViMD_Dataset", "method": "parquet_metadata_only", "status": "pending"}
    rows: List[CandidateRow] = []

    try:
        import pyarrow.parquet as pq
        fpath = hf_hub_download(
            "nguyendv02/ViMD_Dataset",
            "data/test-00000-of-00014.parquet",
            repo_type="dataset",
            token=token,
        )
        schema = pq.read_schema(fpath)
        meta_cols = [c for c in schema.names if c != "audio"]
        table = pq.read_table(fpath, columns=meta_cols)
        df = table.to_pandas()

        max_south = int(limit * 0.6)
        max_north = limit - max_south
        south_rows: List[CandidateRow] = []
        north_rows: List[CandidateRow] = []

        for _, r in df.iterrows():
            reg = str(r.get("region") or "").strip()
            txt = str(r.get("text") or "").strip()
            g = str(r.get("gender") or "").strip()
            if not txt or reg not in ("North", "South"):
                continue

            gender_label = "male" if g == "1" else "female" if g == "2" else None
            region_label = "vi_north" if reg == "North" else "vi_south"

            row = CandidateRow(
                source_en=None,
                reference_target=txt,
                source_corpus="nguyendv02/ViMD_Dataset",
                source_split="test",
                origin_direction="target_first",
                metadata={
                    "region": region_label,
                    "gender": gender_label,
                },
            )

            if region_label == "vi_south" and len(south_rows) < max_south:
                south_rows.append(row)
            elif region_label == "vi_north" and len(north_rows) < max_north:
                north_rows.append(row)

            if len(south_rows) >= max_south and len(north_rows) >= max_north:
                break

        rows.extend(south_rows)
        rows.extend(north_rows)
        manifest_log["status"] = f"loaded {len(rows)} items (South={len(south_rows)}, North={len(north_rows)})"
    except Exception as exc:
        manifest_log["status"] = f"skipped: {exc}"

    return rows, manifest_log
