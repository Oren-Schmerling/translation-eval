"""Hebrew source loaders for casual conversational track."""

from __future__ import annotations

import os
import re
from typing import List, Tuple
from datasets import load_dataset
from dotenv import load_dotenv
from provenance.v2.loaders.base import CandidateRow

load_dotenv()

CLEAN_HE_UTTERANCE = re.compile(r"^[\u0590-\u05FF\s,?!'\".–—\-]+$")


def load_he_opensubtitles(limit: int = 150) -> List[CandidateRow]:
    """Load parallel casual dialogue from opensubtitles en-he."""
    token = os.environ.get("HF_TOKEN")
    rows: List[CandidateRow] = []
    try:
        ds = load_dataset(
            "sentence-transformers/parallel-sentences-opensubtitles",
            "en-he",
            split="train",
            streaming=True,
            token=token,
        )
        for item in ds:
            en = (item.get("english") or "").strip()
            he = (item.get("non_english") or "").strip()
            if not en or not he:
                continue
            rows.append(
                CandidateRow(
                    source_en=en,
                    reference_target=he,
                    source_corpus="sentence-transformers/parallel-sentences-opensubtitles",
                    source_split="train",
                    origin_direction="source_first",
                    metadata={},
                )
            )
            if len(rows) >= limit:
                break
    except Exception as exc:
        print(f"Notice: Failed to stream opensubtitles en-he: {exc}")
    return rows


def _split_clean_hebrew_sentences(text: str) -> List[str]:
    """Split transcript text into clean standalone conversational utterances."""
    chunks = re.split(r"[.?!]\s*", text)
    valid: List[str] = []
    for c in chunks:
        clean = c.strip()
        words = clean.split()
        if 4 <= len(words) <= 18 and CLEAN_HE_UTTERANCE.match(clean):
            # Must contain actual Hebrew letters
            if any("\u0590" <= ch <= "\u05FF" for ch in clean):
                valid.append(clean)
    return valid


def load_he_spontaneous(limit: int = 100) -> Tuple[List[CandidateRow], dict]:
    """Load spontaneous spoken Hebrew transcripts from ivrit-ai with gated fallback."""
    token = os.environ.get("HF_TOKEN")
    rows: List[CandidateRow] = []
    manifest_log = {"attempted": [], "active_source": None, "status": "pending"}

    # Sources in priority order
    sources = [
        ("ivrit-ai/crowd-transcribe-v5", "gated / priority 1"),
        ("ivrit-ai/crowd-transcribe-v4", "gated / priority 2"),
        ("ivrit-ai/audio-v2-40s", "open spoken transcripts / fallback"),
        ("ivrit-ai/turbo-benchmark", "open spoken transcripts / secondary fallback"),
    ]

    for ds_id, note in sources:
        manifest_log["attempted"].append({"dataset": ds_id, "note": note})
        try:
            ds = load_dataset(ds_id, split="train", streaming=True, token=token)
            if "crowd-transcribe" in ds_id:
                ds = ds.select_columns(["sentence"])
            sample_count = 0
            for item in ds:
                raw_text = item.get("sentence") or item.get("text") or item.get("transcript") or ""
                if not raw_text:
                    continue
                sentences = _split_clean_hebrew_sentences(raw_text)
                for s in sentences:
                    rows.append(
                        CandidateRow(
                            source_en=None,
                            reference_target=s,
                            source_corpus=ds_id,
                            source_split="train",
                            origin_direction="target_first",
                            metadata={"original_dataset": ds_id},
                        )
                    )
                    if len(rows) >= limit:
                        break
                sample_count += 1
                if len(rows) >= limit or sample_count > 200:
                    break

            if rows:
                manifest_log["active_source"] = ds_id
                manifest_log["status"] = f"loaded {len(rows)} utterances"
                break
        except Exception as exc:
            manifest_log["attempted"][-1]["error"] = str(exc)[:120]
            continue

    return rows, manifest_log
