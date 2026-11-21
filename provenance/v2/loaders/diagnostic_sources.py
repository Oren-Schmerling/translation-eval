"""Loaders and probes for Track B (register_contrastive) and Track C (diagnostic_morphology)."""

from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from provenance.v2.loaders.base import CandidateRow

load_dotenv()


def probe_cocoa_mt() -> Tuple[List[CandidateRow], dict]:
    """Dynamically probe facebook/CoCoA-MT on Hugging Face and log status."""
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    manifest_log: Dict[str, object] = {
        "source": "facebook/CoCoA-MT",
        "track": "register_contrastive",
        "supported_pairs": [],
        "missing_pairs": ["he", "gu", "vi"],
        "status": "pending",
    }
    try:
        api.repo_info("facebook/CoCoA-MT", repo_type="dataset")
        manifest_log["status"] = "accessible"
    except Exception as exc:
        manifest_log["status"] = f"skipped: not accessible on Hub ({type(exc).__name__})"

    return [], manifest_log


def probe_wino_mt() -> Tuple[List[CandidateRow], dict]:
    """Dynamically probe stanovsky/wino_mt on Hugging Face and log status."""
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    manifest_log: Dict[str, object] = {
        "source": "stanovsky/wino_mt",
        "track": "diagnostic_morphology",
        "status": "pending",
    }
    try:
        api.repo_info("stanovsky/wino_mt", repo_type="dataset")
        manifest_log["status"] = "accessible"
    except Exception as exc:
        manifest_log["status"] = f"skipped: not accessible on Hub ({type(exc).__name__})"

    return [], manifest_log


def load_tatoeba_baseline(lang: str, limit: int = 50) -> Tuple[List[CandidateRow], dict]:
    """Load isolated textbook sentences from Tatoeba (capped at limit items/lang)."""
    manifest_log = {"source": f"Helsinki-NLP/tatoeba en-{lang}", "track": "diagnostic_morphology", "status": "pending"}
    rows: List[CandidateRow] = []

    # Direct OPUS Tatoeba download (upstream of Helsinki-NLP/tatoeba)
    url = f"https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-{lang}.txt.zip"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                en_name = f"Tatoeba.en-{lang}.en"
                tgt_name = f"Tatoeba.en-{lang}.{lang}"
                if en_name in z.namelist() and tgt_name in z.namelist():
                    en_lines = z.read(en_name).decode("utf-8").splitlines()
                    tgt_lines = z.read(tgt_name).decode("utf-8").splitlines()

                    for en_line, tgt_line in zip(en_lines, tgt_lines):
                        en_clean = en_line.strip()
                        tgt_clean = tgt_line.strip()
                        if not en_clean or not tgt_clean:
                            continue
                        rows.append(
                            CandidateRow(
                                source_en=en_clean,
                                reference_target=tgt_clean,
                                source_corpus="Helsinki-NLP/tatoeba",
                                source_split="train",
                                origin_direction="source_first",
                                metadata={"baseline_type": "textbook_morphology"},
                            )
                        )
                        if len(rows) >= limit:
                            break
            manifest_log["status"] = f"loaded {len(rows)} textbook items"
        else:
            manifest_log["status"] = f"download failed HTTP {resp.status_code}"
    except Exception as exc:
        manifest_log["status"] = f"failed: {exc}"

    return rows, manifest_log
