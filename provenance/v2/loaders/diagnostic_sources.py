"""Loaders and probes for Track B (register_contrastive) and Track C (diagnostic_morphology)."""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import requests
from provenance.v2.filters import check_word_count, has_subtitle_artifacts
from provenance.v2.loaders.base import CandidateRow


def load_cocoa_mt_github(limit_pairs: int = 50) -> Tuple[List[CandidateRow], dict]:
    """Fetch EN-VI formality-control contrastive test pairs directly from GitHub."""
    base_url = (
        "https://raw.githubusercontent.com/amazon-science/contrastive-controlled-mt/main"
        "/IWSLT2023/data/test/en-vi/"
    )
    manifest_log: Dict[str, object] = {
        "source": "amazon-science/contrastive-controlled-mt",
        "track": "register_contrastive",
        "supported_pairs": ["vi"],
        "missing_pairs": ["he", "gu"],
        "note": "he and gu unsupported upstream in CoCoA-MT",
        "status": "pending",
    }
    rows: List[CandidateRow] = []

    try:
        def fetch_lines(filename: str) -> List[str]:
            url = base_url + filename
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8").splitlines()

        en_lines = fetch_lines("formality-control.test.en-vi.en")
        formal_lines = fetch_lines("formality-control.test.en-vi.formal.vi")
        informal_lines = fetch_lines("formality-control.test.en-vi.informal.vi")
        formal_ann = fetch_lines("formality-control.test.en-vi.formal.annotated.vi")
        informal_ann = fetch_lines("formality-control.test.en-vi.informal.annotated.vi")

        pair_count = 0
        for i in range(min(len(en_lines), len(formal_lines), len(informal_lines))):
            en = en_lines[i].strip()
            formal = formal_lines[i].strip()
            informal = informal_lines[i].strip()

            if not en or not formal or not informal:
                continue
            if not check_word_count(en, 4, 30) or has_subtitle_artifacts(en):
                continue

            # Extract formality anchors from [F]...[/F] annotations
            f_anchors = list(dict.fromkeys(re.findall(r"\[F\](.*?)\[/F\]", formal_ann[i])))
            inf_anchors = list(dict.fromkeys(re.findall(r"\[F\](.*?)\[/F\]", informal_ann[i])))

            pair_id = f"vi_reg_{pair_count + 1:04d}"

            # Formal variant
            rows.append(
                CandidateRow(
                    source_en=en,
                    reference_target=formal,
                    source_corpus="amazon-science/contrastive-controlled-mt",
                    source_split="test",
                    origin_direction="source_first",
                    metadata={
                        "pair_id": pair_id,
                        "formality": "formal",
                        "contrastive_anchors": f_anchors,
                        "contrastive_forbidden": inf_anchors,
                    },
                )
            )

            # Informal variant
            rows.append(
                CandidateRow(
                    source_en=en,
                    reference_target=informal,
                    source_corpus="amazon-science/contrastive-controlled-mt",
                    source_split="test",
                    origin_direction="source_first",
                    metadata={
                        "pair_id": pair_id,
                        "formality": "informal",
                        "contrastive_anchors": inf_anchors,
                        "contrastive_forbidden": f_anchors,
                    },
                )
            )

            pair_count += 1
            if pair_count >= limit_pairs:
                break

        manifest_log["status"] = f"loaded {len(rows)} contrastive items ({pair_count} pairs) for vi"
    except Exception as exc:
        manifest_log["status"] = f"failed: {exc}"

    return rows, manifest_log


def load_wino_mt_github(
    limit_pro: int = 50, limit_anti: int = 50
) -> Tuple[List[CandidateRow], dict]:
    """Fetch WinoMT grammatical gender test suite directly from raw GitHub."""
    base_url = "https://raw.githubusercontent.com/gabrielStanovsky/mt_gender/master/data/aggregates/"
    manifest_log: Dict[str, object] = {
        "source": "gabrielStanovsky/mt_gender",
        "track": "diagnostic_morphology",
        "status": "pending",
    }
    rows: List[CandidateRow] = []

    try:
        def fetch_wino_split(split_name: str, target_count: int) -> List[CandidateRow]:
            url = base_url + split_name
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                lines = resp.read().decode("utf-8").splitlines()

            half = target_count // 2
            m_rows: List[CandidateRow] = []
            f_rows: List[CandidateRow] = []

            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                gender, sent_idx, sentence, profession = parts[0], parts[1], parts[2], parts[3]
                sentence = sentence.strip()
                if not check_word_count(sentence, 5, 30):
                    continue

                row = CandidateRow(
                    source_en=sentence,
                    reference_target=None,  # Intentionally null per schema specification
                    source_corpus="gabrielStanovsky/mt_gender",
                    source_split=split_name.replace(".txt", ""),
                    origin_direction="source_first",
                    metadata={
                        "gender": gender,
                        "profession": profession,
                        "sent_idx": sent_idx,
                        "stereotype": "pro" if "pro" in split_name else "anti",
                    },
                )

                if gender == "male" and len(m_rows) < half:
                    m_rows.append(row)
                elif gender == "female" and len(f_rows) < (target_count - half):
                    f_rows.append(row)

                if len(m_rows) >= half and len(f_rows) >= (target_count - half):
                    break

            return m_rows + f_rows

        pro_rows = fetch_wino_split("en_pro.txt", limit_pro)
        anti_rows = fetch_wino_split("en_anti.txt", limit_anti)
        rows.extend(pro_rows)
        rows.extend(anti_rows)

        manifest_log["status"] = f"loaded {len(rows)} items (pro={len(pro_rows)}, anti={len(anti_rows)})"
    except Exception as exc:
        manifest_log["status"] = f"failed: {exc}"

    return rows, manifest_log


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
