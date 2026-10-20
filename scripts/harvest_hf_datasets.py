#!/usr/bin/env python3
"""Harvest candidate parallel data for the translation evaluation suite.

The script is intentionally defensive: Hugging Face datasets are optional and
every remote load is wrapped in ``try/except``.  When the ``datasets`` package
is missing, the network is unavailable, or no usable rows are found, the script
falls back to a small deterministic offline fixture covering Hebrew, Gujarati,
and Vietnamese.  It never raises because of a failed dataset load.

Examples
--------
    python3 scripts/harvest_hf_datasets.py --offline-fixture
    python3 scripts/harvest_hf_datasets.py --limit 50 \
        --output data/compiled/harvested.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "compiled" / "harvested.json"

# Candidate datasets explicitly requested by the evaluation plan.  Config names
# vary between dataset versions, so the loader tries a few common spellings and
# then ignores failures.
DATASET_CANDIDATES: List[dict] = [
    {
        "id": "IWSLT/iwslt2023",
        "aliases": ["IWSLT/iwslt2023", "iwslt2023"],
        "configs": ["en-he", "en-gu", "en-vi", "he-en", "gu-en", "vi-en"],
        "languages": ["he", "gu", "vi"],
        "split": "train",
    },
    {
        "id": "CoCoA-MT",
        "aliases": ["facebook/CoCoA-MT", "facebook/cocoamt", "CoCoA-MT/cocoamt"],
        "configs": [None, "en-he", "en-gu", "en-vi"],
        "languages": ["he", "gu", "vi"],
        "split": "train",
    },
    {
        "id": "Biu3010/ViDia2Std",
        "aliases": ["Biu3010/ViDia2Std"],
        "configs": [None],
        "languages": ["vi"],
        "split": "train",
    },
    {
        "id": "ai4bharat/IN22-Conv",
        "aliases": ["ai4bharat/IN22-Conv"],
        "configs": [None, "en-gu", "gu-en"],
        "languages": ["gu"],
        "split": "train",
    },
    {
        "id": "HimangY/BaatCheet-Corpus",
        "aliases": ["HimangY/BaatCheet-Corpus"],
        "configs": [None],
        "languages": ["gu"],
        "split": "train",
    },
    {
        "id": "tatoeba",
        "aliases": ["tatoeba", "Helsinki-NLP/tatoeba"],
        "configs": ["en-he", "en-gu", "en-vi", None],
        "languages": ["he", "gu", "vi"],
        "split": "train",
    },
    {
        "id": "stanovsky/wino_mt",
        "aliases": ["stanovsky/wino_mt", "stanovsky/WinoMT"],
        "configs": [None],
        "languages": ["he"],
        "split": "train",
    },
]


def _offline_fixture(limit: Optional[int]) -> List[dict]:
    """Deterministic synthetic rows so downstream tooling always has data."""
    fixtures: Dict[str, List[Tuple[str, str]]] = {
        "he": [
            ("Good morning, how are you?", "בוקר טוב, מה שלומך?"),
            ("I am going to the market now.", "אני הולך לשוק עכשיו."),
            ("She is reading a book by the window.", "היא קוראת ספר ליד החלון."),
            ("The children are playing in the garden.", "הילדים משחקים בגינה."),
            ("Please pass me the bread.", "בבקשה תעביר לי את הלחם."),
            ("My family lives in the city.", "המשפחה שלי גרה בעיר."),
            ("We will meet tomorrow morning.", "ניפגש מחר בבוקר."),
            ("Thank you for your help.", "תודה על העזרה שלך."),
            ("The train arrives at noon.", "הרכבת מגיעה בצהריים."),
            ("He is drinking coffee with his friend.", "הוא שותה קפה עם החבר שלו."),
            ("This soup is very tasty.", "המרק הזה טעים מאוד."),
            ("Can you open the door?", "אתה יכול לפתוח את הדלת?"),
        ],
        "gu": [
            ("Good morning, how are you?", "સુપ્રભાત, આપ કેમ છો?"),
            ("I am going to the market now.", "હું હવે બજારમાં જાઉં છું."),
            ("She is reading a book by the window.", "તે બારી પાસે પુસ્તક વાંચે છે."),
            ("The children are playing in the garden.", "બાળકો બગીચામાં રમે છે."),
            ("Please pass me the bread.", "કૃપા કરીને મને રોટલી આપો."),
            ("My family lives in the city.", "મારો પરિવાર શહેરમાં રહે છે."),
            ("We will meet tomorrow morning.", "આપણે કાલે સવારે મળીશું."),
            ("Thank you for your help.", "તમારી મદદ માટે આભાર."),
            ("The train arrives at noon.", "ટ્રેન બપોરે આવે છે."),
            ("He is drinking coffee with his friend.", "તે તેના મિત્ર સાથે કોફી પી રહ્યો છે."),
            ("This soup is very tasty.", "આ સૂપ ખૂબ સ્વાદિષ્ટ છે."),
            ("Can you open the door?", "તમે દરવાજો ખોલી શકો છો?"),
        ],
        "vi": [
            ("Good morning, how are you?", "Chào buổi sáng, bạn khỏe không?"),
            ("I am going to the market now.", "Tôi đang đi chợ bây giờ."),
            ("She is reading a book by the window.", "Cô ấy đang đọc sách bên cửa sổ."),
            ("The children are playing in the garden.", "Bọn trẻ đang chơi trong vườn."),
            ("Please pass me the bread.", "Làm ơn đưa cho tôi bánh mì."),
            ("My family lives in the city.", "Gia đình tôi sống ở thành phố."),
            ("We will meet tomorrow morning.", "Chúng ta sẽ gặp nhau vào sáng mai."),
            ("Thank you for your help.", "Cảm ơn bạn đã giúp đỡ."),
            ("The train arrives at noon.", "Tàu đến vào buổi trưa."),
            ("He is drinking coffee with his friend.", "Anh ấy đang uống cà phê với bạn."),
            ("This soup is very tasty.", "Món canh này rất ngon."),
            ("Can you open the door?", "Bạn có thể mở cửa không?"),
        ],
    }

    records: List[dict] = []
    for lang in ("he", "gu", "vi"):
        for index, (source, target) in enumerate(fixtures[lang], start=1):
            if limit is not None and len(records) >= limit:
                return records
            records.append(
                {
                    "id": f"offline-{lang}-{index:03d}",
                    "lang": lang,
                    "source_en": source,
                    "target": target,
                    "dataset": "offline_fixture",
                    "split": "synthetic",
                }
            )
    return records


def _get_first(example: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    translation = example.get("translation")
    if isinstance(translation, dict):
        for key in keys:
            value = translation.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalize_example(example: dict, dataset_id: str, lang: str, split: str) -> Optional[dict]:
    if not isinstance(example, dict):
        return None
    source = _get_first(
        example,
        ("source_en", "source", "src", "en", "english", "sentence_en"),
    )
    target = _get_first(
        example,
        (lang, "target", "tgt", "translation", "sentence", "text"),
    )
    if not source or not target or source == target:
        return None
    return {
        "id": f"{dataset_id}:{split}:" + hashlib.sha1(
            f"{source}\x1f{target}\x1f{lang}".encode("utf-8")
        ).hexdigest()[:16],
        "lang": lang,
        "source_en": source,
        "target": target,
        "dataset": dataset_id,
        "split": split,
    }


def _try_load_dataset(candidate: dict, remaining: Optional[int]) -> Tuple[List[dict], List[str]]:
    """Try a dataset's aliases/configs.  Returns (records, errors)."""
    records: List[dict] = []
    errors: List[str] = []
    if remaining is not None and remaining <= 0:
        return records, errors

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # ImportError or a broken optional dependency
        return records, [f"datasets import failed: {exc}"]

    for alias in candidate["aliases"]:
        for config in candidate["configs"]:
            if remaining is not None and len(records) >= remaining:
                return records, errors
            try:
                kwargs = {"split": candidate.get("split", "train"), "streaming": True}
                if config is not None:
                    kwargs["name"] = config
                dataset = load_dataset(alias, **kwargs)
                for example in dataset:
                    lang = candidate["languages"][0]
                    if config and "-" in config:
                        parts = config.split("-")
                        if parts[-1] in candidate["languages"]:
                            lang = parts[-1]
                    normalized = _normalize_example(
                        example, candidate["id"], lang, candidate.get("split", "train")
                    )
                    if normalized is not None:
                        records.append(normalized)
                        if remaining is not None and len(records) >= remaining:
                            break
            except Exception as exc:
                errors.append(f"{alias} ({config or 'default'}): {exc}")
    return records, errors


def harvest(offline_fixture: bool, limit: Optional[int]) -> dict:
    if offline_fixture:
        return {
            "schema_version": 1,
            "mode": "offline_fixture",
            "records": _offline_fixture(limit),
            "sources": [
                {
                    "dataset": "offline_fixture",
                    "loaded": True,
                    "note": "deterministic synthetic he/gu/vi fixture",
                }
            ],
        }

    records: List[dict] = []
    sources: List[dict] = []
    for candidate in DATASET_CANDIDATES:
        remaining = None if limit is None else limit - len(records)
        if remaining is not None and remaining <= 0:
            break
        loaded, errors = _try_load_dataset(candidate, remaining)
        records.extend(loaded)
        sources.append(
            {
                "dataset": candidate["id"],
                "loaded": bool(loaded),
                "records": len(loaded),
                "errors": errors[:3],
            }
        )

    if not records:
        fixture = _offline_fixture(limit)
        return {
            "schema_version": 1,
            "mode": "offline_fixture_fallback",
            "records": fixture,
            "sources": sources
            + [
                {
                    "dataset": "offline_fixture",
                    "loaded": True,
                    "note": "no online dataset rows were available",
                }
            ],
        }

    return {
        "schema_version": 1,
        "mode": "online",
        "records": records,
        "sources": sources,
    }


def _parse_limit(value: str) -> Optional[int]:
    if value.strip().lower() in ("", "none", "all", "unlimited"):
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--limit must be >= 0")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest parallel he/gu/vi data from Hugging Face datasets."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--offline-fixture",
        action="store_true",
        help="skip network access and emit deterministic synthetic rows",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=_parse_limit,
        help="maximum number of records to emit (default: unlimited)",
    )
    args = parser.parse_args(argv)

    try:
        payload = harvest(args.offline_fixture, args.limit)
    except Exception as exc:  # absolute last-resort guard
        print(f"warning: harvest failed unexpectedly ({exc}); using fixture", file=sys.stderr)
        payload = {
            "schema_version": 1,
            "mode": "offline_fixture_fallback",
            "records": _offline_fixture(args.limit),
            "sources": [],
        }

    output = Path(args.output)
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"wrote {len(payload['records'])} records to {output} "
        f"(mode={payload['mode']})"
    )
    for source in payload.get("sources", []):
        if not source.get("loaded"):
            first_error = (source.get("errors") or ["unavailable"])[0]
            print(f"warning: {source['dataset']} unavailable: {first_error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
