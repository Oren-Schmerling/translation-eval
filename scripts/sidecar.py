#!/usr/bin/env python3
"""Sidecar utilities for COMET scoring and human validation exports.

Subcommands
-----------
    python3 scripts/sidecar.py comet [--input PATH] [--output PATH]
    python3 scripts/sidecar.py export-validation-quiz [--input PATH] ...

The COMET subcommand treats ``unbabel-comet`` as optional: if it is not
installed the command prints a warning and exits successfully.  The quiz export
includes every row flagged as ``high_disagreement_flag`` plus a deterministic
stratified sample (by language and tier), writes a blinded CSV, and emits a
Google-Forms-oriented JSON payload.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "results.jsonl"
DEFAULT_COMET_OUTPUT = REPO_ROOT / "results" / "results.comet.jsonl"
DEFAULT_QUIZ_CSV = REPO_ROOT / "results" / "validation_quiz.csv"
DEFAULT_QUIZ_JSON = REPO_ROOT / "results" / "validation_quiz_forms.json"

QUIZ_COLUMNS = ["quiz_id", "lang", "source_en", "hypothesis", "reference"]


def _resolve(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    return output


def read_jsonl(path: Path) -> List[dict]:
    """Read JSONL defensively, skipping blank and malformed lines."""
    records: List[dict] = []
    if not path.exists():
        print(f"warning: input file not found: {path}", file=sys.stderr)
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    print(
                        f"warning: skipping malformed JSON on {path}:{line_number} ({exc})",
                        file=sys.stderr,
                    )
                    continue
                if isinstance(value, dict):
                    records.append(value)
                else:
                    print(
                        f"warning: skipping non-object JSON on {path}:{line_number}",
                        file=sys.stderr,
                    )
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _first(record: dict, keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return default


def _source(record: dict) -> str:
    return _first(record, ("source_en", "source", "src", "en", "english"))


def _hypothesis(record: dict) -> str:
    return _first(
        record,
        ("hypothesis", "mt", "translation", "output", "system_output", "target"),
    )


def _reference(record: dict) -> str:
    return _first(record, ("reference", "ref", "gold_reference", "gold", "target_ref"))


def comet(args: argparse.Namespace) -> int:
    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    records = read_jsonl(input_path)

    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
    except Exception as exc:  # ImportError or optional dependency failure
        print(
            "warning: unbabel-comet is not installed; skipping COMET scoring "
            f"({exc})",
            file=sys.stderr,
        )
        # Create the expected artifact even when the optional dependency is absent.
        write_jsonl(output_path, [])
        return 0

    if not records:
        write_jsonl(output_path, [])
        return 0

    data = []
    kept: List[dict] = []
    for record in records:
        source = _source(record)
        hypothesis = _hypothesis(record)
        reference = _reference(record)
        if not hypothesis:
            print("warning: skipping result without a hypothesis", file=sys.stderr)
            continue
        data.append({"src": source, "mt": hypothesis, "ref": reference})
        kept.append(record)

    if not data:
        write_jsonl(output_path, [])
        return 0

    try:
        model_path = download_model(args.model)
        model = load_from_checkpoint(model_path)
        predictions = model.predict(data, batch_size=args.batch_size, gpus=0)
        scores = getattr(predictions, "scores", predictions)
        if isinstance(scores, dict):
            scores = scores.get("scores", [])
        scores = list(scores)
    except Exception as exc:
        print(f"warning: COMET scoring failed; writing empty output ({exc})", file=sys.stderr)
        write_jsonl(output_path, [])
        return 0

    output_records: List[dict] = []
    for index, record in enumerate(kept):
        scored = dict(record)
        scored["comet_score"] = float(scores[index]) if index < len(scores) else None
        output_records.append(scored)
    write_jsonl(output_path, output_records)
    print(f"wrote {len(output_records)} COMET-scored rows to {output_path}")
    return 0


def _is_high_disagreement(record: dict) -> bool:
    value = record.get("high_disagreement_flag", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _stratified_sample(
    records: Sequence[dict], sample_rate: float, seed: int
) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for record in records:
        key = (
            _first(record, ("lang", "language"), "unknown").lower(),
            _first(record, ("tier", "eval_tier"), "unknown"),
        )
        groups.setdefault(key, []).append(record)

    rng = random.Random(seed)
    selected: List[dict] = []
    for key in sorted(groups):
        group = groups[key]
        if sample_rate <= 0:
            continue
        count = max(1, int(round(len(group) * sample_rate)))
        count = min(count, len(group))
        selected.extend(rng.sample(group, count))
    return selected


def _text_question(title: str, text: str) -> dict:
    return {
        "title": title,
        "questionItem": {
            "question": {
                "required": False,
                "textQuestion": {"paragraph": True},
                "questionText": text,
            }
        },
    }


def _forms_payload(items: List[dict], seed: int, sample_rate: float) -> dict:
    forms_items: List[dict] = []
    requests: List[dict] = []
    for item in items:
        quiz_id = item["quiz_id"]
        header = {
            "title": (
                f"{quiz_id} | {item['lang']} | {item['tier']}\n"
                f"Source: {item['source_en']}\n"
                f"Translation: {item['hypothesis']}\n"
                f"Reference: {item['reference']}"
            ),
            "questionItem": {
                "question": {
                    "required": True,
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [
                            {"value": "1 - Bad"},
                            {"value": "2"},
                            {"value": "3 - Acceptable"},
                            {"value": "4"},
                            {"value": "5 - Excellent"},
                        ],
                    },
                }
            },
        }
        forms_items.append(header)
        requests.append({"createItem": {"item": header, "location": {"index": len(requests)}}})

    return {
        "title": "Translation Validation Quiz",
        "description": (
            "Blinded pragmatic translation validation. Each row contains the "
            "English source, a machine translation, and a reference."
        ),
        "seed": seed,
        "sample_rate": sample_rate,
        "items": items,
        "google_forms_batch_update": {
            "create": {
                "title": "Translation Validation Quiz",
                "documentTitle": "Translation Validation Quiz",
            },
            "requests": requests,
        },
    }


def export_validation_quiz(args: argparse.Namespace) -> int:
    input_path = _resolve(args.input)
    csv_path = _resolve(args.csv)
    json_path = _resolve(args.forms_json)
    records = read_jsonl(input_path)

    seen: set = set()
    selected: List[Tuple[int, dict]] = []
    for index, record in enumerate(records):
        if _is_high_disagreement(record) and index not in seen:
            seen.add(index)
            selected.append((index, record))

    remaining = [
        (index, record)
        for index, record in enumerate(records)
        if index not in seen
    ]
    sampled = _stratified_sample(
        [record for _, record in remaining], args.sample_rate, args.seed
    )
    sampled_ids = {id(record) for record in sampled}
    for index, record in remaining:
        if id(record) in sampled_ids and index not in seen:
            seen.add(index)
            selected.append((index, record))

    selected.sort(key=lambda pair: pair[0])

    quiz_rows: List[dict] = []
    items: List[dict] = []
    for quiz_number, (_, record) in enumerate(selected, start=1):
        quiz_id = f"Q{quiz_number:03d}"
        row = {
            "quiz_id": quiz_id,
            "lang": _first(record, ("lang", "language"), "unknown"),
            "source_en": _source(record),
            "hypothesis": _hypothesis(record),
            "reference": _reference(record),
        }
        quiz_rows.append(row)
        items.append(
            {
                **row,
                "tier": _first(record, ("tier", "eval_tier"), "unknown"),
                "high_disagreement_flag": _is_high_disagreement(record),
                "item_id": _first(record, ("item_id", "id")),
                "variant_id": _first(record, ("variant_id",)),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUIZ_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(quiz_rows)

    payload = _forms_payload(items, args.seed, args.sample_rate)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"wrote {len(quiz_rows)} quiz rows to {csv_path} "
        f"and Google Forms JSON to {json_path}"
    )
    return 0


def _sample_rate(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("--sample-rate must be between 0 and 1")
    return rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation pipeline sidecar tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    comet_parser = subparsers.add_parser("comet", help="score results with COMET")
    comet_parser.add_argument("--input", default=str(DEFAULT_RESULTS))
    comet_parser.add_argument("--output", default=str(DEFAULT_COMET_OUTPUT))
    comet_parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    comet_parser.add_argument("--batch-size", type=int, default=8)
    comet_parser.set_defaults(func=comet)

    quiz_parser = subparsers.add_parser(
        "export-validation-quiz", help="export blinded validation quiz artifacts"
    )
    quiz_parser.add_argument("--input", default=str(DEFAULT_RESULTS))
    quiz_parser.add_argument("--csv", default=str(DEFAULT_QUIZ_CSV))
    quiz_parser.add_argument("--forms-json", default=str(DEFAULT_QUIZ_JSON))
    quiz_parser.add_argument("--seed", type=int, default=42)
    quiz_parser.add_argument("--sample-rate", type=float, default=0.2)
    quiz_parser.set_defaults(func=export_validation_quiz)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Validate sample rate for the quiz command even when the default is used.
    if args.command == "export-validation-quiz" and not 0.0 <= args.sample_rate <= 1.0:
        print("error: --sample-rate must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except Exception as exc:  # keep sidecars from taking down the pipeline
        print(f"warning: {args.command} failed: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
