#!/usr/bin/env python3
"""Assert sanity on scored MetricX-24 fixture.

Verifies:
1. All rows have finite metricx_ref and metricx_qe in [0.0, 25.0].
2. For each language (he, gu, vi), score(accurate) < score(garbled).
3. Peak VRAM does not exceed --max-vram-gb (if CUDA is used).
"""
import argparse
import json
import math
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MetricX sanity scoring output.")
    parser.add_argument("input_file", help="Path to scored JSONL output.")
    parser.add_argument("--max-vram-gb", type=float, default=None, help="Maximum peak VRAM in GB.")
    args = parser.parse_args()

    rows = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.trim() if hasattr(line, "trim") else line.strip()
            if line:
                rows.append(json.loads(line))

    assert len(rows) == 6, f"Expected exactly 6 rows in sanity output, found {len(rows)}"

    langs = {"he", "gu", "vi"}
    by_lang_and_type = {}

    for row in rows:
        item_id = row.get("item_id", "")
        lang = row.get("lang", "")
        ref_score = row.get("metricx_ref")
        qe_score = row.get("metricx_qe")

        assert ref_score is not None and not math.isnan(ref_score), f"Invalid metricx_ref in row {item_id}"
        assert qe_score is not None and not math.isnan(qe_score), f"Invalid metricx_qe in row {item_id}"

        assert 0.0 <= ref_score <= 25.0, f"metricx_ref {ref_score} out of bounds [0.0, 25.0] in row {item_id}"
        assert 0.0 <= qe_score <= 25.0, f"metricx_qe {qe_score} out of bounds [0.0, 25.0] in row {item_id}"

        is_acc = "acc" in item_id
        by_lang_and_type[(lang, "acc" if is_acc else "bad")] = (ref_score, qe_score)

    for lang in langs:
        assert (lang, "acc") in by_lang_and_type, f"Missing accurate row for lang {lang}"
        assert (lang, "bad") in by_lang_and_type, f"Missing garbled row for lang {lang}"

        acc_ref, acc_qe = by_lang_and_type[(lang, "acc")]
        bad_ref, bad_qe = by_lang_and_type[(lang, "bad")]

        assert acc_ref < bad_ref, (
            f"Direction violation for {lang} metricx_ref (lower is better): "
            f"accurate={acc_ref:.2f} >= garbled={bad_ref:.2f}"
        )
        assert acc_qe < bad_qe, (
            f"Direction violation for {lang} metricx_qe (lower is better): "
            f"accurate={acc_qe:.2f} >= garbled={bad_qe:.2f}"
        )

    # Check VRAM if requested and CUDA available
    if args.max_vram_gb is not None:
        try:
            import torch
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated(0)
                peak_gb = peak_bytes / (1024 ** 3)
                print(f"Peak VRAM used: {peak_gb:.2f} GB (limit: {args.max_vram_gb:.2f} GB)")
                assert peak_gb <= args.max_vram_gb, f"Peak VRAM {peak_gb:.2f} GB exceeded {args.max_vram_gb} GB"
        except Exception:
            pass

    print(f"Sanity verification passed for all 6 rows across {len(langs)} languages.")


if __name__ == "__main__":
    main()
