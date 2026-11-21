#!/usr/bin/env python3
"""Decoupled MetricX-24 Neural Scorer CLI.

Supports:
- Modes: ref, qe, both
- Devices: auto, cuda, cpu
- Issue #10 workaround: explicit temp directory for TrainingArguments
- Deterministic mock backend (--mock-scorer) for CPU-only dev / CI
"""
import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

DEFAULT_MODEL = "google/metricx-24-hybrid-large-v2p6-bfloat16"
DEFAULT_TOKENIZER = "google/mt5-xl"


def normalize_row(row: Dict[str, Any]) -> Tuple[str, str, str]:
    src = row.get("source_en") or row.get("source") or ""
    hyp = row.get("translation") or row.get("hypothesis") or ""
    ref = row.get("reference_target") or row.get("reference") or row.get("gold_reference") or ""
    return str(src).strip(), str(hyp).strip(), str(ref).strip()


def mock_score(src: str, hyp: str, ref: str) -> Tuple[float, float]:
    """Deterministic mock scoring in [0.0, 25.0] where 0.0 is best (lower is better)."""
    if not hyp:
        return 24.5, 24.5

    lower_hyp = hyp.lower()
    bad_tokens = ["garbled", "gibberish", "nonsense", "rác rưởi", "ખોટું"]
    is_bad = any(t in lower_hyp for t in bad_tokens)

    if hyp == ref:
        return 1.10, 1.30

    if is_bad:
        return 22.40, 23.10

    # Character-level similarity metric
    matcher = difflib.SequenceMatcher(None, hyp, ref)
    ratio = matcher.ratio()
    distance = max(0.0, min(1.0, 1.0 - ratio))

    ref_score = round(1.5 + distance * 18.0, 2)
    qe_score = round(2.0 + distance * 17.5, 2)
    return ref_score, qe_score


def run_real_metricx(
    items: List[Tuple[int, str, str, str]],
    mode: str,
    device: str,
    batch_size: int,
    max_input_length: int,
) -> Dict[int, Dict[str, float]]:
    scores: Dict[int, Dict[str, float]] = {idx: {} for idx, _, _, _ in items}
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vendor_dir = os.path.join(repo_root, "scoring", "metricx", "vendor", "metricx")

    if not os.path.exists(vendor_dir):
        raise FileNotFoundError(f"MetricX vendor directory not found at {vendor_dir}. Run setup script first.")

    temp_work_dir = tempfile.mkdtemp(prefix="metricx_work_")
    try:
        modes_to_run = ["ref", "qe"] if mode == "both" else [mode]
        for m in modes_to_run:
            in_file = os.path.join(temp_work_dir, f"input_{m}.jsonl")
            out_file = os.path.join(temp_work_dir, f"output_{m}.jsonl")

            with open(in_file, "w", encoding="utf-8") as f:
                for idx, src, hyp, ref in items:
                    record = {
                        "source": src,
                        "hypothesis": hyp,
                        "reference": ref if m == "ref" else "",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{vendor_dir}:{env.get('PYTHONPATH', '')}"
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""

            cmd = [
                sys.executable,
                "-m",
                "metricx24.predict",
                "--model_name_or_path",
                DEFAULT_MODEL,
                "--tokenizer_name_or_path",
                DEFAULT_TOKENIZER,
                "--input_file",
                in_file,
                "--output_file",
                out_file,
                "--batch_size",
                str(batch_size),
                "--max_input_length",
                str(max_input_length),
            ]
            if m == "qe":
                cmd.append("--qe")

            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                # Automatic OOM fallback to batch_size // 2 on CUDA
                if "CUDA out of memory" in proc.stderr and batch_size > 1:
                    print(f"OOM with batch_size={batch_size}. Retrying with batch_size={batch_size // 2}...")
                    cmd[cmd.index("--batch_size") + 1] = str(batch_size // 2)
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

                if proc.returncode != 0:
                    raise RuntimeError(f"MetricX {m} failed (code {proc.returncode}):\n{proc.stderr}")

            # Read back scores
            with open(out_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        parsed = json.loads(line)
                        score = float(parsed.get("prediction", 0.0))
                        idx = items[i][0]
                        scores[idx][f"metricx_{m}"] = score
    finally:
        shutil.rmtree(temp_work_dir, ignore_errors=True)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="MetricX-24 decoupled neural scoring CLI.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--mode", choices=["ref", "qe", "both"], default="both", help="Scoring mode.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--max-input-length", type=int, default=512, help="Max input length.")
    parser.add_argument("--mock-scorer", action="store_true", help="Run deterministic mock scorer for CPU dev.")
    args = parser.parse_args()

    # Read and index input
    rows: List[Dict[str, Any]] = []
    normalized_items: List[Tuple[int, str, str, str]] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
            src, hyp, ref = normalize_row(r)
            normalized_items.append((idx, src, hyp, ref))

    print(f"Loaded {len(rows)} records from {args.input}")

    if args.mock_scorer:
        print("Scoring with deterministic CPU mock backend (--mock-scorer)...")
        for idx, src, hyp, ref in normalized_items:
            ref_score, qe_score = mock_score(src, hyp, ref)
            if args.mode in ("ref", "both"):
                rows[idx]["metricx_ref"] = ref_score
            if args.mode in ("qe", "both"):
                rows[idx]["metricx_qe"] = qe_score
            rows[idx]["metricx_is_mock"] = True
    else:
        # Determine actual device
        device = args.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        batch_size = args.batch_size
        if batch_size is None:
            batch_size = 8 if device == "cuda" else 2

        print(f"Executing real MetricX-24 [mode={args.mode}, device={device}, batch_size={batch_size}]...")
        scores = run_real_metricx(
            normalized_items,
            mode=args.mode,
            device=device,
            batch_size=batch_size,
            max_input_length=args.max_input_length,
        )
        for idx, score_dict in scores.items():
            rows[idx].update(score_dict)
            rows[idx]["metricx_is_mock"] = False

    # Atomic write to output
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    temp_out = os.path.join(out_dir, f".tmp_{os.path.basename(args.output)}")

    with open(temp_out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.replace(temp_out, args.output)
    print(f"Successfully wrote {len(rows)} scored records to {args.output}")


if __name__ == "__main__":
    main()
