"""Unit tests for MetricX-24 setup, patcher, mock scoring pipeline, and GPU checks."""
import json
import os
import subprocess
import sys
import tempfile
import pytest

from scoring.metricx.patch_vendor import patch_predict_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIXTURE_PATH = os.path.join(REPO_ROOT, "scoring", "metricx", "fixtures", "sanity_6.jsonl")
SCORE_SCRIPT = os.path.join(REPO_ROOT, "scoring", "metricx", "score_metricx.py")
ASSERT_SCRIPT = os.path.join(REPO_ROOT, "scoring", "metricx", "assert_sanity.py")
CHECK_GPU_SCRIPT = os.path.join(REPO_ROOT, "scoring", "metricx", "check_gpu.py")


def test_patch_vendor_idempotency(tmp_path):
    """Test that patch_vendor idempotently inserts DataCollatorWithPadding and does not duplicate it."""
    mock_predict = tmp_path / "predict.py"
    original_code = """
import transformers

def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained("test")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("test")
    trainer = transformers.Trainer(
        model=model,
        args=None,
    )
"""
    mock_predict.write_text(original_code, encoding="utf-8")

    # First run should patch
    changed = patch_predict_file(str(mock_predict))
    assert changed is True
    content_after_first = mock_predict.read_text(encoding="utf-8")
    assert "data_collator=transformers.DataCollatorWithPadding(tokenizer)" in content_after_first
    assert content_after_first.count("DataCollatorWithPadding") == 1

    # Second run should be a no-op and not duplicate
    changed_again = patch_predict_file(str(mock_predict))
    assert changed_again is False
    content_after_second = mock_predict.read_text(encoding="utf-8")
    assert content_after_second.count("DataCollatorWithPadding") == 1


def test_mock_scorer_pipeline_and_assert_sanity(tmp_path):
    """Test full score_metricx.py --mock-scorer pipeline against sanity_6.jsonl."""
    out_file = str(tmp_path / "sanity_scored.jsonl")

    cmd = [
        sys.executable,
        SCORE_SCRIPT,
        "--input",
        FIXTURE_PATH,
        "--output",
        out_file,
        "--mode",
        "both",
        "--mock-scorer",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"score_metricx failed:\n{proc.stderr}"
    assert os.path.exists(out_file)

    # Check rows and mock tags
    rows = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    assert len(rows) == 6
    for r in rows:
        assert r.get("metricx_is_mock") is True
        assert "metricx_ref" in r
        assert "metricx_qe" in r

    # Run assert_sanity.py against the output
    assert_cmd = [sys.executable, ASSERT_SCRIPT, out_file]
    assert_proc = subprocess.run(assert_cmd, capture_output=True, text=True)
    assert assert_proc.returncode == 0, f"assert_sanity failed:\n{assert_proc.stderr}"


def test_check_gpu_exit_codes():
    """Test check_gpu.py behavior on CPU-only machine."""
    # Without --require-cuda, should exit 0 and report CPU_ONLY
    proc_no_req = subprocess.run([sys.executable, CHECK_GPU_SCRIPT], capture_output=True, text=True)
    assert proc_no_req.returncode == 0
    assert "MODE=CPU_ONLY" in proc_no_req.stdout

    # With --require-cuda on CPU machine, should exit non-zero
    proc_req = subprocess.run([sys.executable, CHECK_GPU_SCRIPT, "--require-cuda"], capture_output=True, text=True)
    assert proc_req.returncode != 0
    assert "ERROR" in proc_req.stderr
