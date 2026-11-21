#!/usr/bin/env python3
"""Idempotent patcher for google-research/metricx24 predict.py.

Fixes batch_size > 1 crash caused by missing DataCollatorWithPadding in transformers.Trainer.
"""
import os
import sys

def patch_predict_file(predict_path: str) -> bool:
    if not os.path.exists(predict_path):
        raise FileNotFoundError(f"File not found: {predict_path}")

    with open(predict_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already patched
    if "DataCollatorWithPadding" in content:
        print(f"[patch_vendor] {predict_path} is already patched.")
        return False

    # Look for trainer initialization
    target = "trainer = transformers.Trainer("
    replacement = "trainer = transformers.Trainer(\n      data_collator=transformers.DataCollatorWithPadding(tokenizer),"
    
    if target not in content:
        # Fallback to bare Trainer( if imported directly
        target = "trainer = Trainer("
        replacement = "trainer = Trainer(\n      data_collator=transformers.DataCollatorWithPadding(tokenizer),"
        if target not in content:
            raise ValueError(f"Could not find Trainer initialization in {predict_path}")

    patched_content = content.replace(target, replacement, 1)

    # Verification assertion
    assert "DataCollatorWithPadding(tokenizer)" in patched_content, (
        "Patch failed verification assertion: DataCollatorWithPadding not found in patched content"
    )

    with open(predict_path, "w", encoding="utf-8") as f:
        f.write(patched_content)

    print(f"[patch_vendor] Successfully patched {predict_path}")
    return True


def main() -> None:
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        target_file = os.path.join(
            repo_root, "scoring", "metricx", "vendor", "metricx", "metricx24", "predict.py"
        )

    if not os.path.exists(target_file):
        print(f"[patch_vendor] Target file {target_file} does not exist yet. Run setup_metricx_venv.sh first.")
        sys.exit(0)

    patch_predict_file(target_file)


if __name__ == "__main__":
    main()
