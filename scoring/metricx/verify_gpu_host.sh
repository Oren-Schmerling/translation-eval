#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "================================================================="
echo "  RTX 3060 Ti MetricX-24 Host Bootstrap & Verification Routine   "
echo "================================================================="

echo -e "\n[Step 1/4] Running setup_metricx_venv.sh (CUDA + dependencies)..."
bash "${SCRIPT_DIR}/setup_metricx_venv.sh"

echo -e "\n[Step 2/4] Verifying GPU and bfloat16 capability (--require-cuda)..."
"${REPO_ROOT}/.venv-metricx/bin/python" "${SCRIPT_DIR}/check_gpu.py" --require-cuda

echo -e "\n[Step 3/4] Running MetricX-24 scoring on 6-row sanity fixture (CUDA, batch_size=8)..."
"${REPO_ROOT}/.venv-metricx/bin/python" "${SCRIPT_DIR}/score_metricx.py" \
  --input "${SCRIPT_DIR}/fixtures/sanity_6.jsonl" \
  --output "/tmp/metricx_sanity_out.jsonl" \
  --mode both \
  --device cuda \
  --batch-size 8

echo -e "\n[Step 4/4] Asserting sanity assertions & peak VRAM threshold (<= 5.5 GB)..."
"${REPO_ROOT}/.venv-metricx/bin/python" "${SCRIPT_DIR}/assert_sanity.py" \
  "/tmp/metricx_sanity_out.jsonl" \
  --max-vram-gb 5.5

echo -e "\n================================================================="
echo "  RTX 3060 Ti MetricX-24 harness verified!                       "
echo "================================================================="
