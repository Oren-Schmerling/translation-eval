#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-metricx"
VENDOR_DIR="${SCRIPT_DIR}/vendor/metricx"

CPU_ONLY=0
for arg in "$@"; do
  if [[ "$arg" == "--cpu-only" ]]; then
    CPU_ONLY=1
  fi
done

echo "=== 1. Checking Python 3.10 Environment ==="
if [[ ! -d "${VENV_DIR}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Creating .venv-metricx with uv (Python 3.10)..."
    uv venv --seed --python 3.10 "${VENV_DIR}"
  elif command -v python3.10 >/dev/null 2>&1; then
    echo "Creating .venv-metricx with python3.10..."
    python3.10 -m venv "${VENV_DIR}"
  elif command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "${PY_VER}" == "3.10" ]]; then
      python3 -m venv "${VENV_DIR}"
    else
      echo "ERROR: Python 3.10 or uv is required to initialize .venv-metricx. Please install python3.10 or uv." >&2
      exit 1
    fi
  else
    echo "ERROR: Neither uv nor python3.10 was found. Please install Python 3.10 or uv." >&2
    exit 1
  fi
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" "$@"
  else
    "${VENV_DIR}/bin/pip" install "$@"
  fi
}

echo "=== 2. Installing PyTorch ==="
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi

if [[ "${CPU_ONLY}" -eq 1 ]] || [[ "${HAS_NVIDIA}" -eq 0 ]]; then
  echo "NOTICE: GPU acceleration disabled or nvidia-smi not detected. Installing CPU PyTorch wheels..."
  pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  echo "Detected NVIDIA GPU. Installing CUDA 12.1/12.4 PyTorch wheels..."
  pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo "=== 3. Cloning google-research/metricx Vendor Repo ==="
mkdir -p "${SCRIPT_DIR}/vendor"
if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  echo "Cloning https://github.com/google-research/metricx into ${VENDOR_DIR}..."
  git clone --depth 1 https://github.com/google-research/metricx "${VENDOR_DIR}"
else
  echo "Vendor directory ${VENDOR_DIR} already exists."
fi

echo "=== 4. Installing Pinned metricx24 Dependencies ==="
pip_install \
  "transformers==4.30.2" \
  "datasets==2.13.1" \
  "sentencepiece==0.1.99" \
  "protobuf==3.20.3" \
  "fsspec==2023.9.2" \
  "numpy<2" \
  "accelerate"

echo "=== 5. Applying Idempotent DataCollatorWithPadding Vendor Patch ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/patch_vendor.py" "${VENDOR_DIR}/metricx24/predict.py"

echo "MetricX-24 environment setup complete at ${VENV_DIR}!"
