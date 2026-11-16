#!/usr/bin/env bash
# One-time setup on the rented GPU pod (Ubuntu + NVIDIA driver, 1x 24 GB GPU).
# Run from the repository root:   bash scripts/setup_pod.sh
# This step is preparation (installs, unpacking, model download) and is not part of the timed run.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VENV="${VENV:-.venv}"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -ne 1 ]; then echo "WARNING: $GPU_COUNT GPUs visible; the rule is one GPU."; fi
if [ "$TOTAL_MIB" -gt 24600 ]; then
  echo "WARNING: this GPU has more than 24 GB. The challenge allows at most 24 GB VRAM."
fi

echo "== Python environment =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip freeze > requirements.lock.txt
echo "Pinned versions written to requirements.lock.txt — commit this file for the code review."

echo "== Data =="
if [ ! -f data/Test_Masked.csv ]; then
  echo "ERROR: put the Zindi files in ./data first (see README, 'Get the data onto the pod')."; exit 1
fi
if [ ! -d data/images ] || [ -z "$(ls -A data/images 2>/dev/null | head -1)" ]; then
  if [ -f data/product_images.zip ]; then
    echo "Unpacking product_images.zip ..."
    python3 - <<'PY'
import zipfile
z = zipfile.ZipFile("data/product_images.zip")
z.extractall("data")          # creates data/images/*.jpg and data/image_mapping.csv
print(len(z.namelist()), "files unpacked")
PY
  else
    echo "WARNING: data/product_images.zip not found — the run will work but without photos."
  fi
fi
[ -f data/image_mapping.csv ] || echo "WARNING: data/image_mapping.csv missing (it is inside product_images.zip)."

echo "== Model weights =="
python3 - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("$MODEL")
print("model cached at", path)
PY

echo "== Tests =="
python -m pytest -q tests
echo "Setup complete. Next: bash scripts/run_all.sh --limit 50   (benchmark), then bash scripts/run_all.sh"
