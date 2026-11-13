#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}; W=${WORKERS:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
mkdir -p out; O=out/revalidate.json
[ -e "$O" ] && { echo "refusing to overwrite $O"; exit 1; }
python3 scripts/validate.py genesis ../results/genesis/genesis_general_fit.json --python "$PY" --tracks ../data/tracks --out "$O" --workers "$W"
