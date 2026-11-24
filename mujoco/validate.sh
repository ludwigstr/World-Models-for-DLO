#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-12}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
mkdir -p out; O=out/revalidate.json
[ -e "$O" ] && { echo "refusing to overwrite $O"; exit 1; }
python3 validate.py mujoco ../results/mujoco/mujoco_general_fit.npz --tracks ../data/tracks --out "$O" --workers "$W"
