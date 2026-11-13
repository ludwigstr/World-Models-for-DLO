#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-12}; GENS=${GENS:-60}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
mkdir -p out; P=out/mujoco_general_fit3
for f in $P.npz ${P}_log.jsonl ${P}_validate.json; do
  [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
done
echo "[$(date '+%F %T')] CMA-ES fit ($GENS generations, $W workers, warm start data/mujoco_general_fit3.npz)"
python3 scripts/fit.py --tracks ../data/tracks --mass-max 0.5 --popsize 18 --workers "$W" \
  --resume data/mujoco_general_fit3.npz --sigma0 0.5 --gens "$GENS" --out $P.npz --log ${P}_log.jsonl
echo "[$(date '+%F %T')] fresh-process validation"
python3 scripts/validate.py mujoco $P.npz --tracks ../data/tracks --out ${P}_validate.json --workers "$W"
echo "[$(date '+%F %T')] CHAIN DONE"
