#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}; W=${WORKERS:-8}; GENS=${GENS:-33}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
mkdir -p out; P=out/genesis_fit
for f in $P.json ${P}_log.jsonl ${P}_validate.json; do
  [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
done
echo "[$(date '+%F %T')] CMA-ES fit ($GENS generations, $W workers)"
$PY fit.py --tracks ../data/tracks --mass-max 0.5 --mu-max 20 --popsize 18 --workers "$W" \
  --gens "$GENS" --out $P.json --log ${P}_log.jsonl
echo "[$(date '+%F %T')] fresh-process validation"
python3 validate.py genesis $P.json --python "$PY" --tracks ../data/tracks --out ${P}_validate.json --workers "$W"
echo "[$(date '+%F %T')] CHAIN DONE"
