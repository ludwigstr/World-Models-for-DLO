#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
W=${DLOLAB_WORKERS:-12}; SIGMA0=${SIGMA0:-0.2}; STOP_BY=${STOP_BY:-$(date -d "+24 hours" "+%F %H:%M")}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
P=data/dlolab_general
for f in $P.json ${P}_log.jsonl ${P}_validate.json; do
  [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
done
echo "[$(date '+%F %T')] fit (start = state/dlolab_general_gen21.json, sigma0 $SIGMA0, $W workers, stop by $STOP_BY)"
python3 scripts/fit_cmaes.py --tracks ../data/tracks --mass-max 0.5 --popsize 20 --workers "$W" \
  --resume state/dlolab_general_gen21.json --sigma0 "$SIGMA0" --gens 1000 --stop-by "$STOP_BY" \
  --out $P.json --log ${P}_log.jsonl
echo "[$(date '+%F %T')] fresh-process validation"
python3 scripts/validate.py dlolab $P.json --tracks ../data/tracks --out ${P}_validate.json --workers "$W"
echo "[$(date '+%F %T')] CHAIN DONE"
