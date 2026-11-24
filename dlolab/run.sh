#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
GPUS=${GPUS:-0,1,2}; GW=${GRAD_WORKERS:-36}; MW=${METRIC_WORKERS:-36}
STOP_BY=${STOP_BY:-$(date -d '+8 hours' '+%F %H:%M')}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export GENERAL_FIT_TRACKS="$PWD/../data/tracks"
P=data/dlolab_general_grad_zero
for f in $P.json $P.log.jsonl ${P}_validate.json; do
  [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
done
echo "[$(date '+%F %T')] gradient fit from zero (GPUs $GPUS, $GW gradient workers, $MW metric processes, stop by $STOP_BY)"
python3 fit_gradient.py --tracks ../data/tracks --init zero --lr 2 --max-lr 4 \
  --batch 36 --grad-workers "$GW" --gpus "$GPUS" --backend cuda --metric-workers "$MW" --bptt 120 --stop-by "$STOP_BY" --out $P.json
echo "[$(date '+%F %T')] fresh-process validation"
CUDA_VISIBLE_DEVICES="" python3 validate.py dlolab $P.json --tracks ../data/tracks --out ${P}_validate.json --workers "$MW"
echo "[$(date '+%F %T')] CHAIN DONE"
