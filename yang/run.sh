#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p out; P=out/yang_newton; IDENT=${P}_ident.json
for f in $IDENT ${P}_synth ${P}_inbilstm.pt ${P}_inbilstm_36.json; do [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }; done
echo "[$(date '+%F %T')] 1/4 differential evolution"
python3 fit.py --engine newton --tracks ../data/tracks --clips all --maxiter 40 --popsize 15 --tol 0.01 \
  --bound k_stretch=1,8 --bound d_bend=-6,1 --batch-cands 3 --time-cap-h 0 --out $IDENT
echo "[$(date '+%F %T')] 2/4 simulated training data (Newton)"
python3 dataset.py --engine newton --env yang --ident $IDENT --motion speed --randomize --tracks ../data/tracks \
  --allow-eval-clips --per-clip 100 --batch 512 --out ${P}_synth
echo "[$(date '+%F %T')] 3/4 training"
python3 train.py --data ${P}_synth --out ${P}_inbilstm.pt --epochs 60 --gap-ref-tol 0.010
echo "[$(date '+%F %T')] 4/4 evaluation on the 36 clips"
python3 evaluate.py --ckpt ${P}_inbilstm.pt --tracks ../data/tracks --out ${P}_inbilstm_36.json --synth ${P}_synth
python3 -c "import json; d=json.load(open('${P}_inbilstm_36.json')); print(f\"mittel_warm5 mean {d['summary']['mittel_warm5']['mean']:.3f} mm/node (table: 16.0)\")"
echo "[$(date '+%F %T')] CHAIN DONE"
