#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
GPUS=${GPUS:-"0 0 0"}; G=${GPU:-0}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
A="--tracks ../data/tracks"
mkdir -p data/wm/physworld_mpm_demos data/wm/logs; echo "0.8 1.2" > data/wm/physworld_mpm_demos/SPEED_RANGE
CLIPS=$(ls ../data/tracks/*.npz | xargs -n1 basename | sed 's/.npz$//')
echo "[$(date '+%F %T')] 1/4 one twin on all clips"
python3 fit_shared_twin.py $A --model elastoplastic --global-iters 40 --local-iters 25 --patience 3 --frac 0.33 \
  --gpus "$(echo $GPUS | tr ' ' '\n' | sort -u | paste -sd,)" --workers-per-gpu 4 > data/wm/logs/fit_shared_twin.log 2>&1
echo "[$(date '+%F %T')] 2/4 demos per clip, workers on GPUs: $GPUS"
Q=data/wm/claims; rm -rf $Q; mkdir -p $Q
worker() {
  local gpu=$1
  for c in $CLIPS; do
    [ -f data/wm/physworld_mpm_demos/$c.pt ] && continue
    mkdir $Q/$c 2>/dev/null || continue
    CUDA_VISIBLE_DEVICES=$gpu python3 demos.py --clip $c --n 40 --batch 4 --global-sigma 0.5 $A > data/wm/logs/demos_$c.log 2>&1
    echo "[$(date '+%F %T')] GPU $gpu: $c done"
  done
}
for g in $GPUS; do worker $g & sleep 5; done
wait
echo "[$(date '+%F %T')] 3/4 pooled GNN (60 epochs)"
CUDA_VISIBLE_DEVICES=$G python3 train_pooled.py --out data/wm/physworld_mpm_gnn_pooled --tracks ../data/tracks --epochs 60 --train-by "2099-01-01 00:00"
echo "[$(date '+%F %T')] 4/4 shared Phi (100 iterations)"
CUDA_VISIBLE_DEVICES=$G python3 shared_phi.py --iters 100 --box 1.5 --out data/wm/physworld_shared_phi.json
python3 -c "import json; d=json.load(open('data/wm/physworld_shared_phi.json')); print(f\"shared Phi mean {d['shared_fit']['mean']:.3f} mm/node (table: 7.2)\")"
echo "[$(date '+%F %T')] CHAIN DONE"
