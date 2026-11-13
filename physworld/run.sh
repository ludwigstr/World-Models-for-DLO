#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
GPUS=${GPUS:-"0 0 0"}; G=${GPU:-0}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
A="--tracks ../data/tracks --dir-suffix "
mkdir -p data/wm/physworld_mpm_demos data/wm/logs; echo "0.8 1.2" > data/wm/physworld_mpm_demos/SPEED_RANGE
CLIPS=$(ls ../data/tracks/*.npz | xargs -n1 basename | sed 's/.npz$//')
Q=data/wm/claims; rm -rf $Q; mkdir -p $Q
echo "[$(date '+%F %T')] 1/3 twins + demos per clip, workers on GPUs: $GPUS"
worker() {
  local gpu=$1
  for c in $CLIPS; do
    [ -f data/wm/physworld_mpm_demos/$c.pt ] && continue
    mkdir $Q/$c 2>/dev/null || continue
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/fit_twin.py --clip $c --model elastoplastic --global-iters 10 --local-iters 6 --frac 0.33 $A > data/wm/logs/fit_$c.log 2>&1
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/demos.py --clip $c --n 40 --batch 4 $A > data/wm/logs/demos_$c.log 2>&1
    echo "[$(date '+%F %T')] GPU $gpu: $c done"
  done
}
for g in $GPUS; do worker $g & sleep 5; done
wait
echo "[$(date '+%F %T')] 2/3 pooled GNN (60 epochs) + per-clip Phi fine-tune"
CUDA_VISIBLE_DEVICES=$G python3 scripts/train_pooled.py --out data/wm/physworld_mpm_gnn_pooled --tracks ../data/tracks --epochs 60 --train-by "2099-01-01 00:00"
for c in $CLIPS; do
  [ -f data/wm/physworld_mpm_gnn_pooled/$c.json ] && continue
  CUDA_VISIBLE_DEVICES=$G python3 scripts/gnn.py --clip $c $A --load data/wm/physworld_mpm_gnn_pooled/gnn_pooled.pt --tag pooled > data/wm/logs/phi_$c.log 2>&1
done
python3 scripts/summary.py $A --gnn-tag pooled --out data/wm/physworld_mpm_pooled.json
echo "[$(date '+%F %T')] 3/3 shared Phi (100 iterations)"
CUDA_VISIBLE_DEVICES=$G python3 scripts/shared_phi.py --iters 100 --box 1.5 --out data/wm/physworld_shared_phi.json
python3 -c "import json; d=json.load(open('data/wm/physworld_shared_phi.json')); print(f\"shared Phi mean {d['shared_fit']['mean']:.3f} mm/node (table: 7.5)\")"
echo "[$(date '+%F %T')] CHAIN DONE"
