#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p out
for f in out/dlo_dataset_c36_seq.pt out/adaptigraph_sim_only.pt out/phi_shared4_result.json; do
  [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
done
echo "[$(date '+%F %T')] 1/3 synthetic data"
python3 scripts/make_dataset.py --mode seq --tracks ../data/tracks --chain-len 10 --speed 0.8 1.2 --chains 77143 \
  --batch 10000 --out out/dlo_dataset_c36_seq.pt
echo "[$(date '+%F %T')] 2/3 training"
python3 scripts/train.py --data out/dlo_dataset_c36_seq.pt --out out/adaptigraph_sim_only.pt \
  --k 3 --rest --rot --noise 0.001 --epochs 30 --steps-per-epoch 2000
echo "[$(date '+%F %T')] 3/3 four shared phi + evaluation on the 36 clips"
python3 scripts/phi.py --ckpt out/adaptigraph_sim_only.pt --out out/phi_shared4_result.json
python3 -c "import json; d=json.load(open('out/phi_shared4_result.json')); print(f\"mean {d['mean']:.3f} mm/node (table: 10.3), phi {d['phi']}\")"
echo "[$(date '+%F %T')] CHAIN DONE"
