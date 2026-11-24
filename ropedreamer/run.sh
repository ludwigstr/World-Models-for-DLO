#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-18}; G=${GPU:-0}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p out; S=out/ropedreamer_synth; CK=out/ropedreamer_S.pt; R=out/ropedreamer_eval_36.json
for f in $S $CK $R; do [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }; done
echo "[$(date '+%F %T')] 0/3 check of the MuJoCo parameters (fresh-process validation, expected 9.83 mm/node)"
CUDA_VISIBLE_DEVICES="" python3 validate_params.py mujoco mujoco_general_fit_gen50.npz --tracks ../data/tracks --out out/mujoco_gen50_validate.json --workers 12
echo "[$(date '+%F %T')] 1/3 simulated rollouts"
CUDA_VISIBLE_DEVICES="" python3 make_dataset.py --env general --general-fit mujoco_general_fit_gen50.npz --randomize \
  --motion speed --speed-range 0.8 1.2 --tracks ../data/tracks --allow-eval-clips --frames-per-clip 28000 --twin-reference --workers "$W" --out $S
echo "[$(date '+%F %T')] 2/3 training"
CUDA_VISIBLE_DEVICES=$G python3 train.py --synth $S --size S --seq-len 120 --beta 1.0 --pos-weight 0 --rollout-weight 1000 \
  --lr 1e-4 --batch 32 --val-frac 0.1 --val-by-rollout --gap-ref-tol 0.010 --obs-noise ropedreamer_twins14_noise_params_v3.json \
  --loader-workers 8 --select val_loss --out $CK
echo "[$(date '+%F %T')] 3/3 evaluation on the 36 clips"
CUDA_VISIBLE_DEVICES=$G python3 evaluate.py --ckpt $CK --tracks ../data/tracks --out $R
python3 -c "import json; d=json.load(open('out/ropedreamer_eval_36_summary.json')); print(f\"mean {d['mittel']:.3f} mm/node (table: 18.5)\")"
echo "[$(date '+%F %T')] CHAIN DONE"
