#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=3 PYTHONUNBUFFERED=1
mkdir -p out
DS=out/dlo_dataset_c36_step1.pt; CK=out/caporali_film_c36_step1.pt; RJ=out/caporali_online_c36_results.json
for f in $DS $CK $RJ; do [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }; done
echo "[$(date '+%F %T')] 1/3 synthetic data"
python3 scripts/make_dataset.py --mode step --stride 1 --tracks ../data/tracks --chain-len 10 --speed 0.8 1.2 \
  --chains 154286 --batch 10000 --out $DS
echo "[$(date '+%F %T')] 2/3 training"
python3 scripts/train.py --models film --data $DS --width 256 --epochs 25 --save --save-as $CK --out out/training_bench.json
echo "[$(date '+%F %T')] 3/3 online estimation + evaluation per clip"
export WIRE_NOMINAL="0.04,1.0,20,0.15" WIRE_REST_MODE=frame0 WIRE_REST_CAP=0.6
for f in ../data/tracks/*.npz; do
  c=$(basename $f .npz)
  python3 scripts/estimate.py --clip $c --variant online --arch film --in-sample --base-ckpt $CK --epochs 0 \
    --stride 1 --tracks-npz $f --no-video --out out/neural_video --results-json $RJ
done
python3 -c "
import json; d=json.load(open('$RJ')); v=[x['mittel'] for x in d.values() if isinstance(x,dict) and 'mittel' in x]
print(f'n={len(v)} mean {sum(v)/len(v):.2f} mm/node (table: 15.8)')"
echo "[$(date '+%F %T')] CHAIN DONE"
