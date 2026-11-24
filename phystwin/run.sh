#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
GPUS=${GPUS:-"0 0 0"}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
O=out; L=$O/logs; Q=$O/claims; mkdir -p $L; rm -rf $Q; mkdir -p $Q
CLIPS=$(ls ../data/tracks/*.npz | xargs -n1 basename | sed 's/.npz$//')
echo "[$(date '+%F %T')] start: $(echo $CLIPS | wc -w) clips, workers on GPUs: $GPUS"
worker() {
  local gpu=$1
  for c in $CLIPS; do
    [ -f $O/phystwin_$c.json ] && continue
    mkdir $Q/$c 2>/dev/null || continue
    echo "[$(date '+%F %T')] GPU $gpu: $c start"
    CUDA_VISIBLE_DEVICES=$gpu python3 fit.py --tracks ../data/tracks --clips $c --train-frac 1.0 \
      --stage2-mode window --out $O/phystwin_$c.json --save-twins $O/twins > $L/$c.log 2>&1
    echo "[$(date '+%F %T')] GPU $gpu: $c exit $? $(grep -o 'all frames [0-9.]*' $L/$c.log | tail -1)"
  done
}
for g in $GPUS; do worker $g & sleep 5; done
wait
python3 - <<'PY'
import json, glob, os
r = {}
for f in sorted(glob.glob("out/phystwin_*.json")):
    d = json.load(open(f)); c = next(iter(d)); r[c] = d[c]["mittel"]
m = sum(r.values()) / len(r)
json.dump({"n_clips": len(r), "mittel_mean": m, "per_clip": r}, open("out/summary.json", "w"), indent=1)
print(f"{len(r)} clips | all frames {m:.2f} mm/node (table: 22.0)")
PY
echo "[$(date '+%F %T')] CHAIN DONE"
