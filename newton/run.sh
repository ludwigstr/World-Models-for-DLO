#!/bin/bash
set -u
cd "$(dirname "$0")"
mkdir -p out
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=out/newton_fit
log() { echo "[$(date '+%F %T')] $*"; }
if [ "${RESUME:-0}" = "1" ]; then
  [ -e $P.npz ] || { log "RESUME=1 but $P.npz is missing (copy a snapshot into this folder)"; exit 1; }
else
  [ -e $P.npz ] && { log "$P.npz exists -- refusing to overwrite (RESUME=1 to continue it)"; exit 1; }
fi
log "fit start"
WIRE_TRACKS=../data/tracks WIRE_Z_WEIGHT=0 WIRE_MASS_PIN=0.044 \
WIRE_RESUME=${RESUME:-0} \
WIRE_OUT=$P.npz WIRE_FITLOG=${P}_log.jsonl WIRE_GPUS=${NEWTON_GPUS:-0} WIRE_WORKERS=${NEWTON_WORKERS:-4} \
WIRE_POPSIZE=18 WIRE_GENS=${GENS:-30} WIRE_STOP_BY="${STOP_BY:-$(date -d "+7 days" "+%F %H:%M")}" \
    python3 fit.py >> ${P}_stdout.log 2>&1 || { log "fit failed (see ${P}_stdout.log)"; exit 1; }
grep -q "=== DONE ===" ${P}_stdout.log || { log "fit did not finish"; exit 1; }
log "fit done: $(grep 'final result' ${P}_stdout.log | tail -n 1)"
python3 validate.py --params $P.npz --tracks ../data/tracks --out ${P}_validate.json \
    --gpus ${VALIDATE_GPUS:-0} --workers ${NEWTON_WORKERS:-4} > ${P}_validate.log 2>&1 || { log "validation failed"; exit 1; }
log "fresh-process validation: $(tail -n 1 ${P}_validate.log)"
log "ALL DONE"
