import argparse
import datetime as dt
import json
import time
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parent


import gnn as G


SEC_PER_DEMO_EPOCH = 685.0 / 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--train-by", default="2099-01-01 00:00")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = ROOT / a.out
    if (out / "gnn_pooled.pt").exists():
        raise SystemExit(f"{out}/gnn_pooled.pt exists -- never overwrite, pick a new --out")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(s):
        line = f"[{time.strftime('%F %T')}] {s}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    clips = sorted(p.stem for p in (ROOT / a.tracks).glob("*.npz"))
    files = [ROOT / f"data/wm/physworld_mpm_demos/{c}.pt" for c in clips]
    log(f"waiting for the demos of {len(clips)} clips")
    last, last_n = None, -1
    while True:
        have = [f for f in files if f.exists()]
        sizes = [f.stat().st_size for f in have]
        if len(have) == len(files) and sizes == last:
            break
        if len(have) != last_n:
            log(f"  {len(have)}/{len(files)} demo files present"); last_n = len(have)
        last = sizes if len(have) == len(files) else None
        time.sleep(120 if len(have) == len(files) else 600)
    demos = []
    for c, f in zip(clips, files):
        D = torch.load(f, weights_only=False); demos += D["demos"]
        assert list(D["speed_range"]) == [0.8, 1.2], (c, D["speed_range"])
    budget = (dt.datetime.strptime(a.train_by, "%Y-%m-%d %H:%M") - dt.datetime.now()).total_seconds()
    ep = int(min(a.epochs, budget / (SEC_PER_DEMO_EPOCH * len(demos) * 1.25)))
    log(f"pooled: {len(demos)} demos from {len(clips)} clips; budget {budget / 3600:.1f} h until {a.train_by} "
        f"-> {ep} epochs (cap {a.epochs}), seed {a.seed}")
    if ep < 5:
        raise SystemExit("less than 5 epochs fit before --train-by -- not training, ask the owner")
    torch.manual_seed(a.seed); t0 = time.time()
    model, mu, sd, val = G.train(demos, a.device, ep, 3, 1.0, 0.05, True, a.seed, log)
    t = time.time() - t0
    log(f"training done in {t:.0f}s, val one-step {val:.3f} mm")
    torch.save({"state_dict": model.state_dict(), "phi_mu": mu.cpu(), "phi_sd": sd.cpu(), "val": val, "sec_train": t,
                "clips": clips, "n_demos": len(demos), "epochs": ep, "seed": a.seed}, out / "gnn_pooled.pt")
    (out / "train.json").write_text(json.dumps({"clips": clips, "n_demos": len(demos), "epochs": ep, "val_one_step_mm": val,
                                                "sec_train": t, "seed": a.seed}, indent=1))
    log("POOLED_TRAIN_DONE")


if __name__ == "__main__":
    main()
