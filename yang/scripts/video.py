import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from render import write_results
from evaluate import INBiLSTM, frame_err, rollout

LABEL = "Yang et al. (IN + bi-LSTM on Newton data), 36-clip refit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--ckpt", default="data/wm/yang_newton_inbilstm.pt")
    ap.add_argument("--reported", default="data/wm/yang_newton_inbilstm_36.json")
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; warm = a.warm
    out = ROOT / a.out
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    net = INBiLSTM(use_rest=ck["use_rest"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    rep = json.loads((ROOT / a.reported).read_text())["clips"]
    rows = {}
    for npz in sorted((ROOT / a.tracks).glob("*.npz")):
        clip = npz.stem
        z = np.load(npz)
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        pred = rollout(net, P, ck["dt"], warm=warm)
        e = frame_err(pred, P)
        err = np.array(e, dtype=np.float64); err[:warm] = np.nan
        sim = pred.permute(1, 0, 2, 3).cpu().numpy().astype(np.float64)
        np.savez(out / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=LABEL, short="Yang et al.",
                 score_nodes=np.arange(16))
        v = float(e[warm:].mean())
        rows[clip] = {"reproduced": v, "reported": rep[clip]["mittel_warm5"], "diff": v - rep[clip]["mittel_warm5"],
                      "nan_nodes": int(np.isnan(sim).sum()), "n_frames_scored": int(len(e) - warm)}
        print(clip, json.dumps(rows[clip]), flush=True)
    write_results(out, rows, {"approach": "Yang et al., IN + bi-LSTM trained on Newton rollouts from the DE fit "
                                          "(stored network replayed, no re-training)",
                              "params": a.ckpt, "device": dev, "reported_source": f"{a.reported} ('mittel_warm5')",
                              "metric": f"16 nodes (ends pinned), mean over cables, frames {warm}..T-1 (after warm-up)"})


if __name__ == "__main__":
    main()
