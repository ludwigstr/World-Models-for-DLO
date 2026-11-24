import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from evaluate import dp_of, features
from render import write_results
from wire_model.adaptigraph.model_v2 import AdaptiGraphV2
from wire_model.adaptigraph.features import turning

LABEL = "AdaptiGraph (sim-trained, 4 phi), 36-clip refit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--ckpt", default="data/wm/adaptigraph2/full_k3_rest_rot_noise.pt")
    ap.add_argument("--reported", default="data/wm/adaptigraph2/final_shared4_full_k3_rest_rot_noise.json")
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    out = ROOT / a.out
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    rep = json.loads((ROOT / a.reported).read_text())
    assert rep["ckpt"] == a.ckpt, (rep["ckpt"], a.ckpt)
    phi4 = torch.tensor(rep["phi"], dtype=torch.float32, device=dev)
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    fl = ck.get("flags", {})
    m = AdaptiGraphV2(16, ck["n_hist"], radius=ck["radius"], layers=fl.get("layers", 4),
                      n_extra=int(bool(fl.get("rest"))) + int(bool(fl.get("rot")))).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    H = ck["n_hist"]
    rows = {}
    for npz in sorted((ROOT / a.tracks).glob("*.npz")):
        z = np.load(npz); clip = npz.stem
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        T = P.shape[1]
        rest = torch.zeros(4, 16, device=dev); rest[:, 1:-1] = turning(P[:, 0]).clamp(-0.6, 0.6)
        with torch.no_grad():
            frames = [P[:, t] for t in range(a.warm)]; errs = []
            for t in range(a.warm - 1, T - 1):
                hist = torch.stack([frames[max(0, t - H + j)] for j in range(H)], 1)
                A, Y = P[:, t], P[:, t + 1]
                pred = m(hist, frames[t], phi4, dp_of(A, Y), features(fl, A, Y, rest)).clone()
                pred[:, 0], pred[:, -1] = Y[:, 0], Y[:, -1]
                frames.append(pred)
                errs.append(((pred - Y).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
        e_cab = np.stack(errs, 1)
        e = e_cab.mean(0)
        err = np.full(T, np.nan); err[a.warm:] = e
        sim = torch.stack(frames, 1).permute(1, 0, 2, 3).cpu().numpy().astype(np.float64)
        np.savez(out / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=LABEL, short="AdaptiGraph",
                 score_nodes=np.arange(16), phi=phi4.cpu().numpy())
        r = rep["clips"][clip]
        rows[clip] = {"reproduced": float(e.mean()), "reported": r, "diff": float(e.mean()) - r,
                      "per_cable": e_cab.mean(1).tolist(), "n_frames_scored": int(len(e))}
        print(clip, json.dumps(rows[clip]), flush=True)
    write_results(out, rows, {"approach": "AdaptiGraph, rebuilt (K=3 BPTT, rest + end-rotation inputs, noise), trained on "
                                          "simulation data from the 36 clips only; 4 phi (one per cable) fitted on all clips and frames",
                              "params": a.ckpt, "phi": rep["phi"], "device": dev, "reported_source": f"{a.reported} ('clips')",
                              "metric": f"16 nodes (ends pinned), mean over cables, frames {a.warm}..T-1 (after warm-up)"})


if __name__ == "__main__":
    main()
