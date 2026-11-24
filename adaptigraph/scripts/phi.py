import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from evaluate import WARM, dp_of, features, load_tracks
from wire_model.adaptigraph.model_v2 import AdaptiGraphV2
from wire_model.adaptigraph.features import turning


@torch.no_grad()
def errors(m, flags, P, phi):
    B, Tm = P.shape[:2]
    H = m.n_hist
    rest = torch.zeros(B, 16, device=P.device)
    rest[:, 1:-1] = turning(P[:, 0]).clamp(-0.6, 0.6)
    frames = [P[:, t] for t in range(WARM)]
    out = []
    for t in range(WARM - 1, Tm - 1):
        hist = torch.stack([frames[max(0, t - H + j)] for j in range(H)], 1)
        A, Y = P[:, t], P[:, t + 1]
        pred = m(hist, frames[t], phi, dp_of(A, Y), features(flags, A, Y, rest)).clone()
        pred[:, 0], pred[:, -1] = Y[:, 0], Y[:, -1]
        frames.append(pred)
        out.append((pred - Y).norm(dim=-1).mean(-1) * 1000)
    return torch.stack(out, 1)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists"
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    fl = ck.get("flags", {})
    m = AdaptiGraphV2(16, ck["n_hist"], radius=ck["radius"], layers=fl.get("layers", 4), n_extra=int(bool(fl.get("rest"))) + int(bool(fl.get("rot")))).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    clips, P, Tn = load_tracks(dev)
    grid = torch.linspace(-0.2, 1.2, 29, device=dev)
    C, K = P.shape[0], len(grid)
    E = errors(m, fl, P.repeat_interleave(K, 0), grid.repeat(C)).view(C, K, -1).cpu().numpy()
    n = (Tn - WARM).cpu().numpy()
    cost = np.stack([[E[c, k, : n[c]].mean() for k in range(K)] for c in range(C)])
    g = grid.cpu().numpy()
    per_cable = cost.reshape(len(clips), 4, K)
    best = per_cable.mean(0).argmin(1)
    phi = g[best]
    d = {c: float(np.mean([per_cable[i, k, best[k]] for k in range(4)])) for i, c in enumerate(clips)}
    mean = float(np.mean(list(d.values())))
    res = {"ckpt": a.ckpt, "note": __doc__, "phi": phi.tolist(), "mean": mean, "clips": d,
           "per_cable_curve": per_cable.mean(0).tolist(), "grid": g.tolist()}
    out.write_text(json.dumps(res, indent=1))
    print(f"{a.ckpt}: phi {np.round(phi, 2).tolist()} mean {mean:.2f} mm/node; cable means {np.round(per_cable.mean(0).min(1), 2).tolist()}; worst clip {max(d.values()):.1f}, best clip {min(d.values()):.1f}", flush=True)


if __name__ == "__main__":
    main()
