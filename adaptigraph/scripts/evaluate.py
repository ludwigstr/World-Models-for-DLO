import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from wire_model.adaptigraph.model_v2 import AdaptiGraphV2
from wire_model.adaptigraph.features import end_actions, turning

WARM = 5


def load_tracks(dev, tracks="../data/tracks"):
    clips = sorted(p.stem for p in (ROOT / tracks).glob("*.npz"))
    cab, Tn = [], []
    for c in clips:
        z = np.load(ROOT / tracks / f"{c}.npz")
        for k in range(4):
            cab.append(z[f"c{k}"].astype(np.float32))
            Tn.append(len(cab[-1]))
    Tm = max(Tn)
    P = np.stack([np.concatenate([x, np.repeat(x[-1:], Tm - len(x), 0)]) for x in cab])
    return clips, torch.tensor(P, device=dev), torch.tensor(Tn, device=dev)


def features(flags, X, Y, rest):
    out = []
    if flags.get("rest"):
        out.append(rest[..., None])
    if flags.get("rot"):
        out.append(end_actions(X, Y)[..., 2:])
    return torch.cat(out, -1) if out else None


def dp_of(X, Y):
    d = torch.zeros_like(X)
    d[:, 0], d[:, -1] = Y[:, 0] - X[:, 0], Y[:, -1] - X[:, -1]
    return d


@torch.no_grad()
def score(m, flags, P, Tn, phi_fixed=None):
    m.eval()
    C, Tm = P.shape[:2]
    H = m.n_hist
    rest = torch.zeros(C, 16, device=P.device)
    rest[:, 1:-1] = turning(P[:, 0]).clamp(-0.6, 0.6)
    if phi_fixed is None:
        grid = torch.linspace(-0.2, 1.2, 29, device=P.device)
        cost = torch.zeros(C, len(grid), device=P.device)
        for t in range(WARM - 1):
            idx = [max(0, t - H + j) for j in range(H)]
            X, Y = P[:, t], P[:, t + 1]
            for g, ph in enumerate(grid):
                pred = m(P[:, idx].contiguous(), X, ph.expand(C), dp_of(X, Y), features(flags, X, Y, rest))
                cost[:, g] += (pred[:, 1:-1] - Y[:, 1:-1]).norm(dim=-1).mean(-1)
        phi = grid[cost.argmin(1)]
    else:
        phi = torch.full((C,), float(phi_fixed), device=P.device)
    frames = [P[:, t] for t in range(WARM)]
    total = torch.zeros(C, device=P.device)
    for t in range(WARM - 1, Tm - 1):
        idx = [max(0, t - H + j) for j in range(H)]
        hist = torch.stack([frames[i] for i in idx], 1)
        X, A, B = frames[t], P[:, t], P[:, t + 1]
        pred = m(hist, X, phi, dp_of(A, B), features(flags, A, B, rest))
        pred = pred.clone()
        pred[:, 0], pred[:, -1] = B[:, 0], B[:, -1]
        frames.append(pred)
        total += ((t + 1) < Tn) * (pred - B).norm(dim=-1).mean(-1) * 1000
    return (total / (Tn - WARM)), phi


def by_clip(e, clips):
    return {c: float(e[4 * i:4 * i + 4].mean()) for i, c in enumerate(clips)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--phi-fixed", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = torch.load(ROOT / a.ckpt, map_location=a.device, weights_only=False)
    flags = ck.get("flags", {})
    m = AdaptiGraphV2(16, ck["n_hist"], radius=ck["radius"], layers=flags.get("layers", 4), n_extra=int(bool(flags.get("rest"))) + int(bool(flags.get("rot")))).to(a.device)
    m.load_state_dict(ck["state_dict"])
    clips, P, Tn = load_tracks(a.device)
    e, phi = score(m, flags, P, Tn, a.phi_fixed)
    d = by_clip(e, clips)
    print(f"{a.ckpt} phi {'estimated' if a.phi_fixed is None else a.phi_fixed}: mean {np.mean(list(d.values())):.2f} mm/node, phi mean {phi.mean():.2f}", flush=True)
    if a.out:
        p = ROOT / a.out
        assert not p.exists(), f"{p} exists"
        p.write_text(json.dumps({"mean": float(np.mean(list(d.values()))), "clips": d, "phi": phi.tolist()}, indent=1))
    return d


if __name__ == "__main__":
    main()
