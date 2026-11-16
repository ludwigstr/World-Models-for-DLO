import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import gnn as G
from wire_model.physworld_mpm.gnn import PhysWorldGNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--box", type=float, default=1.5)
    ap.add_argument("--out", default="data/wm/physworld_shared_phi.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gnn", default="data/wm/physworld_mpm_gnn_pooled/gnn_pooled.pt")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    dev, warm = a.device, 5
    ck = torch.load(ROOT / a.gnn, map_location=dev, weights_only=False)
    model = PhysWorldGNN(16, G.H).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mu, sd = ck["phi_mu"].to(dev), ck["phi_sd"].to(dev)

    clips = sorted(p.stem for p in (ROOT / "../data/tracks").glob("*.npz"))
    P, twin = {}, {}
    for c in clips:
        z = np.load(ROOT / f"../data/tracks/{c}.npz")
        P[c] = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        twin[c] = torch.load(ROOT / f"data/wm/physworld_mpm_demos/{c}.pt", weights_only=False)["phi_twin"].float().to(dev)
    mean_twin = torch.stack([twin[c] for c in clips]).mean(0)

    def score(phi_of):
        d = {}
        for c in clips:
            e, _ = G.finetune_and_eval(model, P[c], phi_of(c), mu, sd, warm, dev, 0, True, finetune=False)
            d[c] = float(e.mean())
        return d

    res = {"baselines": {}, "phi_twin_mean": mean_twin.cpu().tolist()}

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(out)

    for name, f in [("twin Phi", lambda c: twin[c]), ("mean twin Phi", lambda c: mean_twin)]:
        t = time.time()
        d = score(f)
        res["baselines"][name] = {"mean": float(np.mean(list(d.values()))), "clips": d}
        print(f"[{time.strftime('%H:%M:%S')}] {name}: mean {res['baselines'][name]['mean']:.2f} mm/node ({time.time() - t:.0f}s)", flush=True)
        save()

    data = []
    for c in clips:
        Y = G.real_yaw(P[c])
        for k in range(4):
            X = P[c][k]
            data.append((k, X, G.dp_of(X[:-1], X[1:]), G.dyaw_nodes(Y[k, 1:] - Y[k, :-1])))
    phi = ((mean_twin - mu) / sd).clone().requires_grad_(True)
    lo, hi = (mean_twin - a.box - mu) / sd, (mean_twin + a.box - mu) / sd
    opt = torch.optim.Adam([phi], lr=0.05)
    best, best_phi, hist = float("inf"), None, []
    for it in range(1, a.iters + 1):
        t = time.time()
        opt.zero_grad()
        errs = []
        for k, X, dp, dy in data:
            pred = G.open_loop(model, X, phi[k], dp, dy, warm)
            tgt = X[warm:]
            mse = ((pred[:, 1:-1] - tgt[:, 1:-1]) ** 2).mean()
            cd = (torch.cdist(tgt, pred).min(-1).values ** 2).mean()
            ((mse + cd) / len(data)).backward()
            errs.append((pred.detach() - tgt).norm(dim=-1).mean().item() * 1000)
        m = float(np.mean(errs))
        if m < best:
            best, best_phi = m, phi.detach().clone()
        opt.step()
        with torch.no_grad():
            phi.copy_(torch.maximum(torch.minimum(phi, hi), lo))
        hist.append(m)
        print(f"[{time.strftime('%H:%M:%S')}] iter {it}: {m:.2f} mm/node, best {best:.2f} ({time.time() - t:.0f}s)", flush=True)
        res["history"], res["best_during_fit"] = hist, best
        res["phi_shared_raw"] = (best_phi * sd + mu).cpu().tolist()
        save()
    raw = best_phi * sd + mu
    d = score(lambda c: raw)
    res["shared_fit"] = {"mean": float(np.mean(list(d.values()))), "clips": d, "n_phi": 128, "iters": a.iters}
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE shared fitted Phi: mean {res['shared_fit']['mean']:.2f} mm/node", flush=True)


if __name__ == "__main__":
    main()
