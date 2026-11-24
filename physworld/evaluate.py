import json
import argparse
import re
from pathlib import Path
import numpy as np


def write_results(out, rows, meta, tol=0.5):
    v = np.array([r["reproduced"] for r in rows.values()]); rv = np.array([r["reported"] for r in rows.values()])
    bad = {c: r["diff"] for c, r in rows.items() if abs(r["diff"]) > tol}
    summ = {**meta, "n_clips": len(rows), "mean_reproduced": float(v.mean()), "mean_reported": float(rv.mean()),
            "max_abs_clip_diff": float(np.abs(v - rv).max()), "tolerance_mm": tol,
            "clips_over_tolerance": bad, "ok": not bad, "clips": rows}
    (out / "results.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: x for k, x in summ.items() if k != "clips"}, indent=1), flush=True)


import argparse
import json
import torch


ROOT = Path(__file__).resolve().parent


from model import H, dp_of, dyaw_nodes, real_yaw
from model import PhysWorldGNN


LABEL = "PhysWorld (pooled GNN, per-node Phi), 36-clip refit"


LABEL_SHARED = "PhysWorld (pooled GNN, one shared Phi for all 36 clips)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--ckpt", default="data/wm/physworld_mpm_gnn_pooled/gnn_pooled.pt")
    ap.add_argument("--reported", default="data/wm/physworld_mpm_gnn_pooled")
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--device", default="cuda")
    ap.add_argument("--shared-phi", default="")
    a = ap.parse_args(); dev = a.device; warm = a.warm
    shared = json.loads((ROOT / a.shared_phi).read_text()) if a.shared_phi else None
    out = ROOT / a.out
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    model = PhysWorldGNN(16, H).to(dev); model.load_state_dict(ck["state_dict"])
    mu, sd = ck["phi_mu"].to(dev), ck["phi_sd"].to(dev)
    rows = {}
    for npz in sorted((ROOT / a.tracks).glob("*.npz")):
        clip = npz.stem
        if shared is None:
            rep = json.loads((ROOT / a.reported / f"{clip}.json").read_text())
        else:
            z0 = np.load(npz)
            rep = {"phi_fitted": shared["phi_shared_raw"], "mittel": shared["shared_fit"]["clips"][clip],
                   "n_frames": z0["c0"].shape[0] - warm, "use_yaw": True, "per_frame": None}
        z = np.load(npz)
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        C, T = P.shape[:2]
        assert rep["n_frames"] == T - warm, (rep["n_frames"], T, warm)
        phi_raw = torch.tensor(rep["phi_fitted"], dtype=torch.float32, device=dev)
        Y = real_yaw(P)
        errs, sim_c = [], []
        with torch.no_grad():
            for c in range(C):
                X = P[c]
                phi = (phi_raw[c] - mu) / sd
                dp = dp_of(X[:-1], X[1:])
                dy = dyaw_nodes((Y[c, 1:] - Y[c, :-1]) if rep["use_yaw"] else torch.zeros_like(Y[c, 1:]))
                frames = [X[t] for t in range(warm)]; e = []
                for t in range(warm - 1, T - 1):
                    h = torch.stack([frames[max(0, t - H + j)] for j in range(H)])
                    pr = model(h[None], frames[t][None], phi[None], dp[t][None], dy[t][None])[0]
                    pr[0] = X[t + 1, 0]; pr[-1] = X[t + 1, -1]
                    frames.append(pr); e.append((pr - X[t + 1]).norm(dim=-1).mean().item() * 1000)
                errs.append(np.array(e)); sim_c.append(torch.stack(frames).cpu().numpy())
        ef = np.mean(errs, 0)
        err = np.full(T, np.nan); err[warm:] = ef
        sim = np.stack(sim_c, 1).astype(np.float64)
        n_nan = int(np.isnan(sim).sum())
        np.savez(out / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=(LABEL if shared is None else LABEL_SHARED), short="PhysWorld",
                 score_nodes=np.arange(16), phi=np.array(rep["phi_fitted"]))
        v = float(ef.mean())
        rows[clip] = {"reproduced": v, "reported": rep["mittel"], "diff": v - rep["mittel"],
                      "max_abs_per_frame_diff": (float(np.abs(ef - np.array(rep["per_frame"])).max())
                                                 if rep["per_frame"] is not None else None),
                      "nan_nodes": n_nan, "n_frames_scored": int(len(ef))}
        print(clip, json.dumps(rows[clip]), flush=True)
    if shared is None:
        meta = {"approach": "PhysWorld, one pooled GNN on the demos of the 36 clips + per-clip per-node Phi (stored fit replayed, no re-fit)",
                "params": f"{a.ckpt} + {a.reported}/<clip>.json ('phi_fitted')", "reported_source": f"{a.reported}/<clip>.json ('mittel')"}
    else:
        meta = {"approach": "PhysWorld, one pooled GNN on the demos of the 36 clips + ONE set of 128 Phi values shared by all 36 clips "
                            "(stored fit replayed, no re-fit)",
                "params": f"{a.ckpt} + {a.shared_phi} ('phi_shared_raw')", "reported_source": f"{a.shared_phi} ('shared_fit.clips')"}
    write_results(out, rows, {**meta, "device": dev,
                              "metric": f"16 nodes (ends pinned), mean over cables, frames {warm}..T-1 (after warm-up)"})


if __name__ == "__main__":
    main()
