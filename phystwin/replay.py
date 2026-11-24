import json
import argparse
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


from twin import SpringMassTwin, build_topology


LABEL = "PhysTwin (per-clip spring-mass twin), 36-clip refit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--tracks", default="../data/tracks"); ap.add_argument("--twins", default="../results/phystwin/twins")
    ap.add_argument("--results-prefix", default="../results/phystwin/per_clip/")
    a = ap.parse_args(); dev = a.device
    out = ROOT / a.out
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    rows = {}
    for npz in sorted((ROOT / a.tracks).glob("*.npz")):
        clip = npz.stem
        ref = json.loads((ROOT / f"{a.results_prefix}{clip}.json").read_text())[clip]["mittel"]
        z = np.load(npz)
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        C, T = P.shape[:2]
        tw = torch.load(ROOT / a.twins / f"{clip}.pt", weights_only=False)
        frame_s = tw["frame_s"]
        assert abs(frame_s - 1.0 / float(z["fps"])) < 1e-12
        r = round(float(tw["x1"][4]) * 2) / 2
        twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), frame_s=frame_s)
        adj_ok = bool(torch.equal(twin.adj.cpu().bool(), tw["adj"].bool()))
        k, gl = tw["k"].to(dev), tw["gl"].to(dev)
        with torch.no_grad():
            X0 = P[:, 0]; V0 = (P[:, 1] - P[:, 0]) / frame_s
            sim = twin.rollout(X0, V0, P[:, :, [0, -1]], k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C))
            ef = torch.nan_to_num((sim[:, :, 1:-1] - P[:, :, 1:-1]).norm(dim=-1), nan=1e3).mean((0, 2)).cpu().numpy() * 1000
        sim_np = sim.cpu().numpy().transpose(1, 0, 2, 3).astype(np.float64)
        n_nan = int(np.isnan(sim_np).sum())
        e = float(ef.mean())
        np.savez(out / "rollouts" / f"{clip}.npz", sim=sim_np, err=ef.astype(float), label=LABEL, short="PhysTwin",
                 score_nodes=np.arange(1, 15))
        rows[clip] = {"reproduced": e, "reported": ref, "diff": e - ref, "topology_matches_saved_adj": adj_ok,
                      "nan_nodes": n_nan, "n_frames": int(T)}
        print(clip, json.dumps(rows[clip]), flush=True)
    write_results(out, rows, {"approach": "PhysTwin, per-clip dense twins refitted on the 36-clip selection",
                              "params": f"{a.twins}/<clip>.pt", "device": dev,
                              "reported_source": f"{a.results_prefix}<clip>.json ('mittel')",
                              "metric": "14 inner nodes (ends kinematic), mean over cables, every frame"})


if __name__ == "__main__":
    main()
