import argparse, json, os, sys
from pathlib import Path

os.environ.update({"WIRE_NOMINAL": "0.04,1.0,20,0.15", "WIRE_REST_MODE": "frame0", "WIRE_REST_CAP": "0.6"})
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import estimate as R
from render import write_results

LABEL = "Caporali et al. (online parameter estimation), 36-clip refit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--ckpt", default="data/bench_film_c36_step1.pt")
    ap.add_argument("--results-json", default="data/neural_video_caporali_c36.json")
    ap.add_argument("--online-lr", type=float, default=0.05)
    a = ap.parse_args()
    out = ROOT / a.out
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    rep = {r["clip"]: r for r in json.loads((ROOT / a.results_json).read_text())
           if r["variant"] == "online" and r["arch"] == "film"}
    trans = torch.load(ROOT / "data/transitions_1s_all.pt", weights_only=False)["transitions"]
    n_in = int(torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False).get("n_params", 4))
    rows = {}
    for npz in sorted((ROOT / a.tracks).glob("*.npz")):
        clip = npz.stem
        d = np.load(npz); shp = {k: d[f"c{k}"] for k in range(4)}; cables = [shp[k] for k in range(4)]
        T = len(cables[0]); n_nodes = shp[0].shape[1]
        model = R.train_fold(trans, None, a.ckpt, 0, 3e-4, len_weight=0.0, nominal=R.nominal_for_width(n_in))
        trace = R.roll_online_traced(model, cables, n_nodes, 1, a.online_lr, trace_params=[], est_rest=False)
        assert [fi for fi, _ in trace] == list(range(T)), "stride-1 trace must cover every frame"
        sim = np.stack([V for _, V in trace]).astype(np.float64)
        err = np.array([np.mean([np.linalg.norm(V[k] - shp[k][fi], axis=1).mean() * 1000 for k in range(4)])
                        for fi, V in trace])
        np.savez(out / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=LABEL, short="Caporali",
                 score_nodes=np.arange(16))
        ref = rep[clip]; ref_f = np.array([e for _, e in ref["errs"]])
        e = float(err.mean())
        rows[clip] = {"reproduced": e, "reported": ref["mittel"], "diff": e - ref["mittel"], "n_frames": T,
                      "max_abs_frame_diff": float(np.abs(err - ref_f).max()) if len(ref_f) == T else None}
        print(clip, json.dumps(rows[clip]), flush=True)
    write_results(out, rows, {"approach": "Caporali et al., FiLM MLP trained on data from the 36 clips, online (m, k_b, k_d)",
                              "params": a.ckpt, "device": R.DEV,
                              "env": {k: os.environ[k] for k in ("WIRE_NOMINAL", "WIRE_REST_MODE", "WIRE_REST_CAP")},
                              "reported_source": f"{a.results_json} ('mittel', variant online, arch film)",
                              "metric": "16 nodes, mean over cables, every frame incl. frame 0"})


if __name__ == "__main__":
    main()
