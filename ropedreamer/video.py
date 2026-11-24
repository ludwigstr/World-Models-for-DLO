import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from render import write_results
from model import RopeDreamer
from data import load_tracks, trajectory_tensors
import data as RD

LABEL = "RopeDreamer (RSSM, quaternion chain), trained on MuJoCo rollouts of the 36 clips"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--ckpt", default="data/ropedreamer_S_noise_seq120_roll1000.pt")
    ap.add_argument("--reported", default="data/ropedreamer_S_noise_seq120_roll1000_tracks36_results.json")
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; warm = a.warm
    out = ROOT / a.out
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    RD.REPR = ck.get("repr", "chain")
    model = RopeDreamer(ck["n_nodes"], ck["size"], beta=ck["beta"], pos_weight=ck.get("pos_weight", 0.0),
                        free_nats=ck.get("free_nats", 0.0), rollout_weight=ck.get("rollout_weight", 0.0),
                        warm=ck.get("warm", 5), repr=ck.get("repr", "chain")).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    off = ck["offset"].to(dev)
    rep = json.loads((ROOT / a.reported).read_text())
    by_clip = {}
    for t in load_tracks([ROOT / a.tracks]):
        by_clip.setdefault(t["clip"], []).append(t)
    rows = {}
    for clip in sorted(by_clip):
        cabs = sorted(by_clip[clip], key=lambda x: x["cable"]); assert [t["cable"] for t in cabs] == [0, 1, 2, 3]
        sims, errs = [], []
        for t in cabs:
            d = trajectory_tensors(t["P"]); T = d["s"].shape[0]
            s = d["s"][None].to(dev) - off; idx = d["idx"][None].to(dev); dp = d["dp"][None].to(dev)
            with torch.no_grad():
                h, z = model.warmup(s[:, :warm], idx[:, :warm], dp[:, :warm])
                pred = model.imagine(h, z, idx[:, warm - 1:T - 1], dp[:, warm - 1:T - 1]) + off
                truth = d["P"][warm:]
                if model.repr == "chain":
                    pred = pred.clone(); pred[0, :, :2] = truth[:, 0].to(dev)
                P = model.positions(pred[0], d["ell"].to(dev))[..., :2].cpu()
            P[:, 0] = truth[:, 0]; P[:, -1] = truth[:, -1]
            errs.append((P - truth).norm(dim=-1).mean(-1).numpy() * 1000)
            full = d["P"][..., :2].numpy().astype(np.float64).copy()
            full[warm:] = P.numpy(); sims.append(full)
        e = np.mean(errs, axis=0)
        err = np.full(T, np.nan); err[warm:] = e
        sim = np.stack(sims, axis=1)
        np.savez(out / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=LABEL, short="RopeDreamer",
                 score_nodes=np.arange(16))
        v = float(e.mean())
        rows[clip] = {"reproduced": v, "reported": rep[clip]["mittel"], "diff": v - rep[clip]["mittel"],
                      "nan_nodes": int(np.isnan(sim).sum()), "n_frames_scored": int(len(e))}
        print(clip, json.dumps(rows[clip]), flush=True)
    write_results(out, rows, {"approach": "RopeDreamer, RSSM over a quaternionic chain, trained on MuJoCo rollouts "
                                          "of the 36 clips only (stored network replayed, no re-training)",
                              "params": a.ckpt, "device": dev, "reported_source": f"{a.reported} ('mittel')",
                              "metric": f"16 nodes (ends pinned), mean over cables, frames {warm}..T-1 (after warm-up)"})


if __name__ == "__main__":
    main()
