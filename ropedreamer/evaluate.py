import argparse
import json
from pathlib import Path
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


from model import RopeDreamer
from model import load_tracks, trajectory_tensors
import model as RD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pin-ends", action="store_true")
    a = ap.parse_args()
    ck = torch.load(ROOT / a.ckpt, map_location=a.device, weights_only=False)
    RD.REPR = ck.get("repr", "chain")
    model = RopeDreamer(ck["n_nodes"], ck["size"], beta=ck["beta"], pos_weight=ck.get("pos_weight", 0.0), free_nats=ck.get("free_nats", 0.0),
                        rollout_weight=ck.get("rollout_weight", 0.0), warm=ck.get("warm", 5), repr=ck.get("repr", "chain")).to(a.device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    off = ck["offset"].to(a.device)
    trajs = load_tracks([ROOT / a.tracks])
    per_clip = {}
    curves, floors = [], []
    by_clip = {}
    for t in trajs:
        by_clip.setdefault(t["clip"], []).append(t)
    for clip, cabs in by_clip.items():
        errs_c, n_frames = [], None
        for t in sorted(cabs, key=lambda x: x["cable"]):
            d = trajectory_tensors(t["P"]); T = d["s"].shape[0]
            s = (d["s"][None].to(a.device) - off); idx = d["idx"][None].to(a.device); dp = d["dp"][None].to(a.device)
            h, z = model.warmup(s[:, :a.warm], idx[:, :a.warm], dp[:, :a.warm])
            pred = model.imagine(h, z, idx[:, a.warm - 1:T - 1], dp[:, a.warm - 1:T - 1]) + off
            truth = d["P"][a.warm:]
            if not a.no_pin_ends and model.repr == "chain":
                pred = pred.clone(); pred[0, :, :2] = truth[:, 0].to(a.device)
            P = model.positions(pred[0], d["ell"].to(a.device))[..., :2].cpu()
            if not a.no_pin_ends:
                P[:, 0] = truth[:, 0]; P[:, -1] = truth[:, -1]
            e = (P - truth).norm(dim=-1).mean(-1).numpy() * 1000
            errs_c.append(e)
            floors.append((model.positions(d["s"].to(a.device), d["ell"].to(a.device))[..., :2].cpu() - d["P"]).norm(dim=-1).mean().item() * 1000)
            n_frames = T
        e = np.mean(errs_c, axis=0)
        curves.append(e[:50] if len(e) >= 50 else np.pad(e, (0, 50 - len(e)), constant_values=np.nan))
        per_clip[clip] = {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()),
                          "n_frames": int(len(e)), "video": "", "warmup_frames": a.warm}
    v = np.array([x["mittel"] for x in per_clip.values()])
    curve = np.nanmean(np.stack(curves), axis=0)
    out = ROOT / a.out
    out.write_text(json.dumps(per_clip, indent=1))
    summary = {"ckpt": a.ckpt, "tracks": a.tracks, "n_clips": len(v), "mittel": float(v.mean()),
               "warmup_frames": a.warm, "ends_pinned": not a.no_pin_ends, "representation_floor_mm": float(np.mean(floors)),
               "rmse_curve_mm_steps_1_50": [float(x) for x in curve]}
    Path(str(out).replace(".json", "_summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"{a.ckpt} on {a.tracks}: {len(v)} clips, mean {v.mean():.2f} mm/node "
          f"(frames after a {a.warm}-frame warm-up); representation floor {np.mean(floors):.2f} mm; "
          f"open-loop step 1/10/50: {curve[0]:.1f}/{curve[9]:.1f}/{curve[49]:.1f} mm")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
