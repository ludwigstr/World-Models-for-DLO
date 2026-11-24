import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_clip import read
from wire_model.perception.harness import track_harness
from wire_model.perception.rectify import find_board
from wire_model.perception.tracking import HarnessTracker
from wire_model.dlo_model import DLOParams, DLOSimulator, NodeConstraints
from train_step_model import turning_angles
from eval_rollout import train_fold, train_fold_seq, step_action, NOMINAL
from long_horizon_all import render_lh

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIR = "/home/lustrel/Python/Hand_tracking/data/WirePrediction"
REST_CAP = 0.35


def guide_batch(sim, p_prev, setpoint, prm, k_p, n_steps, cons_target=None):
    p = p_prev.clone()
    v = torch.zeros_like(p)
    B, N, _ = p.shape
    mask = torch.zeros(B, N, dtype=torch.bool, device=p.device)
    mask[:, 0] = True
    mask[:, -1] = True
    cons = NodeConstraints(mask, setpoint if cons_target is None else cons_target)
    for _ in range(n_steps):
        f_ext = None if k_p == 0.0 else k_p * (setpoint - p)
        p, v = sim.step(p, v, prm, cons, f_ext=f_ext)
    return p


def roll_guided(model, sim, cables, n_nodes, stride, k_p, n_steps,
                exit_prior=False):
    n_frames = len(cables[0])
    idxs = sorted(set(list(range(0, n_frames - 1, stride)) + [n_frames - 1]))
    B = len(cables)
    V = np.stack([s[0] for s in cables]).astype(np.float32)
    p_par = np.tile(NOMINAL, (B, 1)).astype(np.float32)
    lengths = np.array([np.linalg.norm(np.diff(s[0], axis=0), axis=1).sum()
                        for s in cables], np.float32)
    m, k_b, k_d, k_c = NOMINAL

    for ai, bi in zip(idxs, idxs[1:]):
        acts = np.stack([step_action(s[ai], s[bi], n_nodes - 2, 1)
                         for s in cables]).astype(np.float32)
        rest = np.clip(np.stack([turning_angles(x) for x in V]),
                       -REST_CAP, REST_CAP).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(V, device=DEV),
                         torch.tensor(acts, device=DEV),
                         torch.tensor(p_par, device=DEV),
                         torch.tensor(rest, device=DEV)).cpu().numpy()
        pred = pred.astype(np.float32)
        for k, s in enumerate(cables):
            pred[k][0], pred[k][-1] = s[bi][0], s[bi][-1]

        if k_p == 0.0:
            V = pred
            continue

        rest_prm = rest.copy()
        if exit_prior:
            rest_prm[:, 0] = 0.0
            rest_prm[:, -1] = 0.0
        prm = DLOParams(m=m, k_b=k_b, k_d=k_d, k_c=k_c,
                        length=torch.tensor(lengths, device=DEV),
                        rest_angle=torch.tensor(rest_prm, device=DEV),
                        ).to_batched(B, DEV, n_nodes=n_nodes)

        target = torch.tensor(V if k_p < 0 else pred, device=DEV)
        gain = 0.0 if k_p < 0 else k_p
        cons_target = torch.tensor(pred, device=DEV)
        V = guide_batch(sim, torch.tensor(V, device=DEV), target, prm, gain,
                        n_steps, cons_target=cons_target
                        ).cpu().numpy().astype(np.float32)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="data/transitions_1s_all.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--t0-buffer", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--kp", default="0,150,400")
    ap.add_argument("--guide-steps", type=int, default=4000)
    ap.add_argument("--clips", type=int, default=0)
    ap.add_argument("--noise-mm", type=float, default=0.0)
    ap.add_argument("--tau", type=int, default=0)
    ap.add_argument("--pinn-lambda", type=float, default=0.0)
    ap.add_argument("--pinn-warmup", type=float, default=0.3)
    ap.add_argument("--exit-lambda", type=float, default=0.0)
    ap.add_argument("--exit-prior", action="store_true")
    ap.add_argument("--out", default="data/long_horizon_guided_sms.pt")
    args = ap.parse_args()

    kps = [float(x) for x in args.kp.split(",")]
    trans = torch.load(ROOT / args.transitions, weights_only=False)["transitions"]
    clips = sorted(set(t["clip"] for t in trans))
    if args.clips:
        clips = clips[:args.clips]
    print(f"{len(clips)} Clips, Gains {kps}, {args.guide_steps} Sim-Schritte "
          f"je Prognoseschritt\n", flush=True)

    sim = DLOSimulator(n_nodes=16, dt=4e-5, device=DEV)
    results = []
    for ci, clip in enumerate(clips):
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        n_frames = min(n_all, args.max_frames)
        tf = find_board(read(cap, 0))
        tracks, hub_i, _ = track_harness(cap, tf, n_frames, read, clip=clip)
        if tracks is None:
            print(f"[{ci+1}/{len(clips)}] {clip}: Harness nicht gefunden")
            cap.release()
            continue
        order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
        tracks = tracks[:, order, :]
        t0 = min(args.t0_buffer, max(0, n_frames // 4))
        t1 = n_frames - 1
        if t1 - t0 < args.stride * 2:
            print(f"[{ci+1}/{len(clips)}] {clip}: zu kurz")
            cap.release()
            continue
        tracked = HarnessTracker(tf, clip=clip).run(cap, read, tracks, range(t0, t1 + 1))
        cap.release()
        if t0 not in tracked or t1 not in tracked:
            print(f"[{ci+1}/{len(clips)}] {clip}: Rand nicht getrackt")
            continue
        try:
            n_nodes = len(tracked[t0][0]["shape_mm"])
            shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in range(t0, t1 + 1)])
                   for k in range(4)}
        except KeyError:
            print(f"[{ci+1}/{len(clips)}] {clip}: Luecken im Tracking")
            continue

        horizon_s = (t1 - t0) / fps
        print(f"[{ci+1}/{len(clips)}] {clip}: Hold-out-Training, "
              f"Horizont {horizon_s:.0f}s ...", flush=True)
        if args.tau:
            model = train_fold_seq(trans, clip, "data/nn_twoend_transformer.pt",
                                   args.epochs, 3e-4, tau=args.tau,
                                   noise_mm=args.noise_mm)
        else:
            model = train_fold(trans, clip, "data/nn_twoend_transformer.pt",
                               args.epochs, 3e-4, noise_mm=args.noise_mm,
                               pinn_lambda=args.pinn_lambda,
                               pinn_warmup=args.pinn_warmup,
                               exit_lambda=args.exit_lambda)

        cables = [shp[k] for k in range(4)]
        entry = {"clip": clip, "t0": t0, "t1": t1, "fps": fps,
                 "horizon_s": horizon_s, "by_kp": {}}
        for k_p in kps:
            V = roll_guided(model, sim, cables, n_nodes, args.stride,
                            k_p, args.guide_steps, exit_prior=args.exit_prior)
            errs = [1000.0 * float(np.linalg.norm(V[k] - cables[k][-1], axis=1).mean())
                    for k in range(4)]
            entry["by_kp"][k_p] = {"errs": errs, "avg": float(np.mean(errs)),
                                   "pred": V}
            tag = ("nur Physik" if k_p < 0 else
                   "ungeguidet" if k_p == 0 else f"k_p={k_p:g}")
            print(f"    {tag:<14} {np.mean(errs):6.1f} mm/Knoten  "
                  f"(M0-M3: {[round(e,1) for e in errs]})", flush=True)
        results.append(entry)

        outdir = (ROOT / args.out).with_suffix("")
        outdir.mkdir(parents=True, exist_ok=True)
        pos_kps = [k for k in kps if k > 0]
        if pos_kps:
            best_kp = min(pos_kps, key=lambda k: entry["by_kp"][k]["avg"])
            V = entry["by_kp"][best_kp]["pred"]
            render_lh(clip, tf, tracks, t0, t1, fps, shp,
                      {k: V[k] for k in range(4)},
                      {k: entry["by_kp"][best_kp]["errs"][k] for k in range(4)},
                      outdir)

    torch.save({"results": results, "kps": kps,
                "guide_steps": args.guide_steps}, ROOT / args.out)

    print(f"\n{'='*68}")
    print(f"{len(results)} Clips, Within-Run-Vergleich")
    base = np.array([r["by_kp"][0.0]["avg"] for r in results]) if 0.0 in kps else None
    for k_p in kps:
        per_clip = np.array([r["by_kp"][k_p]["avg"] for r in results])
        per_cab = np.array([e for r in results for e in r["by_kp"][k_p]["errs"]])
        tag = "ungeguidet" if k_p <= 0 else f"k_p={k_p:g}"
        line = (f"{tag:<12} Pro-Clip Median {np.median(per_clip):6.1f} Mittel "
                f"{per_clip.mean():6.1f} | Pro-Kabel Median {np.median(per_cab):6.1f} "
                f"Mittel {per_cab.mean():6.1f} mm/Knoten")
        if base is not None and k_p != 0:
            line += f" | besser bei {int((per_clip < base).sum())}/{len(results)} Clips"
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
