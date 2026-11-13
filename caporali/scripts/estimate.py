import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_clip import read
from wire_model.perception.harness import track_harness
from wire_model.perception.rectify import find_board
from wire_model.perception.tracking import HarnessTracker
from wire_model.dlo_model import DLOParams, DLOSimulator, NodeConstraints
from train_step_model import turning_angles
from eval_rollout import train_fold, step_action, NOMINAL
from long_horizon_guided_sms import guide_batch

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIR = os.environ.get("WIRE_VIDEO_DIR", "/home/lustrel/Python/Hand_tracking/data/WirePrediction")
REST_CAP = 0.35
COLORS = [(0.85, 0.15, 0.15), (0.1, 0.6, 0.9), (1.0, 0.6, 0.0),
         (0.2, 0.8, 0.2), (0.9, 0.2, 0.9)][1:]


def roll_guided_traced(model, sim, cables, n_nodes, stride, k_p, n_steps,
                       exit_prior=False):
    n_frames = len(cables[0])
    idxs = sorted(set(list(range(0, n_frames - 1, stride)) + [n_frames - 1]))
    B = len(cables)
    V = np.stack([s[0] for s in cables]).astype(np.float32)
    p_par = np.tile(nominal_for(model), (B, 1)).astype(np.float32)
    lengths = np.array([np.linalg.norm(np.diff(s[0], axis=0), axis=1).sum()
                        for s in cables], np.float32)
    m, k_b, k_d, k_c = NOMINAL
    trace = [(idxs[0], V.copy())]

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
            trace.append((bi, V.copy()))
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
        trace.append((bi, V.copy()))
    return trace


def nominal_for_width(n_in):
    if n_in == 4:
        import os
        if os.environ.get("WIRE_NOMINAL"):
            return np.array([float(x) for x in os.environ["WIRE_NOMINAL"].split(",")], np.float32)
        return NOMINAL.copy()
    n_kb = n_in - 3
    return np.concatenate([NOMINAL[:1], np.full(n_kb, NOMINAL[1], np.float32),
                           NOMINAL[2:]]).astype(np.float32)


def nominal_for(model):
    return nominal_for_width(int(model.net.n_params))


def param_box(model, n_free):
    lo = model.norm.param_lo[:n_free].detach().float().to(DEV)
    hi = model.norm.param_hi[:n_free].detach().float().to(DEV)
    return lo, hi


def roll_online_traced(model, cables, n_nodes, stride, lr, trace_params=None,
                       est_rest=False, est_m=True):
    n_frames = len(cables[0])
    idxs = sorted(set(list(range(0, n_frames - 1, stride)) + [n_frames - 1]))
    B = len(cables)
    V = np.stack([s[0] for s in cables]).astype(np.float32)
    nom = nominal_for(model)
    k_c = float(nom[-1])
    n_free = len(nom) - 1
    lo, hi = param_box(model, n_free)
    p0 = torch.tensor(nom[:n_free], device=DEV).expand(B, n_free)
    z = torch.logit(((p0 - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)).clone()
    z.requires_grad_(True)
    zr = torch.zeros(B, n_nodes - 2, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([z] + ([zr] if est_rest else []), lr=lr)
    trace = [(idxs[0], V.copy())]

    for ai, bi in zip(idxs, idxs[1:]):
        acts = np.stack([step_action(s[ai], s[bi], n_nodes - 2, 1)
                         for s in cables]).astype(np.float32)
        p_free = lo + torch.sigmoid(z) * (hi - lo)
        if not est_m:
            p_free = torch.cat([p0[:, :1], p_free[:, 1:]], dim=1)
        p_par = torch.cat([p_free, torch.full((B, 1), k_c, device=DEV)], dim=1)
        if est_rest:
            rest_t = REST_CAP * torch.tanh(zr)
        elif os.environ.get("WIRE_REST_MODE") == "frame0":
            cap = float(os.environ.get("WIRE_REST_CAP", REST_CAP))
            rest_t = torch.tensor(np.clip(np.stack([turning_angles(c[0]) for c in cables]), -cap, cap).astype(np.float32), device=DEV)
        else:
            rest_t = torch.tensor(
                np.clip(np.stack([turning_angles(x) for x in V]),
                        -REST_CAP, REST_CAP).astype(np.float32), device=DEV)
        pred = model(torch.tensor(V, device=DEV),
                     torch.tensor(acts, device=DEV),
                     p_par,
                     rest_t)

        obs = torch.tensor(np.stack([s[bi] for s in cables]).astype(np.float32),
                           device=DEV)
        pred_fixed = pred.clone()
        for k in range(B):
            pred_fixed[k, 0] = obs[k, 0]
            pred_fixed[k, -1] = obs[k, -1]
        V = pred_fixed.detach().cpu().numpy().astype(np.float32)
        trace.append((bi, V.copy()))

        loss = ((pred_fixed - obs) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if trace_params is not None:
            with torch.no_grad():
                trace_params.append((lo + torch.sigmoid(z) * (hi - lo)
                                     ).cpu().numpy().tolist())
    return trace


def train_fold_seq(trans, hold_clip, ckpt, epochs, lr, nominal):
    from torch.utils.data import DataLoader, TensorDataset
    tr = [t for t in trans if t["clip"] != hold_clip]
    Vi = torch.tensor(np.stack([t["V_in"] for t in tr]).astype(np.float32))
    Vo = torch.tensor(np.stack([t["V_out"] for t in tr]).astype(np.float32))
    A = torch.tensor(np.stack([t["action"] for t in tr]).astype(np.float32))
    H = torch.tensor(np.stack([t["hist"] for t in tr]).astype(np.float32))
    P = torch.tensor(np.tile(np.asarray(nominal, np.float32), (len(tr), 1)))
    R = torch.tensor(np.stack([np.clip(turning_angles(t["V_in"]), -REST_CAP, REST_CAP)
                               for t in tr]).astype(np.float32))
    loader = DataLoader(TensorDataset(Vi, Vo, A, P, R, H), batch_size=128,
                        shuffle=True, drop_last=True)
    from wire_model.estimation import load_predictor
    model, _ = load_predictor(ROOT / ckpt, DEV)
    for q in model.parameters():
        q.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.15)
    for _ in range(epochs):
        model.train()
        for vi, vo, a, pp, r, h in loader:
            vi, vo, a, pp, r, h = (x.to(DEV) for x in (vi, vo, a, pp, r, h))
            loss = torch.nn.functional.mse_loss(model(vi, a, pp, r, hist_V=h), vo)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    return model


def roll_seq_traced(model, cables, n_nodes, n_hist, use_hist=True):
    n_frames = len(cables[0])
    B = len(cables)
    V = np.stack([s[0] for s in cables]).astype(np.float32)
    window = [V.copy() for _ in range(n_hist)]
    p_par = np.tile(nominal_for(model), (B, 1)).astype(np.float32)
    trace = [(0, V.copy())]
    for ai in range(n_frames - 1):
        bi = ai + 1
        acts = np.stack([step_action(s[ai], s[bi], n_nodes - 2, 1)
                         for s in cables]).astype(np.float32)
        rest = np.clip(np.stack([turning_angles(x) for x in V]),
                       -REST_CAP, REST_CAP).astype(np.float32)
        hist = np.stack(window, axis=1)
        with torch.no_grad():
            pred = model(torch.tensor(V, device=DEV), torch.tensor(acts, device=DEV),
                         torch.tensor(p_par, device=DEV), torch.tensor(rest, device=DEV),
                         hist_V=torch.tensor(hist, device=DEV)).cpu().numpy()
        pred = pred.astype(np.float32)
        for k, s in enumerate(cables):
            pred[k][0], pred[k][-1] = s[bi][0], s[bi][-1]
        window = (window[1:] + [V.copy()]) if use_hist else [pred.copy()] * n_hist
        V = pred
        trace.append((bi, V.copy()))
    return trace


def draw_poly(img, pts_px, color_bgr, thick=3, marker=3, dashed=False):
    pts = pts_px.astype(int)
    for i in range(len(pts) - 1):
        if dashed and i % 2 == 1:
            continue
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color_bgr, thick, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), marker, color_bgr, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20260602_102713")
    ap.add_argument("--in-sample", action="store_true")
    ap.add_argument("--variant",
                    choices=["raw", "ablation", "guided", "lossterm", "online",
                             "online_rest", "raw_pj", "online_pj", "seq",
                             "seq_nohist"],
                    default="guided")
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--arch", default="transformer",
                    choices=["transformer", "bilstm", "conv1d", "gnn", "film"])
    ap.add_argument("--kp-guided", type=float, default=150.0)
    ap.add_argument("--online-lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--t0-buffer", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--t1-sec", type=float, default=None)
    ap.add_argument("--guide-steps", type=int, default=4000)
    ap.add_argument("--fps-out", type=float, default=4.0)
    ap.add_argument("--out", default="data/neural_video")
    ap.add_argument("--tracks-npz", default=None)
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    if args.no_video and not args.tracks_npz:
        raise SystemExit("--no-video needs --tracks-npz")

    k_p = {"raw": 0.0, "ablation": -1.0, "guided": args.kp_guided,
          "lossterm": 0.0, "online": 0.0, "online_rest": 0.0,
          "raw_pj": 0.0, "online_pj": 0.0, "seq": 0.0, "seq_nohist": 0.0}[args.variant]
    len_weight = 1.0 if args.variant == "lossterm" else 0.0

    if args.no_video:
        cap = tf = None
    else:
        cap = cv2.VideoCapture(f"{DIR}/{args.clip}.mp4")
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        n_frames = min(n_all, args.max_frames)
        if args.t1_sec is not None:
            n_frames = min(n_frames, int(round(args.t1_sec * fps)) + 1)
        tf = find_board(read(cap, 0))
    if args.tracks_npz:
        d = np.load(args.tracks_npz)
        shp = {k: d[f"c{k}"] for k in range(4)}
        t0, t1 = int(d["t0"]), int(d["t1"])
        if cap is not None:
            cap.release()
        n_nodes = shp[0].shape[1]
    else:
        tracks, hub_i, _ = track_harness(cap, tf, n_frames, read, clip=args.clip)
        if tracks is None:
            raise SystemExit(f"{args.clip}: Harness nicht gefunden")
        order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
        tracks = tracks[:, order, :]
        t0 = min(args.t0_buffer, max(0, n_frames // 4))
        t1 = n_frames - 1
        tracked = HarnessTracker(tf, clip=args.clip).run(cap, read, tracks, range(t0, t1 + 1))
        cap.release()
        n_nodes = len(tracked[t0][0]["shape_mm"])
        shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in range(t0, t1 + 1)])
              for k in range(4)}
    cables = [shp[k] for k in range(4)]
    print(f"{args.clip}: {len(cables[0])} Frames getrackt ({args.variant}), "
          f"trainiere LOCO-Fold ...", flush=True)

    trans = torch.load(ROOT / "data/transitions_1s_all.pt", weights_only=False)["transitions"]
    if args.base_ckpt:
        base_ckpt = args.base_ckpt
    elif args.variant == "seq":
        base_ckpt = "data/nn_seq_transformer.pt"
    elif args.variant == "seq_nohist":
        base_ckpt = "data/nn_seq_transformer_nohist.pt"
    elif args.variant.endswith("_pj"):
        base_ckpt = "data/nn_perjoint_transformer.pt"
    else:
        base_ckpt = ("data/nn_twoend_transformer.pt" if args.arch == "transformer"
                     else f"data/bench_{args.arch}.pt")
    n_in = int(torch.load(ROOT / base_ckpt, map_location="cpu",
                          weights_only=False).get("n_params", 4))
    hold = None if args.in_sample else args.clip
    if args.variant in ("seq", "seq_nohist"):
        trans_seq = torch.load(ROOT / "data/transitions_seq_all.pt",
                               weights_only=False)["transitions"]
        model = train_fold_seq(trans_seq, hold, base_ckpt, args.epochs, 3e-4,
                               nominal_for_width(n_in))
    else:
        model = train_fold(trans, hold, base_ckpt,
                           args.epochs, 3e-4, len_weight=len_weight,
                           nominal=nominal_for_width(n_in))

    sim = DLOSimulator(n_nodes=16, dt=4e-5, device=DEV)
    param_trace = []
    if args.variant in ("seq", "seq_nohist"):
        n_hist = int(torch.load(ROOT / base_ckpt, map_location="cpu",
                                weights_only=False).get("n_hist", 4))
        trace = roll_seq_traced(model, cables, n_nodes, n_hist,
                                use_hist=args.variant == "seq")
    elif args.variant in ("online", "online_rest", "online_pj"):
        trace = roll_online_traced(model, cables, n_nodes, args.stride,
                                   args.online_lr, trace_params=param_trace,
                                   est_rest=args.variant == "online_rest")
        lbl = "(m, k_b, k_d)" if n_in == 4 else f"(m, k_b x {n_in - 3}, k_d)"
        print(f"{args.clip}: p nach dem Rollout {lbl} je Kabel: "
              f"{np.round(np.array(param_trace[-1]), 4).tolist()}", flush=True)
    else:
        trace = roll_guided_traced(model, sim, cables, n_nodes, args.stride, k_p,
                                   args.guide_steps)
    print(f"{args.clip}: Rollout fertig, {len(trace)} Frames, rendere Video ...",
          flush=True)

    arch_tag = "" if args.arch == "transformer" else f"_{args.arch}"
    errs = []
    if args.no_video:
        for frame_idx, V in trace:
            errs.append(float(np.mean([np.linalg.norm(V[k] - shp[k][frame_idx], axis=1).mean() * 1000
                                       for k in range(4)])))
        print(f"\n{args.clip} [{args.variant}{arch_tag}]: Start {errs[0]:.1f}  Ende {errs[-1]:.1f}  "
              f"Mittel {np.mean(errs):.1f} mm/Knoten  (no video)")
    else:
        errs = render_video(args, trace, shp, tf, t0, fps, arch_tag)
    if args.results_json:
        _append_results(args, trace, errs, base_ckpt, n_in, param_trace)


def render_video(args, trace, shp, tf, t0, fps, arch_tag):
    outdir = ROOT / args.out / f"{args.variant}{arch_tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / args.clip
    frames_dir.mkdir(exist_ok=True)

    corners_mm = ((70, 70), (1130, 70), (1130, 1680), (70, 1680))
    pxc = tf.to_px(np.array(corners_mm, dtype=float))
    cap = cv2.VideoCapture(f"{DIR}/{args.clip}.mp4")
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))

    def px(shape_m):
        return tf.to_px(shape_m * 1000.0) - np.array([x0, y0])

    errs = []
    import imageio.v2 as iio
    frame_paths = []
    for fi, (frame_idx, V) in enumerate(trace):
        fr = read(cap, t0 + frame_idx)
        vis = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].copy()
        real = shp
        e = []
        for k in range(4):
            col_rgb = COLORS[k]
            col_bgr = (int(255 * col_rgb[2]), int(255 * col_rgb[1]), int(255 * col_rgb[0]))
            draw_poly(vis, px(real[k][frame_idx]), col_bgr, thick=3, marker=3)
            draw_poly(vis, px(V[k]), col_bgr, thick=2, marker=3, dashed=True)
            e.append(float(np.linalg.norm(V[k] - real[k][frame_idx], axis=1).mean()) * 1000)
        errs.append(float(np.mean(e)))
        cv2.putText(vis, f"{args.variant}  t={(t0+frame_idx)/fps:.1f}s  "
                    f"Abweichung {errs[-1]:.0f} mm/Knoten",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        p = frames_dir / f"f{fi:03d}.png"
        iio.imwrite(p, vis)
        frame_paths.append(p)

    imgs = [iio.imread(p) for p in frame_paths]
    h = min(im.shape[0] for im in imgs); w = min(im.shape[1] for im in imgs)
    video_path = outdir / f"vergleich_{args.clip}.mp4"
    iio.mimsave(video_path, [im[:h, :w] for im in imgs], fps=args.fps_out)

    print(f"\n{args.clip} [{args.variant}{arch_tag}]: Start {errs[0]:.1f}  Ende {errs[-1]:.1f}  "
          f"Mittel {np.mean(errs):.1f} mm/Knoten")
    print(f"-> {video_path}")
    return errs


def _append_results(args, trace, errs, base_ckpt, n_in, param_trace):
    if True:
        import json
        rj = ROOT / args.results_json
        records = json.loads(rj.read_text()) if rj.exists() else []
        rec = {"clip": args.clip, "variant": args.variant, "arch": args.arch,
               "start": errs[0], "ende": errs[-1], "mittel": float(np.mean(errs)),
               "n_frames": len(errs), "guide_steps": args.guide_steps,
               "base_ckpt": base_ckpt, "n_params": n_in,
               "errs": [[int(fi), float(e)] for (fi, _), e in zip(trace, errs)]}
        if param_trace:
            rec["p_final"] = param_trace[-1]
        records.append(rec)
        rj.write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
