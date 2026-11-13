import argparse
import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import track_clip, build_xml
from mujoco_replay import cable_vertices

ROOT = Path(__file__).resolve().parents[1]
DIR = "/home/lustrel/Python/Hand_tracking/data/WirePrediction"

from clip_set import CLIP_TRIMS


def eval_clip(clip, t1_sec=None, weld_solref=None, cable_kw=None,
              jnt_stiffness=None):
    import mujoco
    from mujoco_scene import HUB_CAVS, PLUG_CAV, _rotz

    t1 = None
    if t1_sec is not None:
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_probe = cap.get(cv2.CAP_PROP_FPS) or 4.0
        cap.release()
        t1 = int(round(t1_sec * fps_probe))
    shp, t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])

    xml = build_xml(shp, weld_solref=weld_solref, **(cable_kw or {}))
    m = mujoco.MjModel.from_xml_string(xml)
    if jnt_stiffness is not None:
        from mujoco_scene import set_per_joint_stiffness
        set_per_joint_stiffness(m, jnt_stiffness)
    return rollout_errs(m, shp, fps)


def rollout_errs(m, shp, fps, per_cable=False):
    import mujoco
    from mujoco_scene import HUB_CAVS, PLUG_CAV, _rotz
    from mujoco_replay import cable_vertices
    n_frames = len(shp[0])
    d = mujoco.MjData(m)
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    act = {n: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_{a}a")
               for a in ("x", "y", "t")] for n in names}
    base = {n: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][:2].copy()
            for n in names}
    yaw0 = {}
    for n in names:
        q = m.body_quat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}_mod")]
        yaw0[n] = 2.0 * np.arctan2(q[3], q[0])

    def poses(t):
        r2 = [shp[k][t] for k in range(4)]
        out = np.mean([P[1] - P[0] for P in r2], axis=0)
        hy = np.arctan2(out[1], out[0]) + np.pi / 2
        hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
        res = {"hub": (np.array([r2[0][0][0] - hoff[0], r2[0][0][1] - hoff[1]]), hy)}
        for k, P in enumerate(r2):
            v = P[-1] - P[-2]
            y = np.arctan2(v[1], v[0]) + np.pi / 2
            o = _rotz(np.array(PLUG_CAV), y)
            res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
        return res

    seq, prev = [], {n: poses(0)[n][1] for n in names}
    for t in range(n_frames):
        p = poses(t)
        for n in names:
            prev[n] += np.arctan2(np.sin(p[n][1] - prev[n]), np.cos(p[n][1] - prev[n]))
        seq.append({n: (p[n][0], prev[n]) for n in names})

    def apply(pose):
        for n in names:
            xy, yaw = pose[n]
            ax, ay, at = act[n]
            d.ctrl[ax], d.ctrl[ay] = xy[0] - base[n][0], xy[1] - base[n][1]
            d.ctrl[at] = yaw - yaw0[n]

    apply(seq[0])
    for _ in range(300):
        mujoco.mj_step(m, d)

    errs = []
    sim0 = np.stack([cable_vertices(m, d, k) for k in range(4)])
    e0 = [1000 * np.linalg.norm(sim0[k][:, :2] - shp[k][0], axis=1).mean()
          for k in range(4)]
    errs.append(np.array(e0) if per_cable else float(np.mean(e0)))
    for t in range(n_frames - 1):
        a0, a1 = seq[t], seq[t + 1]
        for s_ in range(sub):
            f = (s_ + 1) / sub
            apply({n: (a0[n][0] + f * (a1[n][0] - a0[n][0]),
                       a0[n][1] + f * (a1[n][1] - a0[n][1])) for n in names})
            mujoco.mj_step(m, d)
        sim = np.stack([cable_vertices(m, d, k) for k in range(4)])
        e = [1000 * np.linalg.norm(sim[k][:, :2] - shp[k][t + 1], axis=1).mean()
             for k in range(4)]
        errs.append(np.array(e) if per_cable else float(np.mean(e)))
    return np.array(errs) if per_cable else errs


def _eval_one_clip(weld_solref, cable_kw, jnt_stiffness, clip):
    t0 = time.time()
    try:
        errs = eval_clip(clip, t1_sec=CLIP_TRIMS[clip],
                         weld_solref=weld_solref, cable_kw=cable_kw,
                         jnt_stiffness=jnt_stiffness)
    except Exception:
        return clip, None, traceback.format_exc(), time.time() - t0
    return clip, errs, None, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weld-fit", default=None)
    ap.add_argument("--out-dir", default="data/mujoco_full_loco")
    ap.add_argument("--mass-per-m", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--damping", type=float, default=None)
    ap.add_argument("--bend", type=float, default=None)
    ap.add_argument("--twist", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--condim", type=int, default=None, choices=[1, 3, 4, 6])
    ap.add_argument("--mu-tor", type=float, default=None)
    ap.add_argument("--mu-roll", type=float, default=None)
    ap.add_argument("--jnt-stiffness", type=float, default=None)
    ap.add_argument("--jnt-stiffness-file", default=None)
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"

    cable_kw = {k: v for k, v in
                dict(mass_per_m=args.mass_per_m, mu=args.mu, damping=args.damping,
                     bend=args.bend, twist=args.twist, condim=args.condim,
                     mu_tor=args.mu_tor, mu_roll=args.mu_roll).items()
                if v is not None}
    if cable_kw:
        print(f"[{args.tag}] build_xml overrides: {cable_kw}", flush=True)

    jnt_stiff = None
    if args.jnt_stiffness_file is not None:
        jnt_stiff = np.load(ROOT / args.jnt_stiffness_file, allow_pickle=True)["best"]
    elif args.jnt_stiffness is not None:
        jnt_stiff = np.full((4, 14), args.jnt_stiffness)

    weld_solref = None
    if args.weld_fit is not None:
        weld_solref = np.load(ROOT / args.weld_fit, allow_pickle=True)["best"]

    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"[{args.tag}] Fortgesetzt: {len(results)} Clips schon vorhanden", flush=True)

    clips = list(CLIP_TRIMS.keys())
    t_start = time.time()
    done0 = len(results)
    todo = [c for c in clips if c not in results]

    _one = partial(_eval_one_clip, weld_solref, cable_kw, jnt_stiff)

    def _record(clip, errs, err_txt, dt):
        if errs is None:
            print(f"[{args.tag}] {clip}: FEHLER\n{err_txt}", flush=True)
            return
        results[clip] = {"start": errs[0], "ende": errs[-1],
                         "mittel": float(np.mean(errs)), "n_frames": len(errs)}
        out_path.write_text(json.dumps(results, indent=1))
        n_done = len(results) - done0
        eta = (time.time() - t_start) / max(1, n_done) * (len(clips) - len(results))
        print(f"[{args.tag}][{len(results)}/{len(clips)}] {clip}: "
              f"Mittel {results[clip]['mittel']:.1f} "
              f"({dt:.0f}s, ETA {eta/60:.0f} min)", flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, c): c for c in todo}
            for f in as_completed(futs):
                _record(*f.result())
    else:
        for clip in todo:
            _record(*_one(clip))

    mittel = np.array([v["mittel"] for v in results.values()])
    print(f"\n[{args.tag}] {len(mittel)} Clips, Median {np.median(mittel):.1f}  Mittel {mittel.mean():.1f} mm/Knoten")
    print(f"MUJOCO_FULL_LOCO_{args.tag.upper()}_FERTIG")


if __name__ == "__main__":
    main()
