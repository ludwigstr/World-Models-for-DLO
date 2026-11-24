import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_set import CLIP_TRIMS
from mujoco_scene import build_xml, cable_ball_joints, set_per_joint_stiffness
from fit_jnt_stiffness import load_clips

ROOT = Path(__file__).resolve().parents[1]
N_C, N_J = 4, 14
K_FLOOR = 0.01
_CACHE = {}
_AXIS = {}


def board_normal_axes(m):
    import mujoco
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ids = cable_ball_joints(m)
    n = np.zeros((N_C, N_J, 3))
    for k in range(N_C):
        for j in range(N_J):
            R = d.xmat[m.jnt_bodyid[ids[k, j]]].reshape(3, 3)
            n[k, j] = R.T @ np.array([0.0, 0.0, 1.0])
            n[k, j] /= np.linalg.norm(n[k, j])
    return n


def rest_quats(delta_deg, n):
    h = np.radians(delta_deg)[..., None] / 2.0
    return np.concatenate([np.cos(h), np.sin(h) * n], axis=-1)


def unpack(x):
    x = np.asarray(x, float)
    k = np.exp(x[:N_C * N_J]).reshape(N_C, N_J)
    delta = x[N_C * N_J:].reshape(N_C, N_J)
    return k, delta


def _eval_one(job):
    cand_i, clip, x = job
    shp, fps = _CACHE[clip]
    import mujoco
    from mujoco_rollout import rollout_errs
    m = mujoco.MjModel.from_xml_string(build_xml(shp))
    if clip not in _AXIS:
        _AXIS[clip] = board_normal_axes(m)
    n = _AXIS[clip]
    k, delta = unpack(x)
    set_per_joint_stiffness(m, k, rest_quat=rest_quats(delta, n))
    return cand_i, clip, float(rollout_errs(m, shp, fps, per_cable=True).mean(0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/jnt_stiffness_fit_r3.npz")
    ap.add_argument("--out", default="data/jnt_spring_rest_fit.npz")
    ap.add_argument("--log", default="data/jnt_spring_rest_fit_log.jsonl")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sigma-logk", type=float, default=0.3)
    ap.add_argument("--sigma-deg", type=float, default=4.0)
    ap.add_argument("--max-deg", type=float, default=25.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--tracks-dir", default=None)
    ap.add_argument("--init-rest", action="store_true")
    args = ap.parse_args()
    import cma, multiprocessing as mp

    log_f = open(ROOT / args.log, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    global _CACHE
    if args.tracks_dir:
        tdir = Path(args.tracks_dir)
        tdir = tdir if tdir.is_absolute() else ROOT / tdir
        clips = sorted(p.stem for p in tdir.glob("*.npz"))
        _CACHE = {}
        for c in clips:
            d = np.load(tdir / f"{c}.npz")
            _CACHE[c] = ({k: d[f"c{k}"] for k in range(4)}, float(d["fps"]))
        log(f"{len(clips)} Clips aus {tdir} geladen (Track-Cache)")
    else:
        clips = list(CLIP_TRIMS.keys())
        _CACHE = load_clips(clips, log)
    if args.probe:
        import mujoco
        shp, fps = _CACHE[clips[0]]
        n = board_normal_axes(mujoco.MjModel.from_xml_string(build_xml(shp)))
        for k in range(N_C):
            print(f"cable {k}: board normal in body frames, joints 0/7/13: "
                  f"{np.round(n[k, 0], 2)} {np.round(n[k, 7], 2)} {np.round(n[k, 13], 2)}")
        return

    init = np.load(ROOT / args.init, allow_pickle=True)
    k0 = np.maximum(np.asarray(init["best"], float), K_FLOOR)
    rest0 = (np.asarray(init["rest_deg"], float).ravel()
             if args.init_rest and "rest_deg" in init else np.zeros(N_C * N_J))
    x0 = np.concatenate([np.log(k0).ravel(), rest0])
    stds = [args.sigma_logk] * (N_C * N_J) + [args.sigma_deg] * (N_C * N_J)
    lo = [-np.inf] * (N_C * N_J) + [-args.max_deg] * (N_C * N_J)
    hi = [np.inf] * (N_C * N_J) + [args.max_deg] * (N_C * N_J)
    log("=== Gelenkfeder + Ruhewinkel (CMA-ES) gestartet ===")
    log(f"112 Parameter: 56 log-Steifigkeiten (Start aus {args.init}, Boden {K_FLOOR}) "
        f"+ 56 Ruhewinkel um die Brettnormale (Start 0, |delta| <= {args.max_deg} deg)")
    log(f"popsize {args.popsize}, gens {args.gens}, workers {args.workers}; "
        f"Fitness = Tabellenmetrik (rollout_errs, alle Frames, {len(clips)} Clips)")

    pool = mp.get_context("fork").Pool(args.workers)

    def evaluate(xs):
        jobs = [(i, c, x) for i, x in enumerate(xs) for c in clips]
        acc = [[] for _ in xs]
        for i, c, e in pool.map(_eval_one, jobs):
            acc[i].append(e)
        return [float(np.mean(a)) for a in acc]

    f0 = evaluate([x0])[0]
    log(f"gen 0 (Start, Ruhewinkel {'aus init' if args.init_rest else '0'}): {f0:.3f} mm/Knoten")
    hist = [{"gen": 0, "best": f0}]
    best_f, best_x = f0, x0.copy()
    es = cma.CMAEvolutionStrategy(x0, 1.0, {"popsize": args.popsize, "verbose": -9,
                                            "CMA_stds": stds, "bounds": [lo, hi]})
    t_start = time.time()
    for gen in range(1, args.gens + 1):
        t_g = time.time()
        xs = es.ask()
        fs = evaluate(xs)
        es.tell(xs, fs)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_x = fs[i], np.asarray(xs[i], float).copy()
        k, delta = unpack(best_x)
        hist.append({"gen": gen, "best": float(fs[i]), "mean": float(np.mean(fs)), "best_ever": best_f})
        log(f"gen {gen}/{args.gens}: best {fs[i]:.3f}  mean {np.mean(fs):.3f}  best_ever {best_f:.3f}  "
            f"({time.time() - t_g:.0f}s)  k {k.min():.3f}..{k.max():.2f}  delta {delta.min():+.1f}..{delta.max():+.1f} deg")
        np.savez(ROOT / args.out, best_x=best_x, best=k, rest_deg=delta, best_f=best_f,
                 base_f=f0, history=json.dumps(hist))
    pool.close(); pool.join()
    log(f"=== FERTIG: {best_f:.3f} mm/Knoten (Start {f0:.3f}), {time.time() - t_start:.0f}s ===")


if __name__ == "__main__":
    main()
