import argparse
import json
import os
import re
import time
from pathlib import Path
import numpy as np
from simulate import CLIP_TRIMS
from simulate import build_xml, cable_ball_joints, set_per_joint_stiffness
from simulate import board_normal_axes, rest_quats


ROOT = Path(__file__).resolve().parent


TRACKS = Path(os.environ.get("GENERAL_FIT_TRACKS") or "../data/tracks")


CUSTOM_TRACKS = bool(os.environ.get("GENERAL_FIT_TRACKS"))


N_C, N_J = 4, 14


N_PJ = N_C * N_J


MASS0, MASS_LO, MASS_HI = 0.15, 0.02, 0.2


RADIUS = 0.003


I_K, I_REST, I_DAMP = 0, N_PJ, 2 * N_PJ


I_BEND, I_TWIST, I_MU = 3 * N_PJ, 3 * N_PJ + N_C, 3 * N_PJ + 2 * N_C


I_MASS = I_MU + 1


N_PAR = I_MASS + N_C


_CACHE, _AXIS = {}, {}


def unpack(x):
    x = np.asarray(x, float)
    pj = lambda i: x[i:i + N_PJ].reshape(N_C, N_J)
    return dict(k=np.exp(pj(I_K)), rest_deg=pj(I_REST), damp=np.exp(pj(I_DAMP)),
                bend=np.exp(x[I_BEND:I_BEND + N_C]), twist=np.exp(x[I_TWIST:I_TWIST + N_C]),
                mu=float(np.exp(x[I_MU])), mass=np.exp(x[I_MASS:I_MASS + N_C]))


def make_model(shp, p):
    import mujoco
    xml = build_xml(shp, mass_per_m=MASS0, mu=p["mu"], radius=RADIUS)
    it = iter(p["mass"] / (np.pi * RADIUS ** 2))
    xml = re.sub(r'density="[^"]*"', lambda _m: f'density="{next(it):.0f}"', xml)
    for key in ("twist", "bend"):
        it = iter(p[key])
        xml = re.sub(rf'<config key="{key}" value="[^"]*"/>',
                     lambda _m: f'<config key="{key}" value="{next(it):g}"/>', xml)
    m = mujoco.MjModel.from_xml_string(xml)
    clip_axes = board_normal_axes(m)
    set_per_joint_stiffness(m, p["k"], rest_quat=rest_quats(p["rest_deg"], clip_axes))
    ids = cable_ball_joints(m)
    for c in range(N_C):
        for j in range(N_J):
            a = m.jnt_dofadr[ids[c, j]]
            m.dof_damping[a:a + 3] = p["damp"][c, j]
    return m


def _eval_one(job):
    cand_i, clip, x = job
    from simulate import rollout_errs
    shp, fps = _CACHE[clip]
    try:
        m = make_model(shp, unpack(x))
        e = float(rollout_errs(m, shp, fps, per_cable=True).mean(0).mean())
        return cand_i, clip, e if np.isfinite(e) else 1e3
    except Exception as exc:
        print(f"  {clip}: {type(exc).__name__}: {exc}", flush=True)
        return cand_i, clip, 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/jnt_spring_rest_fit.npz")
    ap.add_argument("--resume", default="")
    ap.add_argument("--mass-max", type=float, default=MASS_HI)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--out", default="data/mujoco_general_fit.npz")
    ap.add_argument("--log", default="data/mujoco_general_fit_log.jsonl")
    ap.add_argument("--stop-by", default="")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-deg", type=float, default=40.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.tracks:
        global TRACKS, CUSTOM_TRACKS
        TRACKS = (ROOT / a.tracks).resolve(); CUSTOM_TRACKS = True
        os.environ["GENERAL_FIT_TRACKS"] = str(TRACKS)
    import cma
    import multiprocessing as mp

    log_f = open(ROOT / a.log, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    clips = sorted(p.stem for p in TRACKS.glob("*.npz")) if CUSTOM_TRACKS else sorted(CLIP_TRIMS)
    for c in clips:
        z = np.load(TRACKS / f"{c}.npz")
        _CACHE[c] = ({k: z[f"c{k}"] for k in range(4)}, float(z["fps"]))

    init = np.load(ROOT / a.init, allow_pickle=True)
    x0 = np.concatenate([np.log(np.asarray(init["best"], float)).ravel(),
                         np.asarray(init["rest_deg"], float).ravel(),
                         np.full(N_PJ, np.log(0.15)),
                         np.full(N_C, np.log(1e5)), np.full(N_C, np.log(1e5)), [0.0],
                         np.full(N_C, np.log(MASS0))])
    if a.resume:
        x0 = np.asarray(np.load(ROOT / a.resume, allow_pickle=True)["best_x"], float)
    assert len(x0) == N_PAR
    if a.smoke:
        for c in clips[:2]:
            print(_eval_one((0, c, x0)))
        return

    stds = [0.2] * N_PJ + [3.0] * N_PJ + [0.3] * N_PJ + [0.5] * (2 * N_C) + [0.3] + [0.2] * N_C
    lo = ([-np.inf] * N_PJ + [-a.max_deg] * N_PJ + [np.log(1e-4)] * N_PJ + [-np.inf] * (2 * N_C) + [np.log(0.05)]
          + [np.log(MASS_LO)] * N_C)
    hi = ([np.inf] * N_PJ + [a.max_deg] * N_PJ + [np.log(5.0)] * N_PJ + [np.inf] * (2 * N_C) + [np.log(5.0)]
          + [np.log(a.mass_max)] * N_C)
    log(f"=== MuJoCo GENERAL fit: {N_PAR} params, per cable, mass free per cable in {MASS_LO} .. {a.mass_max} kg/m (CMA-ES) ===")
    log(f"popsize {a.popsize}, gens {a.gens}, workers {a.workers}; " + (f"resumed from {a.resume}, sigma0 {a.sigma0}; " if a.resume else f"warm start {a.init} (row 4) + bend/twist 1e5, damping 0.15, mu 1.0; ")
        + f"fitness = table metric over {len(clips)} clips")

    pool = mp.get_context("fork").Pool(a.workers)

    def evaluate(xs):
        acc = [[] for _ in xs]
        for i, c, e in pool.map(_eval_one, [(i, c, x) for i, x in enumerate(xs) for c in clips]):
            acc[i].append(e)
        return [float(np.mean(v)) for v in acc]

    def save(best_x, best_f, f0, hist):
        p = unpack(best_x)
        np.savez(ROOT / a.out, best_x=best_x, best_f=best_f, base_f=f0, history=json.dumps(hist), **p)

    t0 = time.time()
    f0 = evaluate([x0])[0]
    gen0_secs, gen_secs = time.time() - t0, []
    stop_ts = time.mktime(time.strptime(a.stop_by, '%Y-%m-%d %H:%M')) if a.stop_by else 0.0
    log(f"gen 0 ({'resumed best' if a.resume else f'row 4 at mass {MASS0}'}): {f0:.3f} mm/node ({time.time() - t0:.0f}s)")
    hist = [{"gen": 0, "best": f0}]
    best_f, best_x = f0, x0.copy()
    es = cma.CMAEvolutionStrategy(x0, a.sigma0, {"popsize": a.popsize, "verbose": -9, "CMA_stds": stds, "bounds": [lo, hi]})
    for gen in range(1, a.gens + 1):
        if stop_ts and time.time() + 1.15 * (max(gen_secs) if gen_secs else a.popsize * gen0_secs) > stop_ts:
            log(f"wall-clock cap {a.stop_by} reached before gen {gen}: stopping, best_ever {best_f:.3f} kept")
            break
        t_g = time.time()
        xs = es.ask()
        fs = evaluate(xs)
        es.tell(xs, fs)
        gen_secs.append(time.time() - t_g)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_x = fs[i], np.asarray(xs[i], float).copy()
        p = unpack(best_x)
        hist.append({"gen": gen, "best": float(fs[i]), "mean": float(np.mean(fs)), "best_ever": best_f})
        log(f"gen {gen}/{a.gens}: best {fs[i]:.3f}  mean {np.mean(fs):.3f}  best_ever {best_f:.3f}  ({time.time() - t_g:.0f}s)  "
            f"k {p['k'].min():.3f}..{p['k'].max():.2f}  rest {p['rest_deg'].min():+.1f}..{p['rest_deg'].max():+.1f}  "
            f"damp {p['damp'].min():.3f}..{p['damp'].max():.2f}  mu {p['mu']:.2f}  mass {np.round(p['mass'], 4).tolist()}")
        save(best_x, best_f, f0, hist)

    log(f"fit done: {best_f:.3f} (gen 0 {f0:.3f}), {time.time() - t0:.0f}s")
    per_clip = {c: e for _, c, e in pool.map(_eval_one, [(0, c, best_x) for c in clips])}
    pool.close(); pool.join()
    v = np.array(list(per_clip.values()))
    np.savez(ROOT / a.out, best_x=best_x, best_f=best_f, base_f=f0, history=json.dumps(hist),
             per_clip=json.dumps(per_clip), final_avg=v.mean(), **unpack(best_x))
    log(f"final result: {v.mean():.3f} mm/node, {len(clips)} clips IN-SAMPLE")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
