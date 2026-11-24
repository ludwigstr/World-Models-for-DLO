import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent


sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))


from scene import CLIP_TRIMS


TRACKS = Path(os.environ.get("GENERAL_FIT_TRACKS") or "../data/tracks")


CUSTOM_TRACKS = bool(os.environ.get("GENERAL_FIT_TRACKS"))


N_C, N_J = 4, 15


N_PJ = N_C * N_J


MASS0, MASS_LO, MASS_HI = 0.2, 0.02, 0.2


I_K, I_REST, I_DAMP, I_MU = 0, N_PJ, 2 * N_PJ, 3 * N_PJ


I_MASS = I_MU + 1


N_PAR = I_MASS + N_C


_CACHE = {}


def unpack(x):
    x = np.asarray(x, float)
    pj = lambda i: x[i:i + N_PJ].reshape(N_C, N_J)
    return dict(stiffness=np.exp(pj(I_K)), rest_angle=pj(I_REST).astype(np.float32),
                damping=np.exp(pj(I_DAMP)), mu=float(np.exp(x[I_MU])), mass_per_m=np.exp(x[I_MASS:I_MASS + N_C]))


def _eval_one(job):
    cand_i, clip, x = job
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import scene as G
    if clip not in _CACHE:
        z = np.load(TRACKS / f"{clip}.npz")
        _CACHE[clip] = ({k: z[f"c{k}"] for k in range(4)}, float(z["fps"]))
    try:
        e = G.eval_clip(clip, t1_sec=None if CUSTOM_TRACKS else CLIP_TRIMS[clip], shp_fps=_CACHE[clip], **unpack(x))
        e = float(np.mean(list(e.values()) if isinstance(e, dict) else e))
        return cand_i, clip, e if np.isfinite(e) else 1e3
    except Exception as exc:
        print(f"  {clip}: {type(exc).__name__}: {exc}", flush=True)
        return cand_i, clip, 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="genesis_perjoint_sysid.json")
    ap.add_argument("--resume", default="")
    ap.add_argument("--mass-max", type=float, default=MASS_HI)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--mu-max", type=float, default=5.0)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--out", default="data/genesis_general_fit.json")
    ap.add_argument("--log", default="data/genesis_general_fit_log.jsonl")
    ap.add_argument("--stop-by", default="")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rest-cap", type=float, default=0.6)
    ap.add_argument("--smoke", default="")
    a = ap.parse_args()
    if a.tracks:
        global TRACKS, CUSTOM_TRACKS
        TRACKS = (ROOT / a.tracks).resolve(); CUSTOM_TRACKS = True
        os.environ["GENERAL_FIT_TRACKS"] = str(TRACKS)
    import cma
    import multiprocessing as mp

    d = json.loads((ROOT / a.init).read_text())
    t4 = lambda v: np.tile(np.asarray(v, float), N_C)
    x0 = np.concatenate([np.log(t4(d["stiffness"])), np.zeros(N_PJ), np.log(t4(d["damping"])), [0.0], np.full(N_C, np.log(MASS0))])
    if a.resume:
        x0 = np.asarray(json.loads((ROOT / a.resume).read_text())["best_x"], float)
    assert len(x0) == N_PAR
    if a.smoke:
        for c in a.smoke.split(","):
            for m in (d["mass_per_m"], MASS0):
                x = x0.copy(); x[I_MASS:] = np.log(m)
                print(c, "mass", m, _eval_one((0, c, x))[2], flush=True)
        return

    log_f = open(ROOT / a.log, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    clips = sorted(p.stem for p in TRACKS.glob("*.npz")) if CUSTOM_TRACKS else sorted(CLIP_TRIMS)
    stds = [0.3] * N_PJ + [0.08] * N_PJ + [0.3] * N_PJ + [0.3] + [0.2] * N_C
    lo = [np.log(1e-3)] * N_PJ + [-a.rest_cap] * N_PJ + [np.log(1e-4)] * N_PJ + [np.log(0.05)] + [np.log(MASS_LO)] * N_C
    hi = [np.log(100.0)] * N_PJ + [a.rest_cap] * N_PJ + [np.log(10.0)] * N_PJ + [np.log(a.mu_max)] + [np.log(a.mass_max)] * N_C
    log(f"=== Genesis GENERAL fit: {N_PAR} params, per cable, mass free per cable in {MASS_LO} .. {a.mass_max} kg/m (CMA-ES) ===")
    log(f"popsize {a.popsize}, gens {a.gens}, workers {a.workers}; " + (f"resumed from {a.resume}, sigma0 {a.sigma0}, mu <= {a.mu_max}; " if a.resume else f"warm start {a.init} (row 7) on every cable, rest 0, mu 1.0; ")
        + f"fitness = table metric over {len(clips)} clips; mover contact on")

    pool = mp.get_context("spawn").Pool(a.workers)

    def evaluate(xs):
        acc = [[] for _ in xs]
        for i, c, e in pool.map(_eval_one, [(i, c, x) for i, x in enumerate(xs) for c in clips], chunksize=1):
            acc[i].append(e)
        return [float(np.mean(v)) for v in acc]

    def save(best_x, best_f, f0, hist, **extra):
        p = unpack(best_x)
        (ROOT / a.out).write_text(json.dumps({
            "best_x": list(map(float, best_x)), "best_f": best_f, "base_f": f0, "mass_per_m": p["mass_per_m"].tolist(),
            "stiffness": p["stiffness"].tolist(), "damping": p["damping"].tolist(),
            "rest_angle": p["rest_angle"].astype(float).tolist(), "mu": p["mu"], "hist": hist, **extra}, indent=1))

    t0 = time.time()
    f0 = evaluate([x0])[0]
    gen0_secs, gen_secs = time.time() - t0, []
    stop_ts = time.mktime(time.strptime(a.stop_by, '%Y-%m-%d %H:%M')) if a.stop_by else 0.0
    log(f"gen 0 ({'resumed best' if a.resume else f'row 7 at mass {MASS0}, rest 0'}): {f0:.3f} mm/node ({time.time() - t0:.0f}s)")
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
            f"kp {p['stiffness'].min():.3f}..{p['stiffness'].max():.2f}  rest {np.degrees(p['rest_angle']).min():+.1f}..{np.degrees(p['rest_angle']).max():+.1f} deg  "
            f"damp {p['damping'].min():.4f}..{p['damping'].max():.3f}  mu {p['mu']:.2f}  mass {np.round(p['mass_per_m'], 4).tolist()}")
        save(best_x, best_f, f0, hist)

    log(f"fit done: {best_f:.3f} (gen 0 {f0:.3f}), {time.time() - t0:.0f}s")
    per_clip = {c: e for _, c, e in pool.map(_eval_one, [(0, c, best_x) for c in clips], chunksize=1)}
    pool.close(); pool.join()
    v = np.array(list(per_clip.values()))
    save(best_x, best_f, f0, hist, per_clip=per_clip, final_avg=float(v.mean()), final_median=float(np.median(v)))
    log(f"final result: {v.mean():.3f} mm/node (median {np.median(v):.3f}), {len(clips)} clips IN-SAMPLE")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
