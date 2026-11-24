import os
import sys
import time
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import cma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from newton_fit_stiffness import CLIP_TRIMS

ROOT = Path(__file__).resolve().parents[1]
SUBDIV = int(os.environ.get("WIRE_SUBDIV", "1"))
N_JOINTS = 14
GPU_IDS = [int(x) for x in os.environ.get("WIRE_GPUS", "0,1").split(",")]
N_WORKERS = int(os.environ.get("WIRE_WORKERS", "4"))
POPSIZE = int(os.environ.get("WIRE_POPSIZE", "12"))
FIT_SOLVER_ITERATIONS = 10
MAX_GENERATIONS = int(os.environ.get("WIRE_GENS", "12"))
SIGMA0 = float(os.environ.get("WIRE_SIGMA0", "1.0"))
X0_KE = 0.5
X0_KD = 0.05
X0_MASS = 0.15
Z_WEIGHT = 0.5
DAHL_MODE = os.environ.get("WIRE_DAHL_MODE", "scalar")
FIT_REST = os.environ.get("WIRE_REST", "1") == "1"
FIT_TWIST = False
FIT_EXTRA = os.environ.get("WIRE_EXTRA", "0") == "1"
N_EXTRA = 3 if FIT_EXTRA else 0
N_TWIST = 0
N_CABLES = 4
X0_EXTRA = [1.0e5, 1.0e3, 1.0]
REST_STD = float(os.environ.get("WIRE_REST_STD", "0.06"))
N_DAHL = N_JOINTS if DAHL_MODE == "perjoint" else 1
WARM_FILE = os.environ.get("WIRE_WARM", "data/newton_fit_depth.npz")
X0_DAHL_EPS = float(os.environ.get("WIRE_DAHL_EPS", "0.3"))
X0_DAHL_TAU = float(os.environ.get("WIRE_DAHL_TAU", "1.0"))

TRAIN_CLIPS = list(CLIP_TRIMS.keys())

FALLBACK_DIR = Path("/tmp/wire_model_s_fallback")


def safe_savez(path, log=None, **arrays):
    tmp = path.with_name("tmp_" + path.name)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        return path
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        fb = FALLBACK_DIR / path.name
        np.savez(fb, **arrays)
        if log:
            log(f"WARNING: could not write {path} ({e}) -- checkpoint saved to {fb}")
        return fb

_worker_cache = {}


N_KE = N_CABLES * N_JOINTS
N_KD = N_CABLES * N_JOINTS
N_MASS = N_CABLES
N_REST = N_CABLES * N_JOINTS


def _n_core():
    return N_KE + N_KD + N_MASS + 2 * N_DAHL + N_EXTRA


def _from_x(x):
    x = np.asarray(x, float)
    n_core = _n_core()
    row = list(np.exp(x[:n_core]))
    if FIT_REST:
        row += list(x[n_core:n_core + N_REST])
    if FIT_TWIST:
        t0 = n_core + (N_JOINTS if FIT_REST else 0)
        row += list(np.exp(x[t0:t0 + N_TWIST]))
    return np.asarray(row, float)


def _pad_extra(row):
    row = np.asarray(row, float)
    want = _n_core() + (N_REST if FIT_REST else 0)
    if N_EXTRA and len(row) == want - N_EXTRA:
        core = N_KE + N_KD + N_MASS + 2 * N_DAHL
        row = np.concatenate([row[:core], X0_EXTRA, row[core:]])
    return row


def _to_x(row):
    row = _pad_extra(row)
    n_core = _n_core()
    x = list(np.log(row[:n_core]))
    if FIT_REST:
        r = row[n_core:n_core + N_REST]
        x += list(r if len(r) == N_REST else np.zeros(N_REST))
    if FIT_TWIST:
        t0 = n_core + (N_JOINTS if FIT_REST else 0)
        t = row[t0:t0 + N_TWIST]
        x += list(np.log(t if len(t) == N_TWIST else row[:N_JOINTS]))
    return x


def _init_worker(gpu_id):
    _worker_cache["device"] = f"cuda:{gpu_id}"


def _clip_frame_t1(clip, cutoff_sec):
    if cutoff_sec is None:
        return None
    import cv2
    from harness_scene import DIR
    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    cap.release()
    return int(round(cutoff_sec * fps))


def _params_from_log(log_row):
    v = np.asarray(log_row, float)
    n_core = _n_core()
    rest = (v[n_core:n_core + N_REST].reshape(N_CABLES, N_JOINTS)
            if FIT_REST else None)
    row = np.exp(v[:n_core])
    ke = row[:N_KE].reshape(N_CABLES, N_JOINTS)
    kd = row[N_KE:N_KE + N_KD].reshape(N_CABLES, N_JOINTS)
    mass = row[N_KE + N_KD:N_KE + N_KD + N_MASS]
    e0 = N_KE + N_KD + N_MASS
    if DAHL_MODE == "perjoint":
        eps = np.tile(row[e0:e0 + N_JOINTS], (4, 1))
        tau = np.tile(row[e0 + N_JOINTS:e0 + 2 * N_JOINTS], (4, 1))
    else:
        eps = np.full((4, N_JOINTS), float(row[e0]))
        tau = np.full((4, N_JOINTS), float(row[e0 + 1]))
    extra = row[n_core - N_EXTRA:n_core] if N_EXTRA else None
    kt = None
    if FIT_TWIST:
        t0 = n_core + (N_JOINTS if FIT_REST else 0)
        kt = np.tile(np.exp(v[t0:t0 + N_TWIST]), (4, 1))
    return ke, kd, mass, eps, tau, rest, extra, kt


def _z_loss_mm(z_nodes, targets_mm):
    zs = np.stack(z_nodes) * 1000.0
    tg = targets_mm[:zs.shape[0], :, :zs.shape[2]]
    m = np.isfinite(tg)
    if not m.any():
        return 0.0
    return float(np.abs(zs[m] - tg[m]).mean())


def _eval_job(args):
    (log_row,) = args
    from harness_scene import track_clip
    from newton_full_scene import eval_clip
    if "clips" not in _worker_cache:
        _worker_cache["clips"] = {}
        _worker_cache["targets"] = {}
    clips = _worker_cache["clips"]
    targets = _worker_cache["targets"]
    for c in TRAIN_CLIPS:
        if c not in clips:
            t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
            shp, t0, t1_, fps = track_clip(c, t0=8, t1=t1)
            clips[c] = (shp, fps)
            targets[c] = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
    ke, kd, mass, eps, tau, rest, extra, kt = _params_from_log(log_row)
    xy_errs, z_errs = [], []
    for c in TRAIN_CLIPS:
        z_out = []
        errs = eval_clip(c, per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt,
                         mass_per_m=mass,
                         **({} if extra is None else dict(
                             stretch_stiffness=float(extra[0]),
                             twist_stiffness=float(extra[1]),
                             cable_mu=float(extra[2]))),
                         shp_fps=clips[c], solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, subdiv=SUBDIV,
                         dahl_eps_max=eps, dahl_tau=tau,
                         per_joint_rest=rest,
                         device=_worker_cache.get("device"))
        xy_errs.append(float(np.mean(errs)))
        z_errs.append(_z_loss_mm(z_out, targets[c]))
    return float(np.mean(xy_errs)), float(np.mean(z_errs))


def main():
    log_path = ROOT / os.environ.get("WIRE_FITLOG", "data/newton_fit_dahl_log.jsonl")
    log_f = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            log_f.write(line + "\n")
            log_f.flush()
        except OSError:
            pass

    missing = [c for c in TRAIN_CLIPS if not (ROOT / f"data/depth_targets/{c}.npz").exists()]
    if missing:
        log(f"ERROR: depth targets missing for {missing} -- run depth_height_targets.py first")
        sys.exit(1)

    log("=== Newton PER-CABLE full-scene CMA-ES fit (Depth-Anything z objective) started ===")
    log(f"popsize={POPSIZE}, max_generations={MAX_GENERATIONS}, workers={N_WORKERS} on GPUs {GPU_IDS}")
    n_par = _n_core() + (N_REST if FIT_REST else 0)
    what = (f"{N_KE} ke + {N_KD} kd + {N_MASS} Masse je Kabel + Dahl {DAHL_MODE}"
            + (" + stretch/twist/mu" if N_EXTRA else "")
            + (f" + {N_REST} Ruhewinkel je Kabel" if FIT_REST else "")
            + (f" + {N_TWIST} Torsionssteifigkeiten je Gelenk" if N_TWIST else ""))
    log(f"{n_par} shared params ({what}); x0: ke={X0_KE}, kd={X0_KD}, mass={X0_MASS} kg/m; "
        f"fitness = xy mean + {Z_WEIGHT} * |z_sim - z_depth| (mm)")

    ctx = mp.get_context("spawn")
    pools = [ctx.Pool(N_WORKERS // len(GPU_IDS), initializer=_init_worker, initargs=(g,))
            for g in GPU_IDS]

    def eval_population(log_rows):
        async_results = []
        for i, row in enumerate(log_rows):
            pool = pools[i % len(pools)]
            async_results.append(pool.apply_async(_eval_job, ((row,),)))
        return [r.get() for r in async_results]

    out_path = ROOT / os.environ.get("WIRE_OUT", "data/newton_fit_percable.npz")
    warm = ROOT / WARM_FILE
    if warm.exists():
        prev = np.asarray(np.load(warm, allow_pickle=True)["best"], float)
        if len(prev) == N_KE + N_KD + N_MASS + 2 * N_DAHL + (N_REST if FIT_REST else 0):
            row0 = prev
        else:
            sk, sd = prev[:N_JOINTS], prev[N_JOINTS:2 * N_JOINTS]
            smass = float(prev[2 * N_JOINTS])
            e0 = 2 * N_JOINTS + 1
            if len(prev) >= e0 + 2 * N_JOINTS:
                pe = float(np.exp(np.mean(np.log(prev[e0:e0 + N_JOINTS]))))
                pt = float(np.exp(np.mean(np.log(prev[e0 + N_JOINTS:e0 + 2 * N_JOINTS]))))
                tail = prev[e0 + 2 * N_JOINTS:]
            elif len(prev) >= e0 + 2:
                pe, pt = float(prev[e0]), float(prev[e0 + 1])
                tail = prev[e0 + 2:]
            else:
                pe, pt, tail = X0_DAHL_EPS, X0_DAHL_TAU, np.zeros(0)
            srest = tail[:N_JOINTS] if len(tail) >= N_JOINTS else np.zeros(N_JOINTS)
            row0 = np.concatenate([
                np.tile(sk, N_CABLES), np.tile(sd, N_CABLES),
                np.full(N_MASS, smass), [pe] * N_DAHL, [pt] * N_DAHL,
                np.tile(srest, N_CABLES) if FIT_REST else np.zeros(0)])
        x0 = _to_x(row0)
        print(f"warmstart aus {warm.name} ({len(prev)} param, geteilt) -> "
              f"{len(x0)} param je Kabel, Dahl-modus {DAHL_MODE}", flush=True)
    else:
        x0 = ([np.log(X0_KE)] * N_KE + [np.log(X0_KD)] * N_KD
              + [np.log(X0_MASS)] * N_MASS + [np.log(X0_DAHL_EPS)] * N_DAHL
              + [np.log(X0_DAHL_TAU)] * N_DAHL
              + ([0.0] * N_REST if FIT_REST else []))
    opts = {"popsize": POPSIZE, "verbose": -9}
    if FIT_REST:
        opts["CMA_stds"] = [SIGMA0] * _n_core() + [REST_STD] * N_REST
        assert len(opts["CMA_stds"]) == len(x0), (len(opts["CMA_stds"]), len(x0))
        es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
    else:
        es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)
    gen = 0
    gen_history = []
    t_start = time.time()
    while not es.stop() and gen < MAX_GENERATIONS:
        t_g = time.time()
        solutions = es.ask()
        results = eval_population(solutions)
        fitnesses = [xy + Z_WEIGHT * ze for xy, ze in results]
        es.tell(solutions, fitnesses)
        gen += 1
        i_best = int(np.argmin(fitnesses))
        gen_history.append({"gen": gen, "best": float(fitnesses[i_best]),
                            "best_xy": results[i_best][0], "best_z": results[i_best][1],
                            "mean_fit": float(np.mean(fitnesses))})
        log(f"gen {gen}/{MAX_GENERATIONS}: best={fitnesses[i_best]:.2f} "
            f"(xy={results[i_best][0]:.2f}, z-err={results[i_best][1]:.2f}mm) "
            f"mean={np.mean(fitnesses):.2f} ({time.time()-t_g:.0f}s)")
        best_row = _from_x(es.result.xbest)
        safe_savez(out_path, log=log, best=best_row, history=json.dumps(gen_history))

    best_row = _from_x(es.result.xbest)
    log(f"fit done: {gen} generations, {time.time()-t_start:.0f}s, "
        f"ke={np.round(best_row[:N_JOINTS],3).tolist()}, "
        f"kd={np.round(best_row[N_JOINTS:2*N_JOINTS],4).tolist()}, "
        f"mass={best_row[2*N_JOINTS]:.4f} kg/m")
    for p in pools:
        p.close(); p.join()

    log("=== final validation: fitted shared params, full scene, 10 iterations (IN-SAMPLE) ===")
    from harness_scene import track_clip
    from newton_full_scene import eval_clip
    ke, kd, mass, eps, tau, rest, extra, kt = _params_from_log(_to_x(best_row))
    per_clip, per_clip_z = {}, {}
    for c in TRAIN_CLIPS:
        t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
        shp, t0_, t1_, fps = track_clip(c, t0=8, t1=t1)
        tg = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
        z_out = []
        errs = eval_clip(c, per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt,
                         mass_per_m=mass,
                         **({} if extra is None else dict(
                             stretch_stiffness=float(extra[0]),
                             twist_stiffness=float(extra[1]),
                             cable_mu=float(extra[2]))),
                         shp_fps=(shp, fps), z_nodes_out=z_out, subdiv=SUBDIV,
                         solver_iterations=FIT_SOLVER_ITERATIONS,
                         dahl_eps_max=eps, dahl_tau=tau,
                         per_joint_rest=rest)
        per_clip[c] = float(np.mean(errs))
        per_clip_z[c] = _z_loss_mm(z_out, tg)
    final_avg = float(np.mean(list(per_clip.values())))
    final_z = float(np.mean(list(per_clip_z.values())))
    safe_savez(out_path, log=log, best=best_row, history=json.dumps(gen_history),
              final_avg=final_avg, final_z_err=final_z,
              per_clip=json.dumps(per_clip), per_clip_z=json.dumps(per_clip_z))
    log(f"final result: xy {final_avg:.2f} mm/node, z-err vs depth {final_z:.2f} mm (IN-SAMPLE)")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} xy {v:6.1f}  z-err {per_clip_z[c]:6.1f}")
    log(f"-> {out_path}")
    log("=== DONE ===")
    print("NEWTON_FIT_DEPTH_DONE")


if __name__ == "__main__":
    main()
