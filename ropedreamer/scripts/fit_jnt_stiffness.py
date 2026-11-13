import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import track_clip, build_xml, DIR as CLIP_DIR
from mujoco_replay import replay

ROOT = Path(__file__).resolve().parents[1]
N_JOINTS = 14

from clip_set import CLIP_TRIMS
TRAIN_CLIPS_FAST = ["20260602_170108", "20260602_121006", "20260602_162650",
                    "20260602_193953", "20260602_165113"]
CANDIDATES = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]


def clip_frame_t1(clip, cutoff_sec):
    if cutoff_sec is None:
        return None
    import cv2
    cap = cv2.VideoCapture(f"{CLIP_DIR}/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    cap.release()
    return int(round(cutoff_sec * fps))


def load_clips(clip_names, log):
    cache = {}
    for c in clip_names:
        t1 = clip_frame_t1(c, CLIP_TRIMS[c])
        t_start = time.time()
        shp, t0, t1_, fps = track_clip(c, t0=8, t1=t1)
        cache[c] = (shp, fps)
        log(f"  getrackt: {c} ({len(shp[0])} Frames, {time.time()-t_start:.0f}s)")
    return cache


def eval_clips(cache, clip_names, weld_solref):
    out = {}
    for c in clip_names:
        shp, fps = cache[c]
        n = len(shp[0])
        xml = build_xml(shp, weld_solref=weld_solref)
        import mujoco
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
        traj = replay(m, d, shp, n, sub)
        err = np.array([1000.0 * float(np.mean([
            np.linalg.norm(traj[f][k][:, :2] - shp[k][f], axis=1).mean()
            for f in range(len(traj))])) for k in range(4)])
        out[c] = err
    return out


_CACHE = {}


def _eval_one(job):
    cand_i, clip, stiff = job
    shp, fps = _CACHE[clip]
    import mujoco
    from mujoco_scene import set_per_joint_stiffness
    from mujoco_full_loco import rollout_errs
    m = mujoco.MjModel.from_xml_string(build_xml(shp))
    if stiff is not None:
        set_per_joint_stiffness(m, stiff)
    return cand_i, clip, rollout_errs(m, shp, fps, per_cable=True).mean(0)


def eval_clips_par(pool, clip_names, stiff):
    jobs = [(0, c, stiff) for c in clip_names]
    return {c: e for _, c, e in pool.map(_eval_one, jobs)}


def eval_candidates_par(pool, clip_names, trials):
    jobs = [(i, c, t) for i, t in enumerate(trials) for c in clip_names]
    out = [{} for _ in trials]
    for i, c, e in pool.map(_eval_one, jobs):
        out[i][c] = e
    return out


def mean_cable_err(results, cable):
    return float(np.mean([e[cable] for e in results.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jnt_stiffness_fit.npz")
    ap.add_argument("--log", default="data/jnt_stiffness_fit_log.jsonl")
    ap.add_argument("--train-all", action="store_true")
    ap.add_argument("--clips", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--init-value", type=float, default=0.1)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    global CANDIDATES
    if args.candidates:
        CANDIDATES = [float(x) for x in args.candidates.split(",")]

    all_clip_names = list(CLIP_TRIMS.keys())
    if args.clips is not None:
        train_clips = args.clips.split(",")
    else:
        train_clips = all_clip_names if args.train_all else TRAIN_CLIPS_FAST

    log_path = ROOT / args.log
    log_f = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    out_path = ROOT / args.out
    best = np.full((4, N_JOINTS), float(args.init_value))
    history = []
    base_avg = None
    base_per_clip = None
    if args.resume and out_path.exists():
        ckpt = np.load(out_path, allow_pickle=True)
        best = ckpt["best"]
        history = json.loads(str(ckpt["history"]))
        base_avg = float(ckpt["base_avg"])
        if "base_per_clip" in ckpt:
            base_per_clip = json.loads(str(ckpt["base_per_clip"]))
        log(f"=== Gelenkfeder-Fit FORTGESETZT ({len(history)}/{4*N_JOINTS} "
            f"Koordinaten schon fertig) ===")
    else:
        log("=== Gelenkfeder-Fit gestartet ===")
        if args.init is not None:
            init_ckpt = np.load(ROOT / args.init, allow_pickle=True)
            best = init_ckpt["best"]
            log(f"Startpunkt aus {args.init} geladen (history NICHT "
                f"uebernommen -- alle 56 Koordinaten werden mit diesem "
                f"Startwert neu bewertet, nicht uebersprungen)")
    log(f"Trainings-Clips ({len(train_clips)}): {train_clips}")
    log(f"Kandidaten pro Koordinate: {CANDIDATES}")
    log("Tracke alle Clips einmal (Cache fuer den ganzen Lauf)...")
    cache = load_clips(all_clip_names, log)
    global _CACHE
    _CACHE = cache
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(args.workers)
    log(f"Pool: {args.workers} parallele Simulationen "
        f"({4 * len(train_clips)} je Koordinate)")

    if base_avg is None or base_per_clip is None:
        log("Baseline (keine Gelenkfeder) auf allen Clips...")
        base = eval_clips_par(pool, all_clip_names, None)
        base_avg = float(np.mean([e.mean() for e in base.values()]))
        base_per_clip = {c: float(e.mean()) for c, e in base.items()}
        log(f"Baseline Mittel (Endfehler): {base_avg:.2f} mm/Knoten")
    else:
        log(f"Baseline aus Checkpoint: {base_avg:.2f} mm/Knoten")

    done_coords = {(h["cable"], h["joint"]) for h in history}
    n_done_before = len(history)
    n_done_session = 0
    t_run_start = time.time()
    total_coords = 4 * N_JOINTS
    for c in range(4):
        for j in range(N_JOINTS):
            if (c, j) in done_coords:
                continue
            t_c = time.time()
            trials = []
            for cand in CANDIDATES:
                trial = best.copy()
                trial[c, j] = cand
                trials.append(trial)
            results = eval_candidates_par(pool, train_clips, trials)
            scores = [mean_cable_err(r, c) for r in results]
            best_idx = int(np.argmin(scores))
            best[c, j] = CANDIDATES[best_idx]
            history.append({"cable": c, "joint": j, "scores": scores,
                            "chosen": CANDIDATES[best_idx]})
            np.savez(ROOT / args.out, best=best, history=json.dumps(history),
                    base_avg=base_avg, base_per_clip=json.dumps(base_per_clip))
            n_done_session += 1
            coord_i = n_done_before + n_done_session
            elapsed = time.time() - t_run_start
            eta = elapsed / n_done_session * (total_coords - coord_i)
            log(f"[{coord_i}/{total_coords}] M{c} Gelenk{j}: "
                f"scores={[round(s,1) for s in scores]} -> {CANDIDATES[best_idx]:g} "
                f"({time.time()-t_c:.0f}s, ETA {eta/60:.0f} min)")

    if args.resume and out_path.exists() and "final_avg" in np.load(out_path, allow_pickle=True):
        log("=== Endergebnis schon im Checkpoint vorhanden, ueberspringe Validierung ===")
        ckpt = np.load(out_path, allow_pickle=True)
        final_avg = float(ckpt["final_avg"])
        per_clip = json.loads(str(ckpt["per_clip"]))
    else:
        log("=== Pass 1 fertig, Endvalidierung ===")
        final = eval_clips_par(pool, all_clip_names, best)
        final_avg = float(np.mean([e.mean() for e in final.values()]))
        per_clip = {c: float(e.mean()) for c, e in final.items()}
        np.savez(ROOT / args.out, best=best, history=json.dumps(history),
                base_avg=base_avg, base_per_clip=json.dumps(base_per_clip),
                final_avg=final_avg, per_clip=json.dumps(per_clip))
    log(f"Baseline (kein Weld):     {base_avg:.2f} mm/Knoten")
    log(f"Gefittete Gelenkfeder:   {final_avg:.2f} mm/Knoten")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} {v:6.1f}  (Baseline {base_per_clip[c]:6.1f})")
    log(f"-> {ROOT / args.out}")
    pool.close()
    pool.join()
    log("=== FERTIG ===")


if __name__ == "__main__":
    main()
