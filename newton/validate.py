import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
import numpy as np


os.environ.setdefault("WIRE_MASS_PIN", "0.044")


ROOT = Path(__file__).resolve().parent


sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


_dev = {}


def _init(q):
    _dev["device"] = f"cuda:{q.get()}"


def one(job):
    c, params, tracks = job
    t = time.time()
    import fit as E
    import scene as N
    kw = E.unpack(np.load(params, allow_pickle=True)["best"])
    if tracks:
        z = np.load(ROOT / tracks / f"{c}.npz")
        shp, fps = {k: np.asarray(z[f"c{k}"], float) for k in range(4)}, float(z["fps"])
    else:
        from scene import load_clip
        shp, _, _, fps = load_clip(c, t0=8, t1=E._clip_frame_t1(c, E.CLIP_TRIMS[c]))
    errs = N.eval_clip(c, shp_fps=(shp, fps), solver_iterations=10, device=_dev.get("device"), **kw)
    return c, float(np.mean(errs)), len(shp[0]), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--live14", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert bool(a.tracks) != a.live14, "give exactly one of --tracks / --live14"
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    if a.tracks:
        clips = sorted(q.stem for q in (ROOT / a.tracks).glob("*.npz"))
        clips.sort(key=lambda c: -len(np.load(ROOT / a.tracks / f"{c}.npz")["c0"]))
    else:
        from scene import CLIP_TRIMS
        clips = sorted(CLIP_TRIMS)
    gpus = [int(g) for g in a.gpus.split(",")]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(a.workers):
        q.put(gpus[i % len(gpus)])
    t0 = time.time()
    res = {}
    with ctx.Pool(a.workers, initializer=_init, initargs=(q,)) as pool:
        for n, (c, e, T, dt) in enumerate(pool.imap_unordered(
                one, [(c, str(ROOT / a.params), a.tracks) for c in clips]), 1):
            res[c] = {"mittel": e, "n_frames": T, "seconds": dt}
            print(f"[{n}/{len(clips)}] {c}: {e:.3f} mm/node, {T} frames, {dt:.0f}s (total {time.time() - t0:.0f}s)",
                  flush=True)
    res = dict(sorted(res.items()))
    v = np.array([r["mittel"] for r in res.values()])
    out.write_text(json.dumps({"params": a.params, "tracks": a.tracks or "live re-track (14 clips, t0=8, trims)",
                               "protocol": "newton_full_scene.eval_clip, 10 VBD iterations, mass 0.044, "
                                           "15 body origins vs first 15 tracked nodes",
                               "mean": float(v.mean()), "median": float(np.median(v)), "n_clips": len(v),
                               "seconds": time.time() - t0, "clips": res}, indent=1))
    print(f"mean {v.mean():.3f}  median {np.median(v):.3f} mm/node over {len(v)} clips, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
