import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from clip_set import CLIP_TRIMS
from wire_model.yang.sim import THETA_DEFAULT, make_model, theta_dict

TRACKS = Path("../data/tracks")


def _one(job):
    clip, x, TRACKS = job
    from mujoco_full_loco import rollout_errs
    t0 = time.time()
    z = np.load(TRACKS / f"{clip}.npz")
    shp = {k: z[f"c{k}"] for k in range(4)}
    e = np.asarray(rollout_errs(make_model(shp, x), shp, float(z["fps"])), float)
    return clip, e, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", default="data/wm/yang_ident.json")
    ap.add_argument("--default", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--engine", default="mujoco", choices=["mujoco", "newton"])
    a = ap.parse_args()
    tracks = ROOT / a.tracks if a.tracks else TRACKS
    clip_list = sorted(q.stem for q in tracks.glob("*.npz")) if a.tracks else sorted(CLIP_TRIMS)
    print(f"tracks {tracks}: {len(clip_list)} clips", flush=True)
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    import multiprocessing as mp
    if a.engine == "newton":
        from wire_model.yang import sim_newton as SN
        x = SN.THETA_DEFAULT if a.default else np.array(json.loads((ROOT / a.ident).read_text())["theta_log10"])
        clips = clip_list
        F = []
        for c in clips:
            z = np.load(tracks / f"{c}.npz")
            F.append((np.stack([z[f"c{k}"] for k in range(4)], 1).astype(float), float(z["fps"])))
        assert len({f for _, f in F}) == 1
        t0 = time.time()
        Vs = SN.rollouts([fr[0] for fr, _ in F], [SN.schedule_tracked(fr) for fr, _ in F], F[0][1],
                         np.repeat(x[None], len(F), 0), device="cuda:0")
        res = {}
        for c, (fr, _), V in zip(clips, F, Vs):
            e = 1000 * np.linalg.norm(V[..., :2] - fr[:len(V)], axis=-1).mean(axis=(1, 2))
            res[c] = {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()), "n_frames": len(e),
                      "complete": len(V) == len(fr)}
            print(f"{c}: {e.mean():.2f} mm/node ({len(V)}/{len(fr)} frames)", flush=True)
        v = np.array([r["mittel"] for r in res.values()])
        out.write_text(json.dumps({"engine": "newton", "tracks": str(tracks), "theta": SN.theta_dict(x), "theta_log10": list(map(float, x)),
                                   "clips": res, "summary": {"mean": float(v.mean()), "median": float(np.median(v))},
                                   "seconds": time.time() - t0,
                                   "protocol": "sim_newton.rollouts: 300-substep settle, every frame incl. frame 0, "
                                               "16 nodes (15 body origins + last tip)"}, indent=1))
        print(f"Newton simulator alone ({'defaults' if a.default else 'identified'}): mean {v.mean():.2f}  "
              f"median {np.median(v):.2f} mm/node, {len(v)} clips, {time.time() - t0:.0f}s -> {a.out}")
        return
    x = THETA_DEFAULT if a.default else np.array(json.loads((ROOT / a.ident).read_text())["theta_log10"])
    res = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for clip, e, dt in pool.imap_unordered(_one, [(c, x, tracks) for c in clip_list]):
            res[clip] = {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()), "n_frames": len(e)}
            print(f"{clip}: {e.mean():.2f} mm/node ({dt:.0f}s)", flush=True)
    res = dict(sorted(res.items()))
    v = np.array([r["mittel"] for r in res.values()])
    out.write_text(json.dumps({"theta": theta_dict(x), "theta_log10": list(map(float, x)), "clips": res,
                               "summary": {"mean": float(v.mean()), "median": float(np.median(v))},
                               "protocol": "mujoco_full_loco.rollout_errs: every frame incl. frame 0 after settle"}, indent=1))
    print(f"simulator alone ({'defaults' if a.default else 'identified'}): mean {v.mean():.2f}  median {np.median(v):.2f} mm/node, {len(v)} clips -> {a.out}")


if __name__ == "__main__":
    main()
