import argparse, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

LABEL = "MuJoCo general fit (181 parameters), 36-clip refit"
SHORT = "MuJoCo"


def one(clip, out, params):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    x = np.asarray(np.load(params)["best_x"], float)
    import mujoco_replay as R
    import fit as M
    assert M.CUSTOM_TRACKS
    z = np.load(M.TRACKS / f"{clip}.npz")
    shp = {k: z[f"c{k}"] for k in range(4)}
    M._CACHE[clip] = (shp, float(z["fps"]))
    orig, rec = R.cable_vertices, []

    def wrapped(m, d, k, *a, **kw):
        v = orig(m, d, k, *a, **kw)
        rec.append((k, v[:, :2].copy()))
        return v
    R.cable_vertices = wrapped
    e = M._eval_one((0, clip, x))[2]
    T = len(shp[0])
    assert len(rec) == 4 * T and [k for k, _ in rec] == [0, 1, 2, 3] * T, f"eval failed or incomplete: e={e} calls {len(rec)}/{4 * T}"
    sim = np.stack([v for _, v in rec]).reshape(T, 4, 16, 2).astype(np.float64)
    err = np.array([np.mean([1000 * np.linalg.norm(sim[t, k] - shp[k][t], axis=1).mean() for k in range(4)]) for t in range(T)])
    assert abs(err.mean() - e) < 1e-6, (err.mean(), e)
    np.savez(Path(out) / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=LABEL, short=SHORT,
             score_nodes=np.arange(16), scored=e)
    print("RES", e, flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(*sys.argv[2:5]); return
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="data/mujoco_general_fit.npz")
    ap.add_argument("--reference", default="data/mujoco_general_fit_validate.json")
    ap.add_argument("--tracks", default="../data/tracks"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out, params, ref_path, tracks = ROOT / a.out, ROOT / a.params, ROOT / a.reference, ROOT / a.tracks
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    log_dir = out / "rollouts_logs"; log_dir.mkdir()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "GENERAL_FIT_TRACKS": str(tracks.resolve())}
    ref = json.loads(ref_path.read_text())
    clips = sorted(p.stem for p in tracks.glob("*.npz"))

    def run(c):
        t0 = time.time(); lg = log_dir / f"{c}.log"
        with open(lg, "w") as fh:
            p = subprocess.run([sys.executable, __file__, "--one", c, str(out), str(params)], stdout=fh,
                               stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        res = [ln for ln in lg.read_text().splitlines() if ln.startswith("RES")]
        ok = p.returncode == 0 and bool(res)
        print(f"{c}: {float(res[-1].split()[1]):.4f}  ({time.time() - t0:.0f}s)" if ok else f"{c}: CRASH rc={p.returncode}, see {lg}", flush=True)
        return c, ok

    with ThreadPoolExecutor(a.workers) as ex:
        done = dict(ex.map(run, clips))
    rows = {}
    for c in clips:
        if not done[c]:
            continue
        z = np.load(out / "rollouts" / f"{c}.npz", allow_pickle=True)
        e = float(z["err"].mean())
        assert isinstance(ref[c], float), (c, ref[c])
        rows[c] = {"reproduced": e, "reported": ref[c], "diff": e - ref[c], "eval_one_return": float(z["scored"]),
                   "nan_nodes": int(np.isnan(z["sim"]).sum()), "n_frames": int(len(z["err"]))}
    from render import write_results
    write_results(out, rows, {"approach": LABEL, "params": str(params.relative_to(ROOT)),
                              "params_sha256": hashlib.sha256(params.read_bytes()).hexdigest(),
                              "device": "cpu, one fresh process per clip",
                              "reported_source": str(ref_path.relative_to(ROOT)) + " (fresh-process validation)",
                              "clips_failed": [c for c in clips if not done[c]],
                              "metric": "16 nodes, mean over cables, every frame (mujoco_general_fit_cmaes._eval_one)"})


if __name__ == "__main__":
    main()
