import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

LABEL = "Genesis general fit, 36-clip refit (stopped after generation 33)"
SHORT = "Genesis"
PY = str(Path.home() / "venvs/genesis133/bin/python")
CG = Path("/sys/fs/cgroup/user.slice")
GB = 1024 ** 3


def one(clip, out, params):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    x = np.asarray(json.loads(Path(params).read_text())["best_x"], float)
    import genesis_full_scene as G
    import fit as M
    assert M.CUSTOM_TRACKS
    orig, rec, ret = G.eval_clip, [], []

    def wrapped(*a, **kw):
        assert "nodes_out" not in kw
        r = orig(*a, nodes_out=rec, **kw)
        ret.append(r)
        return r
    G.eval_clip = wrapped
    e = M._eval_one((0, clip, x))[2]
    shp = M._CACHE[clip][0]
    T = len(shp[0])
    assert len(ret) == 1 and len(rec) == T, f"eval failed or incomplete: e={e} frames {len(rec)}/{T}"
    sim = np.stack(rec).astype(np.float64)
    err = np.asarray(ret[0], float)
    np.savez(Path(out) / "rollouts" / f"{clip}.npz", sim=sim, err=err, label=LABEL, short=SHORT,
             score_nodes=np.arange(16), scored=e)
    print("RES", e, flush=True)


def margin_gb():
    high = (CG / "memory.high").read_text().strip()
    high = int(high) if high != "max" else int((CG / "memory.max").read_text().strip())
    return (high - int((CG / "memory.current").read_text())) / GB


def rollouts(a, out, params, ref_path, tracks):
    (out / "rollouts").mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "GENERAL_FIT_TRACKS": str(tracks)}
    queue = sorted(p.stem for p in tracks.glob("*.npz"))
    running, tries, lowest, failed = {}, {}, margin_gb(), []
    log_dir = out / "rollouts_logs"; log_dir.mkdir()
    while queue or running:
        for c, (p, t0, lg) in list(running.items()):
            if p.poll() is None:
                continue
            del running[c]
            res = [ln for ln in lg.read_text().splitlines() if ln.startswith("RES")]
            if p.returncode == 0 and res:
                print(f"{c}: {float(res[-1].split()[1]):.4f}  ({time.time() - t0:.0f}s)  margin {margin_gb():.0f} GB", flush=True)
            else:
                failed.append(c)
                print(f"{c}: CRASH rc={p.returncode}, see {lg}", flush=True)
        m = margin_gb(); lowest = min(lowest, m)
        if m < a.min_margin_gb and running:
            c = max(running, key=lambda k: running[k][1])
            running.pop(c)[0].terminate()
            tries[c] = tries.get(c, 0) + 1
            if tries[c] > 3:
                failed.append(c); print(f"{c}: given up after {tries[c]} memory stops", flush=True)
            else:
                queue.insert(0, c); print(f"{c}: stopped at margin {m:.0f} GB, re-queued", flush=True)
            time.sleep(30)
            continue
        cap = a.workers if ref_path.exists() else min(a.workers, 4)
        if queue and len(running) < cap and m >= a.start_margin_gb:
            c = queue.pop(0)
            lg = log_dir / f"{c}.log"
            fh = open(lg, "w")
            p = subprocess.Popen([PY, __file__, "--one", c, str(out), str(params)], stdout=fh, stderr=subprocess.STDOUT,
                                 env=env, cwd=ROOT)
            fh.close()
            running[c] = (p, time.time(), lg)
            time.sleep(3)
            continue
        time.sleep(5)
    print(f"=== ROLLOUTS_DONE === failed {failed}; lowest memory margin {lowest:.0f} GB", flush=True)


def results(out, params, ref_path):
    from render import write_results
    ref = json.loads(ref_path.read_text())
    rows = {}
    for npz in sorted((out / "rollouts").glob("*.npz")):
        clip = npz.stem
        z = np.load(npz, allow_pickle=True)
        err, sim = z["err"], z["sim"]
        e, scored = float(err.mean()), float(z["scored"])
        assert isinstance(ref[clip], float), (clip, ref[clip])
        rows[clip] = {"reproduced": e, "reported": ref[clip], "diff": e - ref[clip], "eval_one_return": scored,
                      "nan_nodes": int(np.isnan(sim).sum()), "n_frames": int(len(err))}
    missing = sorted(set(ref) - set(rows))
    write_results(out, rows, {"approach": LABEL, "params": str(params.relative_to(ROOT)),
                              "params_sha256": hashlib.sha256(params.read_bytes()).hexdigest(),
                              "python": PY + " (genesis-world 1.3.3)", "device": "cpu, one fresh process per clip",
                              "reported_source": str(ref_path.relative_to(ROOT)) + " (fresh-process validation, plain floats)",
                              "clips_missing": missing,
                              "metric": "16 nodes, mean over cables, every frame (genesis_general_fit_cmaes._eval_one)"})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(*sys.argv[2:5]); return
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["rollouts", "results"]); ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="data/genesis_general_fit.json")
    ap.add_argument("--reference", default="data/genesis_general_fit_validate.json")
    ap.add_argument("--tracks", default="../data/tracks"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-margin-gb", type=float, default=40.0); ap.add_argument("--min-margin-gb", type=float, default=20.0)
    a = ap.parse_args()
    out, params, ref_path, tracks = ROOT / a.out, ROOT / a.params, ROOT / a.reference, ROOT / a.tracks
    if a.step == "rollouts":
        rollouts(a, out, params, ref_path, tracks)
    else:
        results(out, params, ref_path)


if __name__ == "__main__":
    main()
