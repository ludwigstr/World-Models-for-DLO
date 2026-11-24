import argparse
import json
import math
import os
import time
import traceback
import multiprocessing as mp
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent


GKEYS = ["lE", "lrho", "lmu", "nu", "lsy"]


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def worker(conn, device, clips, a):
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        import torch
        import twin as F
        from twin import Twin
        tws, Ws = {}, {}
        for c in clips:
            z = np.load(ROOT / a.tracks / f"{c}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)])
            if a.max_frames:
                P = P[:, :a.max_frames]
            tw = Twin(P, device=device, r_grasp=a.r_grasp, mass_scale_dt=a.mass_scale_dt or None)
            blocks = []
            for cab in tw.cab:
                q = np.zeros(cab["n_slices"]); q[cab["slice"]] = cab["arc"]
                blocks.append(np.stack([np.interp(q, cab["s_nodes"], np.eye(16)[j]) for j in range(16)], 1))
            W = np.zeros((tw.sp.n_slices, 64), np.float32)
            for g, b in enumerate(blocks):
                W[tw.sp.slice_off[g]:tw.sp.slice_off[g + 1], 16 * g:16 * g + 16] = b
            tws[c] = tw; Ws[c] = torch.tensor(W, device=device)
        conn.send(("ready", {c: (tws[c].T, tws[c].NP) for c in clips}))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            _, gv, off, need_grad, seed, save_dir = msg
            out = {}
            for ci, c in enumerate(clips):
                tw = tws[c]; t0 = time.time()
                glob = {k: torch.tensor(float(gv[k]), device=device, requires_grad=need_grad and off is None) for k in GKEYS}
                loc = None; o = None
                if off is not None:
                    o = torch.tensor(np.asarray(off, np.float32), device=device, requires_grad=need_grad)
                    per_slice = Ws[c] @ o.reshape(3, 64).T
                    pp_off = per_slice[tw.sp.slice]
                    loc = {"dlE": pp_off[:, 0], "dlrho": pp_off[:, 1], "dlmu": pp_off[:, 2]}
                rng = np.random.default_rng(seed * 1000 + hash(c) % 997)
                L, _, nodes, info, n_sub = F.run_iter(tw, glob, loc, a.model, a.frac, rng, device, need_grad=need_grad)
                bad = bool(info["bad"])
                m = F.twin_metric(nodes, tw.Ptorch) if not bad else np.array([np.inf])
                rec = {"loss": (float(L) / tw.T if L is not None and math.isfinite(L) else float("inf")) if need_grad else None,
                       "twin_mm": float(m[F.WARM:].mean()) if len(m) > F.WARM else float("inf"), "bad": bad,
                       "n_sub": int(n_sub), "sec": time.time() - t0, "T": tw.T}
                if need_grad and not bad:
                    if off is None:
                        g = np.array([0.0 if glob[k].grad is None else float(glob[k].grad) for k in GKEYS])
                    else:
                        g = o.grad.detach().cpu().numpy().astype(np.float64)
                    gn = float(np.sqrt((g ** 2).sum()))
                    rec["grad_norm"] = gn
                    rec["grad"] = (g * (1.0 / gn if gn > 1.0 else 1.0)).tolist()
                if save_dir and not bad:
                    G = dict(glob)
                    if a.model == "elastic":
                        G["elastic"] = True
                    pp = tw.per_particle(G, loc)
                    torch.save({"clip": c, "model": a.model, "glob": {k: v.detach().cpu() for k, v in glob.items()},
                                "node_offsets": None if off is None else np.asarray(off, np.float32),
                                "E": pp["E"].detach().cpu(), "rho": pp["rho"].detach().cpu(), "mu": pp["mu"].detach().cpu(),
                                "nu": float(glob["nu"]), "sy": float(pp["sy"][0]), "perm": tw.sp.perm, "r_grasp": a.r_grasp,
                                "dx": tw.dx, "n_sub": int(n_sub), "mass_scale_dt": a.mass_scale_dt,
                                "rhoi": pp["rhoi"].detach().cpu(), "twin_final": m.tolist(), "nodes_final": nodes.cpu(),
                                "shared": True}, Path(save_dir) / f"{c}_{a.model}.pt")
                out[c] = rec
            conn.send(("done", out))
    except Exception:
        conn.send(("error", traceback.format_exc()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="../data/tracks"); ap.add_argument("--dir-suffix", default="")
    ap.add_argument("--model", default="elastoplastic", choices=["elastic", "elastoplastic"])
    ap.add_argument("--global-iters", type=int, default=10); ap.add_argument("--local-iters", type=int, default=6)
    ap.add_argument("--frac", type=float, default=0.33)
    ap.add_argument("--E0", type=float, default=3e5); ap.add_argument("--mu0", type=float, default=0.3)
    ap.add_argument("--sy0", type=float, default=1e4); ap.add_argument("--nu0", type=float, default=0.3)
    ap.add_argument("--r-grasp", type=float, default=0.012); ap.add_argument("--mass-scale-dt", type=float, default=1e-4)
    ap.add_argument("--gpus", default="0,1,2"); ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--clips", nargs="*", default=None); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--rho0", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--min-rel", type=float, default=0.005)
    a = ap.parse_args()
    import torch
    from twin import BOUNDS, RHO_PHYS
    out_dir = ROOT / f"data/wm/physworld_mpm_twins{a.dir_suffix}"
    out_dir.mkdir(parents=True, exist_ok=False)
    clips = a.clips or sorted(p.stem for p in (ROOT / a.tracks).glob("*.npz"))
    lens = {c: int(np.load(ROOT / a.tracks / f"{c}.npz")["c0"].shape[0]) for c in clips}
    devs = [f"cuda:{g}" for g in a.gpus.split(",") for _ in range(a.workers_per_gpu)][:len(clips)]
    groups = [[] for _ in devs]
    for c in sorted(clips, key=lambda c: -lens[c]):
        groups[int(np.argmin([sum(lens[x] for x in g) for g in groups]))].append(c)
    ctx = mp.get_context("spawn"); conns = []
    for dev, g in zip(devs, groups):
        pc, cc = ctx.Pipe(); ctx.Process(target=worker, args=(cc, dev, g, a), daemon=True).start(); conns.append(pc)
    for pc in conns:
        tag, info = pc.recv()
        if tag != "ready":
            raise SystemExit(info)
    log(f"{len(clips)} clips on {len(devs)} workers ({a.gpus}); model {a.model}; shared twin, {a.global_iters} global + {a.local_iters} local iterations")

    def sweep(gv, off, need_grad, seed, save_dir=None):
        for pc in conns:
            pc.send(("iter", gv, off, need_grad, seed, save_dir))
        res = {}
        for pc in conns:
            tag, r = pc.recv()
            if tag != "done":
                raise SystemExit(r)
            res.update(r)
        return res

    glob = {"lE": torch.tensor(math.log(a.E0), requires_grad=True), "lrho": torch.tensor(math.log(a.rho0 or RHO_PHYS), requires_grad=True),
            "lmu": torch.tensor(math.log(a.mu0), requires_grad=True), "nu": torch.tensor(a.nu0, requires_grad=True),
            "lsy": torch.tensor(math.log(a.sy0), requires_grad=a.model == "elastoplastic")}
    gkeys = [k for k in GKEYS if glob[k].requires_grad]
    glr = {"lE": 0.2, "lrho": 0.2, "lmu": 0.2, "nu": 0.03, "lsy": 0.3}
    hist = []; t_all = time.time()

    def describe():
        return {k: (float(glob[k]) if k == "nu" else float(math.exp(float(glob[k])))) for k in GKEYS}

    def run_stage(name, iters, params, lrs, off_t):
        opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(params, lrs)])
        best = (float("inf"), None, None, -1); ref, stale = float("inf"), 0
        for it in range(iters):
            t0 = time.time()
            gv = {k: float(glob[k]) for k in GKEYS}
            off = None if off_t is None else off_t.detach().numpy()
            res = sweep(gv, off, True, len(hist))
            ok = [c for c in clips if not res[c]["bad"] and math.isfinite(res[c]["loss"])]
            L = float(np.mean([res[c]["loss"] for c in ok])) if ok else float("inf")
            mt = float(np.mean([res[c]["twin_mm"] for c in ok])) if ok else float("inf")
            rec = {"stage": name, "it": it, "loss": L, "twin_mm": mt, "n_ok": len(ok), "bad": [c for c in clips if c not in ok],
                   **describe(), "sec": None}
            if len(ok) == len(clips) and L < best[0]:
                best = (L, [p.detach().clone() for p in params], mt, it)
            if len(ok) < 0.75 * len(clips):
                log(f"  {name} it {it}: only {len(ok)} of {len(clips)} clips stable, stopping stage"); hist.append(rec); break
            g = np.mean([np.asarray(res[c]["grad"]) for c in ok], 0)
            if off_t is None:
                for k, p in zip(gkeys, params):
                    p.grad = torch.tensor(float(g[GKEYS.index(k)]))
            else:
                params[0].grad = torch.tensor(g, dtype=torch.float32)
            opt.step()
            with torch.no_grad():
                for k, (lo, hi) in BOUNDS.items():
                    glob[k].clamp_(lo, hi)
                if off_t is not None:
                    off_t.clamp_(-3.0, 3.0)
            rec["sec"] = time.time() - t0; rec["grad_norm"] = float(np.sqrt((g ** 2).sum()))
            rec["clip_grad_norm_mean"] = float(np.mean([res[c]["grad_norm"] for c in ok]))
            hist.append(rec)
            log(f"  {name} it {it}: loss {L:.6f} twin {mt:.2f} mm ({len(ok)}/{len(clips)} clips)  E {rec['lE']:.3g} rho {rec['lrho']:.4g} "
                f"mu {rec['lmu']:.3g} nu {rec['nu']:.3f} sy {rec['lsy']:.3g}  |g| {rec['grad_norm']:.2e}  {rec['sec']:.0f}s")
            (out_dir / "shared_twin_progress.json").write_text(json.dumps({"hist": hist}, indent=1))
            if L < ref * (1 - a.min_rel):
                ref, stale = L, 0
            else:
                stale += 1
            if a.patience and stale >= a.patience:
                log(f"  {name}: no improvement of {a.min_rel:.1%} for {a.patience} iterations, stage ends"); break
        if best[1] is not None:
            with torch.no_grad():
                for p, b in zip(params, best[1]):
                    p.copy_(b)
        log(f"  {name}: best loss {best[0]:.6f} at it {best[3]} (twin {best[2]} mm)")
        return best

    run_stage("global", a.global_iters, [glob[k] for k in gkeys], [glr[k] for k in gkeys], None)
    off_t = torch.zeros(3, 4, 16, requires_grad=True)
    if a.local_iters > 0:
        run_stage("local", a.local_iters, [off_t], [0.1], off_t)
    gv = {k: float(glob[k]) for k in GKEYS}
    off = off_t.detach().numpy() if a.local_iters > 0 else None
    res = sweep(gv, off, False, 0, save_dir=str(out_dir))
    per = {c: res[c]["twin_mm"] for c in clips}
    summ = {"clips": len(clips), "twin_mm_mean": float(np.mean(list(per.values()))), "per_clip_twin_mm": per,
            "glob": describe(), "glob_raw": gv, "node_offsets": None if off is None else off.tolist(), "hist": hist,
            "sec_total": time.time() - t_all, "args": vars(a),
            "note": "ONE MPM twin for all clips: 5 global values + offsets on log E, log rho, log mu at 16 nodes x 4 cables"}
    (out_dir / "shared_twin.json").write_text(json.dumps(summ, indent=1))
    log(f"shared twin: mean twin metric {summ['twin_mm_mean']:.2f} mm over {len(clips)} clips; {summ['sec_total']/3600:.1f} h; saved {out_dir}")
    for pc in conns:
        pc.send(("stop",))
    print("=== SHARED_TWIN_DONE ===", flush=True)


if __name__ == "__main__":
    main()
