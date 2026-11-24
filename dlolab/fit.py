from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent


sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))


os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("PYTHONUNBUFFERED", "1")


TRACKS = os.environ.get("GENERAL_FIT_TRACKS") or str(ROOT.parent / "data" / "tracks")


N_C, N_H, N_E = 4, 14, 15


MASS0 = 0.045


SEG_MASS0 = 0.002


I_HS = 0


I_SS = I_HS + N_C * N_H


I_TS = I_SS + N_C * N_E


I_KR = I_TS + N_C * N_H


I_K = I_KR + N_C * N_H * 2


I_E, I_G, I_MU, I_DAMP, I_MASS = I_K + 1, I_K + 2, I_K + 3, I_K + 4, I_K + 5


N_PAR = I_MASS + N_C


_CACHE = {}


def unpack(x):
    x = np.asarray(x, float)
    mass = np.exp(x[I_MASS:I_MASS + N_C])
    return dict(hinge_scale=np.exp(x[I_HS:I_SS]).reshape(N_C, N_H),
                stretch_scale=np.exp(x[I_SS:I_TS]).reshape(N_C, N_E),
                twist_scale=np.exp(x[I_TS:I_KR]).reshape(N_C, N_H),
                kappa_rest=x[I_KR:I_K].reshape(N_C, N_H, 2),
                K=float(np.exp(x[I_K])), E=float(np.exp(x[I_E])), G=float(1e5 * x[I_G] ** 2),
                mu=float(np.exp(x[I_MU])), damping=float(np.exp(x[I_DAMP])),
                segment_mass=SEG_MASS0 * mass / MASS0), mass


def _eval_one(job):
    cand_i, clip, x = job
    import simulate as R
    if clip not in _CACHE:
        z = np.load(Path(TRACKS) / f"{clip}.npz")
        _CACHE[clip] = ({k: z[f"c{k}"] for k in range(4)}, float(z["fps"]))
    kw, _ = unpack(x)
    try:
        e = float(np.mean(R.eval_clip(clip, shp_fps=_CACHE[clip], use_inextensible=False, **kw)))
        return cand_i, clip, e if np.isfinite(e) else 1e3
    except Exception as exc:
        print(f"  {clip}: {type(exc).__name__}: {exc}", flush=True)
        return cand_i, clip, 1e3


SETTLE = 300


def _layout():
    import fit as C
    return C


def grad_mask(C):
    m = np.ones(C.N_PAR, bool)
    m[C.I_MU] = m[C.I_DAMP] = False
    m[C.I_MASS:C.I_MASS + N_C] = False
    return m


def cable_block(C, k):
    idx = []
    idx += list(range(C.I_HS + k * N_H, C.I_HS + (k + 1) * N_H))
    idx += list(range(C.I_SS + k * N_E, C.I_SS + (k + 1) * N_E))
    idx += list(range(C.I_TS + k * N_H, C.I_TS + (k + 1) * N_H))
    idx += list(range(C.I_KR + k * N_H * 2, C.I_KR + (k + 1) * N_H * 2))
    idx += [C.I_K, C.I_E, C.I_G]
    return np.asarray(idx)


def zero_x(C):
    x = np.zeros(C.N_PAR)
    x[C.I_K], x[C.I_E], x[C.I_G] = np.log(1e4), np.log(1e5), 0.1
    x[C.I_MU], x[C.I_DAMP] = np.log(0.1), np.log(40.0)
    x[C.I_MASS:C.I_MASS + N_C] = np.log(C.MASS0)
    return x


def grad_worker(gpu, backend, bptt, conn):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    import simulate as R
    import genesis as gs
    C = _layout()
    names = ["hub"] + [f"plug{k}" for k in range(4)]
    cache = {}
    for npz in sorted(Path(TRACKS).glob("*.npz")):
        z = np.load(npz)
        fps = float(z["fps"])
        cache[npz.stem] = ({k: z[f"c{k}"] for k in range(4)}, fps, max(1, round((1.0 / fps) / 1e-3)))
    conn.send(("ready", len(cache)))

    def once(clip, x, max_frames=0):
        shp, fps, spf = cache[clip]
        kw, _ = C.unpack(x)
        g_lin = float(x[C.I_G])
        if getattr(gs, "_initialized", False):
            gs.destroy()
        gs.init(backend=(gs.cuda if backend == "cuda" else gs.cpu), logging_level="warning")
        scene, ents, rods, _ = R.build_rod_scene(
            shp, 0, K=kw["K"], E=kw["E"], G=kw["G"], mu=kw["mu"], damping=kw["damping"],
            segment_mass=kw["segment_mass"], use_inextensible=False,
            requires_grad=True, bptt_window=bptt, grad_clip=1.0)
        scene.build(n_envs=1)
        for k, rod in enumerate(rods):
            n_v = rod.n_vertices
            rod.attach_to_rigid_link(R._mod_link(ents, "hub"), list(range(R.N_BIND)))
            rod.attach_to_rigid_link(R._mod_link(ents, f"plug{k}"), list(range(n_v - R.N_BIND, n_v)))
            assert len(kw["hinge_scale"][k]) == rod.n_internal_vertices and len(kw["stretch_scale"][k]) == rod.n_edges
            rod.set_bending_scale(torch.as_tensor(kw["hinge_scale"][k].astype(np.float32)))
            rod.set_stretching_scale(torch.as_tensor(kw["stretch_scale"][k].astype(np.float32)))
            rod.set_twisting_scale(torch.as_tensor(kw["twist_scale"][k].astype(np.float32)))
            rod.set_kappa_rest_offset(torch.as_tensor(kw["kappa_rest"][k].astype(np.float32)))
        dofs, dtop = {}, {}
        for n in names:
            e = ents[n]
            dofs[n] = e.get_joint(f"{n}_x").dofs_idx_local + e.get_joint(f"{n}_y").dofs_idx_local
            e.set_dofs_kp(np.full(2, 4000.0), dofs[n]); e.set_dofs_kv(np.full(2, 120.0), dofs[n])
            et = ents[f"{n}_top"]
            xy = et.get_joint(f"{n}_x").dofs_idx_local + et.get_joint(f"{n}_y").dofs_idx_local
            tu = et.get_joint(f"{n}_turn").dofs_idx_local
            dtop[n] = (xy, tu)
            et.set_dofs_kp(np.full(2, 4000.0), xy); et.set_dofs_kv(np.full(2, 120.0), xy)
            et.set_dofs_kp(np.full(1, 60.0), tu); et.set_dofs_kv(np.full(1, 4.0), tu)
        p0 = R.MJ_poses(shp, 0)
        base = {n: np.asarray(p0[n][0], float) for n in names}
        by = {n: float(p0[n][1]) for n in names}
        prev = dict(by)

        def movers(t):
            p = R.MJ_poses(shp, t)
            for n in names:
                pos, yaw = p[n]
                prev[n] += np.arctan2(np.sin(yaw - prev[n]), np.cos(yaw - prev[n]))
                rel = np.asarray(pos, float) - base[n]
                ents[n].control_dofs_position(rel, dofs[n])
                ents[f"{n}_top"].control_dofs_position(rel, dtop[n][0])
                ents[f"{n}_top"].control_dofs_position(np.array([prev[n] - by[n]]), dtop[n][1])

        movers(0)
        for _ in range(SETTLE):
            scene.step()
        n_frames = len(shp[0])
        if max_frames:
            n_frames = min(n_frames, max_frames)
        loss = torch.zeros((), dtype=torch.float32)
        per_frame = []
        for t in range(n_frames):
            movers(t)
            for _ in range(spf):
                scene.step()
            d_frame = 0.0
            for k, rod in enumerate(rods):
                pos = rod.get_state().pos.reshape(-1, 3)
                assert pos.shape[0] == len(shp[k][t]), (pos.shape, shp[k][t].shape)
                tgt = torch.as_tensor(np.asarray(shp[k][t], np.float32))
                d = torch.sqrt(((pos[:, :2] - tgt) ** 2).sum(-1) + 1e-12).mean() * 1000.0
                d_frame += float(d)
                loss = loss + d / (n_frames * len(rods))
            per_frame.append(d_frame / len(rods))
        loss.backward()
        g = np.zeros(C.N_PAR)
        for k, rod in enumerate(rods):
            hs, ss, ts = kw["hinge_scale"][k], kw["stretch_scale"][k], kw["twist_scale"][k]
            g[C.I_K] += float(rod.get_stretching_stiffness_grad()[0]) * kw["K"]
            g[C.I_E] += float(rod.get_bending_stiffness_grad()[0]) * kw["E"]
            g[C.I_G] += float(rod.get_twisting_stiffness_grad()[0]) * 2e5 * g_lin
            g[C.I_HS + k * N_H:C.I_HS + (k + 1) * N_H] = rod.get_bending_scale_grad()[:, 0].cpu().numpy() * hs
            g[C.I_SS + k * N_E:C.I_SS + (k + 1) * N_E] = rod.get_stretching_scale_grad()[:, 0].cpu().numpy() * ss
            g[C.I_TS + k * N_H:C.I_TS + (k + 1) * N_H] = rod.get_twisting_scale_grad()[:, 0].cpu().numpy() * ts
            g[C.I_KR + k * N_H * 2:C.I_KR + (k + 1) * N_H * 2] = rod.get_kappa_rest_grad()[:, :, 0].cpu().numpy().reshape(-1)
        out = (float(loss), g, per_frame)
        gs.destroy()
        return out

    while True:
        m = conn.recv()
        if m is None:
            break
        x, clip, max_frames = np.asarray(m[0], float), m[1], int(m[2])
        t0 = time.time()
        try:
            l, g, pf = once(clip, x, max_frames)
            conn.send((clip, l, g.tolist(), time.time() - t0, pf))
        except Exception as e:
            conn.send((clip, None, f"{type(e).__name__}: {e}", time.time() - t0, []))
    conn.close()


def run_batch(conns, x, clips, max_frames=0):
    todo = list(clips)
    busy = {}
    res = []
    free = list(conns)
    while todo or busy:
        while todo and free:
            c = free.pop()
            clip = todo.pop(0)
            c.send((x.tolist(), clip, max_frames))
            busy[c] = clip
        if not busy:
            raise RuntimeError("no gradient workers left")
        for c in wait(list(busy)):
            job = busy.pop(c)
            try:
                res.append(c.recv())
                free.append(c)
            except (EOFError, OSError) as e:
                print(f"  gradient worker died on {job}: {type(e).__name__}; clip re-queued, worker dropped", flush=True)
                conns.remove(c)
                todo.append(job)
    return res


def _metric_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _metric_job(i, clip, x, python, timeout):
    import subprocess
    xs = ",".join(repr(float(v)) for v in x)
    code = ("import sys, numpy as np; sys.path[:0] = ['scripts', '.']; import fit as M\n"
            f"x = np.asarray([{xs}], float)\n"
            f"print('RES', M._eval_one((0, '{clip}', x))[2])")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "GENERAL_FIT_TRACKS": TRACKS}
    try:
        p = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=timeout)
        r = [ln for ln in p.stdout.splitlines() if ln.startswith("RES")]
        if r:
            return i, clip, float(r[-1].split()[1]), None
        return i, clip, 1e3, f"rc={p.returncode} {p.stderr[-200:]!r}"
    except subprocess.TimeoutExpired:
        return i, clip, 1e3, f"timeout {timeout}s"


def real_metric(pool, C, xs, clips, timeout=1500):
    from concurrent.futures import as_completed
    out = np.zeros((len(xs), len(clips)))
    ci = {c: j for j, c in enumerate(clips)}
    futs = [pool.submit(_metric_job, i, c, np.asarray(x, float), sys.executable, timeout)
            for i, x in enumerate(xs) for c in clips]
    fails = []
    for f in as_completed(futs):
        i, c, e, err = f.result()
        out[i, ci[c]] = e
        if err:
            fails.append(f"cand {i} {c}: {err}")
    if fails:
        print(f"  real metric: {len(fails)} job(s) failed (scored 1e3):", *fails[:6], sep="\n   ", flush=True)
    return out.mean(1), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="zero")
    ap.add_argument("--resume", default="")
    ap.add_argument("--tracks", default="../data/tracks")
    ap.add_argument("--out", default="out/dlolab_fit.json")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--stop-by", default="")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--momentum", type=float, default=0.3)
    ap.add_argument("--const-step", type=float, default=0.0)
    ap.add_argument("--single-step", action="store_true")
    ap.add_argument("--max-lr", type=float, default=0.0)
    ap.add_argument("--bptt", type=int, default=120)
    ap.add_argument("--grad-workers", type=int, default=24)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--metric-workers", type=int, default=24)
    ap.add_argument("--smoke", default="")
    ap.add_argument("--cable", type=int, default=0)
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()

    global TRACKS
    TRACKS = str((ROOT / a.tracks).resolve())
    os.environ["GENERAL_FIT_TRACKS"] = TRACKS
    C = _layout()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    lr = a.lr
    if a.resume:
        prev = json.loads((ROOT / a.resume).read_text())
        x = np.asarray(prev["best_x"], float)
        init = {"best_f": prev["best_f"], "resumed_from": a.resume}
        if prev["hist"] and "lr_next" in prev["hist"][-1] and "--lr" not in sys.argv:
            lr = float(prev["hist"][-1]["lr_next"])
    elif a.init == "zero":
        x = zero_x(C)
        init = {"best_f": None}
    else:
        init = json.loads((ROOT / a.init).read_text())
        x = np.asarray(init["best_x"], float)
    assert x.shape == (C.N_PAR,), x.shape
    clips = sorted(p.stem for p in Path(TRACKS).glob("*.npz"))
    gpus = a.gpus.split(",")

    if a.smoke:
        pa, pb = ctx.Pipe()
        pr = ctx.Process(target=grad_worker, args=(gpus[0], a.backend, a.bptt, pb), daemon=True)
        pr.start(); print(pa.recv())
        (clip, l, g, secs, pf), = run_batch([pa], x, [a.smoke], a.frames)
        cable = a.cable
        print(f"smoke {clip} (4 rods) backend={a.backend} frames={a.frames}: loss {l} mm  ({secs:.1f}s, {secs / max(a.frames, 1):.1f} s/frame)")
        if l is None:
            print("ERROR", g); pa.send(None); return
        g = np.asarray(g)
        print("per-frame mm:", np.round(pf, 2).tolist())
        print(f"|grad| {np.linalg.norm(g):.3e}, nonzero {int((g != 0).sum())}/{int(grad_mask(C).sum())}; blocks below for cable {cable}")
        for nm, sl in [("K", [C.I_K]), ("E", [C.I_E]), ("G", [C.I_G]),
                       ("hinge", range(C.I_HS + cable * N_H, C.I_HS + (cable + 1) * N_H)),
                       ("stretch", range(C.I_SS + cable * N_E, C.I_SS + (cable + 1) * N_E)),
                       ("twist", range(C.I_TS + cable * N_H, C.I_TS + (cable + 1) * N_H)),
                       ("kappa", range(C.I_KR + cable * N_H * 2, C.I_KR + (cable + 1) * N_H * 2))]:
            v = g[list(sl)]
            print(f"  {nm:8s} |g| {np.linalg.norm(v):.3e}  nonzero {int((v != 0).sum())}/{len(v)}")
        pa.send(None)
        return

    stop_at = dt.datetime.strptime(a.stop_by, "%Y-%m-%d %H:%M") if a.stop_by else None
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    log = out.with_suffix(".log.jsonl")

    conns = []
    for i in range(a.grad_workers):
        pa, pb = ctx.Pipe()
        ctx.Process(target=grad_worker, args=(gpus[i % len(gpus)], a.backend, a.bptt, pb), daemon=True).start()
        conns.append(pa)
    for c in conns:
        c.recv()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(a.metric_workers)
    print(f"[{time.strftime('%H:%M:%S')}] gradient refinement of {a.resume or a.init}: {len(clips)} clips, "
          f"{a.grad_workers} gradient workers ({a.backend}, dynamic dispatch), {a.metric_workers} metric workers, "
          f"batch {a.batch} clips, bptt {a.bptt}, lr0 {lr}, stop by {a.stop_by or '-'}", flush=True)

    t0 = time.time()
    (f0,), per0 = real_metric(pool, C, [x], clips)
    print(f"[{time.strftime('%H:%M:%S')}] start: real metric {f0:.4f} (init file says {init.get('best_f')}; {time.time() - t0:.0f}s)", flush=True)
    best_x, best_f, per_best = x.copy(), float(f0), per0[0]
    cur_x, cur_f = x.copy(), float(f0)
    mask = grad_mask(C)
    mom = np.zeros(C.N_PAR)
    hist = [{"it": -1, "real": best_f, "lr": lr, "init": a.resume or a.init}]
    rng = np.random.default_rng(0)
    order = list(rng.permutation(len(clips)))
    pos = 0

    def save(tag=""):
        kw, mass = C.unpack(best_x)
        d = {"best_x": best_x.tolist(), "best_f": best_f, "base_f": float(f0), "init": a.resume or a.init,
             "mass_per_m": mass.tolist(), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in kw.items()},
             "per_clip": {c: float(e) for c, e in zip(clips, per_best)}, "hist": hist, "status": tag,
             "cur_x": cur_x.tolist(), "cur_f": cur_f,
             "method": "gradient (rod adjoint, bptt %d, ALL FOUR RODS per scene (fixed fork), unit-normalised per clip) + step rule on the real metric" % a.bptt}
        out.write_text(json.dumps(d, indent=1))

    save("running")
    for it in range(a.iters):
        if stop_at and dt.datetime.now() >= stop_at:
            print(f"[{time.strftime('%H:%M:%S')}] stop-by reached", flush=True); break
        if pos + a.batch > len(order):
            order = list(rng.permutation(len(clips))); pos = 0
        batch = [clips[i] for i in order[pos:pos + a.batch]]; pos += a.batch
        t0 = time.time()
        res = run_batch(conns, cur_x, list(batch), a.max_frames)
        t_grad = time.time() - t0
        G, Ls, fails, norms, secs = np.zeros(C.N_PAR), [], [], [], []
        for clip, l, g, s_, pf in res:
            secs.append(s_)
            if l is None:
                fails.append(f"{clip}: {g}"); continue
            g = np.asarray(g) * mask
            n = np.linalg.norm(g)
            norms.append(n)
            if n > 1e-14 and np.isfinite(n):
                G += g / n
            Ls.append(l)
        if fails:
            print("  gradient failures:", *fails, sep="\n   ", flush=True)
        if not Ls:
            print("  no gradient this iteration, skipping", flush=True); continue
        mom = a.momentum * mom + (1 - a.momentum) * (G / max(len(Ls), 1))
        d = mom / max(np.linalg.norm(mom), 1e-14)
        steps = [a.const_step] if a.const_step else ([lr] if a.single_step else [lr * 0.5, lr, lr * 2.0])
        cands = [cur_x - s * d for s in steps]
        t1 = time.time()
        fs, pers = real_metric(pool, C, cands, clips)
        t_ls = time.time() - t1
        j = int(np.argmin(fs))
        rec = {"it": it, "batch": batch, "grad_loss_mean": float(np.mean(Ls)), "n_rollouts": len(Ls),
               "rollout_norms": [float(v) for v in norms], "rollout_secs": [float(v) for v in secs],
               "consensus": float(np.linalg.norm(G)),
               "steps": steps, "real": [float(v) for v in fs], "best_before": best_f, "lr": lr,
               "t_grad_s": t_grad, "t_linesearch_s": t_ls}
        if a.const_step:
            moved = bool(np.isfinite(fs[0]) and (pers[0] < 1e3).all())
            if moved:
                cur_x, cur_f = cands[0], float(fs[0])
            accepted = bool(fs[0] < best_f)
            if accepted:
                best_x, best_f, per_best = cands[0], float(fs[0]), pers[0]
            rec["moved"] = moved; rec["cur"] = cur_f
        else:
            accepted = bool(fs[j] < best_f)
            if accepted:
                best_x, best_f, per_best = cands[j], float(fs[j]), pers[j]
                cur_x, cur_f = best_x, best_f
                lr = steps[j] * (2.0 if j == 2 else 1.0)
                if a.max_lr:
                    lr = min(lr, a.max_lr)
            else:
                lr *= 0.5
        rec["accepted"] = accepted; rec["best_after"] = best_f; rec["lr_next"] = lr
        hist.append(rec)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] it {it:03d}: grad on {len(batch)} clips ({len(Ls)} rollouts, mean {np.mean(Ls):.2f} mm, "
              f"consensus {np.linalg.norm(G):.1f}/{len(Ls)}, {t_grad:.0f}s, rollouts {min(secs):.0f}-{max(secs):.0f}s)  line search real = "
              + " / ".join(f"{s:.3g}:{v:.4f}" for s, v in zip(steps, fs))
              + f" ({t_ls:.0f}s)  -> {'ACCEPT' if accepted else 'reject'} best {best_f:.4f}  "
              + (f"const step {a.const_step}, {'moved' if rec['moved'] else 'STAYED (crash)'}, current {cur_f:.4f}" if a.const_step else f"lr {lr:.3g}"), flush=True)
        save("running")
        if not a.const_step and lr < 1e-3:
            print("step length exhausted", flush=True); break
    for c in conns:
        c.send(None)
    pool.shutdown()
    save("done")
    print(f"[{time.strftime('%H:%M:%S')}] best real metric {best_f:.4f} (start {f0:.4f}) -> {out}", flush=True)
    print("GRAD_DONE", flush=True)


if __name__ == "__main__":
    main()
