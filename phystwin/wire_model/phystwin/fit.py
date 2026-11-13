from __future__ import annotations

import numpy as np
import torch

from .sim import SpringMassTwin, build_topology

LO = [np.log(1.0), np.log(1e-3), np.log(1e-4), np.log(1e-3), 1.0]
HI = [np.log(1500.0), np.log(1.5), np.log(20.0), np.log(5.0), 4.0]


def velocity0(P: torch.Tensor, frame_s: float) -> torch.Tensor:
    return (P[:, 1] - P[:, 0]) / frame_s if P.shape[1] > 1 else torch.zeros_like(P[:, 0])


def resim_error(twin, P, k, gamma, delta, mu, t0=0, t1=None, frame_s=0.25):
    C = P.shape[0]; B = gamma.shape[0]; pop = B // C; t1 = P.shape[1] if t1 is None else t1
    Pf = P[:, t0:t1]
    X0 = Pf[:, 0].repeat(pop, 1, 1); V0 = velocity0(Pf, frame_s).repeat(pop, 1, 1)
    ends = Pf[:, :, [0, -1]].repeat(pop, 1, 1, 1)
    sim = twin.rollout(X0, V0, ends, k, gamma, delta, mu)
    err = (sim[:, :, 1:-1] - Pf[:, :, 1:-1].repeat(pop, 1, 1, 1)).norm(dim=-1)
    err = torch.nan_to_num(err, nan=1e3, posinf=1e3).mean((1, 2))
    return err.view(pop, C)


def stage1(P, fit_frames, device, gens=20, popsize=16, seed=0, log=print, frame_s=0.25, dt=1e-3):
    import cma
    x0 = [np.log(200.0), np.log(0.3), np.log(0.05), np.log(0.3), 2.0]
    es = cma.CMAEvolutionStrategy(x0, 0.6, {"popsize": popsize, "seed": seed + 1, "verbose": -9, "bounds": [LO, HI]})
    best = (float("inf"), None)
    for g in range(gens):
        xs = es.ask(); costs = np.zeros(len(xs))
        by_r = {}
        for i, x in enumerate(xs):
            by_r.setdefault(round(float(x[4]) * 2) / 2, []).append(i)
        for r, idx in by_r.items():
            twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), dt=dt, frame_s=frame_s)
            xb = torch.tensor(np.array([xs[i] for i in idx]), dtype=torch.float32, device=device)
            C = P.shape[0]; rep = lambda v: v.repeat_interleave(C)
            with torch.no_grad():
                e = resim_error(twin, P, torch.exp(rep(xb[:, 0])).view(-1, 1, 1), torch.exp(rep(xb[:, 1])), torch.exp(rep(xb[:, 2])), torch.exp(rep(xb[:, 3])), 0, fit_frames, frame_s)
            costs[idx] = e.mean(1).cpu().numpy() * 1000
        es.tell(xs, costs.tolist())
        i = int(np.argmin(costs))
        if costs[i] < best[0]:
            best = (float(costs[i]), np.array(xs[i]))
        if g % 5 == 0 or g == gens - 1:
            b = best[1]; log(f"    stage 1 gen {g:2d}: best {best[0]:.2f} mm  k {np.exp(b[0]):.0f} gamma {np.exp(b[1]):.3f} delta {np.exp(b[2]):.3f} mu {np.exp(b[3]):.3f} radius {b[4]:.1f} seg")
    return best


def stage2(P, fit_frames, x1, device, iters=100, win=8, lr=0.05, log=print, frame_s=0.25, dt=1e-3, seed=0, check=10,
           mode="full"):
    r = round(float(x1[4]) * 2) / 2
    twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), dt=dt, frame_s=frame_s)
    C, N = P.shape[0], P.shape[2]
    logk = torch.full((C, N, N), float(x1[0]), device=device, requires_grad=True)
    glob = torch.tensor(x1[1:4], dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([logk, glob], lr=lr); rng = np.random.default_rng(seed)
    if mode == "full":
        return _stage2_full(P, fit_frames, twin, logk, glob, opt, iters, log, frame_s, device)
    assert mode == "window", mode
    best = (float("inf"), None, None, -1)
    def check_full(it):
        nonlocal best
        with torch.no_grad():
            k_ = torch.exp(0.5 * (logk + logk.transpose(1, 2))); gl_ = torch.exp(glob).clamp(max=torch.tensor([1.5, 20.0, 5.0], device=device))
            e_ = resim_error(twin, P, k_, gl_[0].expand(C), gl_[1].expand(C), gl_[2].expand(C), 0, fit_frames, frame_s).mean().item() * 1000
        if e_ < best[0]:
            best = (e_, k_.clone(), gl_.clone(), it)
        log(f"    stage 2 it {it:3d}: full fitting-span error {e_:.2f} mm (best {best[0]:.2f} at it {best[3]})")
    for it in range(iters):
        if it % check == 0:
            check_full(it)
        t0 = int(rng.integers(0, max(1, fit_frames - win))); t1 = min(fit_frames, t0 + win)
        k = torch.exp(0.5 * (logk + logk.transpose(1, 2)))
        gl = torch.exp(glob).clamp(max=torch.tensor([1.5, 20.0, 5.0], device=device))
        e = resim_error(twin, P, k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C), t0, t1, frame_s)
        opt.zero_grad(); e.mean().backward(); torch.nn.utils.clip_grad_norm_([logk, glob], 1.0); opt.step()
        with torch.no_grad():
            logk.clamp_(LO[0], HI[0]); glob.clamp_(torch.tensor(LO[1:4], device=device), torch.tensor(HI[1:4], device=device))
        if it % 25 == 0 or it == iters - 1:
            log(f"    stage 2 it {it:3d}: window error {e.mean().item()*1000:.2f} mm")
    check_full(iters)
    return twin, best[1], best[2]


def _stage2_full(P, fit_frames, twin, logk, glob, opt, iters, log, frame_s, device):
    C = P.shape[0]
    hi_gl = torch.tensor([1.5, 20.0, 5.0], device=device)
    lo_t, hi_t = torch.tensor(LO[1:4], device=device), torch.tensor(HI[1:4], device=device)
    best = (float("inf"), None, None, -1)
    skipped = 0
    for it in range(iters + 1):
        k = torch.exp(0.5 * (logk + logk.transpose(1, 2)))
        gl = torch.exp(glob).clamp(max=hi_gl)
        e = resim_error(twin, P, k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C), 0, fit_frames, frame_s).mean()
        e_mm = e.item() * 1000
        if np.isfinite(e_mm) and e_mm < best[0]:
            best = (e_mm, k.detach().clone(), gl.detach().clone(), it)
        if it % 10 == 0 or it == iters:
            log(f"    stage 2 it {it:3d}: full fitting-span error {e_mm:.2f} mm (best {best[0]:.2f} at it {best[3]}, "
                f"lr {opt.param_groups[0]['lr']:.4f}, skipped {skipped})")
        if it == iters:
            break
        opt.zero_grad()
        if not np.isfinite(e_mm) or (best[1] is not None and e_mm > 2.0 * best[0]):
            skipped += 1; opt.param_groups[0]["lr"] *= 0.5
            if best[1] is not None:
                with torch.no_grad():
                    logk.copy_(torch.log(best[1])); glob.copy_(torch.log(best[2]))
            opt.state.clear()
            continue
        e.backward()
        gn = torch.nn.utils.clip_grad_norm_([logk, glob], 1.0)
        if not torch.isfinite(gn):
            skipped += 1; opt.param_groups[0]["lr"] *= 0.5
            continue
        opt.step()
        with torch.no_grad():
            logk.clamp_(LO[0], HI[0]); glob.clamp_(lo_t, hi_t)
    return twin, best[1], best[2]
