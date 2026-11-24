import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
wp.config.log_level = getattr(wp, "LOG_WARNING", None) or 30
wp.init()
from wire_model.physworld_mpm.twin import Twin, drive_from_nodes, rho_for_dt, MS_CFL
from wire_model import speed_warp as SW


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def slice_chol(cab, sigma, ell, dev):
    out = []
    for c in cab:
        s = np.linspace(0, 1, c["n_slices"])
        K = sigma ** 2 * np.exp(-(s[:, None] - s[None]) ** 2 / (2 * ell ** 2)) + 1e-6 * np.eye(len(s))
        out.append(torch.tensor(np.linalg.cholesky(K), dtype=torch.float32, device=dev))
    return out


def node_phi(tw, lE, lrho):
    v = torch.stack([lE, lrho], -1)
    sm = torch.zeros(tw.sp.n_slices, 2, device=v.device).index_add_(0, tw.sp.slice, v) / tw.sp.slice_cnt[:, None]
    idx = tw.sp.node_slices.clamp_min(0); m = (tw.sp.node_slices >= 0).float()[..., None]
    return (sm[idx] * m).sum(2) / m.sum(2)


def warp_drive(P, taus):
    F = np.transpose(P, (1, 0, 2, 3))
    E = SW._interp(F[:, :, [0, 1, 14, 15]], taus)
    return drive_from_nodes(torch.tensor(np.transpose(E, (1, 0, 2, 3)), dtype=torch.float32))


def broken(nodes, rest_len):
    if not torch.isfinite(nodes).all():
        return "non-finite"
    if (nodes[1:, ..., :2] - nodes[:-1, ..., :2]).norm(dim=-1).max() > 0.15:
        return "node jump > 0.15 m"
    if nodes[..., 2].max() > 0.03:
        return "node height > 3 cm"
    L = (nodes[:, :, 1:, :2] - nodes[:, :, :-1, :2]).norm(dim=-1).sum(-1)
    if (L / rest_len).max() > 1.25:
        return "stretched > 1.25x"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--twin", default="elastoplastic")
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--ell", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out-tag", default="")
    ap.add_argument("--global-sigma", type=float, default=0.0)
    ap.add_argument("--nu-sigma", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--speed-range", type=float, nargs=2, default=None)
    ap.add_argument("--tracks", default="../data/tracks"); ap.add_argument("--dir-suffix", default="")
    a = ap.parse_args(); dev = a.device
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    flag = ROOT / f"data/wm/physworld_mpm_demos{a.dir_suffix}/SPEED_RANGE"
    if a.speed_range is None and flag.exists():
        a.speed_range = [float(v) for v in flag.read_text().split()[:2]]
    if a.speed_range is not None:
        SW.SPEED_RANGE = (float(a.speed_range[0]), float(a.speed_range[1]))
    tf = torch.load(ROOT / f"data/wm/physworld_mpm_twins{a.dir_suffix}/{a.clip}_{a.twin}.pt", weights_only=False)
    z = np.load(ROOT / a.tracks / f"{a.clip}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)]).astype(np.float32)
    T = P.shape[1]
    B = a.batch
    ms = tf.get("mass_scale_dt") or None
    tw = Twin(np.concatenate([P] * B), cables=range(4 * B), device=dev, r_grasp=tf["r_grasp"])
    tw1 = Twin(P, device=dev, r_grasp=tf["r_grasp"])
    n1 = tw1.NP
    lE0 = torch.log(tf["E"]).to(dev); lrho0 = torch.log(tf["rho"]).to(dev); mu0 = tf["mu"].to(dev)
    nu, sy = float(tf["nu"]), float(tf["sy"])
    chol = slice_chol(tw1.cab, a.sigma, a.ell, dev)
    rest_len = torch.tensor([c["length"] for c in tw1.cab], device=dev)
    log(f"{a.clip}: {T} frames, twin {a.twin} (nu {nu:.3f}, sy {sy:.3g}), {a.n} demos in batches of {B}, "
        f"sigma {a.sigma} l {a.ell}, speed range {SW.SPEED_RANGE}")

    def draw():
        out = []
        for _ in range(2):
            pert = torch.cat([chol[g] @ torch.randn(chol[g].shape[0], device=dev) + a.global_sigma * torch.randn((), device=dev)
                              for g in range(4)])
            out.append(pert[tw1.sp.slice])
        if a.global_sigma > 0:
            lmu = torch.log(mu0) + (a.global_sigma * torch.randn(4, device=dev))[tw1.sp.gid.long()]
            nu_d = float(min(0.45, max(0.0, nu + a.nu_sigma * torch.randn(()).item())))
            sy_d = sy * math.exp(a.global_sigma * torch.randn(()).item())
        else:
            lmu, nu_d, sy_d = torch.log(mu0), nu, sy
        return lE0 + out[0], lrho0 + out[1], lmu, nu_d, sy_d

    def simulate(twin, props, drives, nmax):
        lE = torch.cat([p[0] for p in props]); lrho = torch.cat([p[1] for p in props])
        pp = {"E": torch.exp(lE), "rho": torch.exp(lrho), "mu": torch.exp(torch.cat([p[2] for p in props])),
              "nu": torch.cat([torch.full_like(p[0], p[3]) for p in props]),
              "sy": torch.cat([torch.full_like(p[0], p[4]) for p in props])}
        pp["rhoi"] = pp["rho"] if not ms else torch.maximum(pp["rho"], rho_for_dt(pp["E"], pp["nu"], twin.dx, ms, MS_CFL))
        M = pp["E"] * (1 - pp["nu"]) / ((1 + pp["nu"]) * (1 - 2 * pp["nu"]))
        dt = min(1e-4, MS_CFL * twin.dx / torch.sqrt(M / pp["rhoi"]).max().item() * 1.0001)
        twin.mpm.dt = twin.frame_s / int(100 * math.ceil(twin.frame_s / dt / 100))
        ends = torch.cat([d[0] for d in drives]).to(dev); yaw = torch.cat([d[1] for d in drives]).to(dev)
        nodes, info = twin.rollout(twin.to_warp(pp), T=nmax, ends=(ends, yaw))
        return nodes, info

    demos, rejected, t0 = [], [], time.time()
    k = 0
    while k < a.n:
        nb = min(B, a.n - k)
        props, drives, metas = [], [], []
        for _ in range(nb):
            taus, meta = SW.sample_warp(T, rng, offsets=False)
            e, y = warp_drive(P, taus[:, 0])
            props.append(draw()); drives.append((e, y)); metas.append(meta)
        nmax = max(d[0].shape[1] for d in drives)
        pad = lambda x: torch.cat([x, x[:, -1:].expand(-1, nmax - x.shape[1], *x.shape[2:])], 1)
        drives_p = [(pad(e), pad(y)) for e, y in drives]
        if nb == B:
            nodes, info = simulate(tw, props, drives_p, nmax)
            per = [nodes[:, 4 * i:4 * i + 4] if not info["bad"] else None for i in range(nb)]
        else:
            per = [None] * nb; info = {"bad": True}
        if info["bad"]:
            per = []
            for i in range(nb):
                nd, inf = simulate(tw1, [props[i]], [drives[i]], drives[i][0].shape[1])
                per.append(nd if not inf["bad"] else None)
        for i in range(nb):
            n_i = drives[i][0].shape[1]
            nd = per[i][:n_i] if per[i] is not None and per[i].shape[0] >= n_i else None
            why = "rollout aborted (non-finite / exploded)" if nd is None else broken(nd, rest_len)
            if why:
                rejected.append({"demo": k + i, "reason": why, **metas[i]}); continue
            phi = node_phi(tw1, props[i][0], props[i][1])
            demos.append({"nodes": nd[..., :2].cpu(), "ends": drives[i][0].cpu(), "yaw": drives[i][1].cpu(),
                          "phi": phi.cpu(), "meta": metas[i], "nu": props[i][3], "sy": props[i][4],
                          "mu_cable": [float(torch.exp(props[i][2][tw1.sp.gid == g][0])) for g in range(4)]})
        k += nb
        log(f"  {k}/{a.n} demos, kept {len(demos)}, rejected {len(rejected)} ({time.time()-t0:.0f}s)")
    out = ROOT / f"data/wm/physworld_mpm_demos{a.dir_suffix}/{a.clip}{a.out_tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    phi_real = node_phi(tw1, lE0, lrho0).cpu()
    torch.save({"clip": a.clip, "demos": demos, "rejected": rejected, "phi_twin": phi_real, "sigma": a.sigma,
                "ell": a.ell, "global_sigma": a.global_sigma, "twin": a.twin, "speed_range": list(SW.SPEED_RANGE), "sec": time.time() - t0, "n_frames_total": int(sum(d["nodes"].shape[0] for d in demos))}, out)
    log(f"saved {out}: {len(demos)} kept, {len(rejected)} rejected {[r['reason'] for r in rejected]}; "
        f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
