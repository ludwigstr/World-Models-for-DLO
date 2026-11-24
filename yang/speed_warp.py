import numpy as np

SPEED_RANGE = (0.65, 1.4)
PAUSE_PROB = 0.0
USE_OFFSETS = False
OFFSET_MAX = 1.5
MARGIN = 0.03
CLEARANCE = 0.005
N_SUB = 10
MAX_TRIES = 20
FPS = 4.0


def sample_warp(T, rng, offsets=True):
    dt = 1.0 / FPS
    n_max = 3 * T
    t = np.arange(n_max) * dt
    amp = rng.uniform(0.3, 1.5)
    s = np.zeros(n_max)
    for _ in range(3):
        per = rng.uniform(2.0, 15.0)
        s += np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi))
    s = np.clip(amp * s / np.sqrt(1.5), -1.0, 1.0)
    v = 1.0 + np.where(s >= 0, s * (SPEED_RANGE[1] - 1.0), s * (1.0 - SPEED_RANGE[0]))
    pauses = []
    if rng.random() < PAUSE_PROB:
        for _ in range(int(rng.integers(1, 4))):
            t0 = rng.uniform(1.0, max(1.5, T * dt * 0.9)); dur = rng.uniform(0.5, 2.0); ramp = 0.5
            w = np.ones(n_max)
            a = (t > t0 - ramp) & (t < t0); w[a] = 0.5 + 0.5 * np.cos(np.pi * (t[a] - (t0 - ramp)) / ramp)
            w[(t >= t0) & (t <= t0 + dur)] = 0.0
            b = (t > t0 + dur) & (t < t0 + dur + ramp); w[b] = 0.5 - 0.5 * np.cos(np.pi * (t[b] - t0 - dur) / ramp)
            v = v * w
            pauses.append((float(t0), float(dur)))
    tau = np.r_[0.0, np.cumsum(v[:-1])]
    taus = np.repeat(tau[:, None], 5, 1)
    off_amp = []
    if offsets:
        ramp = np.clip(t / 2.0, 0, 1)
        for m in range(5):
            a_m = rng.uniform(0, OFFSET_MAX); off_amp.append(float(a_m))
            dlt = np.zeros(n_max)
            for _ in range(2):
                dlt += np.sin(2 * np.pi * t / rng.uniform(4.0, 20.0) + rng.uniform(0, 2 * np.pi)) / 2
            taus[:, m] = np.maximum.accumulate(np.clip(tau + a_m * dlt * ramp, 0, T - 1))
    taus = np.clip(taus, 0, T - 1)
    done = np.nonzero((taus >= T - 1 - 1e-9).all(1))[0]
    n = int(done[0]) + 1 if len(done) else n_max
    taus = taus[:n]
    if not len(done):
        taus[-1] = T - 1
    return taus, {"speed_amp": float(amp), "pauses": pauses, "offset_amp": off_amp,
                  "n_frames": int(n), "speed_min": float(v[:n].min()), "speed_max": float(v[:n].max()),
                  "duration_ratio": float(n / T)}


def _interp(F, tau):
    i = np.clip(np.floor(tau).astype(int), 0, len(F) - 2)
    f = (tau - i).reshape(-1, *([1] * (F.ndim - 1)))
    return F[i] * (1 - f) + F[i + 1] * f


def end_frames(F, taus):
    F = np.asarray(F, float)
    E = np.empty((len(taus), 4, 4, 2))
    H = _interp(F[:, :, [0, 1]], taus[:, 0])
    E[:, :, 0:2] = H
    for k in range(4):
        E[:, k, 2:4] = _interp(F[:, k, [14, 15]], taus[:, 1 + k])
    return E


def subsample(E, n=N_SUB):
    if len(E) < 2:
        return E
    f = np.linspace(0, 1, n, endpoint=False)
    A = E[:-1, None] * (1 - f.reshape(1, -1, *([1] * (E.ndim - 1)))) + E[1:, None] * f.reshape(1, -1, *([1] * (E.ndim - 1)))
    return np.concatenate([A.reshape(-1, *E.shape[1:]), E[-1:]])


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]], -1)


def mover_poses(E):
    from harness_scene import HUB_CAVS, PLUG_CAV
    out = (E[:, :, 1] - E[:, :, 0]).mean(1)
    hy = np.arctan2(out[:, 1], out[:, 0]) + np.pi / 2
    hm = np.array(HUB_CAVS).mean(0)
    C = np.empty((len(E), 5, 2)); Y = np.empty((len(E), 5))
    C[:, 0] = E[:, 0, 0] - _rot(hm, hy); Y[:, 0] = hy
    for k in range(4):
        v = E[:, k, 3] - E[:, k, 2]
        y = np.arctan2(v[:, 1], v[:, 0]) + np.pi / 2
        C[:, 1 + k] = E[:, k, 3] - _rot(PLUG_CAV, y); Y[:, 1 + k] = y
    return C, Y


def attachments(E, kind):
    if kind == "nodes":
        return np.stack([E[:, :, 0], E[:, :, 3]], 2)
    from harness_scene import HUB_CAVS, PLUG_CAV
    C, Y = mover_poses(E)
    A = np.empty((len(E), 4, 2, 2))
    for k in range(4):
        A[:, k, 0] = C[:, 0] + _rot(HUB_CAVS[k], Y[:, 0])
        A[:, k, 1] = C[:, 1 + k] + _rot(PLUG_CAV, Y[:, 1 + k])
    return A


def footprint_gap(E):
    from harness_scene import PLATFORM_HALF
    C, _ = mover_poses(E)
    i, j = np.triu_indices(5, 1)
    d = np.abs(C[:, i] - C[:, j]).max(-1)
    return d - 2 * PLATFORM_HALF[0]


def real_limits(F, kind):
    F = np.asarray(F, float)
    taus = np.repeat(np.arange(len(F), dtype=float)[:, None], 5, 1)
    E = subsample(end_frames(F, taus))
    A = attachments(E, kind)
    dist = np.linalg.norm(A[:, :, 0] - A[:, :, 1], axis=-1)
    L = np.linalg.norm(np.diff(F[0], axis=1), axis=-1).sum(-1)
    gap = footprint_gap(E)
    d_max = dist.max(0)
    return {"lim_dist": d_max, "arc_len": L, "real_ratio_max": d_max / L,
            "real_exceeds_margin": (d_max > (1 - MARGIN) * L).tolist(),
            "real_min_gap": float(gap.min()), "lim_gap": float(min(CLEARANCE, gap.min()))}


def check(E, lim, kind):
    S = subsample(E)
    A = attachments(S, kind)
    dist = np.linalg.norm(A[:, :, 0] - A[:, :, 1], axis=-1)
    a_ok = bool((dist <= lim["lim_dist"][None] + 1e-9).all())
    g = footprint_gap(S).min()
    b_ok = bool(g >= lim["lim_gap"] - 1e-9)
    return a_ok and b_ok, {"a_ok": a_ok, "b_ok": b_ok, "max_ratio": float((dist / lim["arc_len"]).max()),
                           "min_gap_mm": float(1000 * g)}


def sample_valid(F, rng, kind, lim=None):
    lim = real_limits(F, kind) if lim is None else lim
    rej = {"a": 0, "b": 0}
    for tr in range(1, MAX_TRIES + 1):
        taus, meta = sample_warp(len(F), rng, offsets=USE_OFFSETS)
        E = end_frames(F, taus)
        ok, st = check(E, lim, kind)
        if ok:
            return E, taus, {**meta, "tries": tr, "rejected_a": rej["a"], "rejected_b": rej["b"], "offsets": USE_OFFSETS,
                             **st}
        rej["a"] += not st["a_ok"]; rej["b"] += not st["b_ok"]
    taus, meta = sample_warp(len(F), rng, offsets=False)
    E = end_frames(F, taus)
    ok, st = check(E, lim, kind)
    return E, taus, {**meta, "tries": MAX_TRIES + 1, "rejected_a": rej["a"], "rejected_b": rej["b"],
                     "offsets": False, "fallback_ok": ok, **st}


REL_TOL = 1.5
ABS_TOL = 1.0


def contact_excess(pairs_w, T_w, pairs_r, T_r, mm_w=0, mm_r=0):
    rw = np.asarray(pairs_w, float) / max(T_w, 1); rr = np.asarray(pairs_r, float) / max(T_r, 1)
    new = (rw > 0) & (rr == 0)
    over = rw > REL_TOL * rr + ABS_TOL
    bad = new | over
    if mm_w > 0 and mm_r == 0:
        return "mover-mover contact"
    if bad.any():
        k, m = np.argwhere(bad)[0]
        return f"cable{k}-mover{m}: {rw[k, m]:.2f}/frame vs real {rr[k, m]:.2f}/frame"
    return None
