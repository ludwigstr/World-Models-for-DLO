import numpy as np

DEFAULT = {"sigma_shift": 0.0, "sigma_node": 0.0, "sigma_perp": 0.0, "sigma_tan": 0.0, "end_ext": 0.0, "end_ext_sd": 0.0,
           "straighten": 0.0, "sigma_end": 0.0, "sigma_shape_static": 0.0, "sigma_shape_dyn": 0.0, "rho_shape": 0.9,
           "end_off0": 0.0, "end_off15": 0.0, "smooth_max": 0.0}


def _resample(P, s_new):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    x = np.interp(s_new, d, P[:, 0]); y = np.interp(s_new, d, P[:, 1])
    return np.stack([x, y], 1)


def track_noise(P, rng, p):
    P = np.asarray(P, float)
    T, L, _ = P.shape
    if p.get("smooth_max", 0) > 0:
        w = rng.uniform(0, p["smooth_max"])
        P = P.copy()
        P[:, 1:-1] = P[:, 1:-1] + w * (0.5 * (P[:, :-2] + P[:, 2:]) - P[:, 1:-1])
    Q = P.copy()
    if p.get("sigma_shift", 0) > 0:
        for t in range(T):
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[t], axis=0), axis=1))]
            s = d.copy()
            s[1:-1] += rng.normal(0, p["sigma_shift"], L - 2)
            s = np.clip(s, 0, d[-1])
            s[1:-1] = np.sort(s[1:-1])
            Q[t] = _resample(P[t], s)
    if p.get("end_ext", 0) > 0 or p.get("straighten", 0) > 0:
        ext = rng.normal(p.get("end_ext", 0), p.get("end_ext_sd", 0), 2)
        for (e, n1, n2), x in zip(((0, 1, 2), (L - 1, L - 2, L - 3)), ext):
            u = Q[:, e] - Q[:, n2]
            u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
            Q[:, e] = Q[:, e] + x * u
            a = Q[:, n2] + ((Q[:, n1] - Q[:, n2]) * u).sum(-1, keepdims=True) * u
            Q[:, n1] = Q[:, n1] + p.get("straighten", 0) * (a - Q[:, n1])
    if p.get("sigma_shape_static", 0) > 0 or p.get("sigma_shape_dyn", 0) > 0:
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[0], axis=0), axis=1))]
        u = d / max(d[-1], 1e-9)
        w = np.array([1.0, 0.6, 0.4])
        B = np.stack([np.sin((m + 1) * np.pi * u) for m in range(3)], 1) * w
        a_st = rng.normal(0, p.get("sigma_shape_static", 0), 3)
        rho = p.get("rho_shape", 0.9); sd = p.get("sigma_shape_dyn", 0)
        a_dy = np.zeros((T, 3)); a_dy[0] = rng.normal(0, sd, 3)
        for t in range(1, T):
            a_dy[t] = rho * a_dy[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sd, 3)
        amp = (a_st[None] + a_dy) @ B.T
        tng = np.gradient(P, axis=1)
        tng = tng / np.maximum(np.linalg.norm(tng, axis=-1, keepdims=True), 1e-9)
        nrm = np.stack([-tng[..., 1], tng[..., 0]], -1)
        Q = Q + amp[..., None] * nrm
    if p.get("end_off0", 0) > 0:
        Q[:, 0] += rng.normal(0, p["end_off0"], 2)
    if p.get("end_off15", 0) > 0:
        Q[:, L - 1] += rng.normal(0, p["end_off15"], 2)
    if p.get("sigma_node", 0) > 0:
        Q[:, 1:-1] += rng.normal(0, p["sigma_node"], (T, L - 2, 2))
    if p.get("sigma_perp", 0) > 0 or p.get("sigma_tan", 0) > 0:
        tng = Q[:, 2:] - Q[:, :-2]
        tng = tng / np.maximum(np.linalg.norm(tng, axis=-1, keepdims=True), 1e-9)
        nrm = np.stack([-tng[..., 1], tng[..., 0]], -1)
        Q[:, 1:-1] += (rng.normal(0, p.get("sigma_perp", 0), (T, L - 2, 1)) * nrm
                       + rng.normal(0, p.get("sigma_tan", 0), (T, L - 2, 1)) * tng)
    if p.get("sigma_end", 0) > 0:
        Q[:, [0, L - 1]] += rng.normal(0, p["sigma_end"], (T, 2, 2))
    return Q


def stats(P):
    d = np.diff(P, axis=1); seg = np.linalg.norm(d, axis=-1)
    ang = np.arctan2(d[..., 1], d[..., 0]); turn = np.angle(np.exp(1j * np.diff(ang, axis=1)))
    dp = np.diff(P[:, [0, -1]], axis=0); ddp = np.diff(dp, axis=0)
    return {"turn_dt_mid_deg": np.degrees(np.abs(np.angle(np.exp(1j * np.diff(turn[:, 1:-1], axis=0))))).mean(),
            "seg_dt_mid_mm": 1000 * np.abs(np.diff(seg[:, 1:-1], axis=0)).mean(),
            "seg_end_mm": 1000 * seg[:, [0, -1]].mean(), "seg_mid_mm": 1000 * seg[:, 1:-1].mean(),
            "turn_end_deg": np.degrees(np.abs(turn[:, [0, -1]])).mean(), "turn_mid_deg": np.degrees(np.abs(turn[:, 1:-1])).mean(),
            "ddp_mm": 1000 * np.linalg.norm(ddp, axis=-1).mean()}
