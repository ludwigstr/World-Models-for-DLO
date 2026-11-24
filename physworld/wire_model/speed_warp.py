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
