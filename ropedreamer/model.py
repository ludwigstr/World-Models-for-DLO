from __future__ import annotations
import torch
from torch import Tensor


def q_normalize(q: Tensor, eps: float = 1e-8) -> Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def q_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def q_conj(q: Tensor) -> Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def q_rotate(q: Tensor, v: Tensor) -> Tensor:
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return q_mul(q_mul(q, qv), q_conj(q))[..., 1:]


def q_from_two_vectors(a: Tensor, b: Tensor, eps: float = 1e-9) -> Tensor:
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    w = 1.0 + (a * b).sum(-1, keepdim=True)
    xyz = torch.cross(a, b, dim=-1)
    anti = w.squeeze(-1) < 1e-6
    if bool(anti.any()):
        alt = torch.stack([-a[..., 1], a[..., 0], torch.zeros_like(a[..., 0])], -1)
        alt2 = torch.stack([torch.zeros_like(a[..., 0]), -a[..., 2], a[..., 1]], -1)
        use_alt = alt.norm(dim=-1, keepdim=True) > 1e-6
        axis = torch.where(use_alt, alt, alt2)
        xyz = torch.where(anti.unsqueeze(-1), axis, xyz)
        w = torch.where(anti.unsqueeze(-1), torch.zeros_like(w), w)
    return q_normalize(torch.cat([w, xyz], dim=-1))


def positions_to_chain(P: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if P.shape[-1] == 2:
        P = torch.cat([P, torch.zeros_like(P[..., :1])], dim=-1)
    d = P[..., 1:, :] - P[..., :-1, :]
    seg = d.norm(dim=-1)
    ell = seg.mean(dim=-1)
    u = d / seg.clamp_min(1e-9).unsqueeze(-1)
    ex = torch.zeros_like(u[..., 0, :]); ex[..., 0] = 1.0
    quats = []
    R = q_from_two_vectors(ex, u[..., 0, :])
    quats.append(R)
    for i in range(1, u.shape[-2]):
        local = q_rotate(q_conj(R), u[..., i, :])
        qi = q_from_two_vectors(ex, local)
        quats.append(qi)
        R = q_normalize(q_mul(R, qi))
    q = torch.stack(quats, dim=-2)
    s = torch.cat([P[..., 0, :], q.flatten(-2)], dim=-1)
    return s, ell, seg


def chain_to_positions(s: Tensor, ell: Tensor, n_nodes: int) -> Tensor:
    p0 = s[..., :3]
    q = q_normalize(s[..., 3:].reshape(*s.shape[:-1], n_nodes - 1, 4))
    ex = torch.zeros_like(p0); ex[..., 0] = 1.0
    per_link = ell.ndim == s.ndim and ell.shape[-1] == n_nodes - 1
    L = (lambda i: ell[..., i]) if per_link else (lambda i: ell)
    pts = [p0]
    R = q[..., 0, :]
    p = p0 + L(0).unsqueeze(-1) * q_rotate(R, ex)
    pts.append(p)
    for i in range(1, n_nodes - 1):
        R = q_normalize(q_mul(R, q[..., i, :]))
        p = p + L(i).unsqueeze(-1) * q_rotate(R, ex)
        pts.append(p)
    return torch.stack(pts, dim=-2)


def normalize_state(s: Tensor, n_nodes: int) -> Tensor:
    q = q_normalize(s[..., 3:].reshape(*s.shape[:-1], n_nodes - 1, 4))
    return torch.cat([s[..., :3], q.flatten(-2)], dim=-1)


import numpy as np


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


from pathlib import Path


N_NODES = 16


GRASPS = (0, N_NODES - 2)


def load_tracks(dirs: list[str | Path]) -> list[dict]:
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f)
            for k in range(4):
                out.append({"clip": f.stem, "cable": k, "P": z[f"c{k}"].astype(np.float32),
                            "fps": float(z["fps"]), "set": Path(d).name})
    return out


def load_synth(dirs: list[str | Path], max_gap: float = 0.015, gap_ref_tol: float | None = None) -> list[dict]:
    import re
    out = []
    kept = total = 0
    for d in dirs:
        files = sorted(Path(d).glob("*.npz"))
        ref = {}
        if gap_ref_tol is not None:
            for f in files:
                mt = re.match(r"(.*)_(\d{2,3})$", f.stem)
                if mt and int(mt.group(2)) == 0:
                    ref[mt.group(1)] = float(np.load(f)["gap"].max())
        for f in files:
            z = np.load(f)
            V, gap = z["V"].astype(np.float32), z["gap"]
            thr = max_gap
            if gap_ref_tol is not None:
                r = ref.get(re.sub(r"_\d{2,3}$", "", f.stem))
                if r is not None:
                    thr = max(max_gap, r + gap_ref_tol)
            bad = np.nonzero(gap > thr)[0]
            T = int(bad[0]) if len(bad) else len(V)
            kept += T; total += len(V)
            for k in range(4):
                out.append({"clip": f.stem, "cable": k, "P": V[:T, k], "fps": 4.0, "set": Path(d).name})
    print(f"load_synth: kept {kept}/{total} frames ({100 * kept / max(total, 1):.1f} %), max_gap {1000 * max_gap:.0f} mm"
          + ("" if gap_ref_tol is None else f", tolerant: real-timing reference + {1000 * gap_ref_tol:.0f} mm"), flush=True)
    return out


REPR = "chain"


def trajectory_tensors(P: np.ndarray) -> dict:
    Pt = torch.tensor(P, dtype=torch.float32)
    s, ell_mean, seg = positions_to_chain(Pt)
    if REPR == "positions":
        s = torch.cat([Pt, torch.zeros_like(Pt[..., :1])], -1).flatten(1)
    ell = seg.mean()
    P3 = torch.cat([Pt, torch.zeros_like(Pt[..., :1])], -1)
    nodes = [0, N_NODES - 1]
    dp = P3[1:, nodes] - P3[:-1, nodes]
    dp = torch.cat([dp, torch.zeros_like(dp[:1])], 0)
    idx = torch.tensor(GRASPS, dtype=torch.long).expand(Pt.shape[0], 2).clone()
    return {"s": s, "ell": ell, "idx": idx, "dp": dp, "P": Pt}


class WindowDataset(torch.utils.data.Dataset):

    def __init__(self, trajs: list[dict], seq_len: int, windows_per_traj: int | None = None, seed: int = 0,
                 obs_noise: dict | None = None):
        self.obs_noise = obs_noise
        self.T = [trajectory_tensors(t["P"]) for t in trajs if t["P"].shape[0] >= seq_len]
        self.seq_len = seq_len
        self.n_win = [max(1, (t["s"].shape[0] - seq_len) // (seq_len // 2) + 1) if windows_per_traj is None
                      else windows_per_traj for t in self.T]
        self.index = [(i, w) for i, n in enumerate(self.n_win) for w in range(n)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, _ = self.index[j]
        t = self.T[i]
        T = t["s"].shape[0]
        start = int(self.rng.integers(0, max(1, T - self.seq_len + 1)))
        sl = slice(start, start + self.seq_len)
        if self.obs_noise:
            pass
            Pn = track_noise(t["P"][sl].numpy(), self.rng, self.obs_noise)
            o = trajectory_tensors(Pn.astype(np.float32))
            return t["s"][sl], t["idx"][sl], t["dp"][sl], t["ell"], o["s"], o["dp"]
        return t["s"][sl], t["idx"][sl], t["dp"][sl], t["ell"]


def state_offset(trajs: list[dict]) -> Tensor:
    p0 = np.concatenate([t["P"][:, 0] for t in trajs]).mean(0)
    if REPR == "positions":
        off = torch.zeros(3 * N_NODES); off[0::3], off[1::3] = float(p0[0]), float(p0[1])
        return off
    off = torch.zeros(3 + 4 * (N_NODES - 1)); off[0], off[1] = float(p0[0]), float(p0[1])
    return off


def state_stats(trajs: list[dict]) -> tuple[Tensor, Tensor]:
    S = torch.cat([trajectory_tensors(t["P"])["s"] for t in trajs])
    return S.mean(0), S.std(0)


from torch import Tensor, nn
import torch.nn.functional as F


SIZES = {
    "S": dict(d_embed=1024, d_action=1024, d_rnn=512, d_z=64, d_hidden=512),
    "M": dict(d_embed=1024, d_action=1024, d_rnn=512, d_z=64, d_hidden=1024),
    "L": dict(d_embed=2048, d_action=512, d_rnn=1024, d_z=64, d_hidden=2048),
    "XS": dict(d_embed=256, d_action=256, d_rnn=256, d_z=32, d_hidden=256),
}


def mlp(d_in: int, d_hidden: int, d_out: int, n_hidden: int = 2) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.ELU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class ActionEncoder(nn.Module):

    def __init__(self, n_links: int, d_action: int, d_hidden: int, d_disp: int = 3):
        super().__init__()
        self.emb = nn.Embedding(n_links + 1, d_action, padding_idx=n_links)
        self.n_links = n_links
        self.disp = mlp(d_disp, d_hidden, d_action, n_hidden=1)

    def forward(self, idx: Tensor, dp: Tensor) -> Tensor:
        pad = idx < 0
        idx = torch.where(pad, torch.full_like(idx, self.n_links), idx)
        a = self.emb(idx) + self.disp(dp) * (~pad).unsqueeze(-1)
        return a.sum(dim=1)


class RSSMCell(nn.Module):
    def __init__(self, d_action: int, d_rnn: int, d_z: int, d_hidden: int, d_embed: int,
                 min_std: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_z + d_action, d_hidden), nn.ELU())
        self.gru = nn.GRUCell(d_hidden, d_rnn)
        self.prior = mlp(d_rnn, d_hidden, 2 * d_z, n_hidden=1)
        self.post = mlp(d_rnn + d_embed, d_hidden, 2 * d_z, n_hidden=1)
        self.d_z, self.min_std = d_z, min_std

    def step(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        return self.gru(self.inp(torch.cat([z, a], dim=-1)), h)

    def _dist(self, out: Tensor) -> tuple[Tensor, Tensor]:
        mu, pre = out.chunk(2, dim=-1)
        return mu, F.softplus(pre) + self.min_std

    def prior_dist(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self._dist(self.prior(h))

    def post_dist(self, h: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        return self._dist(self.post(torch.cat([h, e], dim=-1)))


def state_nll(pred: Tensor, target: Tensor, n_nodes: int, inv_var: Tensor) -> Tensor:
    w = inv_var.view(*([1] * (pred.ndim - 1)), -1)
    e0 = ((pred[..., :3] - target[..., :3]) ** 2 * w[..., :3]).sum(-1)
    qp = pred[..., 3:].reshape(*pred.shape[:-1], n_nodes - 1, 4)
    qt = target[..., 3:].reshape(*target.shape[:-1], n_nodes - 1, 4)
    wq = w[..., 3:].reshape(*w.shape[:-1], n_nodes - 1, 4)
    dq = torch.minimum(((qp - qt) ** 2 * wq).sum(-1), ((qp + qt) ** 2 * wq).sum(-1)).sum(-1)
    return 0.5 * (e0 + dq).mean()


def kl_gauss(mu_q, sd_q, mu_p, sd_p) -> Tensor:
    return (torch.log(sd_p / sd_q) + (sd_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sd_p ** 2) - 0.5).sum(-1)


class RopeDreamer(nn.Module):
    def __init__(self, n_nodes: int = 16, size: str = "S", d_disp: int = 3, beta: float = 1.0,
                 free_nats: float = 0.0, pos_weight: float = 0.0, rollout_weight: float = 0.0,
                 warm: int = 5, repr: str = "chain"):
        super().__init__()
        c = SIZES[size]
        self.n_nodes, self.beta, self.free_nats, self.pos_weight = n_nodes, beta, free_nats, pos_weight
        self.rollout_weight, self.warm, self.repr = rollout_weight, warm, repr
        self.d_state = 3 + 4 * (n_nodes - 1) if repr == "chain" else 3 * n_nodes
        self.register_buffer("s_mean", torch.zeros(self.d_state))
        self.register_buffer("s_std", torch.ones(self.d_state))
        self.enc = mlp(self.d_state, c["d_hidden"], c["d_embed"], n_hidden=1)
        self.act = ActionEncoder(n_nodes - 1, c["d_action"], c["d_hidden"], d_disp)
        self.cell = RSSMCell(c["d_action"], c["d_rnn"], c["d_z"], c["d_hidden"], c["d_embed"])
        self.dec_recon = mlp(c["d_rnn"] + c["d_z"], c["d_hidden"], self.d_state)
        self.dec_pred = mlp(c["d_rnn"] + c["d_z"], c["d_hidden"], self.d_state)
        self.d_rnn, self.d_z = c["d_rnn"], c["d_z"]
        self.size = size

    def set_state_stats(self, mean: Tensor, std: Tensor, std_floor: float = 0.05) -> None:
        self.s_mean.copy_(mean); self.s_std.copy_(std.clamp_min(std_floor))

    def encode(self, s: Tensor) -> Tensor:
        return self.enc((s - self.s_mean) / self.s_std)

    def initial(self, B: int, device) -> tuple[Tensor, Tensor]:
        return (torch.zeros(B, self.d_rnn, device=device), torch.zeros(B, self.d_z, device=device))

    def decode(self, dec: nn.Module, h: Tensor, z: Tensor) -> Tensor:
        out = dec(torch.cat([h, z], dim=-1))
        return normalize_state(out, self.n_nodes) if self.repr == "chain" else out

    def positions(self, s: Tensor, ell: Tensor) -> Tensor:
        if self.repr == "chain":
            return chain_to_positions(s, ell, self.n_nodes)
        return s.reshape(*s.shape[:-1], self.n_nodes, 3)

    def forward(self, s: Tensor, idx: Tensor, dp: Tensor, sample: bool = True,
                ell: Tensor | None = None, s_obs: Tensor | None = None) -> dict:
        B, T, _ = s.shape
        h, z = self.initial(B, s.device)
        e = self.encode(s if s_obs is None else s_obs)
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        recon, pred, kls = [], [], []
        for t in range(T):
            if t > 0:
                h = self.cell.step(h, z, a[:, t - 1])
            mu_p, sd_p = self.cell.prior_dist(h)
            mu_q, sd_q = self.cell.post_dist(h, e[:, t])
            z_prior = mu_p + sd_p * torch.randn_like(sd_p) if sample else mu_p
            z = mu_q + sd_q * torch.randn_like(sd_q) if sample else mu_q
            recon.append(self.decode(self.dec_recon, h, z))
            if t > 0:
                pred.append(self.decode(self.dec_pred, h, z_prior))
            kls.append(kl_gauss(mu_q, sd_q, mu_p, sd_p))
        recon = torch.stack(recon, 1)
        pred = torch.stack(pred, 1)
        kl = torch.stack(kls, 1)
        inv_var = 1.0 / self.s_std ** 2
        if self.repr == "chain":
            l_recon = state_nll(recon, s, self.n_nodes, inv_var)
            l_pred = state_nll(pred, s[:, 1:], self.n_nodes, inv_var)
        else:
            w = inv_var.view(1, 1, -1)
            l_recon = 0.5 * (((recon - s) ** 2) * w).sum(-1).mean()
            l_pred = 0.5 * (((pred - s[:, 1:]) ** 2) * w).sum(-1).mean()
        l_kl = torch.clamp(kl, min=self.free_nats).mean()
        loss = l_recon + l_pred + self.beta * l_kl
        l_pos = torch.zeros((), device=s.device)
        if self.pos_weight > 0:
            assert ell is not None, "pos_weight needs the link length per sequence"
            e = ell.view(B, 1)
            P_true = self.positions(s, e)
            P_rec = self.positions(recon, e)
            P_pred = self.positions(pred, e)
            l_pos = F.mse_loss(P_rec, P_true) + F.mse_loss(P_pred, P_true[:, 1:])
            loss = loss + self.pos_weight * l_pos
        l_roll = torch.zeros((), device=s.device)
        if self.rollout_weight > 0 and T > self.warm:
            assert ell is not None
            l_roll = self.open_loop_loss(s, idx, dp, ell)
            loss = loss + self.rollout_weight * l_roll
        return {"loss": loss, "recon": l_recon, "pred": l_pred, "kl": kl.mean(), "pos": l_pos, "roll": l_roll,
                "s_recon": recon, "s_pred": pred}

    def open_loop_loss(self, s: Tensor, idx: Tensor, dp: Tensor, ell: Tensor) -> Tensor:
        B, T, _ = s.shape
        warm = min(self.warm, T - 1)
        h, z = self.initial(B, s.device)
        e = self.encode(s[:, :warm])
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        for t in range(warm):
            if t > 0:
                h = self.cell.step(h, z, a[:, t - 1])
            mu_q, _ = self.cell.post_dist(h, e[:, t])
            z = mu_q
        preds = []
        for t in range(warm, T):
            h = self.cell.step(h, z, a[:, t - 1])
            mu_p, _ = self.cell.prior_dist(h)
            z = mu_p
            preds.append(self.decode(self.dec_pred, h, z))
        pred = torch.stack(preds, 1)
        el = ell.view(B, 1)
        P_true = self.positions(s[:, warm:], el)
        if self.repr == "chain":
            P_pred = chain_to_positions(torch.cat([P_true[..., 0, :], pred[..., 3:]], -1), el, self.n_nodes)
        else:
            P_pred = self.positions(pred, el)
        return F.mse_loss(P_pred[..., 1:-1, :], P_true[..., 1:-1, :])

    @torch.no_grad()
    def warmup(self, s: Tensor, idx: Tensor, dp: Tensor, sample: bool = False) -> tuple[Tensor, Tensor]:
        B, n, _ = s.shape
        h, z = self.initial(B, s.device)
        e = self.encode(s)
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, n, -1)
        for t in range(n):
            if t > 0:
                h = self.cell.step(h, z, a[:, t - 1])
            mu_q, sd_q = self.cell.post_dist(h, e[:, t])
            z = mu_q + sd_q * torch.randn_like(sd_q) if sample else mu_q
        return h, z

    @torch.no_grad()
    def imagine(self, h: Tensor, z: Tensor, idx: Tensor, dp: Tensor, sample: bool = False) -> Tensor:
        B, T = idx.shape[:2]
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        out = []
        for t in range(T):
            h = self.cell.step(h, z, a[:, t])
            mu_p, sd_p = self.cell.prior_dist(h)
            z = mu_p + sd_p * torch.randn_like(sd_p) if sample else mu_p
            out.append(self.decode(self.dec_pred, h, z))
        return torch.stack(out, 1)
