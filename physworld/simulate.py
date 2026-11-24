from __future__ import annotations
import numpy as np
import torch


RADIUS = 0.003


def catmull_rom(P: np.ndarray, n_per_seg: int = 200) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, knots = [], []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-9
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-9
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-9
        t = np.linspace(t1, t2, n_per_seg, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        knots.append(i * n_per_seg)
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    D = np.vstack(out + [P[-1:]])
    knots.append(len(D) - 1)
    return D, np.array(knots)


def resample(D: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = int(np.floor(s[-1] / ds)) + 1
    q = np.linspace(0, s[-1], n)
    x = np.interp(q, s, D[:, 0]); y = np.interp(q, s, D[:, 1])
    return np.stack([x, y], 1), q, s


def densify_track(nodes: np.ndarray, ds: float = 0.003) -> np.ndarray:
    D, _ = catmull_rom(nodes, 50)
    return resample(D, ds)[0]


def build_cable(nodes0: np.ndarray, dx: float = 0.003, radius: float = RADIUS):
    dp = dx / 2
    D, knots = catmull_rom(nodes0, 200)
    C, q, s_dense = resample(D, dp)
    s_nodes = s_dense[knots]
    tang = np.gradient(C, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    k = np.arange(-int(np.ceil(radius / dp)), int(np.ceil(radius / dp))) + 0.5
    off = [(a * dp, b * dp) for a in k for b in k if (a * dp) ** 2 + (b * dp) ** 2 <= radius ** 2 + 1e-12]
    off = np.array(off)
    zc = radius
    X = (C[:, None, :] + off[None, :, 0:1] * nrm[:, None, :])
    Z = zc + np.broadcast_to(off[None, :, 1], X.shape[:2])
    pos = np.concatenate([X, Z[..., None]], -1).reshape(-1, 3)
    S = len(C)
    slice_id = np.repeat(np.arange(S), len(off))
    ds_eff = q[-1] / (S - 1)
    vol = np.full(len(pos), np.pi * radius ** 2 * ds_eff / len(off))
    node_slices = []
    for j, sj in enumerate(s_nodes):
        i = int(np.argmin(np.abs(q - sj)))
        if j == 0:
            sl = [0, 1]
        elif j == len(s_nodes) - 1:
            sl = [S - 2, S - 1]
        else:
            sl = [i - 1, i, i + 1]
        node_slices.append(sl)
    return {"pos": pos.astype(np.float32), "slice": slice_id, "arc": q[slice_id], "vol": vol.astype(np.float32),
            "n_slices": S, "n_sec": len(off), "node_slices": node_slices, "s_nodes": s_nodes, "zc": zc,
            "length": float(q[-1])}


class SceneParticles:

    def __init__(self, cables: list[dict], device):
        self.cables = cables
        self.G = len(cables)
        pos, gid, sl, arc, vol = [], [], [], [], []
        self.offsets = [0]; self.slice_off = [0]
        for g, c in enumerate(cables):
            pos.append(c["pos"]); gid.append(np.full(len(c["pos"]), g)); sl.append(c["slice"] + self.slice_off[-1])
            arc.append(c["arc"] / c["length"]); vol.append(c["vol"])
            self.offsets.append(self.offsets[-1] + len(c["pos"])); self.slice_off.append(self.slice_off[-1] + c["n_slices"])
        perm = np.arange(self.offsets[-1])
        self.perm = perm
        self.pos0 = torch.tensor(np.concatenate(pos)[perm], device=device)
        self.gid = torch.tensor(np.concatenate(gid)[perm], dtype=torch.int32, device=device)
        self.slice = torch.tensor(np.concatenate(sl)[perm], dtype=torch.long, device=device)
        self.arc = torch.tensor(np.concatenate(arc)[perm], dtype=torch.float32, device=device)
        self.vol = torch.tensor(np.concatenate(vol)[perm], device=device)
        self.n_slices = self.slice_off[-1]
        cnt = torch.zeros(self.n_slices, device=device).index_add_(0, self.slice, torch.ones_like(self.vol))
        self.slice_cnt = cnt
        ns = torch.full((self.G, 16, 3), -1, dtype=torch.long)
        for g, c in enumerate(cables):
            for j, sls in enumerate(c["node_slices"]):
                for q, s_ in enumerate(sls):
                    ns[g, j, q] = s_ + self.slice_off[g]
        self.node_slices = ns.to(device)
        self.slice_gid = torch.tensor(np.concatenate([np.full(c["n_slices"], g) for g, c in enumerate(cables)]),
                                      device=device)

    def slice_centroids(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.zeros(self.n_slices, 3, device=x.device, dtype=x.dtype).index_add_(0, self.slice, x)
        return s / self.slice_cnt[:, None]

    def nodes(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.slice_centroids(x)
        idx = self.node_slices.clamp_min(0)
        m = (self.node_slices >= 0).float()[..., None]
        return (sc[idx] * m).sum(2) / m.sum(2)


import math
from dataclasses import dataclass
import warp as wp


BS = wp.constant(4)


BS3 = wp.constant(64)


@wp.func
def nidx(bmap: wp.array4d(dtype=wp.int32), g: int, i: int, j: int, k: int) -> int:
    if i < 0 or j < 0 or k < 0:
        return -1
    bi = i // BS
    bj = j // BS
    bk = k // BS
    if bi >= bmap.shape[1] or bj >= bmap.shape[2] or bk >= bmap.shape[3]:
        return -1
    b = bmap[g, bi, bj, bk]
    if b < 0:
        return -1
    return b * BS3 + (i - bi * BS) * 16 + (j - bj * BS) * 4 + (k - bk * BS)


@wp.func
def qweights(fx: wp.vec3) -> wp.mat33:
    a = wp.vec3(1.5) - fx
    b = fx - wp.vec3(1.0)
    c = fx - wp.vec3(0.5)
    w0 = 0.5 * wp.cw_mul(a, a)
    w1 = wp.vec3(0.75) - wp.cw_mul(b, b)
    w2 = 0.5 * wp.cw_mul(c, c)
    return wp.mat33(w0[0], w0[1], w0[2], w1[0], w1[1], w1[2], w2[0], w2[1], w2[2])


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
        b: wp.array(dtype=wp.mat33), gid: wp.array(dtype=wp.int32),
        E: wp.array(dtype=float), nu: wp.array(dtype=float), rho: wp.array(dtype=float),
        rhoi: wp.array(dtype=float), muf: wp.array(dtype=float), sy: wp.array(dtype=float),
        vol: wp.array(dtype=float), grav: float,
        origin: wp.vec3, inv_dx: float, dt: float, bmap: wp.array4d(dtype=wp.int32), floor_k: int,
        gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gmu: wp.array(dtype=float),
        b_out: wp.array(dtype=wp.mat33), missed: wp.array(dtype=wp.int32)):
    p = wp.tid()
    g = gid[p]
    Cp = C[p]
    I = wp.identity(n=3, dtype=float)
    f = I + dt * Cp
    bt = f * b[p] * wp.transpose(f)
    J2 = wp.determinant(bt)
    J = wp.sqrt(J2)
    jm = wp.pow(J, -2.0 / 3.0)
    bb = bt * jm
    Ie = (bb[0, 0] + bb[1, 1] + bb[2, 2]) / 3.0
    Ee = E[p]
    n = nu[p]
    mu_l = Ee / (2.0 * (1.0 + n))
    kap = Ee / (3.0 * (1.0 - 2.0 * n))
    s = mu_l * (bb - Ie * I)
    ns = wp.sqrt(wp.ddot(s, s) + 1.0e-20)
    r = 0.816496580927726 * sy[p]
    if ns > r:
        s = s * (r / ns)
        bb = s / mu_l + Ie * I
    b_out[p] = bb / jm
    tau = s + (0.5 * kap * (J2 - 1.0)) * I
    m = rhoi[p] * vol[p]
    mg = rho[p] * vol[p]
    dx = 1.0 / inv_dx
    affine = (-dt * vol[p] * 4.0 * inv_dx * inv_dx) * tau + m * Cp
    xp = (x[p] - origin) * inv_dx
    bi = int(xp[0] - 0.5)
    bj = int(xp[1] - 0.5)
    bk = int(xp[2] - 0.5)
    fx = xp - wp.vec3(float(bi), float(bj), float(bk))
    W = qweights(fx)
    mv = m * v[p] + wp.vec3(0.0, 0.0, -dt * grav * mg)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = nidx(bmap, g, bi + i, bj + j, bk + k)
                if idx < 0:
                    wp.atomic_add(missed, 0, 1)
                else:
                    dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                    w = W[i, 0] * W[j, 1] * W[k, 2]
                    wp.atomic_add(gmv, idx, w * (mv + affine * dpos))
                    wp.atomic_add(gm, idx, w * m)
                    if bk + k <= floor_k:
                        wp.atomic_add(gmu, idx, w * m * muf[p])


@wp.kernel
def grid_op(gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gmu: wp.array(dtype=float),
            bcoord: wp.array(dtype=wp.vec4i), origin: wp.vec3, dx: float, dt: float, grav: float,
            floor_k: int, gp0: wp.array(dtype=wp.vec3), gvel: wp.array(dtype=wp.vec3),
            gom: wp.array(dtype=float), n_grasp: int, t_arr: wp.array(dtype=float), r_grasp: float, damp: float, gv: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    t_in = t_arr[0]
    m = gm[t]
    if m <= 1.0e-12:
        gv[t] = wp.vec3(0.0)
        return
    bl = t // BS3
    l = t - bl * BS3
    bc = bcoord[bl]
    g = bc[0]
    i = bc[1] * BS + l // 16
    j = bc[2] * BS + (l // 4) % 4
    k = bc[3] * BS + l % 4
    vv = gmv[t] / m
    vv = vv * damp
    if k <= floor_k:
        vn = vv[2]
        if vn < 0.0:
            mu = gmu[t] / m
            nt = wp.sqrt(vv[0] * vv[0] + vv[1] * vv[1] + 1.0e-20)
            fac = wp.max(0.0, 1.0 + mu * vn / nt)
            vv = wp.vec3(vv[0] * fac, vv[1] * fac, 0.0)
    pos = origin + wp.vec3(float(i), float(j), float(k)) * dx
    for e in range(n_grasp):
        q = gp0[g * n_grasp + e] + gvel[g * n_grasp + e] * t_in
        rel = pos - q
        if wp.length(rel) < r_grasp:
            om = gom[g * n_grasp + e]
            vv = gvel[g * n_grasp + e] + wp.vec3(-om * rel[1], om * rel[0], 0.0)
    gv[t] = vv


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec3), gid: wp.array(dtype=wp.int32), gv: wp.array(dtype=wp.vec3),
        origin: wp.vec3, inv_dx: float, dt: float, bmap: wp.array4d(dtype=wp.int32),
        x_out: wp.array(dtype=wp.vec3), v_out: wp.array(dtype=wp.vec3), C_out: wp.array(dtype=wp.mat33),
        missed: wp.array(dtype=wp.int32)):
    p = wp.tid()
    g = gid[p]
    xp = (x[p] - origin) * inv_dx
    bi = int(xp[0] - 0.5)
    bj = int(xp[1] - 0.5)
    bk = int(xp[2] - 0.5)
    fx = xp - wp.vec3(float(bi), float(bj), float(bk))
    W = qweights(fx)
    nv = wp.vec3(0.0)
    nC = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = nidx(bmap, g, bi + i, bj + j, bk + k)
                if idx < 0:
                    wp.atomic_add(missed, 1, 1)
                else:
                    dpos = wp.vec3(float(i), float(j), float(k)) - fx
                    w = W[i, 0] * W[j, 1] * W[k, 2]
                    gvv = gv[idx]
                    nv = nv + w * gvv
                    nC = nC + (4.0 * inv_dx * w) * wp.outer(gvv, dpos)
    v_out[p] = nv
    C_out[p] = nC
    x_out[p] = x[p] + dt * nv


@wp.kernel
def advance_t(t_arr: wp.array(dtype=float), dt: float):
    t_arr[0] = t_arr[0] + dt


@dataclass
class State:
    x: wp.array
    v: wp.array
    C: wp.array
    b: wp.array

    @staticmethod
    def alloc(n, device, requires_grad=False):
        return State(wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                     wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                     wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=requires_grad),
                     wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=requires_grad))

    def copy(self, requires_grad=False):
        out = State.alloc(self.x.shape[0], self.x.device, requires_grad)
        for a, b_ in zip((out.x, out.v, out.C, out.b), (self.x, self.v, self.C, self.b)):
            wp.copy(a, b_)
        return out


class MPM:

    def __init__(self, G, n_particles, lo_xy, hi_xy, dx=0.003, nz_blocks=2, floor_k=2, dt=1e-4, r_grasp=None,
                 grav=9.81, damp=1.0, device="cuda", use_graph=True):
        self.G, self.dx, self.inv_dx, self.dt, self.floor_k = G, dx, 1.0 / dx, dt, floor_k
        self.device, self.grav, self.damp, self.P = device, grav, damp, n_particles
        self.r_grasp = 1.5 * dx if r_grasp is None else r_grasp
        lo = np.asarray(lo_xy, float) - 16 * dx
        self.origin = wp.vec3(float(lo[0]), float(lo[1]), -floor_k * dx)
        self.NBX = int(math.ceil((hi_xy[0] - lo[0]) / dx / 4)) + 5
        self.NBY = int(math.ceil((hi_xy[1] - lo[1]) / dx / 4)) + 5
        self.NBZ = nz_blocks
        self.missed = wp.zeros(2, dtype=wp.int32, device=device)
        self.use_graph = use_graph
        self.bmap_t = torch.full((G, self.NBX, self.NBY, self.NBZ), -1, dtype=torch.int32, device=device)
        self.bmap = wp.from_torch(self.bmap_t)
        self.cap = 0
        self.stA, self.stB = State.alloc(n_particles, device), State.alloc(n_particles, device)
        self.gp0 = wp.zeros(G * 2, dtype=wp.vec3, device=device)
        self.gvel = wp.zeros(G * 2, dtype=wp.vec3, device=device)
        self.gom = wp.zeros(G * 2, dtype=float, device=device)
        self.t_arr = wp.zeros(1, dtype=float, device=device)
        self.graphs = {}
        self.mats = None

    def set_gid(self, gid_wp):
        self.gid = gid_wp

    def set_mats(self, mats: dict):
        if self.mats is None:
            self.mats = {k: wp.zeros_like(v) for k, v in mats.items() if k != "gid"}
            self.mats["gid"] = self.gid
        for k, v in mats.items():
            if k != "gid":
                wp.copy(self.mats[k], v)
        self.mats_grad = mats

    def _ensure_cap(self, nb):
        if nb > self.cap:
            self.cap = int(nb * 1.3) + 16
            n = self.cap * 64
            self.grid = (wp.zeros(n, dtype=float, device=self.device), wp.zeros(n, dtype=wp.vec3, device=self.device),
                         wp.zeros(n, dtype=float, device=self.device), wp.zeros(n, dtype=wp.vec3, device=self.device))
            self.bcoord_t = torch.zeros(self.cap, 4, dtype=torch.int32, device=self.device)
            self.bcoord = wp.from_torch(self.bcoord_t, dtype=wp.vec4i)
            self.graphs = {}

    def build_blocks(self, x: torch.Tensor, gid: torch.Tensor, dil: int):
        o = torch.tensor([self.origin[0], self.origin[1]], device=x.device)
        bxy = ((x[:, :2] - o) * self.inv_dx / 4).floor().long()
        bx = bxy[:, 0].clamp(0, self.NBX - 1); by = bxy[:, 1].clamp(0, self.NBY - 1)
        mark = torch.zeros(self.G, self.NBX, self.NBY, device=x.device)
        mark[gid.long(), bx, by] = 1.0
        if dil > 0:
            mark = torch.nn.functional.max_pool2d(mark[:, None], 2 * dil + 1, 1, dil)[:, 0]
        act = mark.nonzero()
        A = act.shape[0]
        nb = A * self.NBZ
        self._ensure_cap(nb)
        self.bmap_t.fill_(-1)
        self.bcoord_t.zero_()
        for z in range(self.NBZ):
            ids = torch.arange(A, device=x.device) * self.NBZ + z
            self.bmap_t[act[:, 0], act[:, 1], act[:, 2], z] = ids.int()
            self.bcoord_t[ids, 0] = act[:, 0].int(); self.bcoord_t[ids, 1] = act[:, 1].int()
            self.bcoord_t[ids, 2] = act[:, 2].int(); self.bcoord_t[ids, 3] = z
        return nb

    def substep(self, st: State, out: State, mats: dict, grid, n_nodes, t_arr):
        gm, gmv, gmu, gv = grid
        wp.launch(p2g, dim=self.P, inputs=[st.x, st.v, st.C, st.b, mats["gid"], mats["E"], mats["nu"], mats["rho"],
                                          mats["rhoi"], mats["mu"], mats["sy"], mats["vol"], self.grav, self.origin, self.inv_dx, self.dt,
                                          self.bmap, self.floor_k, gm, gmv, gmu, out.b, self.missed], device=self.device)
        wp.launch(grid_op, dim=n_nodes, inputs=[gm, gmv, gmu, self.bcoord, self.origin, self.dx, self.dt, self.grav,
                                               self.floor_k, self.gp0, self.gvel, self.gom, 2, t_arr, self.r_grasp, self.damp,
                                               gv], device=self.device)
        wp.launch(g2p, dim=self.P, inputs=[st.x, mats["gid"], gv, self.origin, self.inv_dx, self.dt, self.bmap,
                                          out.x, out.v, out.C, self.missed], device=self.device)

    def _untaped(self, n):
        cur, nxt = self.stA, self.stB
        for _ in range(n):
            for arr in self.grid:
                arr.zero_()
            self.substep(cur, nxt, self.mats, self.grid, self.cap * 64, self.t_arr)
            wp.launch(advance_t, dim=1, inputs=[self.t_arr, self.dt], device=self.device)
            cur, nxt = nxt, cur

    def _set_drive(self, gp0, gvel, gom):
        wp.copy(self.gp0, wp.from_torch(gp0.contiguous(), dtype=wp.vec3))
        wp.copy(self.gvel, wp.from_torch(gvel.contiguous(), dtype=wp.vec3))
        wp.copy(self.gom, wp.from_torch(gom.float().contiguous()))

    def _run_graph(self, n):
        if not self.use_graph:
            self._untaped(n); return
        key = (n, float(self.dt))
        if key not in self.graphs:
            t_keep = wp.clone(self.t_arr)
            wp.capture_begin(device=self.device)
            try:
                self._untaped(n)
            finally:
                self.graphs[key] = wp.capture_end(device=self.device)
            wp.copy(self.t_arr, t_keep)
        wp.capture_launch(self.graphs[key])

    def frame(self, st: State, gp0: torch.Tensor, gvel: torch.Tensor, gom: torch.Tensor, frame_s, dil, W=0,
              ckpt_every=0):
        n_sub = int(round(frame_s / self.dt))
        W = min(W, n_sub)
        if st is not self.stA:
            for a, b_ in zip((self.stA.x, self.stA.v, self.stA.C, self.stA.b), (st.x, st.v, st.C, st.b)):
                wp.copy(a, b_)
        nb = self.build_blocks(wp.to_torch(self.stA.x), wp.to_torch(self.gid), dil)
        self._set_drive(gp0, gvel, gom)
        self.t_arr.zero_()
        if ckpt_every:
            K = ckpt_every
            assert n_sub % K == 0 and K % 2 == 0, (n_sub, K)
            ck = []
            for sgm in range(n_sub // K):
                ck.append(tuple(wp.to_torch(a).clone() for a in (self.stA.x, self.stA.v, self.stA.C, self.stA.b)))
                self._run_graph(K)
            rec = {"ckpts": ck, "bmap": self.bmap_t.clone(), "bcoord": self.bcoord_t[:nb].clone(), "nb": nb,
                   "drive": (gp0.clone(), gvel.clone(), gom.clone()), "K": K, "n_sub": n_sub}
            return self.stA, rec
        n_un = n_sub - W
        if n_un > 0:
            self._run_graph(n_un)
        cur = self.stA if n_un % 2 == 0 else self.stB
        tape = None
        if W > 0:
            tape = wp.Tape()
            with tape:
                for k in range(n_un, n_sub):
                    out = State.alloc(self.P, self.device, requires_grad=True)
                    g2 = self.alloc_grid(nb, True)
                    ta = wp.full(1, float(k * self.dt), dtype=float, device=self.device)
                    self.substep(cur, out, self.mats_grad, g2, nb * 64, ta)
                    cur = out
        return cur, tape

    def alloc_grid(self, nb, requires_grad=False):
        n = nb * 64
        return (wp.zeros(n, dtype=float, device=self.device, requires_grad=requires_grad),
                wp.zeros(n, dtype=wp.vec3, device=self.device, requires_grad=requires_grad),
                wp.zeros(n, dtype=float, device=self.device, requires_grad=requires_grad),
                wp.zeros(n, dtype=wp.vec3, device=self.device, requires_grad=requires_grad))

    def frame_backward(self, rec, lam):
        nb, K = rec["nb"], rec["K"]
        self.bmap_t.copy_(rec["bmap"]); self.bcoord_t.zero_(); self.bcoord_t[:nb] = rec["bcoord"]
        self._set_drive(*rec["drive"])
        dts = (wp.vec3, wp.vec3, wp.mat33, wp.mat33)
        lam = list(lam)
        for sgm in reversed(range(len(rec["ckpts"]))):
            inp = State.alloc(self.P, self.device, requires_grad=True)
            for a, c, dt_ in zip((inp.x, inp.v, inp.C, inp.b), rec["ckpts"][sgm], dts):
                wp.copy(a, wp.from_torch(c.contiguous(), dtype=dt_))
            tape = wp.Tape()
            cur = inp
            with tape:
                for k in range(K):
                    out = State.alloc(self.P, self.device, requires_grad=True)
                    g2 = self.alloc_grid(nb, True)
                    ta = wp.full(1, float((sgm * K + k) * self.dt), dtype=float, device=self.device)
                    self.substep(cur, out, self.mats_grad, g2, nb * 64, ta)
                    cur = out
            grads = {}
            for a, l_, dt_ in zip((cur.x, cur.v, cur.C, cur.b), lam, dts):
                if l_ is not None:
                    grads[a] = wp.from_torch(l_.contiguous(), dtype=dt_)
            tape.backward(grads=grads)
            lam = [wp.to_torch(a.grad).clone() for a in (inp.x, inp.v, inp.C, inp.b)]
            del tape
        return lam


BOUNDS = {"lE": (math.log(1e3), math.log(3e6)), "lrho": (math.log(200.0), math.log(8000.0)),
          "lmu": (math.log(0.01), math.log(5.0)), "nu": (0.0, 0.45), "lsy": (math.log(1.0), math.log(1e7))}


RHO_PHYS = 0.044 / (math.pi * 0.003 ** 2)


def smooth_l1(d):
    a = d.abs()
    return torch.where(a < 1.0, 0.5 * a * a, a - 0.5)


def chamfer2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(a, b)
    return d.min(1).values.mean() + d.min(0).values.mean()


MS_CFL = 0.5


def rho_for_dt(E, nu, dx, dt, cfl):
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    return M * (dt / (cfl * dx)) ** 2


def drive_from_nodes(E4: torch.Tensor):
    out = (E4[:, :, 1] - E4[:, :, 0]).mean(0)
    hy = torch.atan2(out[:, 1], out[:, 0])
    v = E4[:, :, 3] - E4[:, :, 2]
    py = torch.atan2(v[..., 1], v[..., 0])
    def unw(a):
        d = torch.remainder(torch.diff(a, dim=-1) + math.pi, 2 * math.pi) - math.pi
        return torch.cat([a[..., :1], a[..., :1] + torch.cumsum(d, -1)], -1)
    hy = unw(hy); py = unw(py)
    yaw = torch.stack([hy.expand_as(py), py], -1)
    return E4[:, :, [0, 3]], yaw


class Twin:
    def __init__(self, P: np.ndarray, cables=(0, 1, 2, 3), dx=0.003, dt=1e-4, frame_s=0.25, device="cuda",
                 damp=1.0, r_grasp=None, mass_scale_dt=None):
        self.P = np.asarray(P, np.float32)[list(cables)]
        self.G, self.T = self.P.shape[0], self.P.shape[1]
        self.dx, self.dt, self.frame_s, self.device = dx, dt, frame_s, device
        self.ms_dt = mass_scale_dt
        self.cab = [build_cable(self.P[g, 0], dx) for g in range(self.G)]
        self.sp = SceneParticles(self.cab, device)
        lo = self.P.reshape(-1, 2).min(0) - 0.1; hi = self.P.reshape(-1, 2).max(0) + 0.1
        self.NP = self.sp.pos0.shape[0]
        torch.cuda.set_device(torch.device(device))
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(device)))
        self.mpm = MPM(self.G, self.NP, lo, hi, dx=dx, dt=dt, device=device, damp=damp, r_grasp=r_grasp)
        self.gid_wp = wp.from_torch(self.sp.gid)
        self.mpm.set_gid(self.gid_wp)
        self.zc = self.cab[0]["zc"]
        self.obs_dense = [[torch.tensor(densify_track(self.P[g, t]), dtype=torch.float32, device=device) for t in range(self.T)]
                          for g in range(self.G)]
        self.Ptorch = torch.tensor(self.P, device=device)
        self.drive = drive_from_nodes(torch.tensor(np.asarray(P, np.float32), device=device)[..., [0, 1, 14, 15], :])
        self.drive = (self.drive[0][list(cables)], self.drive[1][list(cables)])
        self.NP = self.sp.pos0.shape[0]

    def init_state(self) -> State:
        st = State.alloc(self.NP, self.device)
        wp.copy(st.x, wp.from_torch(self.sp.pos0.contiguous(), dtype=wp.vec3))
        b = torch.eye(3, device=self.device).repeat(self.NP, 1, 1).contiguous()
        wp.copy(st.b, wp.from_torch(b, dtype=wp.mat33))
        return st

    def per_particle(self, glob: dict, loc: dict | None):
        one = torch.ones(self.NP, device=self.device)
        lE = glob["lE"] * one; lrho = glob["lrho"] * one; lmu = glob["lmu"] * one
        if loc is not None:
            lE = lE + loc["dlE"]; lrho = lrho + loc["dlrho"]; lmu = lmu + loc["dlmu"]
        sy = torch.exp(glob["lsy"]) * one if glob.get("elastic", None) is None else torch.full_like(one, 1e30)
        E, rho, nu = torch.exp(lE), torch.exp(lrho), glob["nu"] * one
        out = {"E": E, "rho": rho, "mu": torch.exp(lmu), "nu": nu, "sy": sy, "rhoi": rho}
        if self.ms_dt is not None:
            out["rhoi"] = torch.maximum(rho, rho_for_dt(E, nu, self.dx, self.ms_dt, MS_CFL))
        return out

    def to_warp(self, pp: dict, requires_grad=False):
        m = {k: wp.from_torch(v.detach().float().contiguous().clone(), requires_grad=requires_grad) for k, v in pp.items()}
        m["vol"] = wp.from_torch(self.sp.vol.contiguous())
        m["gid"] = self.gid_wp
        return m

    def ends(self, t, ends=None):
        E = self.drive[0] if ends is None else ends[0]
        e = E[:, t]
        return torch.cat([e, torch.full_like(e[..., :1], self.zc)], -1)

    def frame_loss(self, x: torch.Tensor, t: int, target=None):
        sc = self.sp.slice_centroids(x)
        nodes = self.sp.nodes(x)[..., :2]
        tgt = self.Ptorch[:, t] if target is None else target
        cd = 0.0
        for g in range(self.G):
            cd = cd + chamfer2(sc[self.sp.slice_gid == g, :2], self.obs_dense[g][t])
        cd = cd / self.G
        tr = smooth_l1(nodes - tgt).mean()
        return cd + 0.1 * tr, nodes

    def rollout(self, mats, T=None, ends=None, grad=None, loss_cb=None, K=100, W=10, frame_s=None,
                return_x=False, acc=None, st=None):
        acc = {} if acc is None else acc
        T = self.T if T is None else T
        fs = self.frame_s if frame_s is None else frame_s
        st = self.init_state() if st is None else st
        self.mpm.set_mats(mats)
        out = [self.sp.nodes(wp.to_torch(st.x)).detach().clone()]
        xs = [wp.to_torch(st.x).clone()] if return_x else None
        self.mpm.missed.zero_()
        bad = False
        recs = []
        for t in range(T - 1):
            p0 = self.ends(t, ends); p1 = self.ends(t + 1, ends)
            vel = (p1 - p0) / fs
            Y = self.drive[1] if ends is None else ends[1]
            om = (Y[:, t + 1] - Y[:, t]) / fs
            dmax = (p1 - p0).norm(dim=-1).max().item()
            dil = int(math.ceil((dmax + 3 * self.dx) / (4 * self.dx))) + 1
            st, rec = self.mpm.frame(st, p0.reshape(-1, 3), vel.reshape(-1, 3), om.reshape(-1), fs, dil,
                                     W=W if grad == "trunc" else 0,
                                     ckpt_every=K if grad in ("frame", "full") else 0)
            x = wp.to_torch(st.x)
            if not torch.isfinite(x).all() or x[:, 2].abs().max() > 0.5:
                bad = True
                break
            if grad is not None and loss_cb is not None:
                gx = loss_cb(t + 1, x)
                if grad == "trunc" and gx is not None:
                    rec.backward(grads={st.x: wp.from_torch(gx.contiguous(), dtype=wp.vec3)})
                    self._collect(mats, acc); rec.reset()
                elif grad == "frame" and gx is not None:
                    self.mpm.frame_backward(rec, (gx, None, None, None))
                    self._collect(mats, acc)
                elif grad == "full":
                    recs.append((rec, gx))
            out.append(self.sp.nodes(x).detach().clone())
            if return_x:
                xs.append(x.clone())
        if grad == "full" and not bad:
            lam = [None, None, None, None]
            for rec, gx in reversed(recs):
                if gx is not None:
                    lam[0] = gx if lam[0] is None else lam[0] + gx
                lam = self.mpm.frame_backward(rec, lam)
            self._collect(mats, acc)
        missed = self.mpm.missed.numpy().tolist()
        return torch.stack(out), {"missed": missed, "bad": bad, "xs": xs, "state": st}

    def _collect(self, mats, acc):
        for k in acc:
            acc[k] += wp.to_torch(mats[k].grad)
            mats[k].grad.zero_()


import math
import time
import numpy as np
import torch
import warp as wp


wp.config.log_level = getattr(wp, "LOG_WARNING", None) or 30


wp.init()


WARM = 5


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def twin_metric(nodes, P):
    e = (nodes[..., :2] - P.permute(1, 0, 2, 3)).norm(dim=-1)
    e[:, :, [0, -1]] = 0.0
    return e.mean((1, 2)).cpu().numpy() * 1000


CFL = 0.5


def run_iter(tw, glob, loc, model, frac, rng, dev, need_grad=True, dt_scale=1.0):
    G = {k: v for k, v in glob.items()}
    if model == "elastic":
        G["elastic"] = True
    pp = tw.per_particle(G, loc)
    M = pp["E"] * (1 - pp["nu"]) / ((1 + pp["nu"]) * (1 - 2 * pp["nu"]))
    c_max = torch.sqrt(M / pp["rhoi"]).max().item()
    dt = dt_scale * min(1e-4, CFL * tw.dx / c_max * 1.0001)
    n_sub = int(100 * math.ceil(tw.frame_s / dt / 100))
    tw.mpm.dt = tw.frame_s / n_sub
    mats = tw.to_warp(pp, requires_grad=need_grad)
    acc = {k: torch.zeros(tw.NP, device=dev) for k in ("E", "rho", "rhoi", "mu", "nu", "sy")}
    sel = set((rng.random(tw.T) < frac).nonzero()[0].tolist()) if need_grad else set()
    tot = [0.0]
    def cb(t, x):
        xl = x.detach().clone().requires_grad_(t in sel)
        L, _ = tw.frame_loss(xl, t)
        tot[0] += L.item()
        if t not in sel:
            return None
        L.backward(); return xl.grad
    nodes, info = tw.rollout(mats, grad="frame" if need_grad else None, loss_cb=cb, acc=acc)
    if info["bad"]:
        if dt_scale > 0.3:
            log(f"    unstable at n_sub {n_sub}, retrying with dt/2")
            return run_iter(tw, glob, loc, model, frac, rng, dev, need_grad, dt_scale * 0.5)
        return float("inf"), None, nodes, info, n_sub
    if not need_grad:
        return None, None, nodes, info, n_sub
    pairs = [(pp[k], acc[k]) for k in ("E", "rho", "rhoi", "mu", "nu", "sy") if pp[k].requires_grad]
    if pairs:
        torch.autograd.backward([a for a, _ in pairs], [b for _, b in pairs])
    return tot[0], None, nodes, info, n_sub


SPEED_RANGE = (0.65, 1.4)


PAUSE_PROB = 0.0


OFFSET_MAX = 1.5


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
