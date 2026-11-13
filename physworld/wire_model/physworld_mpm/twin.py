from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from .cable import SceneParticles, build_cable, densify_track
from .mpm import MPM, State

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


def max_stable_dt(E, rho, nu, dx, cfl=0.4):
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    return cfl * dx / math.sqrt(M / rho)


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
