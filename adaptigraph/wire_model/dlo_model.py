
from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class DLOParams:

    m: Tensor | float
    k_b: Tensor | float
    k_d: Tensor | float
    k_c: Tensor | float = 0.0
    k_s: Tensor | float = 1.0e5
    length: Tensor | float = 0.5
    rest_angle: Tensor | float = 0.0

    k_yield: Tensor | float | None = None
    k_creep: Tensor | float = 0.0

    def to_batched(self, batch: int, device, dtype=torch.float32,
                   n_nodes: int | None = None) -> "DLOParams":

        def prep(x, per_node=None):
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if per_node is not None:
                if t.ndim == 2:
                    return t.reshape(batch, t.shape[-1], 1)
                if t.ndim == 1 and t.shape[0] == per_node and batch != per_node:
                    return t.unsqueeze(0).expand(batch, -1).reshape(batch, per_node, 1)
            if t.ndim == 0:
                t = t.expand(batch)
            return t.reshape(batch, 1, 1)

        def prep_angle(x):
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.ndim == 2:
                return t
            if n_nodes is None:
                raise ValueError("n_nodes needed to broadcast a scalar/1-D "
                                 "rest_angle to (B, N-2)")
            n_int = n_nodes - 2
            if t.ndim == 0:
                return t.expand(batch, n_int)
            if t.shape[0] != n_int:
                raise ValueError(f"rest_angle has {t.shape[0]} entries, "
                                 f"expected {n_int} interior nodes")
            return t.unsqueeze(0).expand(batch, -1)

        return DLOParams(
            m=prep(self.m),
            k_b=prep(self.k_b, per_node=None if n_nodes is None else n_nodes - 2),
            k_d=prep(self.k_d, per_node=n_nodes),
            k_c=prep(self.k_c),
            k_s=prep(self.k_s),
            length=prep(self.length),
            rest_angle=prep_angle(self.rest_angle),
            k_yield=None if self.k_yield is None else prep(self.k_yield),
            k_creep=prep(self.k_creep),
        )


@dataclass
class NodeConstraints:

    mask: Tensor
    target: Tensor


class DLOSimulator:

    def __init__(
        self,
        n_nodes: int,
        dt: float = 4.0e-5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        symplectic: bool = True,
        damping_mode: str = "implicit",
        eps: float = 1.0e-9,
    ) -> None:
        if n_nodes < 3:
            raise ValueError("n_nodes must be >= 3 for bending to be defined")
        if damping_mode not in ("implicit", "explicit"):
            raise ValueError("damping_mode must be 'implicit' or 'explicit'")
        self.n_nodes = n_nodes
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = dtype
        self.symplectic = symplectic
        self.damping_mode = damping_mode
        self.eps = eps


    def axial_forces(self, p: Tensor, k_s: Tensor, l0: Tensor) -> Tensor:
        d = p[:, 1:] - p[:, :-1]
        length = d.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        direction = d / length
        f_edge = k_s * (length - l0) * direction

        f = torch.zeros_like(p)
        f[:, :-1] += f_edge
        f[:, 1:] -= f_edge
        return f

    def turning_angles(self, p: Tensor) -> Tensor:
        u = p[:, 1:] - p[:, :-1]
        u_prev, u_next = u[:, :-1], u[:, 1:]
        cross = u_prev[..., 0] * u_next[..., 1] - u_prev[..., 1] * u_next[..., 0]
        dot = (u_prev * u_next).sum(-1)
        return torch.atan2(cross, dot)

    def bending_energy(self, p: Tensor, k_b: Tensor,
                       rest_angle: Tensor | float = 0.0) -> Tensor:
        d = self.turning_angles(p) - rest_angle
        d = torch.atan2(torch.sin(d), torch.cos(d))
        return (0.5 * k_b.squeeze(-1) * d.pow(2)).sum()

    def bending_forces(self, p: Tensor, k_b: Tensor,
                       rest_angle: Tensor | float = 0.0) -> Tensor:
        with torch.enable_grad():
            p_ = p.detach().requires_grad_(True)
            energy = self.bending_energy(p_, k_b, rest_angle)
            (grad,) = torch.autograd.grad(energy, p_)
        return -grad

    def elastic_forces(self, p: Tensor, prm: DLOParams) -> Tensor:
        l0 = prm.length / (self.n_nodes - 1)
        return (self.axial_forces(p, prm.k_s, l0)
                + self.bending_forces(p, prm.k_b, prm.rest_angle))

    def total_forces(self, p: Tensor, v: Tensor, prm: DLOParams) -> Tensor:
        return self.elastic_forces(p, prm) - prm.k_d * v


    def _apply_coulomb(self, v_new: Tensor, m_node: Tensor, prm: DLOParams) -> Tensor:
        k_c = getattr(prm, "k_c", 0.0)
        if not torch.is_tensor(k_c):
            if k_c == 0.0:
                return v_new
            k_c = torch.as_tensor(k_c, dtype=v_new.dtype, device=v_new.device)
        elif not bool((k_c != 0).any()):
            return v_new
        dv = self.dt * k_c / m_node
        speed = v_new.norm(dim=-1, keepdim=True)
        scale = (1.0 - dv / speed.clamp_min(self.eps)).clamp_min(0.0)
        return v_new * scale

    def step(
        self,
        p: Tensor,
        v: Tensor,
        prm: DLOParams,
        constraints: NodeConstraints | None = None,
        f_ext: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        m_node = prm.m / self.n_nodes
        f_el = self.elastic_forces(p, prm)
        if f_ext is not None:
            f_el = f_el + f_ext

        if self.damping_mode == "implicit":
            v_new = (m_node * v + self.dt * f_el) / (m_node + self.dt * prm.k_d)
        else:
            v_new = v + self.dt * (f_el - prm.k_d * v) / m_node

        v_new = self._apply_coulomb(v_new, m_node, prm)

        if self.symplectic:
            p_new = p + self.dt * v_new
        else:
            p_new = p + self.dt * v

        if constraints is not None:
            mask = constraints.mask.unsqueeze(-1)
            v_driven = (constraints.target - p) / self.dt
            p_new = torch.where(mask, constraints.target, p_new)
            v_new = torch.where(mask, v_driven, v_new)

        self._apply_plasticity(p_new, prm)
        return p_new, v_new

    def _apply_plasticity(self, p_new: Tensor, prm: DLOParams) -> None:
        if prm.k_yield is None:
            return
        k_yield = torch.as_tensor(prm.k_yield)
        k_yield = k_yield.squeeze(-1) if k_yield.ndim == 3 else k_yield
        k_creep = torch.as_tensor(prm.k_creep)
        k_creep = k_creep.squeeze(-1) if k_creep.ndim == 3 else k_creep
        theta = self.turning_angles(p_new)
        d = theta - prm.rest_angle
        d = torch.atan2(torch.sin(d), torch.cos(d))
        excess = torch.sign(d) * (d.abs() - k_yield).clamp_min(0.0)
        prm.rest_angle = prm.rest_angle + self.dt * k_creep * excess

    def stability_dt(self, prm: DLOParams) -> Tensor:
        m_node = prm.m / self.n_nodes
        dt_axial = 2.0 / (prm.k_s / m_node).sqrt()
        if self.damping_mode == "implicit":
            return dt_axial.reshape(-1)
        dt_damp = 2.0 * m_node / prm.k_d.clamp_min(self.eps)
        return torch.minimum(dt_axial, dt_damp).reshape(-1)
