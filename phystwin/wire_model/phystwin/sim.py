from __future__ import annotations

import torch
from torch import Tensor

G = 9.81


def build_topology(P0: Tensor, radius_segments: float, k_max: int) -> Tensor:
    C, N, _ = P0.shape
    seg = (P0[:, 1:] - P0[:, :-1]).norm(dim=-1).mean(-1)
    d = torch.cdist(P0, P0); eye = torch.eye(N, dtype=torch.bool, device=P0.device)
    d = d.masked_fill(eye, float("inf"))
    adj = d < (radius_segments * seg).view(C, 1, 1)
    keep = torch.zeros_like(adj); keep.scatter_(2, d.argsort(dim=2)[:, :, :k_max], True)
    adj = adj & keep
    ar = torch.arange(N - 1, device=P0.device); adj[:, ar, ar + 1] = True; adj[:, ar + 1, ar] = True
    return adj | adj.transpose(1, 2)


class SpringMassTwin:
    def __init__(self, P0: Tensor, adj: Tensor, mass_per_node: float = 0.002, dt: float = 1e-3, frame_s: float = 0.25, eps_v: float = 0.02):
        self.C, self.N = P0.shape[:2]; self.m, self.dt, self.eps_v = mass_per_node, dt, eps_v
        self.sub = int(round(frame_s / dt)); self.adj = adj.float(); self.L0 = torch.cdist(P0, P0)

    def expand(self, pop: int):
        return self.adj.repeat(pop, 1, 1), self.L0.repeat(pop, 1, 1)

    def rollout(self, X0: Tensor, V0: Tensor, ends: Tensor, k: Tensor, gamma: Tensor, delta: Tensor, mu: Tensor) -> Tensor:
        B, T = ends.shape[:2]; pop = B // self.C; adj, L0 = self.expand(pop)
        kk = k * adj if k.shape[-1] == self.N else k.view(B, 1, 1) * adj
        g, dl, mu = gamma.view(B, 1, 1), delta.view(B, 1, 1), mu.view(B, 1, 1)
        X, V = X0.clone(), V0.clone(); out = [X]
        drag = 1.0 / (1.0 + self.dt * dl / self.m)
        for t in range(1, T):
            e0, e1 = X[:, 0], X[:, -1]; d0 = (ends[:, t, 0] - e0) / self.sub; d1 = (ends[:, t, 1] - e1) / self.sub
            for s in range(self.sub):
                d = X.unsqueeze(1) - X.unsqueeze(2)
                dist = d.norm(dim=-1).clamp_min(1e-6)
                f_spring = (kk * (dist - L0) / dist).unsqueeze(-1) * d
                f_dash = (g * adj).unsqueeze(-1) * (V.unsqueeze(1) - V.unsqueeze(2))
                F = (f_spring + f_dash).sum(2)
                V = (V + self.dt * F / self.m) * drag
                V = V * (1.0 - mu * G * self.dt / (V.norm(dim=-1, keepdim=True) + self.eps_v)).clamp_min(0.0)
                X = X + self.dt * V
                X = torch.cat([(e0 + d0 * (s + 1)).unsqueeze(1), X[:, 1:-1], (e1 + d1 * (s + 1)).unsqueeze(1)], 1)
                V = torch.cat([(d0 / self.dt).unsqueeze(1), V[:, 1:-1], (d1 / self.dt).unsqueeze(1)], 1)
            out.append(X)
        return torch.stack(out, 1)
