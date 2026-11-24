from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


SCALE = 0.1


def mlp(d_in, d_h, d_out, n=2):
    layers, d = [], d_in
    for _ in range(n):
        layers += [nn.Linear(d, d_h), nn.ReLU()]; d = d_h
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class AdaptiGraph(nn.Module):
    def __init__(self, n_nodes=16, n_hist=3, width=128, layers=4, radius=0.12, n_material=1):
        super().__init__()
        self.n_nodes, self.n_hist, self.radius, self.L = n_nodes, n_hist, radius, layers
        d_v = 2 * n_hist + 2 + n_material + 1 + 2
        d_e = 2 + 2
        self.enc_v, self.enc_e = mlp(d_v, width, width, 1), mlp(d_e, width, width, 1)
        self.prop_e = nn.ModuleList([mlp(3 * width, width, width, 1) for _ in range(layers)])
        self.prop_v = nn.ModuleList([mlp(2 * width, width, width, 1) for _ in range(layers)])
        self.dec = mlp(width, width, 2, 1)
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)
        self.register_buffer("eef", torch.zeros(n_nodes, dtype=torch.bool)); self.eef[0] = True; self.eef[-1] = True

    def edges(self, X: Tensor):
        B, N, _ = X.shape
        d = torch.cdist(X, X)
        adj = d < self.radius
        chain = torch.zeros(N, N, dtype=torch.bool, device=X.device)
        ar = torch.arange(N - 1, device=X.device); chain[ar, ar + 1] = True; chain[ar + 1, ar] = True
        adj = adj | chain
        adj &= ~torch.eye(N, dtype=torch.bool, device=X.device)
        return adj

    def forward(self, hist: Tensor, X: Tensor, phi: Tensor, dp_eef: Tensor, material: Tensor | None = None) -> Tensor:
        B, N, _ = X.shape
        hd = ((hist - X.unsqueeze(1)) / SCALE).permute(0, 2, 1, 3).reshape(B, N, -1)
        typ = F.one_hot(self.eef.long(), 2).to(X.dtype).unsqueeze(0).expand(B, N, 2)
        mat = torch.ones(B, N, 1, device=X.device, dtype=X.dtype) if material is None else material
        ph = phi.view(B, N, 1) if phi.dim() == 2 else phi.view(B, 1, 1).expand(B, N, 1)
        v = torch.cat([hd, typ, mat, ph, dp_eef / SCALE], -1)
        adj = self.edges(X)
        rel = (X.unsqueeze(2) - X.unsqueeze(1)) / SCALE
        same = (self.eef.unsqueeze(0) == self.eef.unsqueeze(1)).to(X.dtype)
        etype = torch.stack([same, 1 - same], -1).unsqueeze(0).expand(B, N, N, 2)
        he = self.enc_e(torch.cat([rel, etype], -1)) * adj.unsqueeze(-1)
        hv = self.enc_v(v)
        for l in range(self.L):
            he = he + self.prop_e[l](torch.cat([he, hv.unsqueeze(2).expand(-1, -1, N, -1), hv.unsqueeze(1).expand(-1, N, -1, -1)], -1)) * adj.unsqueeze(-1)
            hv = hv + self.prop_v[l](torch.cat([hv, he.sum(2)], -1))
        dx = self.dec(hv) * SCALE
        out = X + dx
        return torch.where(self.eef.view(1, N, 1), X + dp_eef, out)


from torch import nn


class AdaptiGraphV2(AdaptiGraph):
    def __init__(self, n_nodes=16, n_hist=3, width=128, layers=4, radius=0.12, n_extra=0):
        super().__init__(n_nodes, n_hist, width, layers, radius)
        self.n_extra = n_extra
        if n_extra:
            self.enc_v = mlp(2 * n_hist + 2 + 1 + 1 + 2 + n_extra, width, width, 1)

    def forward(self, hist, X, phi, dp_eef, extra=None):
        B, N, _ = X.shape
        hd = ((hist - X.unsqueeze(1)) / SCALE).permute(0, 2, 1, 3).reshape(B, N, -1)
        typ = F.one_hot(self.eef.long(), 2).to(X.dtype).unsqueeze(0).expand(B, N, 2)
        mat = torch.ones(B, N, 1, device=X.device, dtype=X.dtype)
        ph = phi.view(B, 1, 1).expand(B, N, 1)
        parts = [hd, typ, mat, ph, dp_eef / SCALE] + ([extra] if self.n_extra else [])
        v = torch.cat(parts, -1)
        adj = self.edges(X)
        rel = (X.unsqueeze(2) - X.unsqueeze(1)) / SCALE
        same = (self.eef.unsqueeze(0) == self.eef.unsqueeze(1)).to(X.dtype)
        etype = torch.stack([same, 1 - same], -1).unsqueeze(0).expand(B, N, N, 2)
        he = self.enc_e(torch.cat([rel, etype], -1)) * adj.unsqueeze(-1)
        hv = self.enc_v(v)
        for l in range(self.L):
            he = he + self.prop_e[l](torch.cat([he, hv.unsqueeze(2).expand(-1, -1, N, -1), hv.unsqueeze(1).expand(-1, N, -1, -1)], -1)) * adj.unsqueeze(-1)
            hv = hv + self.prop_v[l](torch.cat([hv, he.sum(2)], -1))
        out = X + self.dec(hv) * SCALE
        return torch.where(self.eef.view(1, N, 1), X + dp_eef, out)


def turning(P):
    u = P[..., 1:, :] - P[..., :-1, :]
    a, b = u[..., :-1, :], u[..., 1:, :]
    return torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1))


def end_actions(P0, P1):
    B, N, _ = P0.shape
    act = torch.zeros(B, N, 3, device=P0.device, dtype=P0.dtype)

    def ang(P, i, j):
        e = P[:, j] - P[:, i]
        return torch.atan2(e[:, 1], e[:, 0])
    for node, (i, j) in ((0, (0, 1)), (N - 1, (N - 2, N - 1))):
        act[:, node, :2] = P1[:, node] - P0[:, node]
        dth = ang(P1, i, j) - ang(P0, i, j)
        act[:, node, 2] = torch.atan2(torch.sin(dth), torch.cos(dth))
    return act


from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent


WARM = 5


def load_tracks(dev, tracks="../data/tracks"):
    clips = sorted(p.stem for p in (ROOT / tracks).glob("*.npz"))
    cab, Tn = [], []
    for c in clips:
        z = np.load(ROOT / tracks / f"{c}.npz")
        for k in range(4):
            cab.append(z[f"c{k}"].astype(np.float32))
            Tn.append(len(cab[-1]))
    Tm = max(Tn)
    P = np.stack([np.concatenate([x, np.repeat(x[-1:], Tm - len(x), 0)]) for x in cab])
    return clips, torch.tensor(P, device=dev), torch.tensor(Tn, device=dev)


def features(flags, X, Y, rest):
    out = []
    if flags.get("rest"):
        out.append(rest[..., None])
    if flags.get("rot"):
        out.append(end_actions(X, Y)[..., 2:])
    return torch.cat(out, -1) if out else None


def dp_of(X, Y):
    d = torch.zeros_like(X)
    d[:, 0], d[:, -1] = Y[:, 0] - X[:, 0], Y[:, -1] - X[:, -1]
    return d


@torch.no_grad()
def score(m, flags, P, Tn, phi_fixed=None):
    m.eval()
    C, Tm = P.shape[:2]
    H = m.n_hist
    rest = torch.zeros(C, 16, device=P.device)
    rest[:, 1:-1] = turning(P[:, 0]).clamp(-0.6, 0.6)
    if phi_fixed is None:
        grid = torch.linspace(-0.2, 1.2, 29, device=P.device)
        cost = torch.zeros(C, len(grid), device=P.device)
        for t in range(WARM - 1):
            idx = [max(0, t - H + j) for j in range(H)]
            X, Y = P[:, t], P[:, t + 1]
            for g, ph in enumerate(grid):
                pred = m(P[:, idx].contiguous(), X, ph.expand(C), dp_of(X, Y), features(flags, X, Y, rest))
                cost[:, g] += (pred[:, 1:-1] - Y[:, 1:-1]).norm(dim=-1).mean(-1)
        phi = grid[cost.argmin(1)]
    else:
        phi = torch.full((C,), float(phi_fixed), device=P.device)
    frames = [P[:, t] for t in range(WARM)]
    total = torch.zeros(C, device=P.device)
    for t in range(WARM - 1, Tm - 1):
        idx = [max(0, t - H + j) for j in range(H)]
        hist = torch.stack([frames[i] for i in idx], 1)
        X, A, B = frames[t], P[:, t], P[:, t + 1]
        pred = m(hist, X, phi, dp_of(A, B), features(flags, A, B, rest))
        pred = pred.clone()
        pred[:, 0], pred[:, -1] = B[:, 0], B[:, -1]
        frames.append(pred)
        total += ((t + 1) < Tn) * (pred - B).norm(dim=-1).mean(-1) * 1000
    return (total / (Tn - WARM)), phi
