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
