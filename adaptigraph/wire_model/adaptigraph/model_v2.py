import torch
from torch import nn
import torch.nn.functional as F

from .model import AdaptiGraph, SCALE, mlp


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
