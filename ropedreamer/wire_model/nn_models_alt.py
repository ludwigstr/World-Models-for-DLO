
from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class _Film(nn.Module):

    def __init__(self, n_global: int, width: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_global, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * width))
        self.width = width

    def forward(self, g: Tensor) -> tuple[Tensor, Tensor]:
        gs = self.net(g)
        scale, shift = gs[:, :self.width], gs[:, self.width:]
        return (1.0 + scale).unsqueeze(1), shift.unsqueeze(1)


class _NodeFeatures(nn.Module):

    def __init__(self, n_nodes: int, n_rest: int, width: int):
        super().__init__()
        self.n_nodes, self.n_rest = n_nodes, n_rest
        self.proj = nn.Linear(3 if n_rest else 2, width)

    def forward(self, V_in: Tensor, rest: Tensor | None) -> Tensor:
        if self.n_rest:
            pad = torch.zeros(V_in.shape[0], 1, device=V_in.device,
                              dtype=V_in.dtype)
            r = torch.cat([pad, rest, pad], dim=1).unsqueeze(-1)
            x = torch.cat([V_in, r], dim=-1)
        else:
            x = V_in
        return self.proj(x)


class _AltBase(nn.Module):

    def __init__(self, n_nodes: int, width: int, n_action: int, n_params: int,
                 n_rest: int):
        super().__init__()
        self.n_nodes, self.width = n_nodes, width
        self.n_params, self.n_rest = n_params, n_rest
        self.n_action = n_action
        self.feat = _NodeFeatures(n_nodes, n_rest, width)
        self.film = _Film(n_action + n_params, width)
        self.head = nn.Linear(width, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _prepare(self, V_in: Tensor, a_n: Tensor, p_n: Tensor):
        scal, rest = p_n[:, :self.n_params], p_n[:, self.n_params:]
        h = self.feat(V_in, rest if self.n_rest else None)
        scale, shift = self.film(torch.cat([a_n, scal], dim=-1))
        return h * scale + shift

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor) -> Tensor:
        h = self._prepare(V_in, a_n, p_n)
        return self.head(self.body(h))


class DLOBiLSTM(_AltBase):

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 2):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.lstm = nn.LSTM(width, width // 2, num_layers=layers,
                            batch_first=True, bidirectional=True)

    def body(self, h: Tensor) -> Tensor:
        out, _ = self.lstm(h)
        return out


class DLOConv1d(_AltBase):

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14,
                 dilations=(1, 2, 4, 8)):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.ReLU(),
                nn.Conv1d(width, width, 3, padding=d, dilation=d))
            for d in dilations])

    def body(self, h: Tensor) -> Tensor:
        x = h.transpose(1, 2)
        for blk in self.blocks:
            x = torch.relu(x + blk(x))
        return x.transpose(1, 2)


class DLOTransformer(_AltBase):

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 3,
                 heads: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.pos = nn.Parameter(torch.zeros(1, n_nodes, width))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def body(self, h: Tensor) -> Tensor:
        return self.enc(h + self.pos)


class DLOTransformerHist(DLOTransformer):

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                n_params: int = 4, n_rest: int = 14, layers: int = 3,
                heads: int = 4, hist_dim: int | None = None,
                hist_hidden: int = 64, z_dim: int = 16):
        super().__init__(n_nodes, width, n_action, n_params, n_rest, layers, heads)
        self.z_dim = z_dim
        self.hist_dim = hist_dim if hist_dim is not None else n_action + 2 * n_rest
        self.hist_mlp = nn.Sequential(
            nn.Linear(self.hist_dim, hist_hidden), nn.ReLU(),
            nn.Linear(hist_hidden, hist_hidden), nn.ReLU())
        self.hist_out = nn.Linear(hist_hidden, z_dim)
        self.film_hist = _Film(z_dim, width)
        nn.init.zeros_(self.film_hist.net[-1].weight)
        nn.init.zeros_(self.film_hist.net[-1].bias)

    def embed_history(self, hist: Tensor, hist_mask: Tensor) -> Tensor:
        he = self.hist_mlp(hist)
        m = hist_mask.unsqueeze(-1)
        pooled = (he * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.hist_out(pooled)

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor,
               hist: Tensor | None = None, hist_mask: Tensor | None = None) -> Tensor:
        h = self._prepare(V_in, a_n, p_n)
        if hist is not None:
            z = self.embed_history(hist, hist_mask)
            scale, shift = self.film_hist(z)
            h = h * scale + shift
        return self.head(self.body(h))


class _GNNLayer(nn.Module):

    def __init__(self, width: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width))
        self.upd = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, h: Tensor) -> Tensor:
        left = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)
        right = torch.cat([h[:, 1:], h[:, :-1]], dim=-1)
        m_fwd, m_bwd = self.msg(left), self.msg(right)
        agg = torch.zeros_like(h)
        agg[:, :-1] += m_bwd
        agg[:, 1:] += m_fwd
        return h + self.upd(torch.cat([h, agg], dim=-1))


class DLOGNN(_AltBase):

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.layers = nn.ModuleList([_GNNLayer(width) for _ in range(layers)])

    def body(self, h: Tensor) -> Tensor:
        for layer in self.layers:
            h = layer(h)
        return h


def _turning_angles(p: Tensor) -> Tensor:
    u = p[:, 1:] - p[:, :-1]
    cross = u[:, :-1, 0] * u[:, 1:, 1] - u[:, :-1, 1] * u[:, 1:, 0]
    dot = (u[:, :-1] * u[:, 1:]).sum(-1)
    return torch.atan2(cross, dot)


class DLOTransformerSeq(_AltBase):
    takes_node_hist = True

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 17, n_rest: int = 14, n_hist: int = 4,
                 layers: int = 3, heads: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.n_hist = n_hist
        self.n_kb = n_params - 3
        assert self.n_kb in (1, n_rest), (n_params, n_rest)
        n_feat = 2 + 4 + 1 + 1 + 1 + 2 * n_hist
        self.feat = nn.Linear(n_feat, width)
        self.film = _Film(n_action + 3, width)
        self.pos = nn.Parameter(torch.zeros(1, n_nodes, width))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    @staticmethod
    def _pad_ends(x: Tensor) -> Tensor:
        z = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return torch.cat([z, x, z], dim=1).unsqueeze(-1)

    def node_features(self, V_in: Tensor, p_n: Tensor,
                      hist: Tensor | None) -> Tensor:
        B, N, _ = V_in.shape
        m_n = p_n[:, 0:1]
        kb_n = p_n[:, 1:1 + self.n_kb]
        kd_kc = p_n[:, 1 + self.n_kb:3 + self.n_kb]
        rest = p_n[:, 3 + self.n_kb:3 + self.n_kb + self.n_rest]
        if self.n_kb == 1:
            kb_n = kb_n.expand(B, self.n_rest)
        e = V_in[:, 1:] - V_in[:, :-1]
        z2 = torch.zeros(B, 1, 2, device=V_in.device, dtype=V_in.dtype)
        e_prev = torch.cat([z2, e], dim=1)
        e_next = torch.cat([e, z2], dim=1)
        theta = self._pad_ends(_turning_angles(V_in))
        if hist is None:
            hist = torch.zeros(B, self.n_hist, N, 2, device=V_in.device,
                               dtype=V_in.dtype)
        h = hist.permute(0, 2, 1, 3).reshape(B, N, 2 * self.n_hist)
        x = torch.cat([V_in, e_prev, e_next, theta, self._pad_ends(rest),
                       self._pad_ends(kb_n), h], dim=-1)
        return x, torch.cat([m_n, kd_kc], dim=-1)

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor,
               hist: Tensor | None = None) -> Tensor:
        x, glob = self.node_features(V_in, p_n, hist)
        h = self.feat(x)
        scale, shift = self.film(torch.cat([a_n, glob], dim=-1))
        h = h * scale + shift
        return self.head(self.enc(h + self.pos))


ARCHITECTURES = {
    "bilstm": DLOBiLSTM,
    "conv1d": DLOConv1d,
    "transformer": DLOTransformer,
    "transformer_hist": DLOTransformerHist,
    "gnn": DLOGNN,
}


class DLOJacobian(_AltBase):

    N_CONT = 3

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 3,
                 heads: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.n_ends = max(1, n_action // 4)
        self.n_cols = self.n_ends * self.N_CONT
        self.pos = nn.Parameter(torch.zeros(1, n_nodes, width))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(width, 2 * self.n_cols)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _state_film(self, p_n: Tensor) -> tuple[Tensor, Tensor]:
        scal = p_n[:, :self.n_params]
        pad = torch.zeros(scal.shape[0], self.n_action, device=scal.device,
                          dtype=scal.dtype)
        return self.film(torch.cat([pad, scal], dim=-1))

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor) -> Tensor:
        rest = p_n[:, self.n_params:]
        h = self.feat(V_in, rest if self.n_rest else None)
        scale, shift = self._state_film(p_n)
        h = self.enc(h * scale + shift + self.pos)

        B, N = V_in.shape[0], self.n_nodes
        J = self.head(h).view(B, N, 2, self.n_cols)
        cont = torch.cat([a_n[:, c + 1:c + 4]
                          for c in range(0, a_n.shape[1], 4)], dim=-1)
        return torch.einsum("bnij,bj->bni", J, cont)


ARCHITECTURES["jacobian"] = DLOJacobian
ARCHITECTURES["transformer_seq"] = DLOTransformerSeq
