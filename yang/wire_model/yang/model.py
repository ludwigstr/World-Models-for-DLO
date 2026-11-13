import torch
import torch.nn as nn


def mlp(n_in, n_hidden, n_out, n_layers=4, out_act=True):
    layers, d = [], n_in
    for i in range(n_layers):
        last = i == n_layers - 1
        layers.append(nn.Linear(d, n_out if last else n_hidden))
        if not last or out_act:
            layers.append(nn.ReLU())
        d = n_hidden
    return nn.Sequential(*layers)


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


class INBiLSTM(nn.Module):
    def __init__(self, n_nodes=16, hidden=150, lstm_layers=2, use_rest=True):
        super().__init__()
        self.N, self.use_rest = n_nodes, use_rest
        n_static = 2 if use_rest else 1
        self.n_node = 2 + 3 + n_static
        n_edge = 2 + 2 + 2 * 3 + 2 * n_static
        self.f_e = mlp(n_edge, hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.enc = mlp(self.n_node, hidden, hidden)
        self.pred = mlp(hidden + 2 * hidden, hidden, 2, out_act=False)
        for name, dim in (("s_pos", 1), ("s_vel", 1), ("s_act", 3), ("s_out", 1)):
            self.register_buffer(name, torch.ones(dim))

    def forward(self, p, v, act, rest):
        B, N, _ = p.shape
        flag = torch.zeros(B, N, 1, device=p.device, dtype=p.dtype)
        flag[:, 0] = flag[:, -1] = 1.0
        static = torch.cat([rest[..., None], flag], -1) if self.use_rest else flag
        pn, vn, an = p / self.s_pos, v / self.s_vel, act / self.s_act
        r = torch.arange(N, device=p.device)
        recv = torch.cat([r[1:], r[:-1]]); send = torch.cat([r[:-1], r[1:]])
        e = torch.cat([pn[:, recv] - pn[:, send], vn[:, recv] - vn[:, send], an[:, recv], an[:, send],
                       static[:, recv], static[:, send]], -1)
        eff = self.f_e(e)
        k = torch.zeros(B, N, eff.shape[-1], device=p.device, dtype=p.dtype).index_add_(1, recv, eff)
        h, _ = self.lstm(k)
        x = torch.cat([vn, an, static], -1)
        out = self.pred(torch.cat([self.enc(x), h], -1))
        return out * self.s_out
