from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class Normalizer:

    action_lo: Tensor
    action_hi: Tensor
    param_lo: Tensor
    param_hi: Tensor
    pos_scale: float = 1.0
    param_log: Tensor | None = None
    rest_scale: float = 0.35

    def _fwd(self, p: Tensor) -> Tensor:
        return torch.where(self.param_log, p.clamp_min(1e-12).log(), p)

    def _lo_hi(self) -> tuple[Tensor, Tensor]:
        return self._fwd(self.param_lo), self._fwd(self.param_hi)

    def norm_action(self, a: Tensor) -> Tensor:
        return 2 * (a - self.action_lo) / (self.action_hi - self.action_lo) - 1

    def norm_params(self, p: Tensor) -> Tensor:
        if self.param_log is None:
            return (p - self.param_lo) / (self.param_hi - self.param_lo)
        lo, hi = self._lo_hi()
        return (self._fwd(p) - lo) / (hi - lo)

    def denorm_params(self, p_n: Tensor) -> Tensor:
        if self.param_log is None:
            return p_n * (self.param_hi - self.param_lo) + self.param_lo
        lo, hi = self._lo_hi()
        raw = p_n * (hi - lo) + lo
        return torch.where(self.param_log, raw.exp(), raw)

    def norm_rest(self, r: Tensor) -> Tensor:
        return r / self.rest_scale

    def to(self, device) -> "Normalizer":
        return Normalizer(
            self.action_lo.to(device), self.action_hi.to(device),
            self.param_lo.to(device), self.param_hi.to(device), self.pos_scale,
            None if self.param_log is None else self.param_log.to(device),
            self.rest_scale)


def canonicalize(V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    centre = V.mean(dim=1, keepdim=True)
    Vc = V - centre
    chord = Vc[:, -1] - Vc[:, 0]
    angle = torch.atan2(chord[:, 1], chord[:, 0])
    return rotate(Vc, -angle), centre, angle


def rotate(V: Tensor, angle: Tensor) -> Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    rot = torch.stack([torch.stack([c, -s], -1),
                       torch.stack([s, c], -1)], dim=-2)
    return torch.einsum("bij,bkj->bki", rot, V)


def rotate_vec(v: Tensor, angle: Tensor) -> Tensor:
    return rotate(v.unsqueeze(1), angle).squeeze(1)


def _mlp(sizes: list[int], out_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if out_act or i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DLONetwork(nn.Module):

    def __init__(
        self,
        n_nodes: int = 16,
        width: int = 256,
        n_action: int = 8,
        n_params: int = 4,
        n_rest: int = 0,
        combine: str = "film",
    ) -> None:
        super().__init__()
        if combine not in ("film", "concat"):
            raise ValueError("combine must be 'film' or 'concat'")
        self.n_nodes, self.width, self.combine = n_nodes, width, combine
        self.n_params = n_params
        self.n_action = n_action

        self.n_rest = n_rest
        self.dlo_block = _mlp([n_nodes * 2, width, width, width])
        self.action_block = _mlp([n_action, width, width])
        self.param_block = _mlp([n_params + n_rest, width, width])

        fused = width if combine == "film" else 3 * width
        self.prediction_block = _mlp([fused, width, width], out_act=True)
        self.head = nn.Linear(width, n_nodes * 2)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor) -> Tensor:
        h_dlo = self.dlo_block(V_in.flatten(1))
        h_act = self.action_block(a_n)
        h_par = self.param_block(p_n)

        if self.combine == "film":
            h = h_dlo * h_act * h_par
        else:
            h = torch.cat([h_dlo, h_act, h_par], dim=-1)

        return self.head(self.prediction_block(h)).view(-1, self.n_nodes, 2)


def _turning_angles_batch(p: Tensor) -> Tensor:
    u = p[:, 1:] - p[:, :-1]
    u_prev, u_next = u[:, :-1], u[:, 1:]
    cross = u_prev[..., 0] * u_next[..., 1] - u_prev[..., 1] * u_next[..., 0]
    dot = (u_prev * u_next).sum(-1)
    return torch.atan2(cross, dot)


class DLOPredictor(nn.Module):

    def __init__(self, net: DLONetwork, norm: Normalizer) -> None:
        super().__init__()
        self.net = net
        self.norm = norm

    def forward(self, V_in: Tensor, action: Tensor, params: Tensor,
                rest_angle: Tensor | None = None,
                hist_V_in: Tensor | None = None, hist_V_out: Tensor | None = None,
                hist_action: Tensor | None = None,
                hist_mask: Tensor | None = None,
                hist_V: Tensor | None = None) -> Tensor:
        Vc, centre, angle = canonicalize(V_in)
        scale = self.norm.pos_scale
        Vc = Vc / scale

        blocks = []
        for c in range(0, action.shape[1], 4):
            disp = rotate_vec(action[:, c + 1:c + 3], -angle) / scale
            blocks += [action[:, c:c + 1], disp, action[:, c + 3:c + 4]]
        a = torch.cat(blocks, dim=-1)
        a_n = self.norm.norm_action(a)
        p_n = self.norm.norm_params(params)
        if rest_angle is not None:
            p_n = torch.cat([p_n, self.norm.norm_rest(rest_angle)], dim=-1)

        net_kwargs = {}
        if getattr(self.net, "takes_node_hist", False):
            if hist_V is not None:
                B, L, N, _ = hist_V.shape
                Vh = rotate((hist_V - centre.unsqueeze(1)).reshape(B * L, N, 2),
                            -angle.repeat_interleave(L)).reshape(B, L, N, 2)
                net_kwargs["hist"] = Vh / scale - Vc.unsqueeze(1)
        if hist_V_in is not None:
            B, L, N = hist_V_in.shape[0], hist_V_in.shape[1], hist_V_in.shape[2]
            Vh_in = hist_V_in.reshape(B * L, N, 2)
            Vh_out = hist_V_out.reshape(B * L, N, 2)
            Ah = hist_action.reshape(B * L, hist_action.shape[-1])
            Vhc, _, angh = canonicalize(Vh_in)
            hblocks = []
            for c in range(0, Ah.shape[1], 4):
                disp = rotate_vec(Ah[:, c + 1:c + 3], -angh) / scale
                hblocks += [Ah[:, c:c + 1], disp, Ah[:, c + 3:c + 4]]
            Ahn = self.norm.norm_action(torch.cat(hblocks, dim=-1))
            ta_in = self.norm.norm_rest(_turning_angles_batch(Vh_in))
            ta_out = self.norm.norm_rest(_turning_angles_batch(Vh_out))
            hist_feat = torch.cat([Ahn, ta_in, ta_out], dim=-1).reshape(B, L, -1)
            net_kwargs = {"hist": hist_feat, "hist_mask": hist_mask}

        dV = self.net(Vc, a_n, p_n, **net_kwargs)
        V_pred_c = (Vc + dV) * scale
        return rotate(V_pred_c, angle) + centre


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


ARCHITECTURES = {
    "bilstm": DLOBiLSTM,
    "conv1d": DLOConv1d,
    "transformer": DLOTransformer,
    "transformer_hist": DLOTransformerHist,
    "gnn": DLOGNN,
}


from dataclasses import dataclass


def load_predictor(path, device="cuda") -> tuple[DLOPredictor, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd, n_rest = ck["state_dict"], ck.get("n_rest", 0)

    n_action = ck.get("n_action", 4)
    n_params = ck.get("n_params")

    if ck.get("arch") in (None, "film", "concat"):
        if n_params is None:
            w = sd.get("param_block.0.weight")
            n_params = (w.shape[1] - n_rest) if w is not None else 3
        net = DLONetwork(n_nodes=ck["n_nodes"], width=ck["width"],
                         n_rest=n_rest, n_params=n_params, n_action=n_action,
                         combine=ck.get("combine", ck.get("arch", "film")))
    else:
        pass
        if n_params is None:
            w = sd.get("film.net.0.weight")
            n_params = (w.shape[1] - n_action) if w is not None else 3
        extra = {}
        if ck["arch"] == "transformer_hist":
            extra = {"hist_hidden": ck.get("hist_hidden", 64),
                     "z_dim": ck.get("z_dim", 16)}
        elif ck["arch"] == "transformer_seq":
            extra = {"n_hist": ck.get("n_hist", 4)}
        net = ARCHITECTURES[ck["arch"]](
            n_nodes=ck["n_nodes"], width=ck["width"], n_rest=n_rest,
            n_params=n_params, n_action=n_action, **extra)
    net.load_state_dict(sd)
    norm = Normalizer(**ck["norm"]).to(device)
    model = DLOPredictor(net, norm).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def build_normalizer(d, n_nodes):
    a, p = d["action"], d["params"]
    lo = a.min(0).values.clone()
    hi = a.max(0).values.clone()
    for c in range(0, a.shape[1], 4):
        lo[c], hi[c] = 0.0, float(n_nodes - 2)
    n_in = p.shape[1] - 1
    n_kb = n_in - 3
    log_mask = torch.zeros(n_in, dtype=torch.bool)
    log_mask[1:1 + n_kb] = True
    return Normalizer(
        action_lo=lo,
        action_hi=hi,
        param_lo=p[:, :n_in].min(0).values.clone(),
        param_hi=p[:, :n_in].max(0).values.clone(),
        param_log=log_mask,
        pos_scale=0.3,
    )


def paper_metric(a, b):
    return (a - b).norm(dim=-1).sum(-1)


import numpy as np


def turning_angles(V):
    e = np.diff(V, axis=0)
    a = np.arctan2(e[:, 1], e[:, 0])
    d = np.diff(a)
    return np.arctan2(np.sin(d), np.cos(d))


def build_tensors(trans, nominal, rest_cap=0.35):
    Vi = np.stack([t["V_in"] for t in trans]).astype(np.float32)
    Vo = np.stack([t["V_out"] for t in trans]).astype(np.float32)
    A = np.stack([t["action"] for t in trans]).astype(np.float32)
    P = np.tile(np.asarray(nominal, np.float32), (len(trans), 1))
    R = np.stack([np.clip(turning_angles(t["V_in"]), -rest_cap, rest_cap)
                  for t in trans]).astype(np.float32)
    return (torch.tensor(Vi), torch.tensor(Vo), torch.tensor(A),
            torch.tensor(P), torch.tensor(R))


ROOT = Path(__file__).resolve().parent


DEV = "cuda" if torch.cuda.is_available() else "cpu"


NOMINAL = np.array([0.02, 0.1, 10.0, 0.15], np.float32)


def edge_length(V):
    return (V[:, 1:] - V[:, :-1]).norm(dim=-1).sum(-1)


def shape_loss(pred, tgt):
    return torch.nn.functional.mse_loss(pred[:, 1:] - pred[:, :-1],
                                        tgt[:, 1:] - tgt[:, :-1])


def arclen_residual(pred, seg_len):
    seg = (pred[:, 1:] - pred[:, :-1]).norm(dim=-1)
    return ((seg - seg_len) ** 2).sum(-1).mean()


def exit_straight_loss(pred):
    th = _turning_angles_batch(pred)
    return (th[:, 0] ** 2 + th[:, -1] ** 2).mean()


def pinn_weight(epoch, epochs, lam_max, warmup=0.3, rate=5.0):
    if epochs <= 1:
        return lam_max
    t = epoch / (epochs - 1)
    if t <= warmup:
        return 0.0
    p = (t - warmup) / max(1e-9, 1.0 - warmup)
    return lam_max * (1.0 - np.exp(-rate * p)) / (1.0 - np.exp(-rate))


def train_fold(trans, hold_clip, ckpt, epochs, lr, len_weight=0.0,
               shape_weight=0.0, noise_mm=0.0, pinn_lambda=0.0,
               pinn_warmup=0.3, exit_lambda=0.0, nominal=None):
    tr = [t for t in trans if t["clip"] != hold_clip]
    Vi, Vo, A, P, R = build_tensors(tr, NOMINAL if nominal is None else nominal)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(Vi, Vo, A, P, R), batch_size=128,
                        shuffle=True, drop_last=True)
    model, _ = load_predictor(ROOT / ckpt, DEV)
    if epochs == 0:
        model.eval()
        return model
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.15)
    for ep in range(epochs):
        model.train()
        lam = pinn_weight(ep, epochs, pinn_lambda, pinn_warmup) if pinn_lambda else 0.0
        for vi, vo, a, p, r in loader:
            vi, vo = vi.to(DEV), vo.to(DEV)
            a, p, r = a.to(DEV), p.to(DEV), r.to(DEV)
            if noise_mm:
                vi = vi + torch.randn_like(vi) * (noise_mm / 1000.0)
                r = _turning_angles_batch(vi).clamp(-0.35, 0.35)
            pred = model(vi, a, p, r)
            loss = torch.nn.functional.mse_loss(pred, vo)
            if len_weight:
                loss = loss + len_weight * torch.nn.functional.mse_loss(
                    edge_length(pred), edge_length(vi))
            if shape_weight:
                loss = loss + shape_weight * shape_loss(pred, vo)
            if exit_lambda:
                loss = loss + exit_lambda * exit_straight_loss(pred)
            if lam:
                ds = (edge_length(vi) / (vi.shape[1] - 1)).unsqueeze(-1)
                loss = loss + lam * arclen_residual(pred, ds)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def step_action(s0, real, k, n_steps):
    def edge(V, at_end):
        a, b = (V[-2], V[-1]) if at_end else (V[0], V[1])
        return 0.5 * (a + b), np.arctan2(b[1] - a[1], b[0] - a[0])
    out = []
    for at_end, alpha in ((True, k), (False, 0.0)):
        m0, a0 = edge(s0, at_end)
        m1, a1 = edge(real, at_end)
        dth = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
        out += [alpha, (m1[0] - m0[0]) / n_steps, (m1[1] - m0[1]) / n_steps,
                dth / n_steps]
    return np.array(out, np.float32)


from simulator import NodeConstraints


def guide_batch(sim, p_prev, setpoint, prm, k_p, n_steps, cons_target=None):
    p = p_prev.clone()
    v = torch.zeros_like(p)
    B, N, _ = p.shape
    mask = torch.zeros(B, N, dtype=torch.bool, device=p.device)
    mask[:, 0] = True
    mask[:, -1] = True
    cons = NodeConstraints(mask, setpoint if cons_target is None else cons_target)
    for _ in range(n_steps):
        f_ext = None if k_p == 0.0 else k_p * (setpoint - p)
        p, v = sim.step(p, v, prm, cons, f_ext=f_ext)
    return p
