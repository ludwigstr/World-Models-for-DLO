
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
