from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .kinematics import normalize_state, chain_to_positions

SIZES = {
    "S": dict(d_embed=1024, d_action=1024, d_rnn=512, d_z=64, d_hidden=512),
    "M": dict(d_embed=1024, d_action=1024, d_rnn=512, d_z=64, d_hidden=1024),
    "L": dict(d_embed=2048, d_action=512, d_rnn=1024, d_z=64, d_hidden=2048),
    "XS": dict(d_embed=256, d_action=256, d_rnn=256, d_z=32, d_hidden=256),
}


def mlp(d_in: int, d_hidden: int, d_out: int, n_hidden: int = 2) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.ELU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class ActionEncoder(nn.Module):

    def __init__(self, n_links: int, d_action: int, d_hidden: int, d_disp: int = 3):
        super().__init__()
        self.emb = nn.Embedding(n_links + 1, d_action, padding_idx=n_links)
        self.n_links = n_links
        self.disp = mlp(d_disp, d_hidden, d_action, n_hidden=1)

    def forward(self, idx: Tensor, dp: Tensor) -> Tensor:
        pad = idx < 0
        idx = torch.where(pad, torch.full_like(idx, self.n_links), idx)
        a = self.emb(idx) + self.disp(dp) * (~pad).unsqueeze(-1)
        return a.sum(dim=1)


class RSSMCell(nn.Module):
    def __init__(self, d_action: int, d_rnn: int, d_z: int, d_hidden: int, d_embed: int,
                 min_std: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_z + d_action, d_hidden), nn.ELU())
        self.gru = nn.GRUCell(d_hidden, d_rnn)
        self.prior = mlp(d_rnn, d_hidden, 2 * d_z, n_hidden=1)
        self.post = mlp(d_rnn + d_embed, d_hidden, 2 * d_z, n_hidden=1)
        self.d_z, self.min_std = d_z, min_std

    def step(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        return self.gru(self.inp(torch.cat([z, a], dim=-1)), h)

    def _dist(self, out: Tensor) -> tuple[Tensor, Tensor]:
        mu, pre = out.chunk(2, dim=-1)
        return mu, F.softplus(pre) + self.min_std

    def prior_dist(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self._dist(self.prior(h))

    def post_dist(self, h: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        return self._dist(self.post(torch.cat([h, e], dim=-1)))


def state_nll(pred: Tensor, target: Tensor, n_nodes: int, inv_var: Tensor) -> Tensor:
    w = inv_var.view(*([1] * (pred.ndim - 1)), -1)
    e0 = ((pred[..., :3] - target[..., :3]) ** 2 * w[..., :3]).sum(-1)
    qp = pred[..., 3:].reshape(*pred.shape[:-1], n_nodes - 1, 4)
    qt = target[..., 3:].reshape(*target.shape[:-1], n_nodes - 1, 4)
    wq = w[..., 3:].reshape(*w.shape[:-1], n_nodes - 1, 4)
    dq = torch.minimum(((qp - qt) ** 2 * wq).sum(-1), ((qp + qt) ** 2 * wq).sum(-1)).sum(-1)
    return 0.5 * (e0 + dq).mean()


def kl_gauss(mu_q, sd_q, mu_p, sd_p) -> Tensor:
    return (torch.log(sd_p / sd_q) + (sd_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sd_p ** 2) - 0.5).sum(-1)


class RopeDreamer(nn.Module):
    def __init__(self, n_nodes: int = 16, size: str = "S", d_disp: int = 3, beta: float = 1.0,
                 free_nats: float = 0.0, pos_weight: float = 0.0, rollout_weight: float = 0.0,
                 warm: int = 5, repr: str = "chain"):
        super().__init__()
        c = SIZES[size]
        self.n_nodes, self.beta, self.free_nats, self.pos_weight = n_nodes, beta, free_nats, pos_weight
        self.rollout_weight, self.warm, self.repr = rollout_weight, warm, repr
        self.d_state = 3 + 4 * (n_nodes - 1) if repr == "chain" else 3 * n_nodes
        self.register_buffer("s_mean", torch.zeros(self.d_state))
        self.register_buffer("s_std", torch.ones(self.d_state))
        self.enc = mlp(self.d_state, c["d_hidden"], c["d_embed"], n_hidden=1)
        self.act = ActionEncoder(n_nodes - 1, c["d_action"], c["d_hidden"], d_disp)
        self.cell = RSSMCell(c["d_action"], c["d_rnn"], c["d_z"], c["d_hidden"], c["d_embed"])
        self.dec_recon = mlp(c["d_rnn"] + c["d_z"], c["d_hidden"], self.d_state)
        self.dec_pred = mlp(c["d_rnn"] + c["d_z"], c["d_hidden"], self.d_state)
        self.d_rnn, self.d_z = c["d_rnn"], c["d_z"]
        self.size = size

    def set_state_stats(self, mean: Tensor, std: Tensor, std_floor: float = 0.05) -> None:
        self.s_mean.copy_(mean); self.s_std.copy_(std.clamp_min(std_floor))

    def encode(self, s: Tensor) -> Tensor:
        return self.enc((s - self.s_mean) / self.s_std)

    def initial(self, B: int, device) -> tuple[Tensor, Tensor]:
        return (torch.zeros(B, self.d_rnn, device=device), torch.zeros(B, self.d_z, device=device))

    def decode(self, dec: nn.Module, h: Tensor, z: Tensor) -> Tensor:
        out = dec(torch.cat([h, z], dim=-1))
        return normalize_state(out, self.n_nodes) if self.repr == "chain" else out

    def positions(self, s: Tensor, ell: Tensor) -> Tensor:
        if self.repr == "chain":
            return chain_to_positions(s, ell, self.n_nodes)
        return s.reshape(*s.shape[:-1], self.n_nodes, 3)

    def forward(self, s: Tensor, idx: Tensor, dp: Tensor, sample: bool = True,
                ell: Tensor | None = None, s_obs: Tensor | None = None) -> dict:
        B, T, _ = s.shape
        h, z = self.initial(B, s.device)
        e = self.encode(s if s_obs is None else s_obs)
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        recon, pred, kls = [], [], []
        for t in range(T):
            if t > 0:
                h = self.cell.step(h, z, a[:, t - 1])
            mu_p, sd_p = self.cell.prior_dist(h)
            mu_q, sd_q = self.cell.post_dist(h, e[:, t])
            z_prior = mu_p + sd_p * torch.randn_like(sd_p) if sample else mu_p
            z = mu_q + sd_q * torch.randn_like(sd_q) if sample else mu_q
            recon.append(self.decode(self.dec_recon, h, z))
            if t > 0:
                pred.append(self.decode(self.dec_pred, h, z_prior))
            kls.append(kl_gauss(mu_q, sd_q, mu_p, sd_p))
        recon = torch.stack(recon, 1)
        pred = torch.stack(pred, 1)
        kl = torch.stack(kls, 1)
        inv_var = 1.0 / self.s_std ** 2
        if self.repr == "chain":
            l_recon = state_nll(recon, s, self.n_nodes, inv_var)
            l_pred = state_nll(pred, s[:, 1:], self.n_nodes, inv_var)
        else:
            w = inv_var.view(1, 1, -1)
            l_recon = 0.5 * (((recon - s) ** 2) * w).sum(-1).mean()
            l_pred = 0.5 * (((pred - s[:, 1:]) ** 2) * w).sum(-1).mean()
        l_kl = torch.clamp(kl, min=self.free_nats).mean()
        loss = l_recon + l_pred + self.beta * l_kl
        l_pos = torch.zeros((), device=s.device)
        if self.pos_weight > 0:
            assert ell is not None, "pos_weight needs the link length per sequence"
            e = ell.view(B, 1)
            P_true = self.positions(s, e)
            P_rec = self.positions(recon, e)
            P_pred = self.positions(pred, e)
            l_pos = F.mse_loss(P_rec, P_true) + F.mse_loss(P_pred, P_true[:, 1:])
            loss = loss + self.pos_weight * l_pos
        l_roll = torch.zeros((), device=s.device)
        if self.rollout_weight > 0 and T > self.warm:
            assert ell is not None
            l_roll = self.open_loop_loss(s, idx, dp, ell)
            loss = loss + self.rollout_weight * l_roll
        return {"loss": loss, "recon": l_recon, "pred": l_pred, "kl": kl.mean(), "pos": l_pos, "roll": l_roll,
                "s_recon": recon, "s_pred": pred}

    def open_loop_loss(self, s: Tensor, idx: Tensor, dp: Tensor, ell: Tensor) -> Tensor:
        B, T, _ = s.shape
        warm = min(self.warm, T - 1)
        h, z = self.initial(B, s.device)
        e = self.encode(s[:, :warm])
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        for t in range(warm):
            if t > 0:
                h = self.cell.step(h, z, a[:, t - 1])
            mu_q, _ = self.cell.post_dist(h, e[:, t])
            z = mu_q
        preds = []
        for t in range(warm, T):
            h = self.cell.step(h, z, a[:, t - 1])
            mu_p, _ = self.cell.prior_dist(h)
            z = mu_p
            preds.append(self.decode(self.dec_pred, h, z))
        pred = torch.stack(preds, 1)
        el = ell.view(B, 1)
        P_true = self.positions(s[:, warm:], el)
        if self.repr == "chain":
            P_pred = chain_to_positions(torch.cat([P_true[..., 0, :], pred[..., 3:]], -1), el, self.n_nodes)
        else:
            P_pred = self.positions(pred, el)
        return F.mse_loss(P_pred[..., 1:-1, :], P_true[..., 1:-1, :])

    @torch.no_grad()
    def warmup(self, s: Tensor, idx: Tensor, dp: Tensor, sample: bool = False) -> tuple[Tensor, Tensor]:
        B, n, _ = s.shape
        h, z = self.initial(B, s.device)
        e = self.encode(s)
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, n, -1)
        for t in range(n):
            if t > 0:
                h = self.cell.step(h, z, a[:, t - 1])
            mu_q, sd_q = self.cell.post_dist(h, e[:, t])
            z = mu_q + sd_q * torch.randn_like(sd_q) if sample else mu_q
        return h, z

    @torch.no_grad()
    def imagine(self, h: Tensor, z: Tensor, idx: Tensor, dp: Tensor, sample: bool = False) -> Tensor:
        B, T = idx.shape[:2]
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        out = []
        for t in range(T):
            h = self.cell.step(h, z, a[:, t])
            mu_p, sd_p = self.cell.prior_dist(h)
            z = mu_p + sd_p * torch.randn_like(sd_p) if sample else mu_p
            out.append(self.decode(self.dec_pred, h, z))
        return torch.stack(out, 1)
