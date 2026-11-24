from __future__ import annotations
import torch
from torch import Tensor, nn


def mlp(d_in, d_h, d_out, n=2):
    layers, d = [], d_in
    for _ in range(n):
        layers += [nn.Linear(d, d_h), nn.ReLU()]; d = d_h
    return nn.Sequential(*layers, nn.Linear(d, d_out))


SCALE = 0.1


ACT = 0.01


YAW = 0.1


class PhysWorldGNN(nn.Module):
    def __init__(self, n_nodes=16, h=3, width=128, steps=7, radius=0.12, n_phi=2):
        super().__init__()
        self.N, self.h, self.radius, self.steps = n_nodes, h, radius, steps
        d_v = 2 + 3 + n_phi
        d_e = 2 * (h + 1) + 2
        self.enc_v, self.enc_e = mlp(d_v, width, width, 1), mlp(d_e, width, width, 1)
        self.prop_e = nn.ModuleList([mlp(3 * width, width, width, 1) for _ in range(steps)])
        self.prop_v = nn.ModuleList([mlp(2 * width, width, width, 1) for _ in range(steps)])
        self.dec = mlp(width, width, 2, 1)
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)
        ctrl = torch.zeros(n_nodes, dtype=torch.bool); ctrl[0] = True; ctrl[-1] = True
        self.register_buffer("ctrl", ctrl)

    def forward(self, hist: Tensor, X: Tensor, phi: Tensor, dp: Tensor, dyaw: Tensor) -> Tensor:
        B, N, _ = X.shape
        allx = torch.cat([hist, X.unsqueeze(1)], 1)
        rel = (allx.unsqueeze(3) - allx.unsqueeze(2)) / SCALE
        rel = rel.permute(0, 2, 3, 1, 4).reshape(B, N, N, -1)
        d = torch.cdist(X, X)
        adj = d < self.radius
        ar = torch.arange(N - 1, device=X.device)
        chain = torch.zeros(N, N, dtype=torch.bool, device=X.device); chain[ar, ar + 1] = True; chain[ar + 1, ar] = True
        adj = (adj | chain) & ~torch.eye(N, dtype=torch.bool, device=X.device)
        typ = torch.stack([~self.ctrl, self.ctrl], -1).to(X.dtype).unsqueeze(0).expand(B, N, 2)
        oc = (self.ctrl.unsqueeze(0) | self.ctrl.unsqueeze(1)).to(X.dtype)
        etype = torch.stack([1 - oc, oc], -1).unsqueeze(0).expand(B, N, N, 2)
        act = torch.cat([dp / ACT, (dyaw / YAW).unsqueeze(-1)], -1) * self.ctrl.view(1, N, 1)
        hv = self.enc_v(torch.cat([typ, act, phi], -1))
        m = adj.unsqueeze(-1).to(X.dtype)
        he = self.enc_e(torch.cat([rel, etype], -1)) * m
        for l in range(self.steps):
            he = he + self.prop_e[l](torch.cat([he, hv.unsqueeze(2).expand(-1, -1, N, -1),
                                                hv.unsqueeze(1).expand(-1, N, -1, -1)], -1)) * m
            hv = hv + self.prop_v[l](torch.cat([hv, he.sum(2)], -1))
        out = X + self.dec(hv) * SCALE * 0.1
        return torch.where(self.ctrl.view(1, N, 1), X + dp, out)


import time
import numpy as np
import torch
from twin import drive_from_nodes


H = 3


def samples(nodes, yaw, tau):
    n = nodes.shape[0]
    idx = torch.arange(n)
    out = []
    for t in range(0, n - tau):
        hi = torch.clamp(torch.arange(t - H, t), min=0)
        out.append((nodes[hi], nodes[t], nodes[t + 1:t + 1 + tau], yaw[t + 1:t + 1 + tau] - yaw[t:t + tau]))
    return out


def dp_of(xa, xb):
    d = torch.zeros_like(xa); d[..., 0, :] = xb[..., 0, :] - xa[..., 0, :]; d[..., -1, :] = xb[..., -1, :] - xa[..., -1, :]
    return d


def dyaw_nodes(dy, N=16):
    out = torch.zeros(*dy.shape[:-1], N, device=dy.device, dtype=dy.dtype)
    out[..., 0] = dy[..., 0]; out[..., -1] = dy[..., 1]
    return out


def build(demos, tau, use_yaw):
    hs, xs, ys, dys, phis = [], [], [], [], []
    for d in demos:
        for c in range(d["nodes"].shape[1]):
            for h_, x_, y_, dy_ in samples(d["nodes"][:, c].float(), d["yaw"][c].float(), tau):
                hs.append(h_); xs.append(x_); ys.append(y_); dys.append(dy_ if use_yaw else torch.zeros_like(dy_))
                phis.append(d["phi"][c].float())
    return torch.stack(hs), torch.stack(xs), torch.stack(ys), torch.stack(dys), torch.stack(phis)


def train(demos, dev, epochs, tau, noise_mm, phi_noise, use_yaw, seed, logf):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(demos), generator=g).tolist()
    nva = max(1, len(demos) // 10)
    va_d, tr_d = [demos[i] for i in perm[:nva]], [demos[i] for i in perm[nva:]]
    tr = build(tr_d, tau, use_yaw); va = build(va_d, tau, use_yaw)
    allphi = torch.cat([d["phi"].reshape(-1, 2) for d in demos])
    mu, sd = allphi.mean(0), allphi.std(0) + 1e-6
    model = PhysWorldGNN(16, H).to(dev); opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = tr[0].shape[0]
    logf(f"  GNN: {n} train / {va[0].shape[0]} val samples from {len(tr_d)}/{len(va_d)} demos, tau {tau}, "
         f"Phi mean {mu.tolist()} sd {sd.tolist()}")
    best = (float("inf"), None, -1)
    tr = [t.to(dev) for t in tr]; va = [t.to(dev) for t in va]; mu, sd = mu.to(dev), sd.to(dev)
    for ep in range(epochs):
        model.train(); t0 = time.time(); p = torch.randperm(n, device=dev); tl = 0.0
        for i in range(0, n, 256):
            b = p[i:i + 256]
            hist = tr[0][b] + torch.randn_like(tr[0][b]) * noise_mm / 1000
            x = tr[1][b]; phi = (tr[4][b] - mu) / sd + torch.randn_like(tr[4][b]) * phi_noise
            loss = 0.0
            for s in range(tau):
                y = tr[2][b, s]
                pred = model(hist, x, phi, dp_of(x, y), dyaw_nodes(tr[3][b, s]))
                loss = loss + ((pred[:, 1:-1] - y[:, 1:-1]) ** 2).mean()
                hist = torch.cat([hist[:, 1:], x.unsqueeze(1)], 1); x = pred
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, va[0].shape[0], 2048):
                sl = slice(i, i + 2048); x = va[1][sl]; y = va[2][sl, 0]
                pr = model(va[0][sl], x, (va[4][sl] - mu) / sd, dp_of(x, y), dyaw_nodes(va[3][sl, 0]))
                errs.append((pr[:, 1:-1] - y[:, 1:-1]).norm(dim=-1).mean(-1))
            v = torch.cat(errs).mean().item() * 1000
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()}, ep)
        if ep % 5 == 0 or ep == epochs - 1:
            logf(f"  GNN ep {ep}: train {tl / n:.3e}  val one-step {v:.3f} mm ({time.time()-t0:.1f}s)")
    model.load_state_dict(best[1])
    logf(f"  GNN best epoch {best[2]} (val one-step {best[0]:.3f} mm)")
    return model, mu, sd, best[0]


def real_yaw(P):
    return drive_from_nodes(P[:, :, [0, 1, 14, 15]])[1]


def open_loop(model, X, phi, dp, dy, warm):
    T = X.shape[0]
    frames = [X[t] for t in range(warm)]
    for t in range(warm - 1, T - 1):
        h = torch.stack([frames[max(0, t - H + j)] for j in range(H)])
        pr = model(h[None], frames[t][None], phi[None], dp[t][None], dy[t][None])[0]
        pr = torch.cat([X[t + 1, :1], pr[1:-1], X[t + 1, -1:]], 0)
        frames.append(pr)
    return torch.stack(frames[warm:])


def finetune_and_eval(model, P, phi0, mu, sd, warm, dev, iters, use_yaw, finetune=True, box=None, objective="rollout"):
    for p in model.parameters():
        p.requires_grad_(False)
    C, T, N, _ = P.shape
    Y = real_yaw(P)
    errs, phis = [], []
    for c in range(C):
        X = P[c]
        hist = torch.stack([X[torch.clamp(torch.arange(t - H, t), min=0)] for t in range(T - 1)])
        dp = dp_of(X[:-1], X[1:])
        dy = dyaw_nodes((Y[c, 1:] - Y[c, :-1]) if use_yaw else torch.zeros_like(Y[c, 1:]))
        phi = ((phi0[c] - mu) / sd).clone().requires_grad_(True)
        lo = hi = None
        if box is not None:
            lo = (phi0[c] - box - mu) / sd; hi = (phi0[c] + box - mu) / sd
        if finetune:
            opt = torch.optim.Adam([phi], lr=0.05)
            for it in range(iters):
                if objective == "rollout":
                    pred = open_loop(model, X, phi, dp, dy, warm); tgt = X[warm:]
                else:
                    pred = model(hist, X[:-1], phi.expand(T - 1, N, 2), dp, dy); tgt = X[1:]
                mse = ((pred[:, 1:-1] - tgt[:, 1:-1]) ** 2).mean()
                cd = (torch.cdist(tgt, pred).min(-1).values ** 2).mean()
                loss = mse + cd
                opt.zero_grad(); loss.backward(); opt.step()
                if lo is not None:
                    with torch.no_grad():
                        phi.copy_(torch.maximum(torch.minimum(phi, hi), lo))
        phis.append((phi.detach() * sd + mu).cpu().numpy())
        with torch.no_grad():
            frames = [X[t] for t in range(warm)]; e = []
            for t in range(warm - 1, T - 1):
                h = torch.stack([frames[max(0, t - H + j)] for j in range(H)])
                pr = model(h[None], frames[t][None], phi.detach()[None], dp[t][None], dy[t][None])[0]
                pr[0] = X[t + 1, 0]; pr[-1] = X[t + 1, -1]
                frames.append(pr); e.append((pr - X[t + 1]).norm(dim=-1).mean().item() * 1000)
        errs.append(np.array(e))
    return np.mean(errs, 0), phis
