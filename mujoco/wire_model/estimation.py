
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .nn_model import DLONetwork, DLOPredictor, Normalizer

Tensor = torch.Tensor


def dist_loss(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).norm(dim=-1).sum(-1)


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
        from .nn_models_alt import ARCHITECTURES
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


def graft_history_predictor(base_path, device="cuda", hist_hidden=64, z_dim=16):
    from .nn_models_alt import DLOTransformerHist
    ck = torch.load(base_path, map_location=device, weights_only=False)
    sd, n_rest = ck["state_dict"], ck.get("n_rest", 0)
    n_action = ck.get("n_action", 4)
    n_params = ck.get("n_params", 4)
    net = DLOTransformerHist(n_nodes=ck["n_nodes"], width=ck["width"],
                             n_rest=n_rest, n_params=n_params, n_action=n_action,
                             hist_hidden=hist_hidden, z_dim=z_dim)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, f"base checkpoint has keys the new net lacks: {unexpected}"
    assert all(k.startswith(("hist_mlp", "hist_out", "film_hist")) for k in missing), \
        f"unexpected missing keys, graft is not a clean superset: {missing}"
    norm = Normalizer(**ck["norm"]).to(device)
    model = DLOPredictor(net, norm).to(device)
    meta = {**ck, "arch": "transformer_hist", "hist_hidden": hist_hidden,
            "z_dim": z_dim}
    return model, meta


def feasible_w(u: Tensor, k_b_rng=(0.01, 1.0), k_d_rng=(2.0, 40.0)) -> Tensor:
    r = u.exp()
    lo = torch.maximum(torch.as_tensor(k_d_rng[0], device=u.device), k_b_rng[0] / r)
    hi = torch.minimum(torch.as_tensor(k_d_rng[1], device=u.device), k_b_rng[1] / r)
    k_d = (lo.clamp_min(1e-9).log() + hi.clamp_min(1e-9).log()).mul(0.5).exp()
    k_d = torch.maximum(k_d, lo)
    k_d = torch.minimum(k_d, hi)
    k_b = (r * k_d).clamp(*k_b_rng)
    return (k_b * k_d).log()


def uw_to_kb_kd(u: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    return torch.exp(0.5 * (w + u)), torch.exp(0.5 * (w - u))


def w_bounds(u: Tensor, k_b_rng=(0.01, 1.0), k_d_rng=(2.0, 40.0)):
    import math
    lo = torch.maximum(2 * math.log(k_b_rng[0]) - u, 2 * math.log(k_d_rng[0]) + u)
    hi = torch.minimum(2 * math.log(k_b_rng[1]) - u, 2 * math.log(k_d_rng[1]) + u)
    return lo, hi


@dataclass
class ParamEstimate:
    rest_angle: Tensor
    ratio: float
    k_b: float
    k_d: float
    loss: float
    k_c: float = 0.0
    loss_history: list = field(default_factory=list)


def estimate_parameters(
    model: DLOPredictor,
    V_in: Tensor,
    action: Tensor,
    V_out: Tensor,
    mass: float,
    n_iters: int = 400,
    lr_rest: float = 0.02,
    lr_ratio: float = 0.05,
    rest_bound: float = 0.35,
    init_rest: Tensor | None = None,
    init_ratio: float = 0.02,
    k_c_bound: float = 0.6,
    lr_kc: float = 0.03,
    estimate_kc: bool | None = None,
    free_w: bool = False,
    lr_w: float = 0.05,
    verbose: bool = False,
) -> ParamEstimate:
    dev = V_in.device
    k = V_in.shape[0]
    n_int = model.net.n_nodes - 2

    if init_rest is None:
        z = torch.zeros(n_int, device=dev)
    else:
        z = torch.atanh((init_rest / rest_bound).clamp(-0.999, 0.999))
    z = z.clone().requires_grad_(True)
    u = torch.tensor(float(torch.log(torch.tensor(init_ratio))),
                     device=dev).requires_grad_(True)

    if estimate_kc is None:
        estimate_kc = getattr(model.net, "n_params", 3) >= 4
    w_free = torch.zeros((), device=dev, requires_grad=True)
    zc = torch.tensor(0.0, device=dev).requires_grad_(True)

    groups = [{"params": [z], "lr": lr_rest}, {"params": [u], "lr": lr_ratio}]
    if estimate_kc:
        groups.append({"params": [zc], "lr": lr_kc})
    if free_w:
        groups.append({"params": [w_free], "lr": lr_w})
    opt = torch.optim.Adam(groups)
    m_col = torch.full((k, 1), mass, device=dev)
    history = []

    for it in range(n_iters):
        rest = rest_bound * torch.tanh(z)
        w = feasible_w(u.detach())
        if free_w:
            lo, hi = w_bounds(u.detach())
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo).clamp_min(1e-6)
            w = mid + half * torch.tanh(w_free)
        k_b, k_d = uw_to_kb_kd(u, w)
        cols = [m_col, k_b.expand(k).unsqueeze(-1), k_d.expand(k).unsqueeze(-1)]
        if estimate_kc:
            k_c = k_c_bound * 0.5 * (1.0 + torch.tanh(zc))
            cols.append(k_c.expand(k).unsqueeze(-1))
        params = torch.cat(cols, dim=-1)
        pred = model(V_in, action, params, rest.unsqueeze(0).expand(k, -1))
        loss = dist_loss(pred, V_out).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
            print(f"    it {it:>4}  D = {1000*loss.item():7.3f} mm  "
                  f"k_b/k_d = {float(u.exp()):.4f}")

    with torch.no_grad():
        rest = rest_bound * torch.tanh(z)
        w = feasible_w(u)
        if free_w:
            lo, hi = w_bounds(u)
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo).clamp_min(1e-6)
            w = mid + half * torch.tanh(w_free)
        k_b, k_d = uw_to_kb_kd(u, w)
        kc_out = float(k_c_bound * 0.5 * (1.0 + torch.tanh(zc))) if estimate_kc else 0.0
    return ParamEstimate(rest_angle=rest.detach(), ratio=float(u.exp()),
                         k_b=float(k_b), k_d=float(k_d), k_c=kc_out,
                         loss=history[-1], loss_history=history)


def degeneracy_report(
    model: DLOPredictor,
    V_in: Tensor,
    action: Tensor,
    V_out: Tensor,
    mass: float,
    rest_angle: Tensor,
    ratio: float,
    span: float = 1.5,
    n: int = 21,
) -> dict:
    dev = V_in.device
    k = V_in.shape[0]
    m_col = torch.full((k, 1), mass, device=dev)
    rest_b = rest_angle.unsqueeze(0).expand(k, -1)
    u0 = torch.tensor(float(torch.log(torch.tensor(ratio))), device=dev)
    w0 = feasible_w(u0)

    def probe(u, w):
        k_b, k_d = uw_to_kb_kd(u, w)
        if not (0.01 <= float(k_b) <= 1.0 and 2.0 <= float(k_d) <= 40.0):
            return float("nan")
        params = torch.cat([m_col, k_b.expand(k).unsqueeze(-1),
                            k_d.expand(k).unsqueeze(-1)], -1)
        with torch.no_grad():
            return float(dist_loss(model(V_in, action, params, rest_b), V_out).mean())

    offs = torch.linspace(-span, span, n, device=dev)
    along_w = [probe(u0, w0 + d) for d in offs]
    along_u = [probe(u0 + d, feasible_w(u0 + d)) for d in offs]
    fin = lambda xs: [x for x in xs if x == x]
    rng_w = (max(fin(along_w)) - min(fin(along_w))) if fin(along_w) else float("nan")
    rng_u = (max(fin(along_u)) - min(fin(along_u))) if fin(along_u) else float("nan")
    return {"offsets": offs.cpu(), "loss_along_w": along_w, "loss_along_u": along_u,
            "range_w_mm": 1000 * rng_w, "range_u_mm": 1000 * rng_u,
            "flatness": rng_u / rng_w if rng_w > 0 else float("inf")}


def estimate_action(
    model: DLOPredictor,
    V_in: Tensor,
    V_target: Tensor,
    mass: float,
    rest_angle: Tensor,
    ratio: float,
    alpha: int | None = None,
    n_iters: int = 300,
    lr: float = 0.02,
    max_disp: float = 0.18,
    max_rot: float = 1.2,
    verbose: bool = False,
) -> tuple[Tensor, float]:
    dev = V_in.device
    n_nodes = model.net.n_nodes
    alphas = (torch.arange(n_nodes - 1, device=dev) if alpha is None
              else torch.tensor([alpha], device=dev))
    b = alphas.shape[0]

    zx = torch.zeros(b, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([zx], lr=lr)
    scale = torch.tensor([max_disp, max_disp, max_rot], device=dev)

    Vi = V_in.expand(b, -1, -1)
    Vt = V_target.expand(b, -1, -1)
    k_b, k_d = uw_to_kb_kd(torch.tensor(float(torch.log(torch.tensor(ratio))),
                                        device=dev),
                           feasible_w(torch.tensor(float(torch.log(
                               torch.tensor(ratio))), device=dev)))
    params = torch.stack([torch.full((b,), mass, device=dev),
                          k_b.expand(b), k_d.expand(b)], -1)
    rest_b = rest_angle.unsqueeze(0).expand(b, -1)

    for it in range(n_iters):
        d = scale * torch.tanh(zx)
        act = torch.cat([alphas.float().unsqueeze(-1), d], -1)
        loss_each = dist_loss(model(Vi, act, params, rest_b), Vt)
        opt.zero_grad(set_to_none=True)
        loss_each.sum().backward()
        opt.step()
        if verbose and it % max(1, n_iters // 5) == 0:
            print(f"    it {it:>4}  best D = {1000*loss_each.min().item():7.3f} mm")

    with torch.no_grad():
        d = scale * torch.tanh(zx)
        act = torch.cat([alphas.float().unsqueeze(-1), d], -1)
        loss_each = dist_loss(model(Vi, act, params, rest_b), Vt)
        i = int(loss_each.argmin())
    return act[i].detach(), float(loss_each[i])
