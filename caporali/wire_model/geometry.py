
from __future__ import annotations

import torch

Tensor = torch.Tensor


def edge_lengths(p: Tensor) -> Tensor:
    return (p[:, 1:] - p[:, :-1]).norm(dim=-1)


def arc_length(p: Tensor) -> Tensor:
    return edge_lengths(p).sum(-1)


def resample_polyline(p: Tensor, n_out: int) -> Tensor:
    batch = p.shape[0]
    seg = (p[:, 1:] - p[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(batch, 1, device=p.device, dtype=p.dtype),
                     seg.cumsum(-1)], dim=-1)
    total = cum[:, -1:].clamp_min(1e-12)
    want = torch.linspace(0, 1, n_out, device=p.device, dtype=p.dtype)
    want = want.unsqueeze(0) * total

    out = torch.empty(batch, n_out, 2, device=p.device, dtype=p.dtype)
    for b in range(batch):
        idx = torch.searchsorted(cum[b], want[b].contiguous()).clamp(1, p.shape[1] - 1)
        lo, hi = idx - 1, idx
        span = (cum[b][hi] - cum[b][lo]).clamp_min(1e-12)
        t = ((want[b] - cum[b][lo]) / span).unsqueeze(-1)
        out[b] = p[b][lo] + t * (p[b][hi] - p[b][lo])
    return out


def _arc_from_turn(n: int, l0: float, phi: Tensor) -> Tensor:
    k = torch.arange(n - 1, device=phi.device, dtype=phi.dtype)
    ang = (k.unsqueeze(0) - (n - 2) / 2.0) * phi.unsqueeze(-1)
    e = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * l0
    zero = torch.zeros(phi.shape[0], 1, 2, device=phi.device, dtype=phi.dtype)
    return torch.cat([zero, e.cumsum(1)], 1)


def polyline_from_angles(theta: Tensor, l0: float,
                         heading0: Tensor | float = 0.0) -> Tensor:
    batch, n_int = theta.shape
    dev, dt = theta.device, theta.dtype
    h0 = torch.as_tensor(heading0, device=dev, dtype=dt).expand(batch)
    headings = torch.cat([h0.unsqueeze(-1),
                          h0.unsqueeze(-1) + theta.cumsum(-1)], -1)
    e = torch.stack([torch.cos(headings), torch.sin(headings)], -1) * l0
    zero = torch.zeros(batch, 1, 2, device=dev, dtype=dt)
    return torch.cat([zero, e.cumsum(1)], 1)


def random_rest_angle(
    batch: int,
    n_nodes: int,
    amp: tuple[float, float] = (0.0, 0.35),
    n_modes: int = 5,
    p_straight: float = 0.15,
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    n_int = n_nodes - 2
    s = torch.linspace(0, 1, n_int, device=device, dtype=dtype)
    k = torch.arange(1, n_modes + 1, device=device, dtype=dtype)

    beta = 0.5 + 1.5 * torch.rand(batch, 1, generator=generator,
                                  device=device, dtype=dtype)
    coef = torch.randn(batch, n_modes, generator=generator, device=device,
                       dtype=dtype) * k.unsqueeze(0).pow(-beta)
    phase = 2 * torch.pi * torch.rand(batch, n_modes, generator=generator,
                                      device=device, dtype=dtype)

    basis = torch.sin(torch.pi * k.view(1, -1, 1) * s.view(1, 1, -1)
                      + phase.unsqueeze(-1))
    theta = (coef.unsqueeze(-1) * basis).sum(1)

    peak = theta.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    lo, hi = amp
    a = lo + (hi - lo) * torch.rand(batch, 1, generator=generator,
                                    device=device, dtype=dtype)
    theta = theta / peak * a

    straight = torch.rand(batch, 1, generator=generator, device=device,
                          dtype=dtype) < p_straight
    return torch.where(straight, torch.zeros_like(theta), theta)


def discrete_arc_batch(
    n: int,
    l0: float,
    chords: Tensor,
    iters: int = 60,
) -> Tensor:
    span = (n - 1) * l0
    if bool((chords > span).any()):
        bad = chords[chords > span][0].item()
        raise ValueError(f"chord {bad} exceeds the fully-straight span {span}")

    dev, dt = chords.device, chords.dtype
    lo = torch.zeros_like(chords)
    hi = torch.full_like(chords, 2.0 * torch.pi / max(n - 1, 1) * 0.999)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = _arc_from_turn(n, l0, mid)
        c = (p[:, -1] - p[:, 0]).norm(dim=-1)
        too_long = c > chords
        lo = torch.where(too_long, mid, lo)
        hi = torch.where(too_long, hi, mid)
    p = _arc_from_turn(n, l0, 0.5 * (lo + hi))
    return p - p[:, :1]


def discrete_arc(
    n: int,
    l0: float,
    chord: float,
    device="cpu",
    dtype=torch.float32,
    iters: int = 60,
) -> Tensor:
    c = torch.tensor([chord], device=device, dtype=torch.float64)
    return discrete_arc_batch(n, l0, c, iters=iters).to(dtype)


def random_slack_wire(
    batch: int,
    n: int,
    length: float,
    slack: tuple[float, float] = (0.55, 0.95),
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    l0 = length / (n - 1)
    lo, hi = slack
    fr = torch.rand(batch, generator=generator, device=device, dtype=torch.float64)
    fracs = lo + fr * (hi - lo)

    p = discrete_arc_batch(n, l0, chords=fracs * length)

    sign = torch.where(
        torch.rand(batch, 1, generator=generator, device=device, dtype=torch.float64) < 0.5,
        -1.0, 1.0)
    p[..., 1] *= sign

    ang = torch.rand(batch, generator=generator, device=device,
                     dtype=torch.float64) * 2 * torch.pi
    c, s = torch.cos(ang), torch.sin(ang)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    p = torch.einsum("bij,bkj->bki", rot, p)
    return p.to(dtype)
