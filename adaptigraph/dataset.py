
from __future__ import annotations

from dataclasses import dataclass, asdict

import math
import torch

from actions import Action, grasped_node_targets, rollout, terminal_edge_action
from dlo_model import DLOParams, DLOSimulator
from geometry import discrete_arc_batch, random_rest_angle

Tensor = torch.Tensor


@dataclass
class SamplingRanges:

    length: tuple[float, float] = (0.25, 0.60)
    mass: tuple[float, float] = (0.01, 0.08)
    k_b: tuple[float, float] = (0.004, 1.00)
    k_b_spread: tuple[float, float] = (1.0, 400.0)
    k_b_p_flat: float = 0.20
    k_d: tuple[float, float] = (2.0, 40.0)
    k_c: tuple[float, float] = (0.0, 0.60)
    k_s: float = 1.0e5
    rest_amp: tuple[float, float] = (0.0, 0.35)
    rest_p_straight: float = 0.15
    slack_init: tuple[float, float] = (0.55, 0.92)
    slack_bounds: tuple[float, float] = (0.30, 0.94)
    max_disp: float = 0.30
    max_rot: float = 1.20
    hub_disp_frac: float = 2.20
    hub_rot_frac: float = 0.15


@dataclass
class SimConfig:

    n_nodes: int = 16
    dt: float = 4.0e-5
    n_move: int = 15000
    n_settle: int = 10000


def _k_b_profile(base, batch, n_nodes, r: "SamplingRanges", device, generator):
    n_int = n_nodes - 2
    t = torch.linspace(0.0, 1.0, n_int, device=device).unsqueeze(0)

    def rnd(*shape):
        return torch.rand(*shape, generator=generator, device=device)

    ph = rnd(batch, 2) * 2 * math.pi
    w = 1.0 + 2.0 * rnd(batch, 2)
    shape = (torch.sin(w[:, :1] * 2 * math.pi * t + ph[:, :1])
             + 0.5 * torch.sin(w[:, 1:] * 2 * math.pi * t + ph[:, 1:]))
    shape = shape / shape.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

    lo, hi = math.log(r.k_b_spread[0]), math.log(r.k_b_spread[1])
    amp = torch.exp(lo + (hi - lo) * rnd(batch, 1)) .log() / 2.0
    flat = (rnd(batch, 1) < r.k_b_p_flat).to(amp.dtype)
    amp = amp * (1.0 - flat)
    prof = base.unsqueeze(1) * torch.exp(shape * amp)
    return prof.clamp(r.k_b[0], r.k_b[1])


def sample_params(batch, r: SamplingRanges, device, n_nodes: int,
                  generator=None) -> DLOParams:
    def uni(lo, hi):
        return lo + (hi - lo) * torch.rand(batch, generator=generator,
                                           device=device)

    log_lo, log_hi = torch.log(torch.tensor(r.k_b, device=device))
    k_b_base = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(
        batch, generator=generator, device=device))
    k_b = _k_b_profile(k_b_base, batch, n_nodes, r, device, generator)

    return DLOParams(
        m=uni(*r.mass),
        k_b=k_b,
        k_d=uni(*r.k_d),
        k_c=uni(*r.k_c),
        k_s=torch.full((batch,), r.k_s, device=device),
        length=uni(*r.length),
        rest_angle=random_rest_angle(batch, n_nodes, amp=r.rest_amp,
                                     p_straight=r.rest_p_straight,
                                     device=device, generator=generator),
    )


def sample_connector_action(
    p: Tensor,
    length: Tensor,
    r: SamplingRanges,
    generator=None,
    max_tries: int = 12,
) -> Action:
    batch, n_nodes, _ = p.shape
    dev = p.device
    lo, hi = r.slack_bounds
    alpha = n_nodes - 2
    sub_len = alpha * length / (n_nodes - 1)

    dx = torch.zeros(batch, device=dev)
    dy = torch.zeros(batch, device=dev)
    dth = torch.zeros(batch, device=dev)
    todo = torch.ones(batch, dtype=torch.bool, device=dev)

    anchor = p[:, 0]
    for _ in range(max_tries):
        k = int(todo.sum())
        if k == 0:
            break
        mag = r.max_disp * length[todo] * torch.rand(k, generator=generator, device=dev)
        ang = 2 * torch.pi * torch.rand(k, generator=generator, device=dev)
        cand_dx, cand_dy = mag * torch.cos(ang), mag * torch.sin(ang)
        cand_dth = r.max_rot * (2 * torch.rand(k, generator=generator, device=dev) - 1)

        probe = Action(
            alpha=torch.full((k,), alpha, dtype=torch.long, device=dev),
            dx=cand_dx, dy=cand_dy, dtheta=cand_dth,
        )
        _, tgt = grasped_node_targets(p[todo], probe)
        span = (tgt[:, 0] - anchor[todo]).norm(dim=-1) / sub_len[todo]
        ok = (span > lo) & (span < hi)

        idx = torch.nonzero(todo, as_tuple=True)[0]
        good = idx[ok]
        dx[good], dy[good], dth[good] = cand_dx[ok], cand_dy[ok], cand_dth[ok]
        todo[good] = False

    return terminal_edge_action(n_nodes, dx, dy, dth, at_end=True)


def generate(
    n_chains: int,
    chain_len: int = 5,
    cfg: SimConfig | None = None,
    r: SamplingRanges | None = None,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Tensor]:
    cfg = cfg or SimConfig()
    r = r or SamplingRanges()
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)

    sim = DLOSimulator(n_nodes=cfg.n_nodes, dt=cfg.dt, device=dev)
    prm = sample_params(n_chains, r, dev, cfg.n_nodes, g)
    lengths = prm.length.clone()

    p = _init_batch(n_chains, cfg.n_nodes, lengths, r, dev, g)

    static = torch.zeros(n_chains, cfg.n_nodes, dtype=torch.bool, device=dev)
    prm_b = prm.to_batched(n_chains, dev, n_nodes=cfg.n_nodes)

    zero = torch.zeros(n_chains, device=dev)
    p = rollout(sim, p, prm_b, terminal_edge_action(cfg.n_nodes, zero, zero, zero),
                static_mask=static, n_move=1, n_settle=cfg.n_settle,
                action2=terminal_edge_action(cfg.n_nodes, zero, zero, zero,
                                             at_end=False))

    V_in, V_out, acts, pars, rests = [], [], [], [], []
    for step in range(chain_len):
        act = sample_connector_action(p, lengths, r, generator=g)
        ang = 2 * torch.pi * torch.rand(n_chains, generator=g, device=dev)
        sc = r.hub_disp_frac * torch.rand(n_chains, generator=g, device=dev)
        mag = sc * torch.stack([act.dx, act.dy], -1).norm(dim=-1)
        hub_dx, hub_dy = mag * torch.cos(ang), mag * torch.sin(ang)
        span_dir = (p[:, -1] + torch.stack([act.dx, act.dy], -1)) \
            - (p[:, 0] + torch.stack([hub_dx, hub_dy], -1))
        span = span_dir.norm(dim=-1)
        limit = r.slack_bounds[1] * lengths
        over = span > limit
        if bool(over.any()):
            shrink = torch.where(over, (limit / span.clamp_min(1e-9)), torch.ones_like(span))
            hub_dx = hub_dx * shrink
            hub_dy = hub_dy * shrink
        hub_act = terminal_edge_action(
            cfg.n_nodes, hub_dx, hub_dy, r.hub_rot_frac * act.dtheta,
            at_end=False)
        pf = rollout(sim, p, prm_b, act, static_mask=static,
                     n_move=cfg.n_move, n_settle=cfg.n_settle, action2=hub_act)

        finite = torch.isfinite(pf).all(dim=(1, 2))
        if not bool(finite.all()):
            n_bad = int((~finite).sum())
            raise RuntimeError(
                f"{n_bad}/{n_chains} rollouts diverged at chain step {step}; "
                f"check dt against DLOSimulator.stability_dt")

        V_in.append(p.cpu())
        V_out.append(pf.cpu())
        acts.append(torch.cat([act.as_vector(), hub_act.as_vector()],
                              dim=-1).cpu())
        def col(x):
            return x if x.ndim == 2 else x.unsqueeze(-1)
        pars.append(torch.cat([col(prm.m), col(prm.k_b), col(prm.k_d),
                               col(prm.k_c), col(prm.length)], -1).cpu())
        rests.append(prm.rest_angle.cpu())
        p = pf
        if verbose:
            print(f"  chain step {step+1}/{chain_len} done", flush=True)

    return {
        "V_in": torch.cat(V_in),
        "V_out": torch.cat(V_out),
        "action": torch.cat(acts),
        "params": torch.cat(pars),
        "rest_angle": torch.cat(rests),
        "config": {**asdict(cfg), **asdict(r)},
    }


def generate_seq(
    n_chains: int,
    chain_len: int = 5,
    n_hist: int = 4,
    n_sub: int = 4,
    cfg: SimConfig | None = None,
    r: SamplingRanges | None = None,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Tensor]:
    cfg = cfg or SimConfig()
    r = r or SamplingRanges()
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)
    total = cfg.n_move + cfg.n_settle
    assert total % n_sub == 0, (total, n_sub)
    sub = total // n_sub
    record_at = [sub * (i + 1) for i in range(n_sub)]
    frac = [min(1.0, t / cfg.n_move) for t in record_at]
    share = [frac[0]] + [frac[i] - frac[i - 1] for i in range(1, n_sub)]

    sim = DLOSimulator(n_nodes=cfg.n_nodes, dt=cfg.dt, device=dev)
    prm = sample_params(n_chains, r, dev, cfg.n_nodes, g)
    lengths = prm.length.clone()
    p = _init_batch(n_chains, cfg.n_nodes, lengths, r, dev, g)
    static = torch.zeros(n_chains, cfg.n_nodes, dtype=torch.bool, device=dev)
    prm_b = prm.to_batched(n_chains, dev, n_nodes=cfg.n_nodes)
    zero = torch.zeros(n_chains, device=dev)
    p = rollout(sim, p, prm_b, terminal_edge_action(cfg.n_nodes, zero, zero, zero),
                static_mask=static, n_move=1, n_settle=cfg.n_settle,
                action2=terminal_edge_action(cfg.n_nodes, zero, zero, zero,
                                             at_end=False))

    def col(x):
        return x if x.ndim == 2 else x.unsqueeze(-1)
    par_row = torch.cat([col(prm.m), col(prm.k_b), col(prm.k_d),
                         col(prm.k_c), col(prm.length)], -1).cpu()
    rest_row = prm.rest_angle.cpu()

    window = [p.cpu().clone() for _ in range(n_hist)]
    H, V_in, V_out, acts, pars, rests, tsub = [], [], [], [], [], [], []
    for step in range(chain_len):
        act = sample_connector_action(p, lengths, r, generator=g)
        ang = 2 * torch.pi * torch.rand(n_chains, generator=g, device=dev)
        sc = r.hub_disp_frac * torch.rand(n_chains, generator=g, device=dev)
        mag = sc * torch.stack([act.dx, act.dy], -1).norm(dim=-1)
        hub_dx, hub_dy = mag * torch.cos(ang), mag * torch.sin(ang)
        span_dir = (p[:, -1] + torch.stack([act.dx, act.dy], -1)) \
            - (p[:, 0] + torch.stack([hub_dx, hub_dy], -1))
        span = span_dir.norm(dim=-1)
        limit = r.slack_bounds[1] * lengths
        over = span > limit
        if bool(over.any()):
            shrink = torch.where(over, (limit / span.clamp_min(1e-9)), torch.ones_like(span))
            hub_dx = hub_dx * shrink
            hub_dy = hub_dy * shrink
        hub_act = terminal_edge_action(
            cfg.n_nodes, hub_dx, hub_dy, r.hub_rot_frac * act.dtheta,
            at_end=False)
        pf, recs = rollout(sim, p, prm_b, act, static_mask=static,
                           n_move=cfg.n_move, n_settle=cfg.n_settle,
                           action2=hub_act, record_at=record_at)
        finite = torch.isfinite(pf).all(dim=(1, 2))
        if not bool(finite.all()):
            raise RuntimeError(
                f"{int((~finite).sum())}/{n_chains} rollouts diverged at chain "
                f"step {step}; check dt against DLOSimulator.stability_dt")

        full = torch.cat([act.as_vector(), hub_act.as_vector()], dim=-1).cpu()
        cur = p.cpu().clone()
        for i in range(n_sub):
            nxt = recs[:, i].cpu()
            a = full.clone()
            a[:, [1, 2, 3, 5, 6, 7]] *= share[i]
            H.append(torch.stack(window, dim=1))
            V_in.append(cur)
            V_out.append(nxt)
            acts.append(a)
            pars.append(par_row)
            rests.append(rest_row)
            tsub.append(torch.full((n_chains,), i, dtype=torch.long))
            window = window[1:] + [cur]
            cur = nxt
        p = pf
        if verbose:
            print(f"  chain step {step+1}/{chain_len} done", flush=True)

    return {
        "hist": torch.cat(H),
        "V_in": torch.cat(V_in),
        "V_out": torch.cat(V_out),
        "action": torch.cat(acts),
        "params": torch.cat(pars),
        "rest_angle": torch.cat(rests),
        "t_sub": torch.cat(tsub),
        "config": {**asdict(cfg), **asdict(r), "n_hist": n_hist, "n_sub": n_sub,
                   "frame_s": cfg.dt * sub},
    }


def _init_batch(batch, n_nodes, lengths, r, dev, g):

    lo, hi = r.slack_init
    fr = lo + (hi - lo) * torch.rand(batch, generator=g, device=dev,
                                     dtype=torch.float64)
    unit = discrete_arc_batch(n_nodes, 1.0 / (n_nodes - 1), chords=fr)
    p = unit * lengths.double().reshape(-1, 1, 1)

    sign = torch.where(torch.rand(batch, 1, generator=g, device=dev,
                                  dtype=torch.float64) < 0.5, -1.0, 1.0)
    p[..., 1] *= sign
    ang = 2 * torch.pi * torch.rand(batch, generator=g, device=dev,
                                    dtype=torch.float64)
    c, s = torch.cos(ang), torch.sin(ang)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return torch.einsum("bij,bkj->bki", rot, p).float()
