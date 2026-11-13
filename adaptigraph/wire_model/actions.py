
from __future__ import annotations

from dataclasses import dataclass

import torch

from .dlo_model import DLOParams, DLOSimulator, NodeConstraints

Tensor = torch.Tensor


@dataclass
class Action:

    alpha: Tensor
    dx: Tensor
    dy: Tensor
    dtheta: Tensor

    def as_vector(self) -> Tensor:
        return torch.stack(
            [self.alpha.to(self.dx.dtype), self.dx, self.dy, self.dtheta], dim=-1
        )

    @staticmethod
    def from_vector(vec: Tensor) -> "Action":
        return Action(
            alpha=vec[:, 0].round().long(),
            dx=vec[:, 1],
            dy=vec[:, 2],
            dtheta=vec[:, 3],
        )

    def to(self, device) -> "Action":
        return Action(
            self.alpha.to(device), self.dx.to(device),
            self.dy.to(device), self.dtheta.to(device),
        )


def grasped_node_targets(p: Tensor, action: Action) -> tuple[Tensor, Tensor]:
    batch = p.shape[0]
    ar = torch.arange(batch, device=p.device)
    idx = torch.stack([action.alpha, action.alpha + 1], dim=-1)

    pa = p[ar, action.alpha]
    pb = p[ar, action.alpha + 1]
    centre = 0.5 * (pa + pb)

    c, s = torch.cos(action.dtheta), torch.sin(action.dtheta)
    rot = torch.stack(
        [torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=-2
    )

    edge = torch.stack([pa, pb], dim=1) - centre.unsqueeze(1)
    rotated = torch.einsum("bij,bkj->bki", rot, edge)
    shift = torch.stack([action.dx, action.dy], dim=-1)
    target = rotated + centre.unsqueeze(1) + shift.unsqueeze(1)
    return idx, target


def rollout(
    sim: DLOSimulator,
    p0: Tensor,
    prm: DLOParams,
    action: Action,
    static_mask: Tensor | None = None,
    n_move: int = 4000,
    n_settle: int = 8000,
    return_traj: bool = False,
    action2: "Action | None" = None,
    record_at: "list[int] | None" = None,
) -> Tensor | tuple[Tensor, Tensor]:
    batch, n_nodes, _ = p0.shape
    device = p0.device
    ar = torch.arange(batch, device=device)

    drives = [grasped_node_targets(p0, action)]
    if action2 is not None:
        drives.append(grasped_node_targets(p0, action2))

    grasp_mask = torch.zeros(batch, n_nodes, dtype=torch.bool, device=device)
    starts = []
    for idx, _ in drives:
        grasp_mask[ar.unsqueeze(-1), idx] = True
        starts.append(torch.stack([p0[ar, idx[:, 0]], p0[ar, idx[:, 1]]], dim=1))

    if static_mask is None:
        static_mask = torch.zeros_like(grasp_mask)
    hold_mask = static_mask & ~grasp_mask
    full_mask = hold_mask | grasp_mask

    p, v = p0.clone(), torch.zeros_like(p0)
    total = n_move + n_settle
    stride = max(1, total // 200)
    traj = []
    record_set = set(record_at) if record_at is not None else set()
    recs = []

    for t in range(total):
        frac = min(1.0, (t + 1) / max(n_move, 1))
        full_target = p0.clone()
        for (idx, target), start in zip(drives, starts):
            full_target[ar.unsqueeze(-1), idx] = start + frac * (target - start)

        p, v = sim.step(p, v, prm, NodeConstraints(full_mask, full_target))
        if return_traj and t % stride == 0:
            traj.append(p.clone())
        if record_at is not None and (t + 1) in record_set:
            recs.append(p.clone())

    if record_at is not None:
        return p, torch.stack(recs, dim=1)
    if return_traj:
        return p, torch.stack(traj, dim=1)
    return p


def terminal_edge_action(
    n_nodes: int,
    dx: Tensor,
    dy: Tensor,
    dtheta: Tensor,
    at_end: bool = True,
) -> Action:
    alpha_val = n_nodes - 2 if at_end else 0
    alpha = torch.full_like(dx, alpha_val, dtype=torch.long)
    return Action(alpha=alpha, dx=dx, dy=dy, dtheta=dtheta)
