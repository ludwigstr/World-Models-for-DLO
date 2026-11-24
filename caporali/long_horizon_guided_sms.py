import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dlo_model import DLOParams, DLOSimulator, NodeConstraints
from train_step_model import turning_angles
from eval_rollout import train_fold, train_fold_seq, step_action, NOMINAL

ROOT = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REST_CAP = 0.35


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


def roll_guided(model, sim, cables, n_nodes, stride, k_p, n_steps,
                exit_prior=False):
    n_frames = len(cables[0])
    idxs = sorted(set(list(range(0, n_frames - 1, stride)) + [n_frames - 1]))
    B = len(cables)
    V = np.stack([s[0] for s in cables]).astype(np.float32)
    p_par = np.tile(NOMINAL, (B, 1)).astype(np.float32)
    lengths = np.array([np.linalg.norm(np.diff(s[0], axis=0), axis=1).sum()
                        for s in cables], np.float32)
    m, k_b, k_d, k_c = NOMINAL

    for ai, bi in zip(idxs, idxs[1:]):
        acts = np.stack([step_action(s[ai], s[bi], n_nodes - 2, 1)
                         for s in cables]).astype(np.float32)
        rest = np.clip(np.stack([turning_angles(x) for x in V]),
                       -REST_CAP, REST_CAP).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(V, device=DEV),
                         torch.tensor(acts, device=DEV),
                         torch.tensor(p_par, device=DEV),
                         torch.tensor(rest, device=DEV)).cpu().numpy()
        pred = pred.astype(np.float32)
        for k, s in enumerate(cables):
            pred[k][0], pred[k][-1] = s[bi][0], s[bi][-1]

        if k_p == 0.0:
            V = pred
            continue

        rest_prm = rest.copy()
        if exit_prior:
            rest_prm[:, 0] = 0.0
            rest_prm[:, -1] = 0.0
        prm = DLOParams(m=m, k_b=k_b, k_d=k_d, k_c=k_c,
                        length=torch.tensor(lengths, device=DEV),
                        rest_angle=torch.tensor(rest_prm, device=DEV),
                        ).to_batched(B, DEV, n_nodes=n_nodes)

        target = torch.tensor(V if k_p < 0 else pred, device=DEV)
        gain = 0.0 if k_p < 0 else k_p
        cons_target = torch.tensor(pred, device=DEV)
        V = guide_batch(sim, torch.tensor(V, device=DEV), target, prm, gain,
                        n_steps, cons_target=cons_target
                        ).cpu().numpy().astype(np.float32)
    return V
