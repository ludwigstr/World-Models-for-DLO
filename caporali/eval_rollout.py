
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from estimation import load_predictor
from nn_model import _turning_angles_batch
from train_step_model import build_tensors, turning_angles, node_err

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


def build_sequences(trans, hold_clip, tau):
    import collections
    by = collections.defaultdict(dict)
    for t in trans:
        if t["clip"] == hold_clip:
            continue
        by[(t["clip"], t["k"])][t["frame"]] = t

    Vi, A, Vo = [], [], []
    for seq in by.values():
        for f in sorted(seq):
            steps = [seq.get(f + 4 * i) for i in range(tau)]
            if any(s is None for s in steps):
                continue
            Vi.append(steps[0]["V_in"])
            A.append([s["action"] for s in steps])
            Vo.append([s["V_out"] for s in steps])
    if not Vi:
        raise ValueError(f"no tau={tau} sequences outside clip {hold_clip}")
    return (torch.tensor(np.stack(Vi), dtype=torch.float32),
            torch.tensor(np.stack(A), dtype=torch.float32),
            torch.tensor(np.stack(Vo), dtype=torch.float32))


def train_fold_seq(trans, hold_clip, ckpt, epochs, lr, tau=3, noise_mm=0.0):
    Vi, A, Vo = build_sequences(trans, hold_clip, tau)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(Vi, A, Vo), batch_size=64,
                        shuffle=True, drop_last=True)
    model, _ = load_predictor(ROOT / ckpt, DEV)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.15)
    for _ in range(epochs):
        model.train()
        for vi, a, vo in loader:
            vi, a, vo = vi.to(DEV), a.to(DEV), vo.to(DEV)
            p_par = torch.tensor(NOMINAL, device=DEV).expand(vi.shape[0], -1)
            V = vi
            if noise_mm:
                V = V + torch.randn_like(V) * (noise_mm / 1000.0)
            loss = 0.0
            for i in range(tau):
                r = _turning_angles_batch(V).clamp(-0.35, 0.35)
                pred = model(V, a[:, i], p_par, r)
                loss = loss + torch.nn.functional.mse_loss(pred, vo[:, i])
                V = torch.cat([vo[:, i, :1], pred[:, 1:-1], vo[:, i, -1:]], 1)
            opt.zero_grad(set_to_none=True)
            (loss / tau).backward()
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


def rollout(model, s0, real, n_nodes, n_steps, rest_cap=0.35):
    V = s0.copy().astype(np.float32)
    a = step_action(s0, real, n_nodes - 2, n_steps)
    p = NOMINAL[None]
    for _ in range(n_steps):
        r = np.clip(turning_angles(V), -rest_cap, rest_cap)[None].astype(np.float32)
        with torch.no_grad():
            V = model(torch.tensor(V[None], device=DEV),
                      torch.tensor(a[None], device=DEV),
                      torch.tensor(p, device=DEV),
                      torch.tensor(r, device=DEV))[0].cpu().numpy()
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="data/transitions_all.pt")
    ap.add_argument("--scenes", default="data/e2e_twoend/neue_szenen.pt")
    ap.add_argument("--ckpt", default="data/nn_twoend_transformer.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    trans = torch.load(ROOT / args.transitions, weights_only=False)["transitions"]
    recs = torch.load(ROOT / args.scenes, weights_only=False)["recs"]
    by_clip = {}
    for r in recs:
        by_clip.setdefault(r["clip"], []).append(r)
    n_nodes = recs[0]["s0"].shape[0]

    roll_err, base_err = [], []
    for ci, (clip, rs) in enumerate(sorted(by_clip.items())):
        model = train_fold(trans, clip, args.ckpt, args.epochs, args.lr)
        for r in rs:
            pred = rollout(model, r["s0"], r["real"], n_nodes, args.steps)
            roll_err.append(1000 * np.linalg.norm(pred - r["real"],
                                                  axis=1).mean())
            base_err.append(r["err_causal"] if r["fitted"] else r["err_single"])
        re = np.array(roll_err); print(
            f"  [{ci+1}/{len(by_clip)}] {clip}: bisher Median Rollout "
            f"{np.median(re):.1f} mm", flush=True)

    re = np.array(roll_err); be = np.array(base_err, float)
    ok = np.isfinite(re) & np.isfinite(be)
    re, be = re[ok], be[ok]
    print(f"\n{len(re)} Kabel (leave-one-clip-out)\n")
    print(f"{'':<34}{'Median':>9}{'Mittel':>9}{'p75':>8}")
    print(f"{'aktuell (Physik-Prognose)':<34}{np.median(be):>9.1f}"
          f"{be.mean():>9.1f}{np.percentile(be,75):>8.1f}")
    print(f"{'Rollout (reales Schrittmodell)':<34}{np.median(re):>9.1f}"
          f"{re.mean():>9.1f}{np.percentile(re,75):>8.1f}")
    print(f"\nAenderung Median {100*(np.median(re)-np.median(be))/np.median(be):+.1f}%"
          f"   besser bei {int((re<be).sum())}/{len(re)} Kabeln")


if __name__ == "__main__":
    main()
