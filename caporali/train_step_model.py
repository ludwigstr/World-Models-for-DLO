
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from estimation import load_predictor
from geometry import polyline_from_angles

ROOT = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def node_err(pred, tgt):
    return 1000 * (pred - tgt).norm(dim=-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="data/transitions_all.pt")
    ap.add_argument("--ckpt", default="data/nn_twoend_transformer.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--test-clips", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = torch.load(ROOT / args.transitions, weights_only=False)
    trans = d["transitions"]
    clips = sorted({t["clip"] for t in trans})
    rng = np.random.default_rng(args.seed)
    test_clips = set(rng.choice(clips, args.test_clips, replace=False))
    tr = [t for t in trans if t["clip"] not in test_clips]
    te = [t for t in trans if t["clip"] in test_clips]
    print(f"{len(trans)} Uebergaenge, {len(clips)} Clips")
    print(f"Test-Clips ({len(test_clips)}): {sorted(test_clips)}")
    print(f"Training {len(tr)}  Test {len(te)}\n")

    n_params = int(torch.load(ROOT / args.ckpt, weights_only=False)
                   .get("net", {}).get("n_params", 4)) if Path(ROOT / args.ckpt).exists() else 4
    n_kb = max(1, n_params - 3)
    nominal = np.array([0.02] + [0.1] * n_kb + [10.0, 0.15], np.float32)
    Vi_tr, Vo_tr, A_tr, P_tr, R_tr = build_tensors(tr, nominal)
    Vi_te, Vo_te, A_te, P_te, R_te = build_tensors(te, nominal)

    model, ck = load_predictor(ROOT / args.ckpt, DEV)
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        ident = node_err(Vi_te.to(DEV), Vo_te.to(DEV)).mean().item()
        phys = node_err(model(Vi_te.to(DEV), A_te.to(DEV), P_te.to(DEV),
                              R_te.to(DEV)), Vo_te.to(DEV)).mean().item()
    print(f"{'identity (V_out=V_in)':<28}{ident:6.2f} mm/Knoten")
    print(f"{'Physik-Surrogat':<28}{phys:6.2f} mm/Knoten  (vor Fine-Tuning)\n")

    loader = DataLoader(TensorDataset(Vi_tr, Vo_tr, A_tr, P_tr, R_tr),
                        batch_size=128, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(loader),
        pct_start=0.15)

    best = phys
    for ep in range(args.epochs):
        model.train()
        for vi, vo, a, p, r in loader:
            vi, vo = vi.to(DEV), vo.to(DEV)
            a, p, r = a.to(DEV), p.to(DEV), r.to(DEV)
            loss = torch.nn.functional.mse_loss(model(vi, a, p, r), vo)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            e = node_err(model(Vi_te.to(DEV), A_te.to(DEV), P_te.to(DEV),
                               R_te.to(DEV)), Vo_te.to(DEV)).mean().item()
        best = min(best, e)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:>3}/{args.epochs}  Test {e:6.2f} mm/Knoten")

    print(f"\n{'real (fine-getunt)':<28}{best:6.2f} mm/Knoten")
    print(f"\nAenderung gegen Physik: {100*(best-phys)/phys:+.0f}%")
    if best < phys - 0.3:
        print("-> echte Dynamik bringt Signal; Rollout lohnt sich")
    else:
        print("-> kein Signal ueber die Physik hinaus")


if __name__ == "__main__":
    main()
