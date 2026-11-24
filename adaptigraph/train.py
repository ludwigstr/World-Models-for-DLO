import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from evaluate import dp_of, features, load_tracks, score
from model_v2 import AdaptiGraphV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dlo_dataset_c36_seq.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rest", action="store_true")
    ap.add_argument("--rot", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-hist", type=int, default=3)
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    dev = a.device
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    d = torch.load(ROOT / a.data, weights_only=False)
    hist, X, Y = d["hist"][:, -a.n_hist:], d["V_in"], d["V_out"]
    rest_all = torch.zeros(len(X), 16)
    rest_all[:, 1:-1] = d["rest_angle"]
    kb = d["params"][:, 1:15].numpy()
    lo, hi = np.log(d["config"]["kb"][0]), np.log(d["config"]["kb"][1])
    phi = torch.tensor(((np.log(kb).mean(-1) - lo) / (hi - lo)).astype(np.float32))
    idx = {X[i].numpy().tobytes(): i for i in range(len(X))}
    nxt = np.array([idx.get(Y[i].numpy().tobytes(), -1) for i in range(len(X))])
    del idx
    chain = np.full((len(X), a.k), -1, dtype=np.int64)
    chain[:, 0] = np.arange(len(X))
    for j in range(1, a.k):
        prev = chain[:, j - 1]
        chain[:, j] = np.where(prev >= 0, nxt[prev], -1)
    starts = np.nonzero(chain[:, -1] >= 0)[0]
    log(f"{len(X)} transitions, {len(starts)} starts with {a.k} steps")
    perm = rng.permutation(starts)
    va = perm[: len(perm) // 20]
    va_set = np.zeros(len(X), dtype=bool)
    va_set[chain[va].ravel()] = True
    tr = np.array([s for s in perm[len(perm) // 20:] if not va_set[chain[s]].any()])
    log(f"{len(tr)} training starts, {len(va)} validation starts")

    flags = {"rest": a.rest, "rot": a.rot, "layers": a.layers}
    n_extra = int(a.rest) + int(a.rot)
    m = AdaptiGraphV2(16, a.n_hist, radius=a.radius, layers=a.layers, n_extra=n_extra).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * a.steps_per_epoch)
    clips, P, Tn = load_tracks(dev)

    def k_step(ix, train):
        c = torch.as_tensor(chain[ix])
        cur = X[c[:, 0]].to(dev)
        if train and a.noise > 0:
            cur = cur + a.noise * torch.randn_like(cur)
        h = hist[c[:, 0]].to(dev)
        ph, rest = phi[c[:, 0]].to(dev), rest_all[c[:, 0]].to(dev)
        loss, err = 0.0, 0.0
        for j in range(a.k):
            x_true, y = X[c[:, j]].to(dev), Y[c[:, j]].to(dev)
            pred = m(h, cur, ph, dp_of(x_true, y), features(flags, x_true, y, rest))
            pred = torch.cat([y[:, :1], pred[:, 1:-1], y[:, -1:]], 1)
            loss = loss + ((pred[:, 1:-1] - y[:, 1:-1]) ** 2).mean()
            err = err + (pred[:, 1:-1] - y[:, 1:-1]).norm(dim=-1).mean().detach()
            h = torch.cat([h[:, 1:], cur[:, None]], 1)
            cur = pred
        return loss / a.k, err / a.k

    def save(path, ep, val):
        torch.save({"state_dict": m.state_dict(), "n_hist": a.n_hist, "radius": a.radius, "flags": flags, "epoch": ep, "val_mm": val,
                    "args": vars(a)}, path)

    best, hist_log, t0 = np.inf, [], time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        tl = []
        for _ in range(a.steps_per_epoch):
            loss, _e = k_step(rng.choice(tr, a.batch, replace=False), True)
            opt.zero_grad()
            (loss * 1e4).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
            opt.step()
            sched.step()
            tl.append(_e.item())
        m.eval()
        with torch.no_grad():
            ve = np.mean([k_step(va[i:i + 1024], False)[1].item() for i in range(0, min(len(va), 10240), 1024)]) * 1000
        est, _ = score(m, flags, P, Tn)
        fix, _ = score(m, flags, P, Tn, 0.5)
        mean = lambda e: float(np.mean([e[4 * i:4 * i + 4].mean().item() for i in range(len(clips))]))
        rec = {"epoch": ep, "train_mm": float(np.mean(tl)) * 1000, "val_mm": float(ve), "phi_estimated": mean(est), "phi_0.5": mean(fix), "sec": time.time() - t0}
        hist_log.append(rec)
        tag = ""
        if ve < best:
            best, tag = ve, " *"
            save(out, ep, ve)
        save(out.with_name(out.stem + "_last.pt"), ep, ve)
        out.with_suffix(".log.json").write_text(json.dumps(hist_log, indent=1))
        log(f"ep {ep}: train {rec['train_mm']:.2f} val {ve:.2f} mm ({a.k}-step) | clips: phi estimated {rec['phi_estimated']:.2f}, phi 0.5 {rec['phi_0.5']:.2f}{tag} ({rec['sec']:.0f}s)")
    log("ADAPTIGRAPH2_TRAIN_FERTIG")


if __name__ == "__main__":
    main()
