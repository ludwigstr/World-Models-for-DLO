import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


from model import RopeDreamer
from model import load_tracks, load_synth, WindowDataset, state_offset, trajectory_tensors, state_stats
import model as RD


OBS_NOISE = {'sigma_perp': 0.00072, 'sigma_tan': 0.0001, 'end_ext': 0.0035, 'end_ext_sd': 0.001, 'straighten': 0.5, 'sigma_end': 0.0005, 'sigma_shape_static': 0.0075, 'sigma_shape_dyn': 0.0095, 'rho_shape': 0.9, 'end_off0': 0.004, 'end_off15': 0.002, 'smooth_max': 0.6}


def open_loop_mm(model, trajs_t, off, warm, horizon, device, n_max=64):
    model.eval()
    errs, recs = [], []
    for t in trajs_t[:n_max]:
        T = t["s"].shape[0]
        n = min(horizon, T - warm)
        if n < 1:
            continue
        s = (t["s"][None] - off).to(device); idx = t["idx"][None].to(device); dp = t["dp"][None].to(device)
        h, z = model.warmup(s[:, :warm], idx[:, :warm], dp[:, :warm])
        pred = model.imagine(h, z, idx[:, warm - 1:warm - 1 + n], dp[:, warm - 1:warm - 1 + n]) + off.to(device)
        truth = t["P"][warm:warm + n].to(device)
        if model.repr == "chain":
            pred = pred.clone(); pred[0, :, :2] = truth[:, 0]
        P = model.positions(pred[0], t["ell"].to(device))[..., :2].clone(); P[:, 0] = truth[:, 0]; P[:, -1] = truth[:, -1]
        errs.append((P - truth).norm(dim=-1).mean(-1).cpu().numpy() * 1000)
        m = warm + n
        out = model(s[:, :m], idx[:, :m], dp[:, :m], sample=False, ell=t["ell"][None].to(device))
        rec = out["s_recon"][0] + off.to(device)
        tr = t["P"][:m].to(device)
        if model.repr == "chain":
            rec = rec.clone(); rec[:, :2] = tr[:, 0]
        Pr = model.positions(rec, t["ell"].to(device))[..., :2].clone(); Pr[:, 0] = tr[:, 0]; Pr[:, -1] = tr[:, -1]
        recs.append((Pr - tr).norm(dim=-1).mean().item() * 1000)
    model.train()
    all_mean = float(np.mean([e.mean() for e in errs]))
    first10 = float(np.mean([e[:10].mean() for e in errs]))
    last = float(np.mean([e[-1] for e in errs]))
    return all_mean, first10, last, float(np.mean(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument
    ap.add_argument("--tracks", nargs="+", default=["../data/tracks"])
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--size", default="S")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-nats", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=0.0)
    ap.add_argument("--rollout-weight", type=float, default=0.0)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--obs-noise", action="store_true")
    ap.add_argument("--gap-ref-tol", type=float, default=None)
    ap.add_argument("--val-by-rollout", action="store_true")
    ap.add_argument("--val-by-clip", action="store_true")
    ap.add_argument("--deadline", default=None)
    ap.add_argument("--out", default="data/ropedreamer_S.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--select", default="val_loss", choices=["val_loss", "openloop"])
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--repr", default="chain", choices=["chain", "positions"])
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    RD.REPR = a.repr
    log_f = open(ROOT / (a.out.replace(".pt", "_log.jsonl")), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    trajs = load_synth([ROOT / d for d in a.synth], gap_ref_tol=a.gap_ref_tol) if a.synth else load_tracks([ROOT / d for d in a.tracks])
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(trajs))
    if a.val_frac > 0 and a.val_by_clip:
        import re as _re
        src = [_re.sub(r"_\d{2,3}$", "", t["clip"]) for t in trajs]
        clips_u = sorted(set(src))
        cperm = rng.permutation(len(clips_u))
        n_vc = max(1, int(round(a.val_frac * len(clips_u))))
        vclips = {clips_u[i] for i in cperm[:n_vc]}
        val = [t for t, c in zip(trajs, src) if c in vclips]; train = [t for t, c in zip(trajs, src) if c not in vclips]
        assert not ({c for c in src if c in vclips} & {c for c in src if c not in vclips})
        print(f"clip-level split: {len(clips_u) - n_vc} train clips / {n_vc} val clips", flush=True)
    elif a.val_frac > 0 and a.val_by_rollout:
        rolls = sorted({t["clip"] for t in trajs})
        rperm = rng.permutation(len(rolls))
        vr = {rolls[i] for i in rperm[:max(1, int(round(a.val_frac * len(rolls))))]}
        val = [t for t in trajs if t["clip"] in vr]; train = [t for t in trajs if t["clip"] not in vr]
        print(f"rollout-level split: {len(rolls) - len(vr)} train rollouts / {len(vr)} val rollouts", flush=True)
    elif a.val_frac > 0:
        n_val = max(1, int(a.val_frac * len(trajs)))
        val = [trajs[i] for i in perm[:n_val]]; train = [trajs[i] for i in perm[n_val:]]
    else:
        val = list(trajs); train = list(trajs)
    off = state_offset(trajs)
    obs_noise = OBS_NOISE if a.obs_noise else None
    if obs_noise:
        print(f"obs-noise augmentation: {obs_noise}", flush=True)
    ds_tr = WindowDataset(train, a.seq_len, seed=a.seed, obs_noise=obs_noise)
    ds_va = WindowDataset(val, a.seq_len, seed=a.seed + 1, obs_noise=obs_noise)
    if obs_noise:
        def _wif(_):
            info = torch.utils.data.get_worker_info()
            info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))
        lkw = {"num_workers": a.loader_workers, "worker_init_fn": _wif, "persistent_workers": True}
    else:
        lkw = {}
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True, drop_last=True, **lkw)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=a.batch, **lkw)
    val_t = [trajectory_tensors(t["P"]) for t in val]
    model = RopeDreamer(16, a.size, beta=a.beta, free_nats=a.free_nats, pos_weight=a.pos_weight,
                        rollout_weight=a.rollout_weight, warm=a.warm, repr=a.repr).to(a.device)
    if a.init:
        ck0 = torch.load(ROOT / a.init, map_location=a.device, weights_only=False)
        model.load_state_dict(ck0["state_dict"]); off = ck0["offset"].cpu()
        if hasattr(model, "pretrain_ae"):
            for p_ in list(model.enc.parameters()) + list(model.dec.parameters()):
                p_.requires_grad_(False)
        log(f"init from {a.init} (epoch {ck0.get('epoch')}, val loss {ck0.get('val_loss', float('nan')):.3f}); state statistics kept")
    else:
        s_mean, s_std = state_stats(train)
        model.set_state_stats((s_mean - off).to(a.device), s_std.to(a.device))
    if hasattr(model, "pretrain_ae") and not a.init:
        S = torch.cat([trajectory_tensors(t["P"])["s"] for t in train]) - off
        model.pretrain_ae(S, a.device, log=log)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"=== RopeDreamer {a.size} ({n_par/1e6:.2f}M params): {len(train)} train / {len(val)} val trajectories "
        f"({sum(t['P'].shape[0] for t in trajs)} frames), seq_len {a.seq_len}, batch {a.batch}, lr {a.lr}, beta {a.beta}, pos_weight {a.pos_weight}, rollout_weight {a.rollout_weight} ===")
    best, bad, hist = float("inf"), 0, []
    off_d = off.to(a.device)
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); acc = {"loss": 0, "recon": 0, "pred": 0, "kl": 0, "roll": 0}; n = 0
        for batch in dl_tr:
            s, idx, dp, ell = batch[:4]
            s, idx, dp, ell = s.to(a.device) - off_d, idx.to(a.device), dp.to(a.device), ell.to(a.device)
            if len(batch) > 4:
                out = model(s, idx, batch[5].to(a.device), ell=ell, s_obs=batch[4].to(a.device) - off_d)
            else:
                out = model(s, idx, dp, ell=ell)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            for k in acc: acc[k] += float(out[k]); 
            n += 1
        model.eval(); va = {"loss": 0, "recon": 0, "pred": 0, "kl": 0, "roll": 0}; m = 0
        with torch.no_grad():
            for batch in dl_va:
                s, idx, dp, ell = batch[:4]
                if len(batch) > 4:
                    out = model(s.to(a.device) - off_d, idx.to(a.device), batch[5].to(a.device), sample=False,
                                ell=ell.to(a.device), s_obs=batch[4].to(a.device) - off_d)
                else:
                    out = model(s.to(a.device) - off_d, idx.to(a.device), dp.to(a.device), sample=False, ell=ell.to(a.device))
                for k in va: va[k] += float(out[k])
                m += 1
        ol, ol10, ol50, rec_mm = open_loop_mm(model, val_t, off, a.warm, a.horizon, a.device)
        rec = {"epoch": ep, **{k: v / n for k, v in acc.items()}, **{"val_" + k: v / m for k, v in va.items()},
               "val_openloop_mm": ol, "val_openloop_mm_1_10": ol10, "val_openloop_mm_50": ol50, "val_recon_mm": rec_mm}
        hist.append(rec)
        score = va["loss"] / m if a.select == "val_loss" else ol
        improved = score < best
        if improved:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "size": a.size, "n_nodes": 16, "offset": off,
                        "seq_len": a.seq_len, "beta": a.beta, "pos_weight": a.pos_weight, "rollout_weight": a.rollout_weight, "warm": a.warm, "repr": a.repr, "model": "ropedreamer", "free_nats": a.free_nats, "epoch": ep, "val_loss": va["loss"] / m, "select": a.select, "best_score": best, "init": a.init, "hist": hist,
                        "tracks": a.tracks, "val_clips": sorted({(t["clip"], t["cable"]) for t in val}.__iter__())}, ROOT / a.out)
        else:
            bad += 1
        log(f"ep {ep:3d}: train loss {rec['loss']:.4f} (recon {rec['recon']:.4f} pred {rec['pred']:.4f} kl {rec['kl']:.2f} roll {rec.get('roll', 0):.5f}) | "
            f"val loss {rec['val_loss']:.4f} pred {rec['val_pred']:.4f} | recon {rec_mm:.1f} mm | open-loop 50 steps: {ol:.1f} mm (1-10: {ol10:.1f}, @50: {ol50:.1f})"
            f"{' *' if improved else ''} ({time.time()-t0:.0f}s)")
        if bad >= a.patience:
            log(f"early stop: no improvement of {a.select} for {a.patience} epochs"); break
        if a.deadline and time.time() + (time.time() - t0) > time.mktime(time.strptime(a.deadline, "%Y-%m-%d %H:%M")):
            log(f"DEADLINE STOP: the next epoch would end after {a.deadline} (epoch {ep}, {bad} epochs since the best)"); break
    log(f"=== ROPEDREAMER_TRAIN_FERTIG: best {a.select} {best:.4f} -> {a.out} ===")


if __name__ == "__main__":
    main()
