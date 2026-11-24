import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


from twin import stage1, stage2, resim_error
from twin import SpringMassTwin, build_topology


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train-frac", type=float, default=0.7); ap.add_argument("--gens", type=int, default=20); ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--stage2-mode", default="full", choices=["full", "window"])
    ap.add_argument("--stage1-from", default=None)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--clips", default=None); ap.add_argument("--no-stage2", action="store_true"); ap.add_argument("--save-twins", default=None)
    a = ap.parse_args(); dev = a.device
    out = ROOT / a.out; res = json.loads(out.read_text()) if out.exists() else {}
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tracks = sorted(Path(ROOT / a.tracks).glob("*.npz"))
    if a.clips: tracks = [t for t in tracks if t.stem in a.clips.split(",")]
    for npz in tracks:
        clip = npz.stem
        if clip in res: continue
        z = np.load(npz); fps = float(z["fps"]); frame_s = 1.0 / fps
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev); C, T = P.shape[:2]
        n_fit = T if a.train_frac >= 1 else max(3, int(round(a.train_frac * T)))
        t_c = time.time(); log(f"{clip}: {T} frames x 4 cables, fit on {n_fit}")
        if a.stage1_from:
            prev = torch.load(ROOT / a.stage1_from / f"{clip}.pt", weights_only=False)
            x1 = np.asarray(prev["x1"], dtype=np.float64); assert abs(float(prev["frame_s"]) - frame_s) < 1e-9
            r0 = round(float(x1[4]) * 2) / 2; tw0 = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r0, 6), frame_s=frame_s)
            with torch.no_grad():
                k0 = torch.full((C, 1, 1), float(np.exp(x1[0])), device=dev); g0 = torch.exp(torch.tensor(x1[1:4], dtype=torch.float32, device=dev))
                cost1 = resim_error(tw0, P, k0, g0[0].expand(C), g0[1].expand(C), g0[2].expand(C), 0, n_fit, frame_s).mean().item() * 1000
            log(f"{clip}: stage 1 reused from {a.stage1_from}: {cost1:.2f} mm, k {np.exp(x1[0]):.0f} radius {x1[4]:.1f} seg")
        else:
            cost1, x1 = stage1(P, n_fit, dev, gens=a.gens, log=log, frame_s=frame_s)
        if a.no_stage2:
            r = round(float(x1[4]) * 2) / 2; twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), frame_s=frame_s)
            k = torch.full((C, 1, 1), float(np.exp(x1[0])), device=dev); gl = torch.exp(torch.tensor(x1[1:4], dtype=torch.float32, device=dev))
        else:
            twin, k, gl = stage2(P, n_fit, x1, dev, iters=a.iters, log=log, frame_s=frame_s, mode=a.stage2_mode)
        with torch.no_grad():
            e = resim_error(twin, P, k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C), 0, T, frame_s)
            X0 = P[:, 0]; V0 = (P[:, 1] - P[:, 0]) / frame_s; sim = twin.rollout(X0, V0, P[:, :, [0, -1]], k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C))
            ef = torch.nan_to_num((sim[:, :, 1:-1] - P[:, :, 1:-1]).norm(dim=-1), nan=1e3).mean((0, 2)).cpu().numpy() * 1000
        res[clip] = {"start": float(ef[0]), "ende": float(ef[-1]), "mittel": float(ef.mean()), "n_frames": int(T), "video": "",
                     "fit_mm": float(ef[:n_fit].mean()), "test_mm": float(ef[n_fit:].mean()) if n_fit < T else None, "n_fit": n_fit,
                     "stage1_mm": cost1, "k_hom": float(np.exp(x1[0])), "radius_seg": float(x1[4]),
                     "gamma": float(gl[0]), "delta": float(gl[1]), "mu": float(gl[2]),
                     "k_dense_mean": [float(k[c][twin.adj[c].bool()].mean()) for c in range(C)] if k.shape[-1] > 1 else None}
        if a.save_twins:
            d = ROOT / a.save_twins; d.mkdir(parents=True, exist_ok=True)
            torch.save({"k": k.cpu(), "gl": gl.cpu(), "radius_seg": float(x1[4]), "x1": np.asarray(x1), "adj": twin.adj.cpu(), "frame_s": frame_s}, d / f"{clip}.pt")
        out.write_text(json.dumps(res, indent=1))
        log(f"{clip}: all frames {ef.mean():.1f} | fitting frames {res[clip]['fit_mm']:.1f} | future {res[clip]['test_mm']} mm  ({time.time()-t_c:.0f}s)")
    v = np.array([r["mittel"] for r in res.values()]); f = np.array([r["fit_mm"] for r in res.values()]); t = np.array([r["test_mm"] for r in res.values() if r["test_mm"] is not None], dtype=float)
    log(f"PhysTwin on {a.tracks}: {len(v)} clips | all frames {v.mean():.2f} | fitting frames {f.mean():.2f} | future prediction {t.mean() if len(t) else float('nan'):.2f} mm/node")
    log("=== PHYSTWIN_FERTIG ===")


if __name__ == "__main__":
    main()
