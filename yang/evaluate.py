import argparse
import glob
import json
import sys
from pathlib import Path
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


from model import INBiLSTM, end_actions, turning
from simulator import CLIP_TRIMS


TRACKS = Path("../data/tracks")


@torch.no_grad()
def rollout(net, P, dt, warm=1, rest_shape=None):
    C, T = P.shape[:2]
    rest = torch.zeros(C, 16, device=P.device); rest[:, 1:-1] = turning(P[:, 0] if rest_shape is None else rest_shape)
    out = [P[:, t] for t in range(warm)]
    for t in range(warm - 1, T - 1):
        p = out[t]; prev = out[t - 1] if t > 0 else out[0]
        v = (p - prev) / dt
        act = end_actions(P[:, t], P[:, t + 1])
        nxt = p + net(p, v, act, rest)
        nxt[:, 0] = P[:, t + 1, 0]; nxt[:, -1] = P[:, t + 1, -1]
        out.append(nxt)
    return torch.stack(out, 1)


def frame_err(pred, P):
    return 1000 * (pred - P).norm(dim=-1).mean(-1).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tracks", default="")
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    net = INBiLSTM(use_rest=ck["use_rest"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    dt = ck["dt"]
    res = {}
    tracks = ROOT / a.tracks if a.tracks else TRACKS
    clip_list = sorted(q.stem for q in tracks.glob("*.npz")) if a.tracks else sorted(CLIP_TRIMS)
    print(f"tracks {tracks}: {len(clip_list)} clips", flush=True)
    for clip in clip_list:
        z = np.load(tracks / f"{clip}.npz")
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        assert abs(1.0 / float(z["fps"]) - dt) < 1e-6
        e = frame_err(rollout(net, P, dt), P)
        ew = frame_err(rollout(net, P, dt, warm=5), P)
        res[clip] = {"start": float(e[1]), "ende": float(e[-1]), "mittel": float(e[1:].mean()),
                     "mittel_incl0": float(e.mean()), "mittel_warm5": float(ew[5:].mean()),
                     "n_frames": int(len(e)), "per_frame": e.round(3).tolist()}
        print(f"{clip}: {res[clip]['mittel']:.2f} mm/node (incl. frame 0 {res[clip]['mittel_incl0']:.2f}, "
              f"5-frame warm-up {res[clip]['mittel_warm5']:.2f}), {len(e)} frames", flush=True)
    summ = {}
    for key in ("mittel", "mittel_incl0", "mittel_warm5"):
        v = np.array([r[key] for r in res.values()])
        summ[key] = {"mean": float(v.mean())}
    report = {"ckpt": a.ckpt, "tracks": str(tracks), "clips": res, "summary": summ, "n_clips": len(res),
              "protocol": "frame 0 measured, v0 = 0, open loop, measured end motions, ends pinned; "
                          "mittel = mean over predicted frames 1..T-1"}
    if a.synth:
        files = sorted(glob.glob(str(ROOT / a.synth / "*.npz")))
        rng = np.random.default_rng(ck["args"]["seed"])
        if "val_files" in ck:
            names = set(ck["val_files"])
            val = [i for i, f in enumerate(files) if Path(f).name in names]
        else:
            val = sorted(rng.permutation(len(files))[: max(1, len(files) // 10)].tolist())
        sv = []
        for i in val:
            zz = np.load(files[i])
            V = zz["V"]
            thr = ck["args"]["max_gap"]
            if ck["args"].get("gap_ref_tol") is not None:
                import re
                r0 = Path(re.sub(r"_\d{2}$", "", files[i][:-4]) + "_00.npz")
                if r0.exists():
                    thr = max(thr, float(np.load(r0)["gap"].max()) + ck["args"]["gap_ref_tol"])
            ok = np.nonzero(zz["gap"] > thr)[0]
            T = int(ok[0]) if len(ok) else len(V)
            if T < 3:
                continue
            Pv = torch.tensor(V[:T].transpose(1, 0, 2, 3), dtype=torch.float32, device=dev)
            R = torch.tensor(zz["rest"], dtype=torch.float32, device=dev)
            sv.append(float(frame_err(rollout(net, Pv, dt, rest_shape=R), Pv)[1:].mean()))
        report["synthetic_val_rollouts"] = {"n": len(sv), "mean": float(np.mean(sv))}
        print(f"held-out synthetic rollouts (source/source): {len(sv)}, mean {np.mean(sv):.2f} mm/node")
    out.write_text(json.dumps(report, indent=1))
    for key, s in summ.items():
        print(f"{key}: mean {s['mean']:.2f} mm/node over {len(res)} clips")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
