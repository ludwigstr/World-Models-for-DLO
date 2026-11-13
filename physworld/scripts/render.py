import json
import argparse, glob, re, sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from wire_model.perception.rectify import find_board

SCALE = 0.8
COLORS = [(0.1, 0.6, 0.9), (1.0, 0.6, 0.0), (0.2, 0.8, 0.2), (0.9, 0.2, 0.9)]


def draw_poly(img, pts_px, color_bgr, thick=3, marker=3, dashed=False):
    pts_px = pts_px[np.isfinite(pts_px).all(-1)]
    pts = pts_px.astype(int)
    for i in range(len(pts) - 1):
        if dashed and i % 2 == 1:
            continue
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color_bgr, thick, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), marker, color_bgr, -1, cv2.LINE_AA)


def panel_dirs(clip):
    pan = {}
    for d in glob.glob(str(ROOT / f"data/videos/{clip}/visualization/panels/*_frame_*")):
        pan.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    return pan


def render_clip(row_dir, clip, tracks, fps_out=4.0):
    r = np.load(row_dir / "rollouts" / f"{clip}.npz", allow_pickle=True)
    sim, err, label, short = r["sim"], r["err"], str(r["label"]), str(r["short"])
    sel = r["score_nodes"]
    z = np.load(ROOT / tracks / f"{clip}.npz"); fps = float(z["fps"]); sf = [int(s) for s in z["source_frames"]]
    real = np.stack([z[f"c{k}"] for k in range(4)], 1)
    T = real.shape[0]
    assert sim.shape == real.shape and err.shape == (T,), (sim.shape, real.shape, err.shape)
    scored = ~np.isnan(err)
    e_cab = 1000 * np.linalg.norm(sim[:, :, sel] - real[:, :, sel], axis=-1).mean(-1)
    drawn = np.isfinite(sim).any(axis=(0, 2, 3))
    e_chk = e_cab[:, drawn].mean(-1)
    assert np.allclose(e_chk[scored], err[scored], atol=1e-3), np.abs(e_chk[scored] - err[scored]).max()
    mean = float(err[scored].mean())

    vp = row_dir / f"{short.lower()}_{clip}.mp4"
    worst = int(np.nanargmax(err)); mid = T // 2
    stills = [row_dir / f"{short.lower()}_{clip}_frame{worst:03d}_worst.png",
              row_dir / f"{short.lower()}_{clip}_frame{mid:03d}_middle.png"]
    for p in [vp] + stills:
        if p.exists():
            raise SystemExit(f"{p} exists -- never overwrite rendered videos")
    pan = panel_dirs(clip)
    tf = None; imgs = []
    for f in range(T):
        raw = cv2.imread(f"{pan[sf[f]]}/01_raw.png")
        raw = cv2.resize(raw, (int(raw.shape[1] * SCALE), int(raw.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
        tf = tf or find_board(raw)
        px = lambda s: np.clip(tf.to_px(s * 1000.0), -5000, 5000)
        vis = raw.copy()
        for k in range(4):
            col = COLORS[k]; bgr = (int(255 * col[2]), int(255 * col[1]), int(255 * col[0]))
            draw_poly(vis, px(real[f, k]), bgr, thick=4, marker=5)
            if scored[f]:
                draw_poly(vis, px(sim[f, k]), bgr, thick=3, marker=5, dashed=True)
        tag = f"error {err[f]:.1f} mm/node" if scored[f] else "warm-up (measured, not scored)"
        imgs.append(cv2.cvtColor(cv2.resize(vis, (800, 1064), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    iio.mimsave(vp, imgs, fps=fps_out)
    iio.imwrite(stills[0], imgs[worst]); iio.imwrite(stills[1], imgs[mid])
    return vp, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("row_dir"); ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--tracks", default="../data/tracks"); ap.add_argument("--fps-out", type=float, default=4.0)
    a = ap.parse_args()
    row_dir = Path(a.row_dir).resolve()
    clips = a.clips or sorted(p.stem for p in (row_dir / "rollouts").glob("*.npz"))
    for c in clips:
        if (row_dir / f"{str(np.load(row_dir / 'rollouts' / f'{c}.npz')['short']).lower()}_{c}.mp4").exists():
            print(f"{c}: video exists, skipped (never overwritten)", flush=True); continue
        vp, m = render_clip(row_dir, c, a.tracks, a.fps_out)
        print(f"{c}: {m:.3f} mm/node -> {vp}", flush=True)
    print("=== RENDER_DONE ===", flush=True)



def write_results(out, rows, meta, tol=0.5):
    v = np.array([r["reproduced"] for r in rows.values()]); rv = np.array([r["reported"] for r in rows.values()])
    bad = {c: r["diff"] for c, r in rows.items() if abs(r["diff"]) > tol}
    summ = {**meta, "n_clips": len(rows), "mean_reproduced": float(v.mean()), "mean_reported": float(rv.mean()),
            "max_abs_clip_diff": float(np.abs(v - rv).max()), "tolerance_mm": tol,
            "clips_over_tolerance": bad, "ok": not bad, "clips": rows}
    (out / "results.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: x for k, x in summ.items() if k != "clips"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
