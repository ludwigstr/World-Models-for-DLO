import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wire_model.estimation import feasible_w, load_predictor, uw_to_kb_kd
from wire_model.perception.rectify import board_mask, find_board
from wire_model.perception.segment import detect_modules, segment, wire_components
from wire_model.perception.centerline import (_resample_pixels, strand_centerline,
                                              strand_quality)
from wire_model.perception.tracer import trace_cable_goal
from wire_model.perception.track import detect_hub, strand_target

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = ROOT / "data" / "dlo_nn_rest.pt"
N = 16
COLORS = ["deepskyblue", "orange", "limegreen", "magenta", "gold"]

def read(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()
    return fr if ok else None

def wires_disc_mask(frame, tf, module_centres, conn_radius_px=135):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bm = board_mask(frame, tf, inset_mm=30)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    conn = np.zeros(g.shape, bool)
    for c in module_centres:
        conn |= ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) < conn_radius_px ** 2
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    valid = bm & (~conn)
    thr = max(30.0, float(np.percentile(bh[valid], 99.0)))
    w = ((bh > thr) & valid).astype(np.uint8)
    w = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return w.astype(bool)


def strands_of(frame, tf, min_slack=0.55, module_dilate=12, hub_cut_px=150,
               min_area=600, conn_radius_px=None):
    seg = segment(frame, tf, module_dilate=module_dilate)
    if len(seg.module_centres) < 2:
        return [], seg, None
    wires = (wires_disc_mask(frame, tf, seg.module_centres, conn_radius_px)
             if conn_radius_px else seg.wires)
    ys, xs = np.nonzero(wires)
    if len(xs) == 0:
        return [], seg, None
    P = np.stack([xs, ys], -1).astype(float)
    hub = max(seg.module_centres,
              key=lambda c: int((np.linalg.norm(P - c, axis=1) < 170).sum()))

    cut = wires.copy()
    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
    cut[((xx - hub[0]) ** 2 + (yy - hub[1]) ** 2) < hub_cut_px ** 2] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(cut.astype(np.uint8), 8)
    peripheral = [c for c in seg.module_centres if np.linalg.norm(c - hub) > 60]

    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area or not peripheral:
            continue
        m = (lab == i)
        pts = np.stack(np.nonzero(m)[::-1], -1).astype(float)
        far = pts[int(np.linalg.norm(pts - hub, axis=1).argmax())]
        mod = peripheral[int(np.linalg.norm(np.array(peripheral) - far,
                                            axis=1).argmin())]
        cl = strand_centerline(m, N, hub, mod)
        q = strand_quality(cl, tf)
        sh = tf.to_mm(cl) / 1000.0
        out.append({"shape": sh, "far": sh[-1], "hub": sh[0],
                    "clean": q["slack"] >= min_slack and q["monotonic"],
                    "plug_px": mod})
    return out, seg, hub

def wire_mask_of(frame, tf, conn_radius=135, close=11):
    seg = segment(frame, tf, module_dilate=12)
    w = wires_disc_mask(frame, tf, seg.module_centres, conn_radius)
    w = cv2.morphologyEx(w.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((close, close), np.uint8)).astype(bool)
    return w, seg


def hub_of(wires, module_centres):
    ys, xs = np.nonzero(wires)
    if len(xs) == 0:
        return None
    P = np.stack([xs, ys], -1).astype(float)
    return max(module_centres,
               key=lambda c: int((np.linalg.norm(P - c, axis=1) < 170).sum()))


def traced_cable(wires, hub_px, mod_px, tf, graph=None):
    hub_px = np.asarray(hub_px, float)
    mod_px = np.asarray(mod_px, float)
    poly = trace_cable_goal(wires, mod_px, hub_px, graph=graph)
    if poly is None or len(poly) < 20:
        return None, None
    if np.linalg.norm(poly[0] - hub_px) > np.linalg.norm(poly[-1] - hub_px):
        poly = poly[::-1]
    poly = np.vstack([hub_px[None, :], poly, mod_px[None, :]])
    cl = _resample_pixels(poly, N)
    return tf.to_mm(cl) / 1000.0, strand_quality(cl, tf)


def corridor_cable(wires, hub_px, mod_px, tf, half_width=90):
    ys, xs = np.nonzero(wires)
    if len(xs) < 50:
        return None, None
    P = np.stack([xs, ys], -1).astype(float)
    hub_px = np.asarray(hub_px, float)
    mod_px = np.asarray(mod_px, float)
    axis = mod_px - hub_px
    L = float(np.linalg.norm(axis))
    if L < 1:
        return None, None
    u = axis / L
    rel = P - hub_px
    t = rel @ u
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    sel = (t >= -20) & (t <= L + 20) & (perp < half_width)
    if int(sel.sum()) < 50:
        return None, None
    mask = np.zeros(wires.shape, bool)
    pts = P[sel].astype(int)
    mask[pts[:, 1], pts[:, 0]] = True
    cl = strand_centerline(mask, N, hub_px, mod_px)
    return tf.to_mm(cl) / 1000.0, strand_quality(cl, tf)


def module_centres_px(frame, tf):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detect_modules(g, board_mask(frame, tf, inset_mm=30))

def module_centres_mm(frame, tf):
    mc = module_centres_px(frame, tf)
    return tf.to_mm(mc) / 1000.0 if len(mc) else np.zeros((0, 2))

def track_module_px(cap, tf, start_px, t0, t1, half=95, search=90):
    fr0 = read(cap, t0)
    g0 = cv2.cvtColor(fr0, cv2.COLOR_BGR2GRAY)
    x, y = int(round(start_px[0])), int(round(start_px[1]))
    y_lo, y_hi = max(0, y - half), min(g0.shape[0], y + half)
    x_lo, x_hi = max(0, x - half), min(g0.shape[1], x + half)
    tmpl = g0[y_lo:y_hi, x_lo:x_hi]

    pos = np.asarray(start_px, float)
    path = [pos.copy()]
    worst = 1.0
    for idx in range(t0 + 1, t1 + 1):
        fr = read(cap, idx)
        if fr is None:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        wx0, wy0 = max(0, cx - half - search), max(0, cy - half - search)
        wx1, wy1 = min(g.shape[1], cx + half + search), min(g.shape[0], cy + half + search)
        win = g[wy0:wy1, wx0:wx1]
        if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
            path.append(pos.copy())
            continue
        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        pos = np.array([wx0 + loc[0] + tmpl.shape[1] / 2.0,
                        wy0 + loc[1] + tmpl.shape[0] / 2.0], float)
        worst = min(worst, float(mx))
        path.append(pos.copy())
    return pos, np.array(path), worst

def turning_angles(shape):
    u = np.diff(shape, axis=0); a, b = u[:-1], u[1:]
    return np.arctan2(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0], (a*b).sum(1))

def _param_row(model, mass, k_b, k_d, k_c=0.0):
    row = [float(mass), float(k_b), float(k_d)]
    if getattr(model.net, "n_params", 3) >= 4:
        row.append(float(k_c))
    return torch.tensor([row], device=DEV)


def _action_vec(model, shape0, mover_future, hub_future=None):
    d = mover_future - shape0[-1]
    a = [N - 2, d[0], d[1], 0.0]
    if getattr(model.net, "n_action", 4) >= 8:
        dh = (hub_future - shape0[0]) if hub_future is not None else np.zeros(2)
        a += [0.0, dh[0], dh[1], 0.0]
    return np.array(a, dtype=np.float32)


def predict(model, shape0, mover_future, mass, ratio=0.02, rest_cap=0.35,
            hub_future=None):
    rest = np.clip(turning_angles(shape0), -rest_cap, rest_cap)
    u = torch.log(torch.tensor(ratio)); k_b, k_d = uw_to_kb_kd(u, feasible_w(u))
    action = _action_vec(model, shape0, mover_future, hub_future)
    with torch.no_grad():
        pred = model(torch.tensor(shape0[None], dtype=torch.float32, device=DEV),
                     torch.tensor(action[None], device=DEV),
                     _param_row(model, mass, float(k_b), float(k_d)),
                     torch.tensor(rest[None], dtype=torch.float32, device=DEV))
    return pred[0].cpu().numpy()


def predict_fitted(model, shape0, mover_future, fp, mass,
                   hub_future=None):
    action = _action_vec(model, shape0, mover_future, hub_future)
    with torch.no_grad():
        pred = model(torch.tensor(shape0[None], dtype=torch.float32, device=DEV),
                     torch.tensor(action[None], device=DEV),
                     _param_row(model, mass, fp["k_b"], fp["k_d"],
                                fp.get("k_c", 0.0)),
                     fp["rest"].to(DEV).unsqueeze(0))
    return pred[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--t1", type=int, required=True)
    ap.add_argument("--mass", type=float, default=0.02)
    ap.add_argument("--fitted", default=None)
    ap.add_argument("--conn-radius", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    vid = args.video; name = Path(vid).stem
    out = args.out or str(ROOT / "data" / f"forecast_frames_{name}_{args.t0}_{args.t1}.png")
    cap = cv2.VideoCapture(vid); fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    tf = find_board(read(cap, 0))
    frame0 = read(cap, args.t0); frame1 = read(cap, args.t1)
    w0, seg0 = wire_mask_of(frame0, tf, args.conn_radius or 135)
    w1, seg1w = wire_mask_of(frame1, tf, args.conn_radius or 135)
    hub0 = hub_of(w0, seg0.module_centres)
    peripheral0 = [c for c in seg0.module_centres
                   if np.linalg.norm(c - hub0) > 60]
    s0 = []
    for mod in peripheral0:
        sh, q = traced_cable(w0, hub0, mod, tf)
        if sh is None:
            continue
        s0.append({"shape": sh, "far": sh[-1], "hub": sh[0],
                   "clean": True, "plug_px": mod, "q": q})
    modules0 = module_centres_mm(frame0, tf)
    print(f"{name}: t0={args.t0} ({args.t0/fps:.0f}s) {len(s0)} clean strands, "
          f"t1={args.t1} ({args.t1/fps:.0f}s)")

    mods0_px = module_centres_px(frame0, tf)
    mods1_px = module_centres_px(frame1, tf)
    print(f"  Module erkannt: t0 {len(mods0_px)}, t1 {len(mods1_px)}")

    hub0_px = np.asarray(hub0, float)
    hub1_px, _, hub_corr = track_module_px(cap, tf, hub0_px, args.t0, args.t1)
    hub_disp = (tf.to_mm(hub1_px) - tf.to_mm(hub0_px)) / 1000.0
    hub_move_px = float(np.linalg.norm(hub1_px - hub0_px))
    print(f"  HUB bewegt sich: ({hub0_px[0]:.0f},{hub0_px[1]:.0f}) -> "
          f"({hub1_px[0]:.0f},{hub1_px[1]:.0f}) px = {hub_move_px:.0f} px "
          f"/ {1000*np.linalg.norm(hub_disp):.0f} mm  (Korrelation {hub_corr:.3f})")

    fitted = None
    if args.fitted:
        fitted = torch.load(ROOT / args.fitted, weights_only=False)["results"]
        print(f"  gefittete Parameter geladen fuer Kabel {sorted(fitted)}")

    results = []
    for ci, st in enumerate(s0):
        far_px = tf.to_px(st["far"] * 1000.0)
        plug0_px = (mods0_px[int(np.linalg.norm(mods0_px - far_px, axis=1).argmin())]
                    if len(mods0_px) else far_px)
        plug1_px, path_px, corr = track_module_px(cap, tf, plug0_px, args.t0, args.t1)

        if len(mods1_px):
            k = int(np.linalg.norm(mods1_px - plug1_px, axis=1).argmin())
            det1_px = mods1_px[k]
            check_px = float(np.linalg.norm(det1_px - plug1_px))
        else:
            det1_px, check_px = plug1_px, float("nan")

        disp_mm = (tf.to_mm(plug1_px) - tf.to_mm(plug0_px)) / 1000.0
        mf = st["far"] + disp_mm
        move_px = float(np.linalg.norm(plug1_px - plug0_px))
        move_mm = float(np.linalg.norm(disp_mm) * 1000)

        mf_rel = st["far"] + (disp_mm - hub_disp)
        fp = fitted.get(ci) if fitted else None
        if fp is not None:
            pred = predict_fitted(model, st["shape"], mf_rel, fp,
                                  args.mass) + hub_disp
        else:
            pred = predict(model, st["shape"], mf_rel, args.mass) + hub_disp
        real, qreal = traced_cable(w1, hub1_px, plug1_px, tf)
        err = (1000*np.linalg.norm(pred - real, axis=1).mean()) if real is not None else None
        results.append({"s0": st["shape"], "mover0": st["far"], "mover1": mf,
                        "pred": pred, "real": real, "move": move_mm,
                        "err": err, "plug0_px": plug0_px, "plug1_px": plug1_px,
                        "det1_px": det1_px, "check_px": check_px,
                        "move_px": move_px, "corr": corr})
    cap.release()

    corners = np.array([[70, 70], [1130, 70], [1130, 1680], [70, 1680]])
    pxc = tf.to_px(corners)
    Hn, Wn = frame0.shape[:2]
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(Wn, int(pxc[:, 0].max())), min(Hn, int(pxc[:, 1].max()))
    def px(s): return tf.to_px(s * 1000.0) - np.array([x0, y0])
    rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 8.0))
    ax[0].imshow(rgb0)
    ax[1].imshow((0.35 * rgb1 + 60).clip(0, 255).astype(np.uint8))

    errs, moves = [], []
    for i, r in enumerate(results):
        col = COLORS[i % len(COLORS)]
        p0, pp = px(r["s0"]), px(r["pred"])
        off_ = np.array([x0, y0])
        m0 = r["plug0_px"] - off_; m1 = r["plug1_px"] - off_
        ax[0].plot(p0[:, 0], p0[:, 1], "--", color=col, lw=2.6)
        ax[0].plot([m0[0]], [m0[1]], "o", color=col, ms=12, mec="k")

        ax[1].plot(p0[:, 0], p0[:, 1], "--", color="0.55", lw=1.2, alpha=0.75)
        ax[1].annotate("", xy=(m1[0], m1[1]), xytext=(m0[0], m0[1]),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        if r["real"] is not None:
            prr = px(r["real"])
            ax[1].plot(prr[:, 0], prr[:, 1], ":", color=col, lw=3.4, alpha=0.95)
        ax[1].plot(pp[:, 0], pp[:, 1], "--", color=col, lw=2.6)
        ax[1].plot([m1[0]], [m1[1]], "*", color=col, ms=19, mec="k")
        moves.append(r["move"])
        if r["err"] is not None:
            errs.append(r["err"])

        off = np.array([x0, y0])
        a0, a1 = r["plug0_px"] - off, r["plug1_px"] - off
        box = dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65, ec="none")
        ax[0].text(a0[0] + 18, a0[1] - 14,
                   f"M{i} ({r['plug0_px'][0]:.0f}, {r['plug0_px'][1]:.0f}) px",
                   color=col, fontsize=9, weight="bold", bbox=box)
        ax[1].text(a1[0] + 18, a1[1] - 14,
                   f"M{i} ({r['plug1_px'][0]:.0f}, {r['plug1_px'][1]:.0f}) px\n"
                   f"Δ {r['move_px']:.0f} px = {r['move']:.0f} mm",
                   color=col, fontsize=9, weight="bold", bbox=box)
    off = np.array([x0, y0])
    h0d, h1d = hub0_px - off, hub1_px - off
    hbox = dict(boxstyle="round,pad=0.25", fc="black", alpha=0.72, ec="none")
    ax[0].plot([h0d[0]], [h0d[1]], "s", color="white", ms=13, mec="k")
    ax[0].text(h0d[0] + 18, h0d[1] + 34,
               f"HUB ({hub0_px[0]:.0f}, {hub0_px[1]:.0f}) px",
               color="white", fontsize=8.5, weight="bold", bbox=hbox)
    ax[1].annotate("", xy=(h1d[0], h1d[1]), xytext=(h0d[0], h0d[1]),
                   arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
    ax[1].plot([h1d[0]], [h1d[1]], "s", color="white", ms=13, mec="k")
    ax[1].text(h1d[0] + 18, h1d[1] + 34,
               f"HUB ({hub1_px[0]:.0f}, {hub1_px[1]:.0f}) px\n"
               f"\u0394 {hub_move_px:.0f} px = {1000*np.linalg.norm(hub_disp):.0f} mm",
               color="white", fontsize=8.5, weight="bold", bbox=hbox)

    for a in ax: a.set_xticks([]); a.set_yticks([])
    ax[0].set_title(f"Ausgangsbild  (t = {args.t0/fps:.0f} s, Frame {args.t0})\n"
                    f"{len(results)} Kabel gemessen  |  Punkte = Mover, "
                    f"Quadrat = Hub", fontsize=11)
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[
        Line2D([0], [0], color="0.25", ls="--", lw=2.6, label="Prognose"),
        Line2D([0], [0], color="0.25", ls=":", lw=3.4, label="real gemessen")],
        loc="lower left", fontsize=9, framealpha=0.85)
    escore = (f"  |  Abweichung {np.mean(errs):.0f} mm/Knoten ({len(errs)}/{len(results)} Kabel)"
              if errs else "")
    ax[1].set_title(f"Prognosezielbild  (t = {args.t1/fps:.0f} s, Frame {args.t1})\n"
                    f"Pfeile = Moverbewegung{escore}", fontsize=11)
    fig.suptitle(f"Kabelprognose {args.t0/fps:.0f} s \u2192 {args.t1/fps:.0f} s   |   {name}   |   "
                 f"Mover bis {max(moves):.0f} mm, HUB {1000*np.linalg.norm(hub_disp):.0f} mm",
                 fontsize=13, y=0.99)
    fig.tight_layout(); fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"\n{'Mover':<6} {'t0 (px)':>16} {'t1 (px)':>16} {'Weg':>16} {'Korr':>7}")
    for i, r in enumerate(results):
        e = f"{r['err']:.1f} mm/Knoten" if r["err"] is not None else "kein sauberes t1-Kabel"
        print(f"M{i:<5} ({r['plug0_px'][0]:6.0f},{r['plug0_px'][1]:6.0f}) "
              f"({r['plug1_px'][0]:6.0f},{r['plug1_px'][1]:6.0f}) "
              f"{r['move_px']:6.0f}px/{r['move']:4.0f}mm {r['corr']:7.3f}")

if __name__ == "__main__":
    model, _ = load_predictor(CKPT, DEV)
    main()
