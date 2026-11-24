import sys
from pathlib import Path

import os
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_scene import (track_clip, _rotz, DIR, COLORS,
                          BOARD_W, BOARD_H, HUB_CAVS, PLUG_CAV, CAV_Z,
                          PLATFORM_HALF, PLATE_HALF, PLATE_Z, MODULE_HALF,
                          MODULE_Z, PLATFORM_Z, CAV_R)

ROOT = Path(__file__).resolve().parent
N_NODES = 16
HUB_RED = (0.85, 0.15, 0.15, 1)
GREY = (0.55, 0.55, 0.55, 1)
BOARD_RGBA = (0.72, 0.72, 0.70, 1)


def platform_xml(name):
    return f'''<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <body name="{name}" pos="0 0 {PLATFORM_Z:.5f}">
      <joint name="{name}_x" type="slide" axis="1 0 0" damping="8"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" damping="8"/>
      <geom type="box" size="{PLATFORM_HALF[0]} {PLATFORM_HALF[1]} {PLATFORM_HALF[2]}"
            mass="2"/>
    </body>
  </worldbody>
</mujoco>'''


def topper_xml(name, yaw):
    return f'''<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <body name="{name}" pos="0 0 {PLATFORM_Z:.5f}">
      <joint name="{name}_x" type="slide" axis="1 0 0" damping="8"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" damping="8"/>
      <geom type="box" pos="0 0 {PLATE_Z:.5f}"
            size="{PLATE_HALF[0]} {PLATE_HALF[1]} {PLATE_HALF[2]}"
            mass="0.5"/>
      <body name="{name}_mod" pos="0 0 {MODULE_Z:.5f}" euler="0 0 {np.degrees(yaw):.4f}">
        <joint name="{name}_turn" type="hinge" axis="0 0 1" damping="0.5"/>
        <geom type="box" size="{MODULE_HALF} {MODULE_HALF} {MODULE_HALF}"
              mass="0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>'''


def cable_xml(route_m, stiffness=0.1, damping=0.5, radius=0.003, mass_per_m=0.0368,
             rest_angle=None,
             anchor_stiffness=200.0, weld_anchor=True):
    n = len(route_m)
    density = mass_per_m / (np.pi * radius ** 2)

    def _per_joint(v, name):
        if np.isscalar(v):
            return lambda i: f"{float(v):.6g}"
        a = np.asarray(v, float).ravel()
        if a.size != n - 1:
            raise ValueError(f"{name}: {a.size} Werte, erwartet {n - 1}")
        return lambda i: f"{a[i - 1]:.6g}"

    _st = _per_joint(stiffness, "stiffness")
    _dp = _per_joint(damping, "damping")

    def seg_geom(i):
        if i == n - 1:
            return f'<geom type="sphere" size="{radius:.5f}"/>'
        d = route_m[i + 1] - route_m[i]
        return (f'<geom type="capsule" fromto="0 0 0 {d[0]:.5f} {d[1]:.5f} {d[2]:.5f}" '
               f'size="{radius:.5f}" density="{density:.0f}"/>')

    body = (f'<body name="seg0" pos="{route_m[0][0]:.5f} {route_m[0][1]:.5f} {route_m[0][2]:.5f}">\n'
           f'  <joint type="free"/>\n  {seg_geom(0)}')
    tail = ""
    for i in range(1, n):
        d = route_m[i] - route_m[i - 1]
        body += (f'\n  <body name="seg{i}" pos="{d[0]:.5f} {d[1]:.5f} {d[2]:.5f}">\n'
                f'    <joint name="h{i}z" type="hinge" axis="0 0 1" '
                f'stiffness="{0.0 if rest_angle is not None else _st(i)}" '
                f'damping="{_dp(i)}"/>\n'
                f'    <joint name="h{i}y" type="hinge" axis="0 1 0" '
                f'stiffness="{0.0 if rest_angle is not None else _st(i)}" '
                f'damping="{_dp(i)}"/>\n'
                f'    {seg_geom(i)}')
        tail += "</body>"
    body += tail + "</body>"
    last = n - 1
    _eq = (f'<weld body1="anchor" body2="seg{last}" solref="0.004 1"/>'
           if weld_anchor else
           f'<connect body1="anchor" body2="seg{last}" anchor="0 0 0" solref="0.004 1"/>')

    return f'''<mujoco>
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    {body}
    <body name="anchor" pos="{route_m[-1][0]:.5f} {route_m[-1][1]:.5f} {route_m[-1][2]:.5f}">
      <joint type="free"/>
      <geom type="sphere" size="{radius*0.5:.5f}" contype="0" conaffinity="0" mass="0.0001"/>
    </body>
  </worldbody>
  <equality>
    {_eq}
  </equality>
</mujoco>'''


def board_xml(mu=1.0):
    return f'''<mujoco>
  <worldbody>
    <geom type="plane" size="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0.02" pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0"
          friction="{mu} 0.05 0.002"/>
    <light pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 2.5" dir="0.2 0.3 -1" diffuse=".7 .7 .7"/>
    <light pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 1.5" dir="0 0 -1" diffuse=".3 .3 .3"/>
  </worldbody>
</mujoco>'''


def mover_pose(P, is_hub):
    if is_hub:
        raise ValueError("hub pose needs all 4 routes, see build_scene()")
    v = P[-1] - P[-2]
    yaw = np.arctan2(v[1], v[0]) + np.pi / 2
    off = _rotz(np.array(PLUG_CAV), yaw)
    return np.array([P[-1][0] - off[0], P[-1][1] - off[1]]), yaw


def _sweep_stale_pid_dirs(root, keep_below=64):
    try:
        dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("pid")]
    except OSError:
        return
    if len(dirs) < keep_below:
        return
    import shutil
    for d in dirs:
        try:
            pid = int(d.name[3:])
        except ValueError:
            continue
        if pid == os.getpid() or Path(f"/proc/{pid}").exists():
            continue
        shutil.rmtree(d, ignore_errors=True)


def cable_link_order(entity, n_nodes=16):
    idx = {l.name: i for i, l in enumerate(entity.links)}
    return [idx[f"seg{i}"] for i in range(n_nodes)]


def apply_rest_angles(ents, rest, stiffness=0.1, kv=0.5):
    import numpy as _np
    if rest is None:
        return 0
    rest = _np.asarray(rest, float)
    n_set = 0
    for k in range(4):
        e = ents.get(f"cable{k}")
        if e is None:
            continue
        idx, tgt, gains = [], [], []
        r = rest[k] if rest.ndim == 2 else rest
        st = stiffness[k] if _np.ndim(stiffness) == 2 else stiffness
        for i in range(1, len(r) + 1):
            for ax in ("z", "y"):
                try:
                    j = e.get_joint(f"h{i}{ax}")
                except Exception:
                    continue
                idx += list(j.dofs_idx_local)
                tgt.append(float(r[i - 1]) if ax == "z" else 0.0)
                gains.append(float(st[i - 1] if _np.ndim(st) else st))
        if idx:
            e.set_dofs_kp(_np.asarray(gains, _np.float32), idx)
            e.set_dofs_kv(_np.full(len(idx), kv, _np.float32), idx)
            e.control_dofs_position(_np.asarray(tgt, _np.float32), idx)
            n_set += len(idx)
    return n_set


def _per_cable(cable_kw, k):
    out = dict(cable_kw)
    for key in ("stiffness", "damping", "rest_angle"):
        v = out.get(key)
        if v is not None and np.ndim(v) == 2:
            out[key] = np.asarray(v)[k]
    if np.ndim(out.get("mass_per_m", 0.0)) == 1:
        out["mass_per_m"] = float(np.asarray(out["mass_per_m"])[k])
    return out


def build_scene(shp, frame_idx=0, mu=1.0, **cable_kw):
    import genesis as gs

    routes2 = [shp[k][frame_idx].astype(float) for k in range(4)]
    z = CAV_Z
    routes = [np.hstack([P, np.full((len(P), 1), z)]) for P in routes2]

    hub_xy = routes2[0][0]
    out_dir = np.mean([P[1] - P[0] for P in routes2], axis=0)
    hub_yaw = np.arctan2(out_dir[1], out_dir[0]) + np.pi / 2
    hub_off = _rotz(np.array(HUB_CAVS).mean(0), hub_yaw)
    hub_pos = np.array([hub_xy[0] - hub_off[0], hub_xy[1] - hub_off[1]])

    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=1e-3))

    xml_root = ROOT / "data" / "genesis_full"
    xml_dir = xml_root / f"pid{os.getpid()}"
    xml_dir.mkdir(parents=True, exist_ok=True)
    _sweep_stale_pid_dirs(xml_root)

    def write(name, xml):
        p = xml_dir / f"{name}.xml"
        p.write_text(xml)
        return p

    ents = {}
    p = write("board", board_xml(mu=mu))
    ents["board"] = scene.add_entity(gs.morphs.MJCF(file=str(p)),
                                     surface=gs.surfaces.Default(color=BOARD_RGBA))

    def add_mover(name, pos, yaw, color):
        p = write(name, platform_xml(name))
        ents[name] = scene.add_entity(
            gs.morphs.MJCF(file=str(p), pos=(float(pos[0]), float(pos[1]), 0)),
            surface=gs.surfaces.Default(color=color))
        p = write(f"{name}_top", topper_xml(name, yaw))
        ents[f"{name}_top"] = scene.add_entity(
            gs.morphs.MJCF(file=str(p), pos=(float(pos[0]), float(pos[1]), 0)),
            surface=gs.surfaces.Default(color=GREY))

    add_mover("hub", hub_pos, hub_yaw, HUB_RED)

    plug_yaws, plug_poss = {}, {}
    for k, P in enumerate(routes2):
        pos, yaw = mover_pose(P, is_hub=False)
        plug_yaws[k] = yaw
        plug_poss[k] = pos
        add_mover(f"plug{k}", pos, yaw, (*COLORS[k], 1))

    for k in range(4):
        p = write(f"cable{k}", cable_xml(routes[k], **_per_cable(cable_kw, k)))
        ents[f"cable{k}"] = scene.add_entity(
            gs.morphs.MJCF(file=str(p)),
            surface=gs.surfaces.Default(color=(*COLORS[k], 1)))

    return scene, ents, hub_pos, hub_yaw, plug_poss, plug_yaws


def render_full_video(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
                      kp_mover=4000.0, kv_mover=120.0, kp_turn=60.0, kv_turn=4.0,
                      kp_cable=80.0, kv_cable=4.0, fps_out=4.0, mu=1.0,
                      pin_cable_ends=False,
                      out_dir="data/genesis_full_video", **cable_kw):
    import cv2
    import genesis as gs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import imageio.v2 as iio
    from fit_clip import read
    from rectify import find_board

    t1 = None
    if t1_sec is not None:
        cap_probe = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 4.0
        cap_probe.release()
        t1 = int(round(t1_sec * fps_probe))
    shp, t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))

    scene, ents, hub_pos0, hub_yaw0, plug_pos0, plug_yaw0 = build_scene(
        shp, frame_idx=0, mu=mu, **cable_kw)

    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    tf = find_board(read(cap, 0))
    cam_pos, cam_lookat, cam_up, cam_fov = genesis_camera(tf)
    cam = scene.add_camera(res=(560, 784), pos=tuple(cam_pos), lookat=tuple(cam_lookat),
                           up=tuple(cam_up), fov=cam_fov)
    scene.build()


    names = ["hub"] + [f"plug{k}" for k in range(4)]
    dofs, dofs_top = {}, {}
    for n in names:
        e = ents[n]
        dofs[n] = e.get_joint(f"{n}_x").dofs_idx_local + e.get_joint(f"{n}_y").dofs_idx_local
        e.set_dofs_kp(np.full(2, kp_mover), dofs[n])
        e.set_dofs_kv(np.full(2, kv_mover), dofs[n])

        et = ents[f"{n}_top"]
        xy_top = et.get_joint(f"{n}_x").dofs_idx_local + et.get_joint(f"{n}_y").dofs_idx_local
        turn_top = et.get_joint(f"{n}_turn").dofs_idx_local
        dofs_top[n] = (xy_top, turn_top)
        et.set_dofs_kp(np.full(2, kp_mover), xy_top)
        et.set_dofs_kv(np.full(2, kv_mover), xy_top)
        et.set_dofs_kp(np.full(1, kp_turn), turn_top)
        et.set_dofs_kv(np.full(1, kv_turn), turn_top)
    cable_dofs = {}
    for k in range(4):
        e = ents[f"cable{k}"]
        e.set_dofs_kp(np.full(e.n_dofs, kp_cable))
        e.set_dofs_kv(np.full(e.n_dofs, kv_cable))
        cable_dofs[k] = (np.arange(0, 3), np.arange(e.n_dofs - 6, e.n_dofs - 3),
                         np.arange(3, 6), np.arange(e.n_dofs - 3, e.n_dofs))

    apply_rest_angles(ents, cable_kw.get("rest_angle"),
                      cable_kw.get("stiffness", 0.1))

    _ref_ang = {}
    for _k in range(4):
        _r0 = shp[_k][0]
        _ref_ang[_k] = (float(np.arctan2(*(_r0[1] - _r0[0])[::-1])),
                        float(np.arctan2(*(_r0[-1] - _r0[-2])[::-1])))

    def end_angles(k, t):
        r = shp[k][t]
        a0 = float(np.arctan2(*(r[1] - r[0])[::-1]))
        a1 = float(np.arctan2(*(r[-1] - r[-2])[::-1]))
        _w = lambda a: float(np.arctan2(np.sin(a), np.cos(a)))
        return _w(a0 - _ref_ang[k][0]), _w(a1 - _ref_ang[k][1])

    def unwrap(prev, cur):
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    def poses(t):
        r2 = [shp[k][t] for k in range(4)]
        out = np.mean([P[1] - P[0] for P in r2], axis=0)
        hy = np.arctan2(out[1], out[0]) + np.pi / 2
        hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
        res = {"hub": (np.array([r2[0][0][0] - hoff[0], r2[0][0][1] - hoff[1]]), hy)}
        for k, P in enumerate(r2):
            v = P[-1] - P[-2]
            y = np.arctan2(v[1], v[0]) + np.pi / 2
            o = _rotz(np.array(PLUG_CAV), y)
            res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
        return res

    base = {n: (hub_pos0 if n == "hub" else plug_pos0[int(n[-1])]) for n in names}
    base_yaw = {n: (hub_yaw0 if n == "hub" else plug_yaw0[int(n[-1])]) for n in names}
    prev_yaw = dict(base_yaw)

    def apply(pose):
        for n in names:
            xy, yaw = pose[n]
            yaw_u = unwrap(prev_yaw[n], yaw)
            prev_yaw[n] = yaw_u
            rel_xy = xy - base[n]
            rel_yaw = np.array([yaw_u - base_yaw[n]])
            ents[n].control_dofs_position(rel_xy, dofs[n])
            ents[f"{n}_top"].control_dofs_position(rel_xy, dofs_top[n][0])
            ents[f"{n}_top"].control_dofs_position(rel_yaw, dofs_top[n][1])
        for k in range(4):
            z = np.full((16, 1), CAV_Z)
            route = np.hstack([shp[k][t_cur], z])
            e = ents[f"cable{k}"]
            if pin_cable_ends:
                e.set_dofs_position(route[0], cable_dofs[k][0])
                e.set_dofs_position(route[-1], cable_dofs[k][1])
                a0, a1 = end_angles(k, t)
                e.set_dofs_position(np.array([0.0, 0.0, a0]), cable_dofs[k][2])
                e.set_dofs_position(np.array([0.0, 0.0, a1]), cable_dofs[k][3])
                e.set_dofs_velocity(np.zeros(3), cable_dofs[k][0])
                e.set_dofs_velocity(np.zeros(3), cable_dofs[k][1])
            else:
                e.control_dofs_position(route[0], cable_dofs[k][0])
                e.control_dofs_position(route[-1], cable_dofs[k][1])
                a0, a1 = end_angles(k, t)
                e.control_dofs_position(np.array([0.0, 0.0, a0]), cable_dofs[k][2])
                e.control_dofs_position(np.array([0.0, 0.0, a1]), cable_dofs[k][3])

    t_cur = 0
    apply(poses(0))
    for _ in range(300):
        scene.step()

    outdir = ROOT / out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / clip
    frames_dir.mkdir(exist_ok=True)

    corners_mm = ((70, 70), (1130, 70), (1130, 1680), (70, 1680))
    pxc = tf.to_px(np.array(corners_mm, dtype=float))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))

    def px(shape_m):
        return tf.to_px(shape_m * 1000.0) - np.array([x0, y0])

    crop_h, crop_w = y1 - y0, x1 - x0
    panel_h_in = 7.0
    panel_w_in = panel_h_in * crop_w / crop_h
    figsize = (2 * panel_w_in + 0.6, panel_h_in + 0.5)

    errs, frame_paths = [], []
    for t in range(n_frames):
        t_cur = t
        apply(poses(t))
        for _ in range(steps_per_frame):
            scene.step()
        cam.render(rgb=True)

        sim_shapes = []
        for k in range(4):
            e = ents[f"cable{k}"]
            pos = e.get_links_pos().cpu().numpy()[cable_link_order(e)]
            sim_shapes.append(pos[:, :2])
        e_all = [float(np.linalg.norm(sim_shapes[k] - shp[k][t], axis=1).mean()) * 1000
                for k in range(4)]
        errs.append(float(np.mean(e_all)))

        fr = read(cap, t0 + t)
        photo = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        img = cam.render(rgb=True)[0]

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(photo)
        for k in range(4):
            col = COLORS[k]
            P, Q = px(shp[k][t]), px(sim_shapes[k])
            ax[0].plot(P[:, 0], P[:, 1], ":", color=col, lw=3.2, alpha=0.95)
            ax[0].plot(Q[:, 0], Q[:, 1], "--", color=col, lw=2.4)
        ax[0].legend(handles=[
            Line2D([0], [0], color="0.25", ls=":", lw=3.2, label="real gemessen"),
            Line2D([0], [0], color="0.25", ls="--", lw=2.4, label="Genesis")],
            loc="lower left", fontsize=9, framealpha=0.85)
        ax[0].set_title(f"Video t = {(t0+t)/fps:.2f} s   Abweichung {errs[-1]:.0f} mm/Knoten",
                        fontsize=11)
        ax[1].imshow(img)
        ax[1].set_title("Genesis-Rendering", fontsize=11)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{clip}  |  Frame {t+1}/{n_frames}", fontsize=12)
        fig.tight_layout()
        p = frames_dir / f"f{t:03d}.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(p)
    cap.release()

    pngs = [iio.imread(p) for p in frame_paths]
    h = min(p.shape[0] for p in pngs); w = min(p.shape[1] for p in pngs)
    video_path = outdir / f"vergleich_{clip}.mp4"
    iio.mimsave(video_path, [p[:h, :w] for p in pngs], fps=fps_out)

    print(f"{clip}: Start {errs[0]:.1f}  Ende {errs[-1]:.1f}  Mittel {np.mean(errs):.1f} mm/Knoten")
    print(f"-> {video_path}")
    return video_path, errs


def eval_clip(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
              kp_mover=4000.0, kv_mover=120.0, kp_turn=60.0, kv_turn=4.0,
              kp_cable=80.0, kv_cable=4.0, mu=1.0, pin_cable_ends=False,
              nodes_out=None, nodes3_out=None, shp_fps=None, **cable_kw):
    import cv2
    from harness_scene import track_clip as _track_clip

    t1 = None
    if t1_sec is not None:
        cap_probe = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 4.0
        cap_probe.release()
        t1 = int(round(t1_sec * fps_probe))
    if shp_fps is not None:
        shp, fps = shp_fps
    else:
        shp, t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))

    scene, ents, hub_pos0, hub_yaw0, plug_pos0, plug_yaw0 = build_scene(
        shp, frame_idx=0, mu=mu, **cable_kw)
    scene.build()

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    dofs, dofs_top = {}, {}
    for n in names:
        e = ents[n]
        dofs[n] = e.get_joint(f"{n}_x").dofs_idx_local + e.get_joint(f"{n}_y").dofs_idx_local
        e.set_dofs_kp(np.full(2, kp_mover), dofs[n])
        e.set_dofs_kv(np.full(2, kv_mover), dofs[n])

        et = ents[f"{n}_top"]
        xy_top = et.get_joint(f"{n}_x").dofs_idx_local + et.get_joint(f"{n}_y").dofs_idx_local
        turn_top = et.get_joint(f"{n}_turn").dofs_idx_local
        dofs_top[n] = (xy_top, turn_top)
        et.set_dofs_kp(np.full(2, kp_mover), xy_top)
        et.set_dofs_kv(np.full(2, kv_mover), xy_top)
        et.set_dofs_kp(np.full(1, kp_turn), turn_top)
        et.set_dofs_kv(np.full(1, kv_turn), turn_top)
    cable_dofs = {}
    for k in range(4):
        e = ents[f"cable{k}"]
        e.set_dofs_kp(np.full(e.n_dofs, kp_cable))
        e.set_dofs_kv(np.full(e.n_dofs, kv_cable))
        cable_dofs[k] = (np.arange(0, 3), np.arange(e.n_dofs - 6, e.n_dofs - 3),
                         np.arange(3, 6), np.arange(e.n_dofs - 3, e.n_dofs))

    apply_rest_angles(ents, cable_kw.get("rest_angle"),
                      cable_kw.get("stiffness", 0.1))

    _ref_ang = {}
    for _k in range(4):
        _r0 = shp[_k][0]
        _ref_ang[_k] = (float(np.arctan2(*(_r0[1] - _r0[0])[::-1])),
                        float(np.arctan2(*(_r0[-1] - _r0[-2])[::-1])))

    def end_angles(k, t):
        r = shp[k][t]
        a0 = float(np.arctan2(*(r[1] - r[0])[::-1]))
        a1 = float(np.arctan2(*(r[-1] - r[-2])[::-1]))
        _w = lambda a: float(np.arctan2(np.sin(a), np.cos(a)))
        return _w(a0 - _ref_ang[k][0]), _w(a1 - _ref_ang[k][1])

    def unwrap(prev, cur):
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    def poses(t):
        r2 = [shp[k][t] for k in range(4)]
        out = np.mean([P[1] - P[0] for P in r2], axis=0)
        hy = np.arctan2(out[1], out[0]) + np.pi / 2
        hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
        res = {"hub": (np.array([r2[0][0][0] - hoff[0], r2[0][0][1] - hoff[1]]), hy)}
        for k, P in enumerate(r2):
            v = P[-1] - P[-2]
            y = np.arctan2(v[1], v[0]) + np.pi / 2
            o = _rotz(np.array(PLUG_CAV), y)
            res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
        return res

    base = {n: (hub_pos0 if n == "hub" else plug_pos0[int(n[-1])]) for n in names}
    base_yaw = {n: (hub_yaw0 if n == "hub" else plug_yaw0[int(n[-1])]) for n in names}
    prev_yaw = dict(base_yaw)

    def apply_movers(pose):
        for n in names:
            xy, yaw = pose[n]
            yaw_u = unwrap(prev_yaw[n], yaw)
            prev_yaw[n] = yaw_u
            rel_xy = xy - base[n]
            rel_yaw = np.array([yaw_u - base_yaw[n]])
            ents[n].control_dofs_position(rel_xy, dofs[n])
            ents[f"{n}_top"].control_dofs_position(rel_xy, dofs_top[n][0])
            ents[f"{n}_top"].control_dofs_position(rel_yaw, dofs_top[n][1])

    def pin_cable(t):
        for k in range(4):
            z = np.full((16, 1), CAV_Z)
            route = np.hstack([shp[k][t], z])
            e = ents[f"cable{k}"]
            e.set_dofs_position(route[0], cable_dofs[k][0])
            e.set_dofs_position(route[-1], cable_dofs[k][1])
            e.set_dofs_velocity(np.zeros(3), cable_dofs[k][0])
            e.set_dofs_velocity(np.zeros(3), cable_dofs[k][1])

    def pin_cable_pd(t):
        for k in range(4):
            z = np.full((16, 1), CAV_Z)
            route = np.hstack([shp[k][t], z])
            e = ents[f"cable{k}"]
            e.control_dofs_position(route[0], cable_dofs[k][0])
            e.control_dofs_position(route[-1], cable_dofs[k][1])
            a0, a1 = end_angles(k, t)
            e.control_dofs_position(np.array([0.0, 0.0, a0]), cable_dofs[k][2])
            e.control_dofs_position(np.array([0.0, 0.0, a1]), cable_dofs[k][3])

    def step(t):
        if pin_cable_ends:
            pin_cable(t)
        scene.step()

    apply_movers(poses(0))
    if not pin_cable_ends:
        pin_cable_pd(0)
    else:
        pin_cable(0)
    for _ in range(300):
        step(0)

    errs = []
    for t in range(n_frames):
        apply_movers(poses(t))
        if not pin_cable_ends:
            pin_cable_pd(t)
        for _ in range(steps_per_frame):
            step(t)
        sim_shapes, sim3 = [], []
        for k in range(4):
            e = ents[f"cable{k}"]
            pos = e.get_links_pos().cpu().numpy()[cable_link_order(e)]
            sim_shapes.append(pos[:, :2])
            if nodes3_out is not None:
                sim3.append(pos[:, :3])
        e_all = [float(np.linalg.norm(sim_shapes[k] - shp[k][t], axis=1).mean()) * 1000
                for k in range(4)]
        errs.append(float(np.mean(e_all)))
        if nodes3_out is not None:
            nodes3_out.append(np.stack(sim3))
        if nodes_out is not None:
            nodes_out.append(np.stack(sim_shapes))

    return errs


def genesis_camera(tf, height_m=15.0, pad_mm=60.0):
    from board_camera import board_camera, BOARD_H_MM
    pos, right_3d, up_3d = board_camera(tf, height_m=height_m)
    fovy_m = BOARD_H_MM / 1000.0 + 2 * pad_mm / 1000.0
    fov_deg = 2 * np.degrees(np.arctan(fovy_m / (2 * height_m)))
    lookat = pos.copy(); lookat[2] = 0.0
    up = np.array([0.0, 1.0, 0.0])
    return pos, lookat, up, fov_deg
