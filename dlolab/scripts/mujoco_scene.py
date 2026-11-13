import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_clip import read
from wire_model.perception.harness import track_harness
from wire_model.perception.rectify import find_board
from wire_model.perception.tracking import HarnessTracker

ROOT = Path(__file__).resolve().parents[1]
DIR = "data/WirePrediction"
BOARD_W, BOARD_H = 1.200, 1.680
COLORS = [(0.1, 0.6, 0.9), (1.0, 0.6, 0.0), (0.2, 0.8, 0.2), (0.9, 0.2, 0.9)]


def track_clip(clip, t0=8, t1=None, max_frames=400):
    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    n_frames = min(n_all, max_frames)
    tf = find_board(read(cap, 0))
    tracks, hub_i, _ = track_harness(cap, tf, n_frames, read, clip=clip)
    order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
    tracks = tracks[:, order, :]
    t1 = (n_frames - 1) if t1 is None else t1
    tracked = HarnessTracker(tf, clip=clip).run(cap, read, tracks, range(t0, t1 + 1))
    cap.release()
    shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in range(t0, t1 + 1)])
           for k in range(4)}
    return shp, t0, t1, fps


def _fmt_vertices(P):
    rel = P - P[0]
    return "\n              ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in rel)


SC = 0.5
PLATFORM_HALF = (0.15 * SC, 0.15 * SC, 0.01 * SC)
PLATE_HALF    = (0.08 * SC, 0.08 * SC, 0.02 * SC)
PLATE_Z       = 0.03 * SC
MODULE_HALF   = 0.05 * SC
MODULE_Z      = 0.10 * SC
CAV_R         = 0.01 * SC
PLATFORM_Z    = 0.01 * SC
HUB_CAVS  = [(0.0, -0.05 * SC, 0.0), (-0.02 * SC, -0.05 * SC, 0.0),
             (0.02 * SC, -0.05 * SC, 0.0), (0.0, -0.05 * SC, 0.02 * SC)]
PLUG_CAV  = (0.0, 0.05 * SC, 0.0)
CAV_Z = PLATFORM_Z + MODULE_Z


def _yaw_quat(yaw):
    return (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


def snap_yaw(yaw):
    return np.round(yaw / (np.pi / 2)) * (np.pi / 2)


def _rotz(v, yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2]])


def mover_body(name, pos, yaw, rgba, points, cav_rgba):
    cav_xml = "\n".join(
        f'''        <body name="{name}_{tag}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}">
          <geom type="sphere" size="{CAV_R}" rgba="{cav_rgba}" contype="0" conaffinity="0"/>
        </body>''' for tag, c in points)
    return f'''    <body name="{name}" pos="{pos[0]:.5f} {pos[1]:.5f} {PLATFORM_Z:.5f}">
      <joint name="{name}_x" type="slide" axis="1 0 0" damping="8"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" damping="8"/>
      <geom type="box" size="{PLATFORM_HALF[0]} {PLATFORM_HALF[1]} {PLATFORM_HALF[2]}"
            rgba="{rgba}" mass="2"/>
      <geom type="box" pos="0 0 {PLATE_Z:.5f}"
            size="{PLATE_HALF[0]} {PLATE_HALF[1]} {PLATE_HALF[2]}"
            rgba="0.5 0.5 0.5 1" mass="0.5"/>
      <body name="{name}_mod" pos="0 0 {MODULE_Z:.5f}" euler="0 0 {np.degrees(yaw):.4f}">
        <joint name="{name}_turn" type="hinge" axis="0 0 1" damping="0.5"/>
        <geom type="box" size="{MODULE_HALF} {MODULE_HALF} {MODULE_HALF}"
              rgba="0.5 0.5 0.5 1" mass="0.3"/>
{cav_xml}
      </body>
    </body>'''


def lift_profile(P2, radius, lift_frac=0.18):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P2, axis=0), axis=1))]
    t = d / max(d[-1], 1e-9)
    ramp = np.minimum(t / lift_frac, (1.0 - t) / lift_frac).clip(0.0, 1.0)
    z = CAV_Z + (radius - CAV_Z) * ramp
    return np.column_stack([P2, z])


def build_xml(shp, frame_idx=0, radius=0.003, bend=1.0e5, twist=1.0e5,
              damping=0.15, mass_per_m=0.15, mu=1.0, weld_solref=None,
              condim=3, mu_tor=0.05, mu_roll=0.002):
    routes2 = [shp[k][frame_idx].astype(float) for k in range(4)]
    routes = [lift_profile(P, radius) for P in routes2]

    hub_xy = routes2[0][0]
    out_dir = np.mean([P[1] - P[0] for P in routes2], axis=0)
    hub_yaw = np.arctan2(out_dir[1], out_dir[0]) + np.pi / 2
    hub_off = _rotz(np.array(HUB_CAVS).mean(0), hub_yaw)
    hub_pos = np.array([hub_xy[0] - hub_off[0], hub_xy[1] - hub_off[1]])

    hub_relief_dist = [float(np.linalg.norm(routes[k][1] - routes[k][0])) for k in range(4)]
    plug_relief_dist = [float(np.linalg.norm(routes[k][-1] - routes[k][-2])) for k in range(4)]

    plugs, plug_cav_world = [], []
    for k, P in enumerate(routes2):
        v = P[-1] - P[-2]
        yaw = np.arctan2(v[1], v[0]) + np.pi / 2
        off = _rotz(np.array(PLUG_CAV), yaw)
        pos = np.array([P[-1][0] - off[0], P[-1][1] - off[1]])
        cav_y = PLUG_CAV[1]
        relief_pt = (PLUG_CAV[0], cav_y + plug_relief_dist[k], PLUG_CAV[2])
        plugs.append(mover_body(f"plug{k}", pos, yaw,
                                f"{COLORS[k][0]} {COLORS[k][1]} {COLORS[k][2]} 1",
                                [("cav0", PLUG_CAV), ("relief0", relief_pt)], "1 1 0 1"))
        plug_cav_world.append(pos)

    hub_relief_pts = [("cav" + str(i), c) for i, c in enumerate(HUB_CAVS)]
    hub_relief_pts += [(f"relief{i}", (c[0], c[1] - hub_relief_dist[i], c[2]))
                       for i, c in enumerate(HUB_CAVS)]
    hub = mover_body("hub", hub_pos, hub_yaw, "0.85 0.15 0.15 1", hub_relief_pts, "1 0 0 1")

    cables = "\n".join(
        f'''    <composite prefix="Wire{k}" type="cable" initial="free"
      offset="{r[0][0]:.5f} {r[0][1]:.5f} {r[0][2]:.5f}"
      vertex="{_fmt_vertices(r)}">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{twist:g}"/>
        <config key="bend" value="{bend:g}"/>
        <config key="vmax" value="0.05"/>
      </plugin>
      <joint kind="main" damping="{damping}"/>
      <geom type="capsule" size="{radius}" rgba="{COLORS[k][0]} {COLORS[k][1]} {COLORS[k][2]} 1"
            condim="{condim}" friction="{mu} {mu_tor} {mu_roll}" margin="0.001"
            solref="-150000 -500" solimp="0.999 0.9999 0.0001"
            density="{mass_per_m / (np.pi * radius ** 2):.0f}"/>
    </composite>''' for k, r in enumerate(routes))

    reliefs = "\n".join(
        f'''    <body name="hub_relief_wire{k}" pos="{routes[k][1][0]:.5f} {routes[k][1][1]:.5f} {routes[k][1][2]:.5f}">
      <joint type="free" damping="{damping}"/>
      <geom type="sphere" size="0.0025" rgba="1 1 0.3 1" contype="0" conaffinity="0" mass="0.0005"/>
    </body>
    <body name="plug{k}_relief_wire" pos="{routes[k][-2][0]:.5f} {routes[k][-2][1]:.5f} {routes[k][-2][2]:.5f}">
      <joint type="free" damping="{damping}"/>
      <geom type="sphere" size="0.0025" rgba="1 1 0.3 1" contype="0" conaffinity="0" mass="0.0005"/>
    </body>''' for k in range(4))

    actuators = "\n".join(
        f'''    <position name="{nm}_xa" joint="{nm}_x" kp="4000" kv="120" ctrlrange="-3 3"/>
    <position name="{nm}_ya" joint="{nm}_y" kp="4000" kv="120" ctrlrange="-3 3"/>
    <position name="{nm}_ta" joint="{nm}_turn" kp="60" kv="4" ctrlrange="-20 20"/>'''
        for nm in ["hub"] + [f"plug{k}" for k in range(4)])

    seg_len = [float(np.linalg.norm(routes[k][-1] - routes[k][-2])) for k in range(4)]
    stiff = 'solimp="0.99 0.999 0.0001" solref="0.002 1"'
    connects = "\n".join(
        f'''    <connect name="hub_Wire{k}" body1="Wire{k}B_first" body2="hub_cav{k}" anchor="0 0 0"/>
    <connect name="plug_Wire{k}" body1="Wire{k}B_last" body2="plug{k}_cav0"
             anchor="{seg_len[k]:.5f} 0 0"/>
    <connect name="hub_relief_a{k}" body1="hub_relief_wire{k}" body2="Wire{k}B_1" anchor="0 0 0" {stiff}/>
    <connect name="hub_relief_b{k}" body1="hub_relief_wire{k}" body2="hub_relief{k}" anchor="0 0 0" {stiff}/>
    <connect name="plug_relief_a{k}" body1="plug{k}_relief_wire" body2="Wire{k}B_last" anchor="0 0 0" {stiff}/>
    <connect name="plug_relief_b{k}" body1="plug{k}_relief_wire" body2="plug{k}_relief0" anchor="0 0 0" {stiff}/>'''
        for k in range(4))

    welds = ""
    if weld_solref is not None:
        n_joints = 14
        if np.isscalar(weld_solref):
            wsr = [[float(weld_solref)] * n_joints for _ in range(4)]
        elif np.ndim(weld_solref) == 1:
            assert len(weld_solref) == n_joints, \
                f"weld_solref braucht {n_joints} Werte, hat {len(weld_solref)}"
            wsr = [list(weld_solref) for _ in range(4)]
        else:
            assert len(weld_solref) == 4 and all(len(row) == n_joints for row in weld_solref), \
                f"weld_solref braucht Form (4, {n_joints})"
            wsr = weld_solref
        tags = ["first"] + [str(i) for i in range(1, 13 + 1)] + ["last"]
        weld_lines = []
        for k in range(4):
            for j in range(n_joints):
                b1, b2 = f"Wire{k}B_{tags[j]}", f"Wire{k}B_{tags[j + 1]}"
                weld_lines.append(
                    f'    <weld name="Wire{k}_seg{j}" body1="{b1}" body2="{b2}" '
                    f'anchor="0 0 0" solimp="0.99 0.999 0.0001" solref="{wsr[k][j]:g} 1"/>')
        welds = "\n" + "\n".join(weld_lines)


    return f'''<mujoco model="WireHarnessMeasured">
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <compiler autolimits="true"/>
  <size njmax="2000" nconmax="1000"/>
  <!-- Thin, stiff cables need a tighter integration/contact solve than the
       MuJoCo defaults.  At 2 ms with the default soft contact they could move
       through the board by more than two cable diameters before the solver
       arrested them. -->
  <option gravity="0 0 -9.81" integrator="implicit" timestep="0.0005"
          iterations="100" ls_iterations="50"/>
  <statistic center="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0.05" extent="1.2"/>
  <visual><global elevation="-70" azimuth="90" offwidth="1200" offheight="1400"/></visual>

  <worldbody>
    <light diffuse=".7 .7 .7" pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 3" dir="0 0 -1"/>
    <!-- mu is identified against the footage, not assumed. An earlier version
         fixed mass_per_m at a plausible wire value and swept mu alone, landing
         on 3.0 -- high for a dry contact pair, the tell that it was absorbing
         physics the elastic cable does not carry (plastic set above all,
         measured in this project to be the dominant shape parameter and not
         representable in mujoco.elasticity.cable). Sweeping mu and mass_per_m
         together instead, holding mu*mass_per_m*g (max static grip) roughly
         fixed and letting cable inertia rise, found a clearly better and more
         robust optimum on the 18-clip LOCO set (mu=1.0, mass_per_m=0.15;
         Median 41.0->36.3, Mittel 44.2->40.5 mm/node; won on 12/18 clips vs.
         mu=3.0). Pushing further (mu=0.75, mass_per_m=0.20) rolled back
         slightly, so this is a real optimum, not a monotone trend to chase
         further. Still an effective parameter pair, not a material one -- the
         two hardest clips (already >65mm at baseline) do better with MORE
         friction, suggesting they need grip that inertia alone can't supply. -->
    <!-- The friction TRIPLE is sliding / torsional / rolling. MuJoCo reads the
         2nd only at condim>=4 and the 3rd only at condim=6, so with the
         condim=3 this scene used everywhere, the 0.05 and 0.002 above were
         never evaluated: two of the three friction types were switched OFF,
         not merely left unfitted. `condim` is a parameter now so that can be
         measured instead of assumed. -->
    <geom name="board" type="plane" size="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0.02"
          pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0" rgba="0.86 0.86 0.83 1"
          condim="{condim}" friction="{mu} {mu_tor} {mu_roll}"/>

{hub}

{chr(10).join(plugs)}

{cables}

{reliefs}
  </worldbody>

  <equality>
{connects}{welds}
  </equality>

  <actuator>
{actuators}
  </actuator>
</mujoco>
'''


def cable_ball_joints(m):
    import mujoco
    ids = np.zeros((4, 14), int)
    for k in range(4):
        names = [f"Wire{k}J_{j}" for j in range(1, 14)] + [f"Wire{k}J_last"]
        for j, n in enumerate(names):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if i < 0:
                raise KeyError(f"joint {n} not in the compiled model")
            ids[k, j] = i
    return ids


def set_per_joint_stiffness(m, values, rest_quat=None):
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = np.tile(v, (4, 1))
    ids = cable_ball_joints(m)
    for k in range(4):
        for j in range(14):
            m.jnt_stiffness[ids[k, j]] = float(v[k, j])
            if rest_quat is not None:
                a = m.jnt_qposadr[ids[k, j]]
                m.qpos_spring[a:a + 4] = np.asarray(rest_quat[k][j], float)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20260602_114019")
    ap.add_argument("--t0", type=int, default=8)
    ap.add_argument("--bend", type=float, default=1.0e5)
    ap.add_argument("--twist", type=float, default=1.0e5)
    ap.add_argument("--damping", type=float, default=0.15)
    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--out", default="data/mujoco/scene.xml")
    args = ap.parse_args()

    shp, t0, t1, fps = track_clip(args.clip, t0=args.t0)
    xml = build_xml(shp, bend=args.bend, twist=args.twist,
                    damping=args.damping, mu=args.mu)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)

    lens = [1000 * np.linalg.norm(np.diff(shp[k][0], axis=0), axis=1).sum()
            for k in range(4)]
    print(f"{args.clip}: Frames {t0}..{t1} ({fps:.0f} fps)")
    print(f"Hub {np.round(1000*shp[0][0][0], 1)} mm, "
          f"Kabellaengen {[round(l) for l in lens]} mm")
    print(f"-> {out}")

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
    print(f"laedt: {m.nbody} bodies, {m.nq} DoF, {m.neq} Gleichungs-Constraints; "
          f"200 Schritte stabil: {np.isfinite(d.qpos).all()}")


if __name__ == "__main__":
    main()
