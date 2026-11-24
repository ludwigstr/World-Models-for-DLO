EXCLUDED_TRACKING = {"20260602_110528", "20260602_172926"}


ALL_CLIP_TRIMS = {
    "20260602_102713": None, "20260602_110528": None, "20260602_134613": None,
    "20260605_112939": None, "20260602_183620": 20, "20260602_172435": 19,
    "20260602_170108": 9, "20260602_175847": 14, "20260602_172926": 20,
    "20260602_193329": 15, "20260602_141210": 13, "20260602_114019": 13,
    "20260602_162650": 12, "20260602_165113": 13, "20260602_193953": 12,
    "20260602_121006": 11,
}


CLIP_TRIMS = {k: v for k, v in ALL_CLIP_TRIMS.items()
              if k not in EXCLUDED_TRACKING}


import numpy as np


BOARD_W, BOARD_H = 1.200, 1.680


COLORS = [(0.1, 0.6, 0.9), (1.0, 0.6, 0.0), (0.2, 0.8, 0.2), (0.9, 0.2, 0.9)]


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
  <option gravity="0 0 -9.81" integrator="implicit" timestep="0.0005"
          iterations="100" ls_iterations="50"/>
  <statistic center="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0.05" extent="1.2"/>
  <visual><global elevation="-70" azimuth="90" offwidth="1200" offheight="1400"/></visual>

  <worldbody>
    <light diffuse=".7 .7 .7" pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 3" dir="0 0 -1"/>
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


def cable_vertices(m, d, k, n_nodes=16):
    import mujoco
    names = ([f"Wire{k}B_first"]
             + [f"Wire{k}B_{i}" for i in range(1, n_nodes - 2)]
             + [f"Wire{k}B_last"])
    ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names])
    if (ids < 0).any():
        raise RuntimeError(f"cable {k}: unresolved bodies {names}")
    P = d.xpos[ids].copy()
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ids[-1]][0]
    tip = P[-1] + d.xmat[ids[-1]].reshape(3, 3) @ np.array([2 * m.geom_size[g][1], 0, 0])
    return np.vstack([P, tip])


def rollout_errs(m, shp, fps, per_cable=False):
    import mujoco
    pass
    pass
    n_frames = len(shp[0])
    d = mujoco.MjData(m)
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    act = {n: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_{a}a")
               for a in ("x", "y", "t")] for n in names}
    base = {n: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][:2].copy()
            for n in names}
    yaw0 = {}
    for n in names:
        q = m.body_quat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}_mod")]
        yaw0[n] = 2.0 * np.arctan2(q[3], q[0])

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

    seq, prev = [], {n: poses(0)[n][1] for n in names}
    for t in range(n_frames):
        p = poses(t)
        for n in names:
            prev[n] += np.arctan2(np.sin(p[n][1] - prev[n]), np.cos(p[n][1] - prev[n]))
        seq.append({n: (p[n][0], prev[n]) for n in names})

    def apply(pose):
        for n in names:
            xy, yaw = pose[n]
            ax, ay, at = act[n]
            d.ctrl[ax], d.ctrl[ay] = xy[0] - base[n][0], xy[1] - base[n][1]
            d.ctrl[at] = yaw - yaw0[n]

    apply(seq[0])
    for _ in range(300):
        mujoco.mj_step(m, d)

    errs = []
    sim0 = np.stack([cable_vertices(m, d, k) for k in range(4)])
    e0 = [1000 * np.linalg.norm(sim0[k][:, :2] - shp[k][0], axis=1).mean()
          for k in range(4)]
    errs.append(np.array(e0) if per_cable else float(np.mean(e0)))
    for t in range(n_frames - 1):
        a0, a1 = seq[t], seq[t + 1]
        for s_ in range(sub):
            f = (s_ + 1) / sub
            apply({n: (a0[n][0] + f * (a1[n][0] - a0[n][0]),
                       a0[n][1] + f * (a1[n][1] - a0[n][1])) for n in names})
            mujoco.mj_step(m, d)
        sim = np.stack([cable_vertices(m, d, k) for k in range(4)])
        e = [1000 * np.linalg.norm(sim[k][:, :2] - shp[k][t + 1], axis=1).mean()
             for k in range(4)]
        errs.append(np.array(e) if per_cable else float(np.mean(e)))
    return np.array(errs) if per_cable else errs


N_C, N_J = 4, 14


def board_normal_axes(m):
    import mujoco
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ids = cable_ball_joints(m)
    n = np.zeros((N_C, N_J, 3))
    for k in range(N_C):
        for j in range(N_J):
            R = d.xmat[m.jnt_bodyid[ids[k, j]]].reshape(3, 3)
            n[k, j] = R.T @ np.array([0.0, 0.0, 1.0])
            n[k, j] /= np.linalg.norm(n[k, j])
    return n


def rest_quats(delta_deg, n):
    h = np.radians(delta_deg)[..., None] / 2.0
    return np.concatenate([np.cos(h), np.sin(h) * n], axis=-1)
