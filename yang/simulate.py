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


from pathlib import Path
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


DIR = str(ROOT.parent / "data" / "videos")


BOARD_W, BOARD_H = 1.200, 1.680


COLORS = [(0.1, 0.6, 0.9), (1.0, 0.6, 0.0), (0.2, 0.8, 0.2), (0.9, 0.2, 0.9)]


TRACKS = ROOT.parent / "data" / "tracks"


def load_clip(clip, t0=0, t1=None, max_frames=400):
    z = np.load(TRACKS / f"{clip}.npz")
    shp = {k: np.asarray(z[f"c{k}"], float) for k in range(4)}
    return shp, 0, len(shp[0]) - 1, float(z["fps"])


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


import warp as wp


def subdivide_path(P, subdiv):
    P = np.asarray(P, float)
    if subdiv <= 1:
        return P
    out = []
    for a, b in zip(P[:-1], P[1:]):
        for j in range(subdiv):
            out.append(a + (b - a) * (j / subdiv))
    out.append(P[-1])
    return np.array(out)


def resample_per_joint(vals, n_joints, subdiv=1, scale=None):
    vals = np.asarray(vals, float)
    if scale is None:
        scale = float(subdiv)
    if vals.shape[-1] == n_joints and scale == 1.0:
        return vals
    src = np.linspace(0.0, 1.0, vals.shape[-1])
    dst = np.linspace(0.0, 1.0, n_joints)
    out = np.stack([np.interp(dst, src, row) for row in np.atleast_2d(vals)])
    out = out * float(scale)
    return out if vals.ndim > 1 else out[0]


def initial_route3(shp, k, table_fraction=0.6, z_table=0.003, z_conn=None,
                   subdiv=1):
    pass
    if z_conn is None:
        z_conn = CAV_Z
    P = subdivide_path(shp[k][0].astype(float), subdiv)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    ramp = max(1e-6, (1.0 - table_fraction) / 2.0 * L)
    z = np.empty(len(P))
    for i, si in enumerate(s):
        if si < ramp:
            f = si / ramp
        elif si > L - ramp:
            f = (L - si) / ramp
        else:
            f = 1.0
        z[i] = z_conn + f * (z_table - z_conn)
    return np.hstack([P, z[:, None]])


def make_kinematic(builder, body_idx):
    builder.body_mass[body_idx] = 0.0
    builder.body_inv_mass[body_idx] = 0.0
    builder.body_inertia[body_idx] = wp.mat33(0.0)
    builder.body_inv_inertia[body_idx] = wp.mat33(0.0)


def build_scene(builder, shp, *, radius, stretch_stiffness, bend_stiffness, bend_damping, twist_stiffness, twist_damping, per_joint_ke, per_joint_kd, per_joint_kt, mass_per_m, scene_collision, table_fraction, subdiv, joint_scale, mass_exact, per_joint_rest, cable_mu, shape_mu, mu_tors, mu_roll, dahl_eps_max, dahl_tau):
    import newton
    builder.default_shape_cfg.mu = 1.0 if shape_mu is None else float(shape_mu)
    if mu_tors is not None:
        builder.default_shape_cfg.mu_torsional = float(mu_tors)
    if mu_roll is not None:
        builder.default_shape_cfg.mu_rolling = float(mu_roll)
    _use_dahl = dahl_eps_max is not None and dahl_tau is not None
    if _use_dahl:
        newton.solvers.SolverVBD.register_custom_attributes(builder)

    if mass_per_m is not None:
        mpm = np.broadcast_to(np.asarray(mass_per_m, float).ravel(), (4,))
    use_per_joint = per_joint_ke is not None
    cable_joints = []

    n_joints_sim = (len(shp[0][0]) - 1) * subdiv - 1
    if per_joint_ke is not None:
        per_joint_ke = resample_per_joint(per_joint_ke, n_joints_sim, subdiv,
                                          scale=joint_scale)
    if per_joint_kd is not None:
        per_joint_kd = resample_per_joint(per_joint_kd, n_joints_sim, subdiv,
                                          scale=joint_scale)
    if per_joint_kt is not None:
        per_joint_kt = resample_per_joint(per_joint_kt, n_joints_sim, subdiv,
                                          scale=joint_scale)

    cable_bodies = []
    for k in range(4):
        r0 = initial_route3(shp, k, table_fraction=table_fraction,
                            subdiv=subdiv)
        positions = [wp.vec3(*p) for p in r0]
        rod_cfg = None
        if mass_per_m is not None:
            rod_cfg = builder.ShapeConfig()
            rod_cfg.mu = 1.0 if cable_mu is None else float(cable_mu)
            if mu_tors is not None:
                rod_cfg.mu_torsional = float(mu_tors)
            if mu_roll is not None:
                rod_cfg.mu_rolling = float(mu_roll)
            seg_len = np.linalg.norm(np.diff(r0[:, :3], axis=0), axis=1)
            if mass_exact:
                vol = (np.pi * radius ** 2 * seg_len.sum()
                       + len(seg_len) * (4.0 / 3.0) * np.pi * radius ** 3)
                rod_cfg.density = float(mpm[k] * seg_len.sum() / vol)
            else:
                rod_cfg.density = float(mpm[k] / (np.pi * radius ** 2))
        bodies, joints = builder.add_rod(
            positions=positions, radius=radius, cfg=rod_cfg,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=(0.0 if use_per_joint else bend_stiffness),
            bend_damping=(0.0 if per_joint_kd is not None else bend_damping),
            twist_stiffness=twist_stiffness, twist_damping=twist_damping,
            body_frame_origin="start", label=f"cable{k}")
        if per_joint_rest is not None:
            rk = np.asarray(per_joint_rest, float)
            rk = rk[k] if rk.ndim == 2 else rk
            jb = len(builder.joint_X_c) - len(joints)
            for j in range(len(joints)):
                ang = float(rk[j])
                q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang)
                cur = builder.joint_X_c[jb + j]
                builder.joint_X_c[jb + j] = wp.transform(
                    wp.transform_get_translation(cur),
                    wp.mul(q, wp.transform_get_rotation(cur)))
        if use_per_joint:
            base = len(builder.joint_target_ke) - 4 * len(joints)
            for j in range(len(joints)):
                builder.joint_target_ke[base + 4 * j + 2] = float(per_joint_ke[k][j])
                builder.joint_target_ke[base + 4 * j + 3] = float(
                    per_joint_ke[k][j] if per_joint_kt is None
                    else per_joint_kt[k][j])
                if per_joint_kd is not None:
                    builder.joint_target_kd[base + 4 * j + 2] = float(per_joint_kd[k][j])
                    builder.joint_target_kd[base + 4 * j + 3] = float(per_joint_kd[k][j])
        make_kinematic(builder, bodies[0])
        make_kinematic(builder, bodies[-1])
        cable_bodies.append(bodies)
        cable_joints.append(list(joints))

    platform_bodies, module_bodies = {}, {}
    if scene_collision:
        pass
        from newton_pseudo_render import mover_poses as _mover_poses
        board_body = builder.add_body(
            xform=wp.transform(wp.vec3(BOARD_W / 2, BOARD_H / 2, -0.005), wp.quat_identity()),
            is_kinematic=True, label="board")
        builder.add_shape_box(board_body, hx=BOARD_W / 2, hy=BOARD_H / 2, hz=0.005,
                              label="board_geo")
        for name in ["hub", "plug0", "plug1", "plug2", "plug3"]:
            pb = builder.add_body(is_kinematic=True, label=f"{name}_platform")
            builder.add_shape_box(pb, hx=PLATFORM_HALF[0], hy=PLATFORM_HALF[1],
                                  hz=PLATFORM_HALF[2], label=f"{name}_platform_geo")
            builder.add_shape_box(pb, xform=wp.transform(wp.vec3(0.0, 0.0, PLATE_Z), wp.quat_identity()),
                                  hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2],
                                  label=f"{name}_plate_geo")
            platform_bodies[name] = pb
            mb = builder.add_body(is_kinematic=True, label=f"{name}_module")
            builder.add_shape_box(mb, hx=MODULE_HALF, hy=MODULE_HALF, hz=MODULE_HALF,
                                  label=f"{name}_module_geo")
            module_bodies[name] = mb
    return cable_bodies, cable_joints, platform_bodies, module_bodies


def set_dahl(model, cable_joints, dahl_eps_max, dahl_tau):
    def _per_joint(v, name):
        a = np.asarray(v, float)
        n_j = len(cable_joints[0]) if cable_joints else 0
        if a.ndim == 0:
            return np.full((4, n_j), float(a))
        if a.ndim == 1 and a.size == 4:
            return np.repeat(a[:, None], n_j, axis=1)
        if a.ndim == 1 and a.size == n_j:
            return np.repeat(a[None, :], 4, axis=0)
        if a.shape == (4, n_j):
            return a
        raise ValueError(f"{name}: shape {a.shape} fits neither a scalar, "
                         f"4 cables nor (4, {n_j}) joints")

    eps_pc = _per_joint(dahl_eps_max, "dahl_eps_max")
    tau_pc = _per_joint(dahl_tau, "dahl_tau")
    eps = np.zeros(model.joint_count, dtype=np.float32)
    tau = np.zeros(model.joint_count, dtype=np.float32)
    for k, jts in enumerate(cable_joints):
        for j, jid in enumerate(jts):
            eps[jid] = eps_pc[k, j]
            tau[jid] = tau_pc[k, j]
    model.vbd.dahl_eps_max.assign(eps)
    model.vbd.dahl_tau.assign(tau)


def end_pose_of(shp, t, scene_collision=True):
    from newton.utils import create_parallel_transport_cable_quaternions

    def route3(k, t):
        P = shp[k][t].astype(float)
        z = np.full((len(P), 1), CAV_Z)
        return np.hstack([P, z])
    out = []
    for k in range(4):
        r = route3(k, t)
        pts = [wp.vec3(*p) for p in r]
        edge_q = create_parallel_transport_cable_quaternions(pts)
        out.append((r[0], np.array(list(edge_q[0])), r[-2], np.array(list(edge_q[-1]))))
    if scene_collision:
        from newton_pseudo_render import mover_poses as _mover_poses
    movers = _mover_poses(shp, t) if scene_collision else None
    return out, movers


def nlerp_quat(q0, q1, f):
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = q0 + f * (q1 - q0)
    return q / np.linalg.norm(q)


def eval_clip(clip, t1_sec=None, radius=0.003,
             stretch_stiffness=1.0e5, bend_stiffness=0.5, bend_damping=0.05,
             twist_stiffness=None, twist_damping=None,
             per_joint_ke=None, per_joint_kd=None, per_joint_kt=None,
             mass_per_m=None,
             dt=1e-3, steps_per_frame=None, device=None,
             scene_collision=True, solver_iterations=None,
             return_per_cable=False, shp_fps=None, z_max_out=None,
             z_nodes_out=None, table_fraction=0.6, subdiv=1,
             contact_buffer=64, joint_scale=None, mass_exact=False,
             nodes_xyz_out=None, dahl_eps_max=None, dahl_tau=None,
             per_joint_rest=None, cable_mu=None, shape_mu=None,
             mu_tors=None, mu_roll=None):
    import newton
    import cv2
    from newton.utils import create_parallel_transport_cable_quaternions

    if shp_fps is not None:
        shp, fps = shp_fps
    else:
        t1 = None
        if t1_sec is not None:
            cap_probe = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
            fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 4.0
            cap_probe.release()
            t1 = int(round(t1_sec * fps_probe))
        shp, t0, t1, fps = load_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))
    if solver_iterations is None:
        solver_iterations = 10 if scene_collision else 5

    def route3(k, t):
        P = shp[k][t].astype(float)
        z = np.full((len(P), 1), CAV_Z)
        return np.hstack([P, z])

    builder = newton.ModelBuilder()
    _use_dahl = dahl_eps_max is not None and dahl_tau is not None
    cable_bodies, cable_joints, platform_bodies, module_bodies = build_scene(
        builder, shp, radius=radius, stretch_stiffness=stretch_stiffness,
        bend_stiffness=bend_stiffness, bend_damping=bend_damping,
        twist_stiffness=twist_stiffness, twist_damping=twist_damping,
        per_joint_ke=per_joint_ke, per_joint_kd=per_joint_kd, per_joint_kt=per_joint_kt,
        mass_per_m=mass_per_m, scene_collision=scene_collision,
        table_fraction=table_fraction, subdiv=subdiv, joint_scale=joint_scale,
        mass_exact=mass_exact, per_joint_rest=per_joint_rest, cable_mu=cable_mu,
        shape_mu=shape_mu, mu_tors=mu_tors, mu_roll=mu_roll,
        dahl_eps_max=dahl_eps_max, dahl_tau=dahl_tau)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    if _use_dahl:
        set_dahl(model, cable_joints, dahl_eps_max, dahl_tau)

    collision_pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverVBD(
        model, iterations=solver_iterations,
        rigid_body_contact_buffer_size=contact_buffer)
    state0, state1 = model.state(), model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    def end_pose(t):
        return end_pose_of(shp, t, scene_collision)


    def pin_ends_interp(pose_a, pose_b, f):
        cable_a, movers_a = pose_a
        cable_b, movers_b = pose_b
        bq = state0.body_q.numpy()
        for k in range(4):
            p0a, q0a, p0b, q0b = cable_a[k]
            p1a, q1a, p1b, q1b = cable_b[k]
            bodies = cable_bodies[k]
            bq[bodies[0]] = np.concatenate([p0a + f * (p1a - p0a), nlerp_quat(q0a, q1a, f)])
            bq[bodies[-1]] = np.concatenate([p0b + f * (p1b - p0b), nlerp_quat(q0b, q1b, f)])
        if scene_collision:
            pass
            for name in movers_a:
                pos_a, yaw_a = movers_a[name]
                pos_b, yaw_b = movers_b[name]
                p = pos_a + f * (pos_b - pos_a)
                bq[platform_bodies[name]] = [p[0], p[1], PLATFORM_Z, 0.0, 0.0, 0.0, 1.0]
                qa = np.array(list(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw_a))))
                qb = np.array(list(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw_b))))
                bq[module_bodies[name]] = np.concatenate(
                    [[p[0], p[1], PLATFORM_Z + MODULE_Z], nlerp_quat(qa, qb, f)])
        state0.body_q.assign(bq)
        state1.body_q.assign(bq)

    def step_physics():
        nonlocal state0, state1
        state0.clear_forces()
        collision_pipeline.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    pose0 = end_pose(0)
    pin_ends_interp(pose0, pose0, 0.0)
    for _ in range(300):
        step_physics()

    errs = []
    pose_prev = pose0
    for t in range(n_frames):
        pose_next = end_pose(t)
        for s in range(steps_per_frame):
            f = (s + 1) / steps_per_frame
            pin_ends_interp(pose_prev, pose_next, f)
            step_physics()
        pose_prev = pose_next
        bq = state0.body_q.numpy()
        e_all = []
        for k in range(4):
            bodies = cable_bodies[k][::subdiv]
            sim_pts = bq[bodies][:, :2]
            real_pts = shp[k][t][:len(bodies)]
            e_all.append(float(np.linalg.norm(sim_pts - real_pts, axis=1).mean()) * 1000)
        errs.append(e_all if return_per_cable else float(np.mean(e_all)))
        if z_nodes_out is not None:
            z_nodes_out.append(np.stack([bq[cable_bodies[k][::subdiv]][:, 2]
                                         for k in range(4)]))
        if nodes_xyz_out is not None:
            nodes_xyz_out.append(np.stack([bq[cable_bodies[k]][:, :3]
                                           for k in range(4)]))
        if z_max_out is not None:
            z_max_out.append(np.array([bq[cable_bodies[k]][:, 2].max()
                                       for k in range(4)]))

    return np.array(errs) if return_per_cable else errs


import inspect
import os


FIT_FILE = ROOT / "data/newton_fit_everything_pinned.npz"


MASS_PIN = "0.044"


N_KIN = 18


NAMES = ["hub"] + [f"plug{k}" for k in range(4)]


def fit_kwargs(path=FIT_FILE):
    if os.environ.get("WIRE_MASS_PIN") not in (None, MASS_PIN):
        raise RuntimeError(f"WIRE_MASS_PIN={os.environ['WIRE_MASS_PIN']}, row 12 needs {MASS_PIN}")
    os.environ["WIRE_MASS_PIN"] = MASS_PIN
    import newton_fit_everything_cmaes as E
    assert E.MASS_PIN == float(MASS_PIN) and E.N_PAR == 231, (E.MASS_PIN, E.N_PAR)
    best = np.asarray(np.load(path, allow_pickle=True)["best"], float)
    assert len(best) == 231, len(best)
    return E.unpack(best)


def _scene_kwargs(kw):
    pass
    defaults = {n: p.default for n, p in inspect.signature(eval_clip).parameters.items()
                if p.default is not inspect.Parameter.empty}
    names = [n for n, p in inspect.signature(build_scene).parameters.items()
             if p.kind == inspect.Parameter.KEYWORD_ONLY]
    out = {n: defaults[n] for n in names}
    for n, v in kw.items():
        assert n in out, f"fit parameter {n} is not a build_scene argument"
        out[n] = v
    return out


def _quat_yaw(y):
    return np.array([0.0, 0.0, np.sin(y / 2), np.cos(y / 2)])


def kin_poses(drive, t):
    pass
    pass
    cab, movers = end_pose_of(drive, t, True)
    P = np.zeros((N_KIN, 7))
    for k, (p0, q0, p1, q1) in enumerate(cab):
        P[k] = np.r_[p0, q0]
        P[4 + k] = np.r_[p1, q1]
    for m, n in enumerate(NAMES):
        pos, yaw = movers[n]
        P[8 + m] = [pos[0], pos[1], PLATFORM_Z, 0.0, 0.0, 0.0, 1.0]
        P[13 + m] = np.r_[[pos[0], pos[1], PLATFORM_Z + MODULE_Z], _quat_yaw(float(yaw))]
    return P


def contact_violations(bq, cab, seglen, plat, mod, radius=0.003, own=2, nsamp=9, active=None):
    pass
    B = bq[cab]
    p, q = B[..., :3], B[..., 3:7]
    u, w = q[..., :3], q[..., 3:4]
    ez = np.zeros_like(p); ez[..., 2] = 1.0
    ax = ez + 2.0 * np.cross(u, np.cross(u, ez) + w * ez)
    sgrid = np.linspace(0.0, 1.0, nsamp)
    P = p[..., None, :] + ax[..., None, :] * (seglen[..., None, None] * sgrid[:, None])
    W = len(cab)
    seg = np.arange(15)[None, None, :, None]
    kk = np.arange(4)[None, :, None, None]
    nviol = np.zeros(W, int); dmin = np.full(W, np.inf); pairs = np.zeros((W, 4, 5), int)
    for m in range(5):
        own_mask = (seg < own) if m == 0 else ((kk == m - 1) & (seg > 14 - own))
        own_mask = np.broadcast_to(own_mask, P.shape[:-1])
        boxes = []
        Pp = bq[plat[:, m]]
        boxes.append((Pp[:, :3], None, np.array(PLATFORM_HALF)))
        boxes.append((Pp[:, :3] + np.array([0, 0, PLATE_Z]), None, np.array(PLATE_HALF)))
        Pm = bq[mod[:, m]]
        boxes.append((Pm[:, :3], Pm[:, 3:7], np.full(3, MODULE_HALF)))
        for c, bqq, half in boxes:
            L = P - c[:, None, None, None, :]
            if bqq is not None:
                bu = -bqq[:, None, None, None, :3]; bw = bqq[:, None, None, None, 3:4]
                L = L + 2.0 * np.cross(bu, np.cross(bu, L) + bw * L)
            ex = np.maximum(np.abs(L) - half, 0.0)
            dist = np.linalg.norm(ex, axis=-1)
            v = (dist < radius) & ~own_mask
            if active is not None:
                v &= active[:, None, None, None]
            nviol += v.reshape(W, -1).sum(1)
            pairs[:, :, m] += v.reshape(W, 4, -1).sum(-1)
            dm = np.where(v, dist, np.inf).reshape(W, -1).min(1)
            dmin = np.minimum(dmin, dm)
    return nviol, dmin, pairs


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def pin(idx: wp.array(dtype=wp.int32), pa: wp.array(dtype=wp.transform), pb: wp.array(dtype=wp.transform),
                f: float, q0: wp.array(dtype=wp.transform), q1: wp.array(dtype=wp.transform)):
            i = wp.tid()
            a = pa[i]
            b = pb[i]
            ta = wp.transform_get_translation(a)
            tb = wp.transform_get_translation(b)
            ra = wp.transform_get_rotation(a)
            rb = wp.transform_get_rotation(b)
            if wp.dot(ra, rb) < 0.0:
                rb = -rb
            r = wp.normalize(ra + f * (rb - ra))
            X = wp.transform(ta + f * (tb - ta), r)
            q0[idx[i]] = X
            q1[idx[i]] = X
        _KERNEL = pin
    return _KERNEL


class Batch:

    def __init__(self, rests, kw=None, device=None):
        import newton
        import warp as wp
        pass
        wp.config.quiet = True
        kw = fit_kwargs() if kw is None else kw
        kws = list(kw) if isinstance(kw, (list, tuple)) else [kw] * len(rests)
        assert len(kws) == len(rests)
        sks = [_scene_kwargs(k) for k in kws]
        sk = sks[0]
        self.device = device
        self.W = len(rests)
        use_dahl = sk["dahl_eps_max"] is not None and sk["dahl_tau"] is not None
        assert all((q["dahl_eps_max"] is not None and q["dahl_tau"] is not None) == use_dahl for q in sks)
        main = newton.ModelBuilder()
        if use_dahl:
            newton.solvers.SolverVBD.register_custom_attributes(main)
        self.cab = np.zeros((self.W, 4, 15), np.int64)
        self.kin = np.zeros((self.W, N_KIN), np.int64)
        self.L_last = np.zeros((self.W, 4))
        self.seglen = np.zeros((self.W, 4, 15))
        jids = []
        pass
        for w, R in enumerate(rests):
            b = newton.ModelBuilder()
            if use_dahl:
                newton.solvers.SolverVBD.register_custom_attributes(b)
            cb, cj, pb, mb = build_scene(b, R, **sks[w])
            off_b, off_j = main.body_count, main.joint_count
            main.add_world(b)
            self.cab[w] = np.asarray(cb) + off_b
            jids.append([[j + off_j for j in js] for js in cj])
            self.kin[w] = np.r_[[c[0] for c in cb], [c[-1] for c in cb], [pb[n] for n in NAMES],
                                [mb[n] for n in NAMES]] + off_b
            for k in range(4):
                r0 = initial_route3(R, k, table_fraction=sks[w]["table_fraction"], subdiv=sks[w]["subdiv"])
                self.L_last[w, k] = np.linalg.norm(r0[-1] - r0[-2])
                self.seglen[w, k] = np.linalg.norm(np.diff(r0, axis=0), axis=1)
        main.color()
        self.model = model = main.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        if use_dahl:
            eps = np.zeros(model.joint_count, dtype=np.float32)
            tau = np.zeros(model.joint_count, dtype=np.float32)
            for w, cj in enumerate(jids):
                e_pc = np.broadcast_to(np.asarray(sks[w]["dahl_eps_max"], float), (4, 14))
                t_pc = np.broadcast_to(np.asarray(sks[w]["dahl_tau"], float), (4, 14))
                for k, jts in enumerate(cj):
                    for j, jid in enumerate(jts):
                        eps[jid] = e_pc[k, j]; tau[jid] = t_pc[k, j]
            model.vbd.dahl_eps_max.assign(eps)
            model.vbd.dahl_tau.assign(tau)
        self.pipe = newton.CollisionPipeline(model, broad_phase="sap")
        self.contacts = self.pipe.contacts()
        self.solver = newton.solvers.SolverVBD(model, iterations=10, rigid_body_contact_buffer_size=64)
        self.s0, self.s1 = model.state(), model.state()
        self.ctrl = model.control()
        self.idx = wp.array(self.kin.ravel().astype(np.int32), dtype=wp.int32, device=device)
        self.dt = 1e-3

    def _pin(self, A, B, f):
        import warp as wp
        dev = self.device
        wp.launch(_kernel(), dim=self.W * N_KIN,
                  inputs=[self.idx, wp.array(np.ascontiguousarray(A.reshape(-1, 7)), dtype=wp.transform, device=dev),
                          wp.array(np.ascontiguousarray(B.reshape(-1, 7)), dtype=wp.transform, device=dev), float(f),
                          self.s0.body_q, self.s1.body_q], device=dev)

    def _pin_arrays(self, A, B):
        import warp as wp
        dev = self.device
        return (wp.array(np.ascontiguousarray(A.reshape(-1, 7)), dtype=wp.transform, device=dev),
                wp.array(np.ascontiguousarray(B.reshape(-1, 7)), dtype=wp.transform, device=dev))

    def _step(self):
        self.s0.clear_forces()
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def nodes(self):
        bq = self.s0.body_q.numpy()
        B = bq[self.cab]
        V = np.empty((self.W, 4, 16, 3))
        V[:, :, :15] = B[..., :3]
        u = B[:, :, 14, 3:6]; w_ = B[:, :, 14, 6:7]
        v = np.zeros_like(u); v[..., 2] = self.L_last
        V[:, :, 15] = B[:, :, 14, :3] + v + 2.0 * np.cross(u, np.cross(u, v) + w_ * v)
        return V

    def run(self, poses, sub=250, settle=300, check_every=0, lengths=None):
        import warp as wp
        T = len(poses)
        self.contact_n = np.zeros(self.W, int); self.contact_dmin = np.full(self.W, np.inf)
        self.contact_pairs = np.zeros((self.W, 4, 5), int)
        Ls = np.full(self.W, T) if lengths is None else np.asarray(lengths)
        a0, _ = self._pin_arrays(poses[0], poses[0])
        wp.launch(_kernel(), dim=self.W * N_KIN, inputs=[self.idx, a0, a0, 0.0, self.s0.body_q, self.s1.body_q],
                  device=self.device)
        for _ in range(settle):
            self._step()
        out = []
        prev = poses[0]
        for t in range(T):
            A, B = self._pin_arrays(prev, poses[t])
            for s_ in range(sub):
                wp.launch(_kernel(), dim=self.W * N_KIN,
                          inputs=[self.idx, A, B, (s_ + 1) / sub, self.s0.body_q, self.s1.body_q], device=self.device)
                self._step()
                if check_every and (s_ + 1) % check_every == 0:
                    n, dm, pr = contact_violations(self.s0.body_q.numpy(), self.cab, self.seglen, self.kin[:, 8:13],
                                                   self.kin[:, 13:18], active=t < Ls)
                    self.contact_n += n; self.contact_dmin = np.minimum(self.contact_dmin, dm); self.contact_pairs += pr
            prev = poses[t]
            out.append(self.nodes())
        return np.stack(out, 1)


def rollouts(rests, drives, kw=None, device=None, fps=4.0, check_every=0, stats=None):
    b = Batch(rests, kw=kw, device=device)
    Ts = [len(d[0]) for d in drives]
    T = max(Ts)
    poses = np.zeros((T, len(drives), N_KIN, 7))
    for w, d in enumerate(drives):
        for t in range(T):
            poses[t, w] = kin_poses(d, min(t, Ts[w] - 1)) if t < Ts[w] else poses[t - 1, w]
    V = b.run(poses, sub=max(1, round((1.0 / fps) / b.dt)), check_every=check_every, lengths=Ts)
    if stats is not None:
        stats.extend(zip(b.contact_n.tolist(), b.contact_dmin.tolist(), list(b.contact_pairs)))
    res = []
    for w, Tw in enumerate(Ts):
        v = V[w, :Tw]
        bad = np.nonzero(~np.isfinite(v).reshape(Tw, -1).all(1))[0]
        res.append(v[:bad[0]] if len(bad) else v)
    return res


def drive_from_poses(rest_frame, seq):
    R = np.asarray(rest_frame, float)
    rot = lambda v, a: np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])
    hp0, hy0 = seq[0]["hub"]
    off = []
    for k in range(4):
        pp0, py0 = seq[0][f"plug{k}"]
        off.append(([rot(R[k, i] - hp0, -hy0) for i in (0, 1)], [rot(R[k, i] - pp0, -py0) for i in (14, 15)]))
    out = {k: np.zeros((len(seq), 4, 2)) for k in range(4)}
    for t, p in enumerate(seq):
        hp, hy = p["hub"]
        for k in range(4):
            pp, py = p[f"plug{k}"]
            out[k][t, 0:2] = [np.asarray(hp) + rot(o, hy) for o in off[k][0]]
            out[k][t, 2:4] = [np.asarray(pp) + rot(o, py) for o in off[k][1]]
    return out


THETA_NAMES = ["E_bend", "G_twist", "zeta_bend", "zeta_twist", "mass_per_m", "mu"]


THETA_BOUNDS = [(3.0, 7.0),
                (3.0, 7.0),
                (-4.0, 0.7),
                (-4.0, 0.7),
                (np.log10(0.01), np.log10(0.5)),
                (np.log10(0.05), np.log10(5.0))]


THETA_DEFAULT = np.log10([1e5, 1e5, 0.15, 0.15, 0.15, 1.0])


RADIUS = 0.003


def theta_dict(x):
    v = 10.0 ** np.asarray(x, float)
    return dict(zip(THETA_NAMES, v.tolist()))


def make_model(shp, x, frame_idx=0):
    import mujoco
    pass
    p = theta_dict(x)
    xml = build_xml(shp, frame_idx=frame_idx, radius=RADIUS, bend=p["E_bend"], twist=p["G_twist"],
                    mass_per_m=p["mass_per_m"], mu=p["mu"])
    m = mujoco.MjModel.from_xml_string(xml)
    ids = cable_ball_joints(m)
    for c in range(4):
        for j in range(ids.shape[1]):
            a = m.jnt_dofadr[ids[c, j]]
            m.dof_damping[a] = p["zeta_twist"]
            m.dof_damping[a + 1:a + 3] = p["zeta_bend"]
    return m


class Drive:

    def __init__(self, m):
        import mujoco
        self.m = m
        self.act = {n: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_{a}a") for a in ("x", "y", "t")]
                    for n in NAMES}
        self.base = {n: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][:2].copy() for n in NAMES}
        self.yaw0 = {}
        for n in NAMES:
            q = m.body_quat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}_mod")]
            self.yaw0[n] = 2.0 * np.arctan2(q[3], q[0])
        self.jnt = {n: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_{a}")]
                        for a in ("x", "y", "turn")] for n in NAMES}

    @staticmethod
    def poses(frame):
        pass
        out = np.mean([P[1] - P[0] for P in frame], axis=0)
        hy = np.arctan2(out[1], out[0]) + np.pi / 2
        hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
        res = {"hub": (np.array([frame[0][0][0] - hoff[0], frame[0][0][1] - hoff[1]]), hy)}
        for k, P in enumerate(frame):
            v = P[-1] - P[-2]
            y = np.arctan2(v[1], v[0]) + np.pi / 2
            o = _rotz(np.array(PLUG_CAV), y)
            res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
        return res

    @classmethod
    def pose_seq(cls, frames):
        seq, prev = [], {n: cls.poses(frames[0])[n][1] for n in NAMES}
        for fr in frames:
            p = cls.poses(fr)
            for n in NAMES:
                prev[n] += np.arctan2(np.sin(p[n][1] - prev[n]), np.cos(p[n][1] - prev[n]))
            seq.append({n: (p[n][0], prev[n]) for n in NAMES})
        return seq

    def apply(self, d, pose):
        for n in NAMES:
            xy, yaw = pose[n]
            ax, ay, at = self.act[n]
            d.ctrl[ax], d.ctrl[ay] = xy[0] - self.base[n][0], xy[1] - self.base[n][1]
            d.ctrl[at] = yaw - self.yaw0[n]

    def set_qpos(self, d, pose):
        for n in NAMES:
            xy, yaw = pose[n]
            jx, jy, jt = self.jnt[n]
            d.qpos[jx], d.qpos[jy] = xy[0] - self.base[n][0], xy[1] - self.base[n][1]
            d.qpos[jt] = yaw - self.yaw0[n]

    def step_frame(self, d, a0, a1, sub):
        import mujoco
        for s_ in range(sub):
            f = (s_ + 1) / sub
            self.apply(d, {n: (a0[n][0] + f * (a1[n][0] - a0[n][0]), a0[n][1] + f * (a1[n][1] - a0[n][1]))
                           for n in NAMES})
            mujoco.mj_step(self.m, d)


def unstable(d):
    import mujoco
    return d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0


def vertices(m, d):
    from mujoco_replay import cable_vertices
    return np.stack([cable_vertices(m, d, k) for k in range(4)])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_from_to(u, v):
    c = float(np.dot(u, v))
    ax = np.cross(u, v)
    if c < -0.999999:
        ax = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    q = np.array([1.0 + c, *ax])
    return q / np.linalg.norm(q)


def _resample_fixed(P3, seg):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P3, axis=0), axis=1))]
    s = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((len(s), 3))
    for i, si in enumerate(s):
        j = int(np.clip(np.searchsorted(d, si) - 1, 0, len(d) - 2))
        f = (si - d[j]) / max(d[j + 1] - d[j], 1e-12)
        out[i] = P3[j] + f * (P3[j + 1] - P3[j])
    return out


class OneStep:

    def __init__(self, shp, fps, x, settle=200):
        import mujoco
        pass
        self.shp, self.fps, self.settle = shp, fps, settle
        self.m = make_model(shp, x)
        self.d = mujoco.MjData(self.m)
        self.drive = Drive(self.m)
        self.sub = max(1, int(round((1.0 / fps) / self.m.opt.timestep)))
        self.T = len(shp[0])
        self.seq = Drive.pose_seq([[shp[k][t] for k in range(4)] for t in range(self.T)])
        self.lift = lift_profile
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.cab = []
        for k in range(4):
            names = [f"Wire{k}B_first"] + [f"Wire{k}B_{i}" for i in range(1, 14)] + [f"Wire{k}B_last"]
            ids = np.array([bid(n) for n in names])
            g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ids[-1]][0]
            seg = np.r_[[m.body_pos[i][0] for i in ids[1:]], 2 * m.geom_size[g][1]]
            R0 = d.xquat[ids].copy()
            x0 = np.stack([_quat_rotate(q, [1.0, 0, 0]) for q in R0])
            qadr = [m.jnt_qposadr[m.body_jntadr[i]] for i in ids]
            rel = [(m.jnt_qposadr[m.body_jntadr[bid(f"hub_relief_wire{k}")]]),
                   (m.jnt_qposadr[m.body_jntadr[bid(f"plug{k}_relief_wire")]])]
            self.cab.append(dict(ids=ids, seg=seg, R0=R0, x0=x0, qadr=qadr, rel=rel))

    def reset(self, i):
        import mujoco
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for k, c in enumerate(self.cab):
            V = _resample_fixed(self.lift(self.shp[k][i].astype(float), RADIUS), c["seg"])
            dirs = np.diff(V, axis=0)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            R = [_quat_mul(_quat_from_to(c["x0"][j], dirs[j]), c["R0"][j]) for j in range(15)]
            a = c["qadr"][0]
            d.qpos[a:a + 3] = V[0]; d.qpos[a + 3:a + 7] = R[0]
            for j in range(1, 15):
                q = _quat_mul(_quat_conj(_quat_mul(R[j - 1], m.body_quat[c["ids"][j]])), R[j])
                a = c["qadr"][j]
                d.qpos[a:a + 4] = q / np.linalg.norm(q)
            for a, v in zip(c["rel"], (V[1], V[-2])):
                d.qpos[a:a + 3] = v; d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.drive.set_qpos(d, self.seq[i])
        d.qvel[:] = 0
        self.drive.apply(d, self.seq[i])
        for _ in range(self.settle):
            mujoco.mj_step(m, d)

    def error(self, i):
        self.reset(i)
        self.drive.step_frame(self.d, self.seq[i], self.seq[i + 1], self.sub)
        V = vertices(self.m, self.d)[:, :, :2]
        tgt = np.stack([self.shp[k][i + 1] for k in range(4)])
        dist = np.linalg.norm(V - tgt, axis=2)
        if not np.isfinite(dist).all() or unstable(self.d):
            return np.inf, np.inf
        return float((dist ** 2).mean()), float(dist.mean())

    def nochange(self, i):
        tgt = np.stack([self.shp[k][i + 1] for k in range(4)]); cur = np.stack([self.shp[k][i] for k in range(4)])
        dist = np.linalg.norm(cur - tgt, axis=2)
        return float((dist ** 2).mean()), float(dist.mean())

    def reset_error(self, i):
        self.reset(i)
        V = vertices(self.m, self.d)[:, :, :2]
        return float(np.linalg.norm(V - np.stack([self.shp[k][i] for k in range(4)]), axis=2).mean())


def _quat_rotate(q, v):
    qv = np.array([0.0, *v])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


sim_newton_THETA_NAMES = ["k_stretch", "d_stretch", "k_bend", "d_bend", "k_twist", "d_twist", "mass_per_m", "mu"]


sim_newton_THETA_BOUNDS = [(1.0, 6.0),
                (-5.0, 0.0),
                (-4.0, 1.0),
                (-6.0, 0.0),
                (-4.0, 1.0),
                (-6.0, 0.0),
                (np.log10(0.01), np.log10(0.5)),
                (np.log10(0.05), np.log10(5.0))]


sim_newton_THETA_DEFAULT = np.log10([1e5, 1e-5, 0.5, 0.05, 0.5, 0.05, 0.044, 1.0])


DT = 1e-3


ITERATIONS = 10


TABLE_FRACTION = 0.6


def theta_dict(x):
    v = 10.0 ** np.asarray(x, float)
    return dict(zip(sim_newton_THETA_NAMES, v.tolist()))


def _qmul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def _qrot(q, v):
    u = np.asarray(q[:3]); w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _qz(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def _qbetween(u, v):
    c = float(np.dot(u, v))
    if c < -0.999999:
        ax = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        return np.array([*ax, 0.0])
    q = np.array([*np.cross(u, v), 1.0 + c])
    return q / np.linalg.norm(q)


def _rotz2(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def lift(P2):
    pass
    return initial_route3({0: [np.asarray(P2, float)]}, 0, table_fraction=TABLE_FRACTION)


def _yaw(v):
    return float(np.arctan2(v[1], v[0]))


def _unwrap(a):
    return np.unwrap(np.asarray(a, float), axis=0)


def mover_pose_seq(frames):
    pass
    return Drive.pose_seq([[frames[t][k] for k in range(4)] for t in range(len(frames))])


def schedule_tracked(frames):
    F = np.asarray(frames, float)
    T = len(F)
    xy = np.zeros((T, N_KIN, 2)); yaw = np.zeros((T, N_KIN))
    xy[:, 0:4] = F[:, :, 0]
    xy[:, 4:8] = F[:, :, 14]
    yaw[:, 0:4] = _unwrap(np.arctan2(F[:, :, 1, 1] - F[:, :, 0, 1], F[:, :, 1, 0] - F[:, :, 0, 0]))
    yaw[:, 4:8] = _unwrap(np.arctan2(F[:, :, 15, 1] - F[:, :, 14, 1], F[:, :, 15, 0] - F[:, :, 14, 0]))
    seq = mover_pose_seq(F)
    for t, p in enumerate(seq):
        for m, n in enumerate(NAMES):
            xy[t, 8 + m] = p[n][0]; xy[t, 13 + m] = p[n][0]; yaw[t, 13 + m] = p[n][1]
    return xy, yaw


def schedule_from_poses(rest_frame, seq):
    R = np.asarray(rest_frame, float)
    p0 = seq[0]
    ends = []
    for k in range(4):
        hp, hy = p0["hub"]
        ends.append(("hub", _rotz2(R[k, 0] - hp, -hy), _yaw(R[k, 1] - R[k, 0]) - hy))
    for k in range(4):
        pp, py = p0[f"plug{k}"]
        ends.append((f"plug{k}", _rotz2(R[k, 14] - pp, -py), _yaw(R[k, 15] - R[k, 14]) - py))
    T = len(seq)
    xy = np.zeros((T, N_KIN, 2)); yaw = np.zeros((T, N_KIN))
    for t, p in enumerate(seq):
        for e, (n, off, hd) in enumerate(ends):
            mp, my = p[n]
            xy[t, e] = np.asarray(mp) + _rotz2(off, my); yaw[t, e] = my + hd
        for m, n in enumerate(NAMES):
            xy[t, 8 + m] = p[n][0]; xy[t, 13 + m] = p[n][0]; yaw[t, 13 + m] = p[n][1]
    return xy, yaw


def sim_newton_kernel():
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def set_kinematic(idx: wp.array(dtype=wp.int32), xy_a: wp.array(dtype=wp.vec2),
                          xy_b: wp.array(dtype=wp.vec2), yaw_a: wp.array(dtype=float),
                          yaw_b: wp.array(dtype=float), yaw_ref: wp.array(dtype=float),
                          z: wp.array(dtype=float), qb: wp.array(dtype=wp.quat), f: float,
                          q0: wp.array(dtype=wp.transform), q1: wp.array(dtype=wp.transform)):
            i = wp.tid()
            p = xy_a[i] + f * (xy_b[i] - xy_a[i])
            a = yaw_a[i] + f * (yaw_b[i] - yaw_a[i]) - yaw_ref[i]
            q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a), qb[i])
            X = wp.transform(wp.vec3(p[0], p[1], z[i]), q)
            q0[idx[i]] = X
            q1[idx[i]] = X
        _KERNEL = set_kinematic
    return _KERNEL


class Engine:

    def __init__(self, rests, device=None, iterations=ITERATIONS, dt=DT, theta_build=sim_newton_THETA_DEFAULT):
        import newton
        import warp as wp
        pass
        pass
        wp.config.quiet = True
        self.W = len(rests)
        self.dt, self.iterations, self.device = dt, iterations, device
        self.rests = [np.asarray(r, float) for r in rests]
        p = theta_dict(theta_build)
        self.m_build = p["mass_per_m"]
        main = newton.ModelBuilder()
        self.cab = np.zeros((self.W, 4, 15), np.int64)
        self.jnt = []
        self.kin = np.zeros((self.W, N_KIN), np.int64)
        self.kz = np.zeros((self.W, N_KIN)); self.kq = np.zeros((self.W, N_KIN, 4)); self.kyaw = np.zeros((self.W, N_KIN))
        self.L_last = np.zeros((self.W, 4))
        self.L_rest = np.zeros((self.W, 4))
        self.rest3 = []
        for w, R in enumerate(self.rests):
            b = newton.ModelBuilder()
            b.default_shape_cfg.mu = p["mu"]
            r3 = [lift(R[k]) for k in range(4)]
            self.rest3.append(r3)
            jw = []
            for k in range(4):
                cfg = b.ShapeConfig()
                cfg.mu = p["mu"]
                seg = np.linalg.norm(np.diff(r3[k], axis=0), axis=1)
                vol = np.pi * RADIUS ** 2 * seg.sum() + len(seg) * (4.0 / 3.0) * np.pi * RADIUS ** 3
                cfg.density = float(self.m_build * seg.sum() / vol)
                bodies, joints = b.add_rod(positions=[wp.vec3(*q) for q in r3[k]], radius=RADIUS, cfg=cfg,
                                           stretch_stiffness=p["k_stretch"], stretch_damping=p["d_stretch"],
                                           bend_stiffness=p["k_bend"], bend_damping=p["d_bend"],
                                           twist_stiffness=p["k_twist"], twist_damping=p["d_twist"],
                                           body_frame_origin="start", label=f"cable{k}")
                assert len(bodies) == 15, len(bodies)
                make_kinematic(b, bodies[0]); make_kinematic(b, bodies[-1])
                self.cab[w, k] = bodies
                jw.append(list(joints))
                self.L_last[w, k] = seg[-1]; self.L_rest[w, k] = seg.sum()
            board = b.add_body(xform=wp.transform(wp.vec3(BOARD_W / 2, BOARD_H / 2, -0.005), wp.quat_identity()),
                               is_kinematic=True, label="board")
            b.add_shape_box(board, hx=BOARD_W / 2, hy=BOARD_H / 2, hz=0.005)
            plat, mod = [], []
            for n in NAMES:
                pb = b.add_body(is_kinematic=True, label=f"{n}_platform")
                b.add_shape_box(pb, hx=PLATFORM_HALF[0], hy=PLATFORM_HALF[1], hz=PLATFORM_HALF[2])
                b.add_shape_box(pb, xform=wp.transform(wp.vec3(0.0, 0.0, PLATE_Z), wp.quat_identity()),
                                hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2])
                mb = b.add_body(is_kinematic=True, label=f"{n}_module")
                b.add_shape_box(mb, hx=MODULE_HALF, hy=MODULE_HALF, hz=MODULE_HALF)
                plat.append(pb); mod.append(mb)
            off_b, off_j = main.body_count, main.joint_count
            main.add_world(b)
            self.cab[w] += off_b
            self.jnt.append([[j + off_j for j in js] for js in jw])
            self.kin[w] = np.r_[self.cab[w, :, 0], self.cab[w, :, -1], plat, mod] + np.r_[np.zeros(8, int), np.full(10, off_b)]
            bq = np.array([list(main.body_q[i]) for i in self.kin[w][:8]])
            self.kz[w, :8] = bq[:, 2]; self.kq[w, :8] = bq[:, 3:7]
            self.kyaw[w, 0:4] = [_yaw(R[k, 1] - R[k, 0]) for k in range(4)]
            self.kyaw[w, 4:8] = [_yaw(R[k, 15] - R[k, 14]) for k in range(4)]
            self.kz[w, 8:13] = PLATFORM_Z; self.kz[w, 13:18] = PLATFORM_Z + MODULE_Z
            self.kq[w, 8:] = [0, 0, 0, 1.0]
        main.color()
        self.model = model = main.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        self._mass0 = model.body_mass.numpy().copy(); self._imass0 = model.body_inv_mass.numpy().copy()
        self._I0 = model.body_inertia.numpy().copy(); self._iI0 = model.body_inv_inertia.numpy().copy()
        self._mu0 = model.shape_material_mu.numpy().copy()
        self.shape_world = model.shape_world.numpy()
        self.qd_start = model.joint_qd_start.numpy()
        self.pipe = newton.CollisionPipeline(model, broad_phase="sap")
        self.contacts = self.pipe.contacts()
        self.kin_flat = wp.array(self.kin.ravel().astype(np.int32), dtype=wp.int32, device=device)
        self.kz_d = wp.array(self.kz.ravel(), dtype=float, device=device)
        self.kq_d = wp.array(self.kq.reshape(-1, 4), dtype=wp.quat, device=device)
        self.kyaw_d = wp.array(self.kyaw.ravel(), dtype=float, device=device)
        self.solver = None
        self.theta = None

    def set_theta(self, X):
        import newton
        X = np.broadcast_to(np.asarray(X, float), (self.W, len(sim_newton_THETA_NAMES)))
        self.theta = X.copy()
        m = self.model
        ke = m.joint_target_ke.numpy(); kd = m.joint_target_kd.numpy()
        mass, imass = self._mass0.copy(), self._imass0.copy()
        I, iI = self._I0.copy(), self._iI0.copy()
        mu = self._mu0.copy()
        if not hasattr(self, "_jd"):
            self._jd = np.stack([self.qd_start[np.concatenate([np.asarray(js) for js in self.jnt[w]])]
                                 for w in range(self.W)])
            self._dyn = self.cab[:, :, 1:-1].reshape(self.W, -1)
            self._sw = self.shape_world.copy()
        V = 10.0 ** X
        col = lambda name: V[:, sim_newton_THETA_NAMES.index(name)][:, None]
        for off, name in ((0, "k_stretch"), (1, "k_stretch"), (2, "k_bend"), (3, "k_twist")):
            ke[self._jd + off] = col(name)
        for off, name in ((0, "d_stretch"), (1, "d_stretch"), (2, "d_bend"), (3, "d_twist")):
            kd[self._jd + off] = col(name)
        sc = col("mass_per_m") / self.m_build
        mass[self._dyn] *= sc; imass[self._dyn] /= sc
        I[self._dyn] *= sc[:, :, None, None]; iI[self._dyn] /= sc[:, :, None, None]
        ok = self._sw >= 0
        mu[ok] = V[self._sw[ok], sim_newton_THETA_NAMES.index("mu")]
        m.joint_target_ke.assign(ke); m.joint_target_kd.assign(kd)
        m.body_mass.assign(mass); m.body_inv_mass.assign(imass)
        m.body_inertia.assign(I); m.body_inv_inertia.assign(iI)
        m.shape_material_mu.assign(mu)
        self.solver = newton.solvers.SolverVBD(m, iterations=self.iterations, rigid_body_contact_buffer_size=64)

    def reset_state(self, frames_i=None, sched_xy=None, sched_yaw=None):
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.ctrl = self.model.control()
        if frames_i is None:
            return
        key = getattr(self, "_reset_key", None)
        if key is not None and key == getattr(self, "_reset_key_req", None):
            self.s0.body_q.assign(self._reset_bq); self.s1.body_q.assign(self._reset_bq)
            self._set_kin(sched_xy, sched_xy, sched_yaw, sched_yaw, 0.0)
            return
        bq = self.s0.body_q.numpy()
        for w in range(self.W):
            for k in range(4):
                P = lift(frames_i[w][k])
                q = _qmul(_qz(sched_yaw[w, k] - self.kyaw[w, k]), self.kq[w, k])
                d_prev = _qrot(q, np.array([0, 0, 1.0]))
                ids = self.cab[w, k]
                for j in range(15):
                    d = P[j + 1] - P[j]; d /= np.linalg.norm(d)
                    if j > 0:
                        q = _qmul(_qbetween(d_prev, d), q)
                    q /= np.linalg.norm(q)
                    bq[ids[j]] = np.r_[P[j], q]
                    d_prev = d
        self.s0.body_q.assign(bq); self.s1.body_q.assign(bq)
        self._reset_bq, self._reset_key = bq.copy(), getattr(self, "_reset_key_req", None)
        if sched_xy is not None:
            self._set_kin(sched_xy, sched_xy, sched_yaw, sched_yaw, 0.0)

    def _set_kin(self, xya, xyb, ya, yb, f):
        import warp as wp
        dev = self.device
        wp.launch(sim_newton_kernel(), dim=self.W * N_KIN,
                  inputs=[self.kin_flat, wp.array(np.ascontiguousarray(xya.reshape(-1, 2)), dtype=wp.vec2, device=dev),
                          wp.array(np.ascontiguousarray(xyb.reshape(-1, 2)), dtype=wp.vec2, device=dev),
                          wp.array(ya.ravel(), dtype=float, device=dev), wp.array(yb.ravel(), dtype=float, device=dev),
                          self.kyaw_d, self.kz_d, self.kq_d, float(f), self.s0.body_q, self.s1.body_q], device=dev)

    def _step(self):
        self.s0.clear_forces()
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def hold(self, xy, yaw, n):
        import warp as wp
        dev = self.device
        xa = wp.array(np.ascontiguousarray(xy.reshape(-1, 2)), dtype=wp.vec2, device=dev)
        ya = wp.array(yaw.ravel(), dtype=float, device=dev)
        for _ in range(n):
            wp.launch(sim_newton_kernel(), dim=self.W * N_KIN, inputs=[self.kin_flat, xa, xa, ya, ya, self.kyaw_d, self.kz_d,
                                                              self.kq_d, 0.0, self.s0.body_q, self.s1.body_q], device=dev)
            self._step()

    def frame(self, xya, ya, xyb, yb, sub, check_every=0, active=None):
        import warp as wp
        dev = self.device
        arr = lambda v, t: wp.array(np.ascontiguousarray(v), dtype=t, device=dev)
        xa, xb = arr(xya.reshape(-1, 2), wp.vec2), arr(xyb.reshape(-1, 2), wp.vec2)
        a, b = arr(ya.ravel(), float), arr(yb.ravel(), float)
        for s_ in range(sub):
            wp.launch(sim_newton_kernel(), dim=self.W * N_KIN, inputs=[self.kin_flat, xa, xb, a, b, self.kyaw_d, self.kz_d,
                                                              self.kq_d, (s_ + 1) / sub, self.s0.body_q, self.s1.body_q],
                      device=dev)
            self._step()
            if check_every and (s_ + 1) % check_every == 0:
                pass
                if not hasattr(self, "seglen"):
                    self.seglen = np.stack([[np.linalg.norm(np.diff(r[k], axis=0), axis=1) for k in range(4)]
                                            for r in self.rest3])
                if not hasattr(self, "contact_n"):
                    self.contact_n = np.zeros(self.W, int); self.contact_dmin = np.full(self.W, np.inf)
                    self.contact_pairs = np.zeros((self.W, 4, 5), int)
                n, dm, pr = contact_violations(self.s0.body_q.numpy(), self.cab, self.seglen, self.kin[:, 8:13],
                                               self.kin[:, 13:18], active=active)
                self.contact_n += n; self.contact_dmin = np.minimum(self.contact_dmin, dm); self.contact_pairs += pr

    def vertices(self):
        bq = self.s0.body_q.numpy()
        B = bq[self.cab]
        V = np.empty((self.W, 4, 16, 3))
        V[:, :, :15] = B[..., :3]
        q = B[:, :, 14, 3:7]
        u = q[..., :3]; w_ = q[..., 3:4]
        v = np.zeros_like(u); v[..., 2] = self.L_last
        tip = v + 2.0 * np.cross(u, np.cross(u, v) + w_ * v)
        V[:, :, 15] = B[:, :, 14, :3] + tip
        return V


def _sub(fps, dt=DT):
    return max(1, int(round((1.0 / fps) / dt)))


def onestep_errors(engine_cache, transitions, X, settle=0, device=None):
    W = len(transitions)
    key = tuple((id(fr), i) for fr, _, i in transitions)
    eng = engine_cache.get(key)
    if eng is None:
        if len(engine_cache) >= 3:
            engine_cache.clear()
        eng = Engine([fr[0] for fr, _, _ in transitions], device=device)
        eng._sched = [schedule_tracked(fr) for fr, _, _ in transitions]
        engine_cache[key] = eng
    eng.set_theta(X)
    xa = np.stack([s[0][i] for s, (_, _, i) in zip(eng._sched, transitions)])
    ya = np.stack([s[1][i] for s, (_, _, i) in zip(eng._sched, transitions)])
    xb = np.stack([s[0][i + 1] for s, (_, _, i) in zip(eng._sched, transitions)])
    yb = np.stack([s[1][i + 1] for s, (_, _, i) in zip(eng._sched, transitions)])
    eng._reset_key_req = key
    eng.reset_state(np.stack([fr[i] for fr, _, i in transitions]), xa, ya)
    if settle:
        eng.hold(xa, ya, settle)
    subs = {_sub(f) for _, f, _ in transitions}
    assert len(subs) == 1, f"mixed frame rates {subs}"
    eng.frame(xa, ya, xb, yb, subs.pop())
    V = eng.vertices()[..., :2]
    tgt = np.stack([fr[i + 1] for fr, _, i in transitions])
    dist = np.linalg.norm(V - tgt, axis=-1).reshape(W, -1)
    ok = np.isfinite(dist).all(1)
    sq = np.where(ok, (np.nan_to_num(dist) ** 2).mean(1), np.inf)
    ab = np.where(ok, np.nan_to_num(dist).mean(1), np.inf)
    return sq, ab


def sim_newton_rollouts(rests, schedules, fps, X, settle=300, device=None, engine=None, check_every=0, stats=None):
    eng = engine if engine is not None else Engine(rests, device=device)
    eng.set_theta(X)
    eng.reset_state()
    T = max(len(s[0]) for s in schedules)
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], T - len(a), 0)])
    XY = np.stack([pad(s[0]) for s in schedules], 1)
    YW = np.stack([pad(s[1]) for s in schedules], 1)
    sub = _sub(fps)
    eng.hold(XY[0], YW[0], settle)
    out = [eng.vertices()]
    Ls = np.array([len(s[0]) for s in schedules])
    eng.contact_n = np.zeros(eng.W, int); eng.contact_dmin = np.full(eng.W, np.inf)
    eng.contact_pairs = np.zeros((eng.W, 4, 5), int)
    for t in range(T - 1):
        eng.frame(XY[t], YW[t], XY[t + 1], YW[t + 1], sub, check_every=check_every, active=(t + 1) < Ls)
        out.append(eng.vertices())
    if stats is not None:
        stats.extend(zip(eng.contact_n.tolist(), eng.contact_dmin.tolist(), list(eng.contact_pairs)))
    V = np.stack(out, 1)
    res = []
    for w, s in enumerate(schedules):
        v = V[w, :len(s[0])]
        bad = np.nonzero(~np.isfinite(v).reshape(len(v), -1).all(1))[0]
        res.append(v[:bad[0]] if len(bad) else v)
    return res


SPEED_RANGE = (0.65, 1.4)


PAUSE_PROB = 0.0


USE_OFFSETS = False


OFFSET_MAX = 1.5


MARGIN = 0.03


CLEARANCE = 0.005


N_SUB = 10


MAX_TRIES = 20


FPS = 4.0


def sample_warp(T, rng, offsets=True):
    dt = 1.0 / FPS
    n_max = 3 * T
    t = np.arange(n_max) * dt
    amp = rng.uniform(0.3, 1.5)
    s = np.zeros(n_max)
    for _ in range(3):
        per = rng.uniform(2.0, 15.0)
        s += np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi))
    s = np.clip(amp * s / np.sqrt(1.5), -1.0, 1.0)
    v = 1.0 + np.where(s >= 0, s * (SPEED_RANGE[1] - 1.0), s * (1.0 - SPEED_RANGE[0]))
    pauses = []
    if rng.random() < PAUSE_PROB:
        for _ in range(int(rng.integers(1, 4))):
            t0 = rng.uniform(1.0, max(1.5, T * dt * 0.9)); dur = rng.uniform(0.5, 2.0); ramp = 0.5
            w = np.ones(n_max)
            a = (t > t0 - ramp) & (t < t0); w[a] = 0.5 + 0.5 * np.cos(np.pi * (t[a] - (t0 - ramp)) / ramp)
            w[(t >= t0) & (t <= t0 + dur)] = 0.0
            b = (t > t0 + dur) & (t < t0 + dur + ramp); w[b] = 0.5 - 0.5 * np.cos(np.pi * (t[b] - t0 - dur) / ramp)
            v = v * w
            pauses.append((float(t0), float(dur)))
    tau = np.r_[0.0, np.cumsum(v[:-1])]
    taus = np.repeat(tau[:, None], 5, 1)
    off_amp = []
    if offsets:
        ramp = np.clip(t / 2.0, 0, 1)
        for m in range(5):
            a_m = rng.uniform(0, OFFSET_MAX); off_amp.append(float(a_m))
            dlt = np.zeros(n_max)
            for _ in range(2):
                dlt += np.sin(2 * np.pi * t / rng.uniform(4.0, 20.0) + rng.uniform(0, 2 * np.pi)) / 2
            taus[:, m] = np.maximum.accumulate(np.clip(tau + a_m * dlt * ramp, 0, T - 1))
    taus = np.clip(taus, 0, T - 1)
    done = np.nonzero((taus >= T - 1 - 1e-9).all(1))[0]
    n = int(done[0]) + 1 if len(done) else n_max
    taus = taus[:n]
    if not len(done):
        taus[-1] = T - 1
    return taus, {"speed_amp": float(amp), "pauses": pauses, "offset_amp": off_amp,
                  "n_frames": int(n), "speed_min": float(v[:n].min()), "speed_max": float(v[:n].max()),
                  "duration_ratio": float(n / T)}


def _interp(F, tau):
    i = np.clip(np.floor(tau).astype(int), 0, len(F) - 2)
    f = (tau - i).reshape(-1, *([1] * (F.ndim - 1)))
    return F[i] * (1 - f) + F[i + 1] * f


def end_frames(F, taus):
    F = np.asarray(F, float)
    E = np.empty((len(taus), 4, 4, 2))
    H = _interp(F[:, :, [0, 1]], taus[:, 0])
    E[:, :, 0:2] = H
    for k in range(4):
        E[:, k, 2:4] = _interp(F[:, k, [14, 15]], taus[:, 1 + k])
    return E


def subsample(E, n=N_SUB):
    if len(E) < 2:
        return E
    f = np.linspace(0, 1, n, endpoint=False)
    A = E[:-1, None] * (1 - f.reshape(1, -1, *([1] * (E.ndim - 1)))) + E[1:, None] * f.reshape(1, -1, *([1] * (E.ndim - 1)))
    return np.concatenate([A.reshape(-1, *E.shape[1:]), E[-1:]])


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]], -1)


def mover_poses(E):
    pass
    out = (E[:, :, 1] - E[:, :, 0]).mean(1)
    hy = np.arctan2(out[:, 1], out[:, 0]) + np.pi / 2
    hm = np.array(HUB_CAVS).mean(0)
    C = np.empty((len(E), 5, 2)); Y = np.empty((len(E), 5))
    C[:, 0] = E[:, 0, 0] - _rot(hm, hy); Y[:, 0] = hy
    for k in range(4):
        v = E[:, k, 3] - E[:, k, 2]
        y = np.arctan2(v[:, 1], v[:, 0]) + np.pi / 2
        C[:, 1 + k] = E[:, k, 3] - _rot(PLUG_CAV, y); Y[:, 1 + k] = y
    return C, Y


def attachments(E, kind):
    if kind == "nodes":
        return np.stack([E[:, :, 0], E[:, :, 3]], 2)
    pass
    C, Y = mover_poses(E)
    A = np.empty((len(E), 4, 2, 2))
    for k in range(4):
        A[:, k, 0] = C[:, 0] + _rot(HUB_CAVS[k], Y[:, 0])
        A[:, k, 1] = C[:, 1 + k] + _rot(PLUG_CAV, Y[:, 1 + k])
    return A


def footprint_gap(E):
    pass
    C, _ = mover_poses(E)
    i, j = np.triu_indices(5, 1)
    d = np.abs(C[:, i] - C[:, j]).max(-1)
    return d - 2 * PLATFORM_HALF[0]


def real_limits(F, kind):
    F = np.asarray(F, float)
    taus = np.repeat(np.arange(len(F), dtype=float)[:, None], 5, 1)
    E = subsample(end_frames(F, taus))
    A = attachments(E, kind)
    dist = np.linalg.norm(A[:, :, 0] - A[:, :, 1], axis=-1)
    L = np.linalg.norm(np.diff(F[0], axis=1), axis=-1).sum(-1)
    gap = footprint_gap(E)
    d_max = dist.max(0)
    return {"lim_dist": d_max, "arc_len": L, "real_ratio_max": d_max / L,
            "real_exceeds_margin": (d_max > (1 - MARGIN) * L).tolist(),
            "real_min_gap": float(gap.min()), "lim_gap": float(min(CLEARANCE, gap.min()))}


def check(E, lim, kind):
    S = subsample(E)
    A = attachments(S, kind)
    dist = np.linalg.norm(A[:, :, 0] - A[:, :, 1], axis=-1)
    a_ok = bool((dist <= lim["lim_dist"][None] + 1e-9).all())
    g = footprint_gap(S).min()
    b_ok = bool(g >= lim["lim_gap"] - 1e-9)
    return a_ok and b_ok, {"a_ok": a_ok, "b_ok": b_ok, "max_ratio": float((dist / lim["arc_len"]).max()),
                           "min_gap_mm": float(1000 * g)}


def sample_valid(F, rng, kind, lim=None):
    lim = real_limits(F, kind) if lim is None else lim
    rej = {"a": 0, "b": 0}
    for tr in range(1, MAX_TRIES + 1):
        taus, meta = sample_warp(len(F), rng, offsets=USE_OFFSETS)
        E = end_frames(F, taus)
        ok, st = check(E, lim, kind)
        if ok:
            return E, taus, {**meta, "tries": tr, "rejected_a": rej["a"], "rejected_b": rej["b"], "offsets": USE_OFFSETS,
                             **st}
        rej["a"] += not st["a_ok"]; rej["b"] += not st["b_ok"]
    taus, meta = sample_warp(len(F), rng, offsets=False)
    E = end_frames(F, taus)
    ok, st = check(E, lim, kind)
    return E, taus, {**meta, "tries": MAX_TRIES + 1, "rejected_a": rej["a"], "rejected_b": rej["b"],
                     "offsets": False, "fallback_ok": ok, **st}


REL_TOL = 1.5


ABS_TOL = 1.0


def contact_excess(pairs_w, T_w, pairs_r, T_r, mm_w=0, mm_r=0):
    rw = np.asarray(pairs_w, float) / max(T_w, 1); rr = np.asarray(pairs_r, float) / max(T_r, 1)
    new = (rw > 0) & (rr == 0)
    over = rw > REL_TOL * rr + ABS_TOL
    bad = new | over
    if mm_w > 0 and mm_r == 0:
        return "mover-mover contact"
    if bad.any():
        k, m = np.argwhere(bad)[0]
        return f"cable{k}-mover{m}: {rw[k, m]:.2f}/frame vs real {rr[k, m]:.2f}/frame"
    return None
