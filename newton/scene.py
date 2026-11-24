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


TRACKS = ROOT.parent / "data" / "tracks"


def load_clip(clip, t0=0, t1=None, max_frames=400):
    z = np.load(TRACKS / f"{clip}.npz")
    shp = {k: np.asarray(z[f"c{k}"], float) for k in range(4)}
    return shp, 0, len(shp[0]) - 1, float(z["fps"])


SC = 0.5


PLATFORM_HALF = (0.15 * SC, 0.15 * SC, 0.01 * SC)


PLATE_HALF    = (0.08 * SC, 0.08 * SC, 0.02 * SC)


PLATE_Z       = 0.03 * SC


MODULE_HALF   = 0.05 * SC


MODULE_Z      = 0.10 * SC


PLATFORM_Z    = 0.01 * SC


HUB_CAVS  = [(0.0, -0.05 * SC, 0.0), (-0.02 * SC, -0.05 * SC, 0.0),
             (0.02 * SC, -0.05 * SC, 0.0), (0.0, -0.05 * SC, 0.02 * SC)]


PLUG_CAV  = (0.0, 0.05 * SC, 0.0)


CAV_Z = PLATFORM_Z + MODULE_Z


def _rotz(v, yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2]])


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
        pass
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
        pass
    movers = mover_poses(shp, t) if scene_collision else None
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


def mover_poses(shp, t):
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


import os


FALLBACK_DIR = Path("/tmp/wire_model_s_fallback")


def safe_savez(path, log=None, **arrays):
    tmp = path.with_name("tmp_" + path.name)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        return path
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        fb = FALLBACK_DIR / path.name
        np.savez(fb, **arrays)
        if log:
            log(f"WARNING: could not write {path} ({e}) -- checkpoint saved to {fb}")
        return fb


_worker_cache = {}


def _init_worker(gpu_id):
    _worker_cache["device"] = f"cuda:{gpu_id}"


def _clip_frame_t1(clip, cutoff_sec):
    if cutoff_sec is None:
        return None
    import cv2
    pass
    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    cap.release()
    return int(round(cutoff_sec * fps))


def _z_loss_mm(z_nodes, targets_mm):
    zs = np.stack(z_nodes) * 1000.0
    tg = targets_mm[:zs.shape[0], :, :zs.shape[2]]
    m = np.isfinite(tg)
    if not m.any():
        return 0.0
    return float(np.abs(zs[m] - tg[m]).mean())
