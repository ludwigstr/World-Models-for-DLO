from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dlolab_path import DLO_LAB
sys.path.insert(0, str(DLO_LAB))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from mujoco_scene import (track_clip, _rotz, DIR, CAV_Z, HUB_CAVS, PLUG_CAV,
                          COLORS,
                          BOARD_W, BOARD_H, PLATFORM_HALF, PLATE_HALF, PLATE_Z,
                          MODULE_HALF, MODULE_Z, PLATFORM_Z)
import genesis_full_scene as MJ

N_BIND = 2


def resample_route(P, n):
    P = np.asarray(P, float)
    m = len(P)
    u = np.linspace(0.0, m - 1.0, n)
    i0 = np.clip(np.floor(u).astype(int), 0, m - 2)
    w = (u - i0)[:, None]
    return P[i0] * (1.0 - w) + P[i0 + 1] * w


def sim_at_tracked(sim, n_track):
    return resample_route(sim, n_track)


def table_route(route2, table_fraction, z_conn, z_table=0.003):
    P = np.asarray(route2, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_arc = np.concatenate([[0.0], np.cumsum(seg)])
    L = s_arc[-1]
    ramp = max(1e-6, (1.0 - table_fraction) / 2.0 * L)
    f = np.ones(len(P))
    f = np.where(s_arc < ramp, s_arc / ramp, f)
    f = np.where(s_arc > L - ramp, (L - s_arc) / ramp, f)
    z = z_conn + f * (z_table - z_conn)
    return np.hstack([P, z[:, None]])


def rod_file(route_m, path):
    v = np.asarray(route_m, np.float32)
    np.save(path, v)
    return str(path)


def build_rod_scene(shp, frame_idx=0, K=1.0e4, E=1.0e5, G=0.0,
                    plastic_yield=10.0, plastic_creep=1.0,
                    segment_radius=0.003, segment_mass=0.002,
                    static_friction=0.3, kinetic_friction=0.25,
                    restitution=0.0,
                    use_inextensible=True, mu=1.0, requires_grad=False,
                    scratch=None, bptt_window=20, grad_clip=1.0,
                    only_cable=None, n_nodes=None, table_fraction=None,
                    damping=40.0, rigid_material=True):
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=1e-3, requires_grad=requires_grad,
            bptt_window=(bptt_window if requires_grad else 0)),
        rod_options=gs.options.RODOptions(
            damping=damping, angular_damping=min(damping, 10.0),
            n_pbd_iters=10,
            grad_clip=(grad_clip if requires_grad else 0.0),
            requires_grad_K=requires_grad, requires_grad_E=requires_grad,
            requires_grad_G=requires_grad),
        rigid_options=gs.options.RigidOptions(skip_backward=True),
    )

    scratch = Path(scratch or (ROOT / "data" / "genesis_rod" / f"pid{os.getpid()}"))
    scratch.mkdir(parents=True, exist_ok=True)
    MJ._sweep_stale_pid_dirs(scratch.parent)

    def write(name, xml):
        p = scratch / f"{name}.xml"
        p.write_text(xml)
        return str(p)

    rigid_mat = (gs.materials.Rigid(friction=float(np.clip(mu, 0.01, 5.0)),
                                    coup_friction=float(mu))
                 if rigid_material else None)

    ents = {}
    ents["board"] = scene.add_entity(
        gs.morphs.MJCF(file=write("board", MJ.board_xml(mu=mu))),
        **({} if rigid_mat is None else {"material": rigid_mat}))

    poses0 = MJ_poses(shp, frame_idx)
    for name in ["hub"] + [f"plug{k}" for k in range(4)]:
        pos, yaw = poses0[name]
        xy = (float(pos[0]), float(pos[1]), 0.0)
        ents[name] = scene.add_entity(
            gs.morphs.MJCF(file=write(name, MJ.platform_xml(name)), pos=xy),
            **({} if rigid_mat is None else {"material": rigid_mat}))
        ents[f"{name}_top"] = scene.add_entity(
            gs.morphs.MJCF(file=write(f"{name}_top", MJ.topper_xml(name, yaw)),
                           pos=xy),
            **({} if rigid_mat is None else {"material": rigid_mat}))

    rods = []
    cable_ids = range(4) if only_cable is None else [int(only_cable)]
    for k in cable_ids:
        r2 = shp[k][frame_idx]
        if table_fraction is None:
            route = np.hstack([r2, np.full((len(r2), 1), CAV_Z)])
        else:
            route = table_route(r2, table_fraction, CAV_Z)
        if n_nodes:
            route = resample_route(route, int(n_nodes))
        f = rod_file(route, scratch / f"cable{k}.npy")
        rods.append(scene.add_entity(
            material=gs.materials.ROD.Base(
                K=K, E=E, G=G, plastic_yield=plastic_yield,
                plastic_creep=plastic_creep, segment_radius=segment_radius,
                segment_mass=(float(segment_mass[k]) if np.ndim(segment_mass) else segment_mass),
                use_inextensible=use_inextensible,
                static_friction=static_friction,
                kinetic_friction=kinetic_friction,
                restitution=restitution),
            morph=gs.morphs.Rod(file=f, rest_state="default", fixed=False),
            surface=gs.surfaces.Default(color=(*COLORS[k], 1.0))))
    return scene, ents, rods, poses0


def _per_cable(v, k, shared_ndim):
    a = np.asarray(v, float)
    return a[k] if a.ndim > shared_ndim else a


def MJ_poses(shp, t):
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


def _mod_link(ents, name):
    for l in ents[f"{name}_top"].links:
        if l.name == f"{name}_mod":
            return l
    raise KeyError(f"{name}_mod not found")


def eval_clip(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
              kp_mover=4000.0, kv_mover=120.0, kp_turn=60.0, kv_turn=4.0,
              shp_fps=None, verts_out=None, settle=300, hinge_scale=None,
              slack=1.0, stretch_scale=None, twist_scale=None,
              kappa_rest=None, pin_ends=False, hinge_damping=None,
              damping=None, state_a=0.0, state_b=0.0,
              state_a_load=None, state_a_unload=None,
              codebook=None, **rod_kw):
    import genesis as gs
    import cv2

    t1 = None
    if t1_sec is not None:
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_p = cap.get(cv2.CAP_PROP_FPS) or 4.0
        cap.release()
        t1 = int(round(t1_sec * fps_p))
    if shp_fps is not None:
        shp, fps = shp_fps
    else:
        shp, _t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))

    if damping is not None:
        rod_kw["damping"] = damping
    scene, ents, rods, _ = build_rod_scene(shp, 0, **rod_kw)
    scene.build(n_envs=1)

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    dofs, dofs_top = {}, {}
    for n in names:
        e = ents[n]
        dofs[n] = (e.get_joint(f"{n}_x").dofs_idx_local
                   + e.get_joint(f"{n}_y").dofs_idx_local)
        e.set_dofs_kp(np.full(2, kp_mover), dofs[n])
        e.set_dofs_kv(np.full(2, kv_mover), dofs[n])
        et = ents[f"{n}_top"]
        xy = (et.get_joint(f"{n}_x").dofs_idx_local
              + et.get_joint(f"{n}_y").dofs_idx_local)
        turn = et.get_joint(f"{n}_turn").dofs_idx_local
        dofs_top[n] = (xy, turn)
        et.set_dofs_kp(np.full(2, kp_mover), xy)
        et.set_dofs_kv(np.full(2, kv_mover), xy)
        et.set_dofs_kp(np.full(1, kp_turn), turn)
        et.set_dofs_kv(np.full(1, kv_turn), turn)

    n_v = rods[0].n_vertices
    for k, rod in enumerate(rods):
        rod.attach_to_rigid_link(_mod_link(ents, "hub"), list(range(N_BIND)))
        rod.attach_to_rigid_link(_mod_link(ents, f"plug{k}"),
                                 list(range(n_v - N_BIND, n_v)))
        if hinge_scale is not None:
            import torch
            hs = _per_cable(hinge_scale, k, 1)
            if len(hs) != rod.n_internal_vertices:
                hs = resample_route(hs[:, None],
                                    rod.n_internal_vertices)[:, 0]
            rod.set_bending_scale(torch.as_tensor(hs.astype(np.float32)))
        if stretch_scale is not None:
            import torch
            ss = _per_cable(stretch_scale, k, 1)
            if len(ss) != rod.n_edges:
                ss = resample_route(ss[:, None], rod.n_edges)[:, 0]
            rod.set_stretching_scale(torch.as_tensor(ss.astype(np.float32)))
        if twist_scale is not None:
            import torch
            ts = _per_cable(twist_scale, k, 1)
            if len(ts) != rod.n_internal_vertices:
                ts = resample_route(ts[:, None], rod.n_internal_vertices)[:, 0]
            rod.set_twisting_scale(torch.as_tensor(ts.astype(np.float32)))
        if state_a or state_b:
            rod.set_state_coeffs(state_a, state_b)
        if state_a_load is not None or state_a_unload is not None:
            import torch
            n_h = rod.n_internal_vertices
            lo = np.full(n_h, 0.0) if state_a_load is None else np.asarray(state_a_load, float)
            un = np.full(n_h, 0.0) if state_a_unload is None else np.asarray(state_a_unload, float)
            if lo.size == 1: lo = np.full(n_h, float(lo))
            if un.size == 1: un = np.full(n_h, float(un))
            un = np.minimum(un, lo)
            rod.set_hinge_state_a(torch.as_tensor(lo.astype(np.float32)),
                                  torch.as_tensor(un.astype(np.float32)))
        if codebook is not None:
            import torch
            cb = np.asarray(codebook, float)
            K, F = 4, 4
            rod.set_codebook(cb[:K], cb[K:2 * K],
                             cb[2 * K:2 * K + K * F].reshape(K, F),
                             tau=0.7, on=True)
        if hinge_damping is not None:
            import torch
            rod.set_bending_damping(torch.as_tensor(
                np.full(rod.n_internal_vertices, hinge_damping, np.float32)))
        if kappa_rest is not None:
            import torch
            kr = _per_cable(kappa_rest, k, 2)
            if len(kr) != rod.n_internal_vertices:
                kr = np.stack([resample_route(kr[:, i:i + 1],
                                              rod.n_internal_vertices)[:, 0]
                               for i in range(kr.shape[1])], axis=1)
            rod.set_kappa_rest_offset(torch.as_tensor(kr.astype(np.float32)))
        if slack != 1.0:
            rod.set_rest_length_scale(slack)

    def unwrap(prev, cur):
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    p0 = MJ_poses(shp, 0)
    base = {n: np.asarray(p0[n][0], float) for n in names}
    base_yaw = {n: float(p0[n][1]) for n in names}
    prev_yaw = dict(base_yaw)

    def apply_movers(t, f=1.0):
        p = MJ_poses(shp, t)
        for n in names:
            pos, yaw = p[n]
            prev_yaw[n] = unwrap(prev_yaw[n], yaw)
            rel_xy = np.asarray(pos, float) - base[n]
            rel_yaw = np.array([prev_yaw[n] - base_yaw[n]])
            if pin_ends:
                cur = ents[n].get_dofs_position(dofs[n]).cpu().numpy()
                tgt = cur + f * (rel_xy - cur)
                ents[n].set_dofs_position(tgt, dofs[n])
                ents[n].set_dofs_velocity(np.zeros(2), dofs[n])
                et = ents[f"{n}_top"]
                cur_t = et.get_dofs_position(dofs_top[n][0]).cpu().numpy()
                et.set_dofs_position(cur_t + f * (rel_xy - cur_t),
                                     dofs_top[n][0])
                cur_y = et.get_dofs_position(dofs_top[n][1]).cpu().numpy()
                et.set_dofs_position(cur_y + f * (rel_yaw - cur_y),
                                     dofs_top[n][1])
                et.set_dofs_velocity(np.zeros(2), dofs_top[n][0])
                et.set_dofs_velocity(np.zeros(1), dofs_top[n][1])
            else:
                ents[n].control_dofs_position(rel_xy, dofs[n])
                ents[f"{n}_top"].control_dofs_position(rel_xy, dofs_top[n][0])
                ents[f"{n}_top"].control_dofs_position(rel_yaw, dofs_top[n][1])

    apply_movers(0)
    for _ in range(settle):
        scene.step()

    errs = []
    for t in range(n_frames):
        if pin_ends:
            for s_i in range(steps_per_frame):
                apply_movers(t, (s_i + 1) / steps_per_frame)
                scene.step()
        else:
            apply_movers(t)
            for _ in range(steps_per_frame):
                scene.step()
        sim = [r.get_state().pos.cpu().numpy().reshape(-1, 3) for r in rods]
        e_all = [float(np.linalg.norm(
            sim_at_tracked(sim[k], len(shp[k][t]))[:, :2] - shp[k][t],
            axis=1).mean()) * 1000 for k in range(4)]
        errs.append(float(np.mean(e_all)))
        if verts_out is not None:
            verts_out.append(np.stack(sim))
    return errs
