import os
import re
from pathlib import Path
import numpy as np
from simulate import build_xml, cable_ball_joints, set_per_joint_stiffness
from simulate import board_normal_axes, rest_quats


ROOT = Path(__file__).resolve().parent


TRACKS = Path(os.environ.get("GENERAL_FIT_TRACKS") or "../data/tracks")


CUSTOM_TRACKS = bool(os.environ.get("GENERAL_FIT_TRACKS"))


N_C, N_J = 4, 14


N_PJ = N_C * N_J


MASS0, MASS_LO, MASS_HI = 0.15, 0.02, 0.2


RADIUS = 0.003


I_K, I_REST, I_DAMP = 0, N_PJ, 2 * N_PJ


I_BEND, I_TWIST, I_MU = 3 * N_PJ, 3 * N_PJ + N_C, 3 * N_PJ + 2 * N_C


I_MASS = I_MU + 1


N_PAR = I_MASS + N_C


_CACHE, _AXIS = {}, {}


def unpack(x):
    x = np.asarray(x, float)
    pj = lambda i: x[i:i + N_PJ].reshape(N_C, N_J)
    return dict(k=np.exp(pj(I_K)), rest_deg=pj(I_REST), damp=np.exp(pj(I_DAMP)),
                bend=np.exp(x[I_BEND:I_BEND + N_C]), twist=np.exp(x[I_TWIST:I_TWIST + N_C]),
                mu=float(np.exp(x[I_MU])), mass=np.exp(x[I_MASS:I_MASS + N_C]))


def make_model(shp, p):
    import mujoco
    xml = build_xml(shp, mass_per_m=MASS0, mu=p["mu"], radius=RADIUS)
    it = iter(p["mass"] / (np.pi * RADIUS ** 2))
    xml = re.sub(r'density="[^"]*"', lambda _m: f'density="{next(it):.0f}"', xml)
    for key in ("twist", "bend"):
        it = iter(p[key])
        xml = re.sub(rf'<config key="{key}" value="[^"]*"/>',
                     lambda _m: f'<config key="{key}" value="{next(it):g}"/>', xml)
    m = mujoco.MjModel.from_xml_string(xml)
    clip_axes = board_normal_axes(m)
    set_per_joint_stiffness(m, p["k"], rest_quat=rest_quats(p["rest_deg"], clip_axes))
    ids = cable_ball_joints(m)
    for c in range(N_C):
        for j in range(N_J):
            a = m.jnt_dofadr[ids[c, j]]
            m.dof_damping[a:a + 3] = p["damp"][c, j]
    return m


def _eval_one(job):
    cand_i, clip, x = job
    from simulate import rollout_errs
    shp, fps = _CACHE[clip]
    try:
        m = make_model(shp, unpack(x))
        e = float(rollout_errs(m, shp, fps, per_cable=True).mean(0).mean())
        return cand_i, clip, e if np.isfinite(e) else 1e3
    except Exception as exc:
        print(f"  {clip}: {type(exc).__name__}: {exc}", flush=True)
        return cand_i, clip, 1e3
