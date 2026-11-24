import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_scene import (load_clip, _rotz, DIR, COLORS, BOARD_W, BOARD_H,
                          HUB_CAVS, PLUG_CAV, CAV_Z, PLATFORM_HALF, MODULE_HALF)
from newton_full_scene import make_kinematic

ROOT = Path(__file__).resolve().parent
HUB_RGBA = (0.85, 0.15, 0.15)
GREY = (0.55, 0.55, 0.55)
BOARD_RGBA = (0.80, 0.79, 0.75)


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


def draw_box(ax, center, half, yaw, facecolor, zorder):
    from matplotlib.patches import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    corners = local @ R.T + center
    shadow = corners + np.array([0.006, -0.006])
    ax.add_patch(Polygon(shadow, closed=True, facecolor="black", alpha=0.28,
                         linewidth=0, zorder=zorder - 0.2))
    ax.add_patch(Polygon(corners, closed=True, facecolor=facecolor,
                         edgecolor=np.array(facecolor) * 0.5, linewidth=1.1, zorder=zorder))
    hl = 0.55 * corners[3] + 0.45 * center
    hl_pts = np.array([corners[3], 0.5 * (corners[3] + corners[0]), hl, 0.5 * (corners[3] + corners[2])])
    ax.add_patch(Polygon(hl_pts, closed=True, facecolor="white", alpha=0.20,
                         linewidth=0, zorder=zorder + 0.1))


def draw_mover(ax, name, pos, yaw):
    color = HUB_RGBA if name == "hub" else COLORS[int(name[-1])]
    draw_box(ax, pos, PLATFORM_HALF[0], 0.0, color, zorder=2)
    draw_box(ax, pos, MODULE_HALF, yaw, GREY, zorder=3)


def draw_cable(ax, pts, color, zorder):
    ax.plot(pts[:, 0], pts[:, 1], "-", color="black", alpha=0.25, linewidth=9,
           zorder=zorder - 0.1, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=6.5,
           zorder=zorder, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(pts[:, 0], pts[:, 1], "-", color="white", alpha=0.35, linewidth=2.2,
           zorder=zorder + 0.1, solid_capstyle="round", solid_joinstyle="round")


