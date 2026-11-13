
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .rectify import board_mask

FASTDLO_DIR = Path(os.environ.get(
    "FASTDLO_DIR",
    "extern/fastdlo"))

NET_W, NET_H = 640, 848

_PIPE = None


def available() -> bool:
    return (FASTDLO_DIR / "weights" / "CP_seg.pth").exists()


def _pipeline():
    global _PIPE
    if _PIPE is None:
        if not available():
            raise RuntimeError(f"FASTDLO-Gewichte nicht gefunden in {FASTDLO_DIR}")
        sys.path.insert(0, str(FASTDLO_DIR))
        cwd = os.getcwd()
        os.chdir(FASTDLO_DIR)
        try:
            from fastdlo.core import Pipeline
            _PIPE = Pipeline(
                checkpoint_siam=str(FASTDLO_DIR / "weights" / "CP_siam.pth"),
                checkpoint_seg=str(FASTDLO_DIR / "weights" / "CP_seg.pth"),
                img_w=NET_W, img_h=NET_H)
        finally:
            os.chdir(cwd)
    return _PIPE


def fastdlo_wires(frame, tf, conn_radius=135, module_centres=(), close=9,
                  mask_th=77):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    out, _ = _pipeline().run(source_img=small, mask_th=mask_th)

    m_small = np.any(np.asarray(out) > 0, axis=2) if out.ndim == 3 else out > 0
    m = cv2.resize(m_small.astype(np.uint8), (w, h),
                   interpolation=cv2.INTER_NEAREST).astype(bool)

    m &= board_mask(frame, tf, inset_mm=30)
    if len(module_centres):
        yy, xx = np.mgrid[0:h, 0:w]
        for c in np.asarray(module_centres, float).reshape(-1, 2):
            m &= ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) >= conn_radius ** 2

    if close > 1:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((close, close), np.uint8)).astype(bool)
    return m


def instances(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    p = _pipeline()
    p.run(source_img=small, mask_th=77)
    sx, sy = w / NET_W, h / NET_H
    res = []
    for v in getattr(p, "last_splines", {}).values():
        pts = np.asarray(v["points"], float)
        res.append(np.stack([pts[:, 1] * sx, pts[:, 0] * sy], -1))
    return res
