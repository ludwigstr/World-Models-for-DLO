
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BOARD_W_MM = 1200.0
BOARD_H_MM = 1680.0


@dataclass
class BoardTransform:

    origin_px: np.ndarray
    e_x: np.ndarray
    e_y: np.ndarray
    mm_per_px: float
    residual_rms_px: float
    implied_height_mm: float

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        d = np.asarray(pts_px, dtype=np.float64) - self.origin_px
        x = d @ self.e_x
        y = d @ self.e_y
        return np.stack([x, y], -1) * self.mm_per_px

    def to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_mm, dtype=np.float64) / self.mm_per_px
        return self.origin_px + p[..., :1] * self.e_x + p[..., 1:2] * self.e_y


def _fit_line_robust(u: np.ndarray, v: np.ndarray, iters: int = 5):
    p = np.polyfit(u, v, 1)
    keep = np.ones_like(u, dtype=bool)
    for _ in range(iters):
        r = np.abs(v - np.polyval(p, u))
        keep = r < max(2.5 * float(np.median(r)), 1.0)
        if keep.sum() < 4:
            break
        p = np.polyfit(u[keep], v[keep], 1)
    rms = float((v - np.polyval(p, u))[keep].std()) if keep.sum() else float("inf")
    return p, rms


def _edge_track(grad: np.ndarray, axis_idx: np.ndarray, lo: int, hi: int,
                positive: bool, along_rows: bool):
    pos, strength = [], []
    for i in axis_idx:
        strip = grad[i, lo:hi] if along_rows else grad[lo:hi, i]
        j = int(np.argmax(strip) if positive else np.argmin(strip))
        pos.append(lo + j)
        strength.append(abs(float(strip[j])))
    return np.array(pos, float), np.array(strength, float)


def find_board(frame: np.ndarray, margin: int = 220) -> BoardTransform:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    h, w = g.shape
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)

    rows = np.arange(int(0.07 * h), int(0.96 * h), 20)
    xl, sl = _edge_track(gx, rows, 0, margin, True, True)
    xr, sr = _edge_track(gx, rows, w - margin, w, False, True)

    cols = np.arange(int(0.15 * w), int(0.85 * w), 20)
    yb, sb = _edge_track(gy, cols, h - margin, h, False, False)

    def keep_strong(vals, strength, other=None):
        k = strength > np.percentile(strength, 25)
        return (vals[k], other[k]) if other is not None else vals[k]

    kl = sl > np.percentile(sl, 25)
    kr = sr > np.percentile(sr, 25)
    kb = sb > np.percentile(sb, 25)

    pl, rms_l = _fit_line_robust(rows[kl], xl[kl])
    pr, rms_r = _fit_line_robust(rows[kr], xr[kr])
    pb, rms_b = _fit_line_robust(cols[kb], yb[kb])

    def vert_dir(pv):
        d = np.array([pv[0], 1.0])
        d = d / np.linalg.norm(d)
        return -d

    e_y = vert_dir(pl) + vert_dir(pr)
    e_y = e_y / np.linalg.norm(e_y)
    e_x = np.array([-e_y[1], e_y[0]])

    y_ref = float(np.median(rows))
    p_left = np.array([np.polyval(pl, y_ref), y_ref])
    p_right = np.array([np.polyval(pr, y_ref), y_ref])
    width_px = float(abs((p_right - p_left) @ e_x))
    mm_per_px = BOARD_W_MM / width_px

    if e_x @ (p_right - p_left) < 0:
        e_x = -e_x

    a_v, b_v = pl
    a_b, b_b = pb
    denom = 1.0 - a_b * a_v
    y0 = (a_b * b_v + b_b) / denom
    bl = np.array([a_v * y0 + b_v, y0])

    return BoardTransform(
        origin_px=bl,
        e_x=e_x,
        e_y=e_y,
        mm_per_px=mm_per_px,
        residual_rms_px=float(max(rms_l, rms_r)),
        implied_height_mm=float(bl[1] * mm_per_px),
    )


_MASK_CACHE: dict = {}


def board_mask(frame: np.ndarray, tf: BoardTransform, inset_mm: float = 25.0):
    h, w = frame.shape[:2]
    key = (h, w, round(float(tf.origin_px[0]), 3), round(float(tf.origin_px[1]), 3),
           round(float(tf.e_x[0]), 6), round(float(tf.e_x[1]), 6),
           round(float(tf.mm_per_px), 8), float(inset_mm))
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float64)
    mm = tf.to_mm(pts).reshape(h, w, 2)
    m = ((mm[..., 0] > inset_mm) & (mm[..., 0] < BOARD_W_MM - inset_mm)
         & (mm[..., 1] > inset_mm) & (mm[..., 1] < BOARD_H_MM - inset_mm))
    if len(_MASK_CACHE) > 8:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = m
    return m
