
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rectify import BoardTransform, board_mask


@dataclass
class Segmentation:
    wires: np.ndarray
    connectors: np.ndarray
    board: np.ndarray
    connector_boxes: list
    module_centres: np.ndarray


def detect_modules(
    g: np.ndarray,
    board: np.ndarray,
    dark_thresh: int = 130,
    open_kernel: int = 140,
    min_core_area: int = 2000,
) -> np.ndarray:
    solid = ((g < dark_thresh) & board).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    filled = np.zeros_like(solid)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    op = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                          np.ones((open_kernel, open_kernel), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(op, 8)

    centres = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w_ / max(h_, 1)
        if a < max(min_core_area, 30000) or a > 50000:
            continue
        if not (0.80 <= aspect <= 1.25):
            continue
        centres.append(cent[i])
    return np.array(centres, dtype=float).reshape(-1, 2)


def _solid_mask(frame, tf, dark_thresh=130):
    from rectify import board_mask
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bm = board_mask(frame, tf, inset_mm=30)
    solid = ((g < dark_thresh) & bm).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    filled = np.zeros_like(solid)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled, g, bm


def static_mask(cap, tf, read_fn, n_frames, samples=9, dark_thresh=130):
    acc, used = None, 0
    for idx in np.linspace(0, max(0, n_frames - 1), samples).astype(int):
        fr = read_fn(cap, int(idx))
        if fr is None:
            continue
        f, _, _ = _solid_mask(fr, tf, dark_thresh)
        acc = f.astype(np.int32) if acc is None else acc + f
        used += 1
    if not used:
        return None
    return (acc > used // 2).astype(np.uint8)


def scene_quality(frame, tf, over_thresh=0.40, blob_area=70000,
                  dark_thresh=130, static=None, modules=None,
                  module_radius=170.0):
    filled, g, bm = _solid_mask(frame, tf, dark_thresh)
    over = float((g[bm] >= 250).mean()) if bm.any() else 1.0
    if static is not None:
        filled = (filled.astype(bool) & ~static.astype(bool)).astype(np.uint8)
    if modules is not None:
        drop = np.zeros(filled.shape, np.uint8)
        for c in np.asarray(modules, float).reshape(-1, 2):
            if not np.isfinite(c).all():
                continue
            cv2.circle(drop, (int(round(c[0])), int(round(c[1]))),
                       int(module_radius), 1, thickness=-1)
        filled = (filled.astype(bool) & ~drop.astype(bool)).astype(np.uint8)
    op = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((140, 140), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(op, 8)
    biggest = max([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)],
                  default=0)

    info = {"overexposed": over, "biggest_blob": biggest}
    ok = over <= over_thresh and biggest <= blob_area
    return ok, info


def segment(
    frame: np.ndarray,
    tf: BoardTransform,
    inset_mm: float = 30.0,
    dark_thresh: int = 100,
    module_min_area: int = 8000,
    module_dilate: int = 45,
    blackhat_kernel: int = 31,
    wire_percentile: float = 99.0,
    wire_min_response: int = 30,
) -> Segmentation:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    board = board_mask(frame, tf, inset_mm=inset_mm)

    module_centres = detect_modules(g, board)

    solid = ((g < dark_thresh) & board).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE,
                             np.ones((31, 31), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(solid, 8)

    conn = np.zeros_like(solid)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < module_min_area:
            continue
        holds_module = any(lab[int(cy), int(cx)] == i
                           for cx, cy in module_centres
                           if 0 <= int(cy) < lab.shape[0]
                           and 0 <= int(cx) < lab.shape[1])
        if holds_module:
            conn[lab == i] = 1
            boxes.append((int(x), int(y), int(w), int(h), int(a),
                          float(cent[i][0]), float(cent[i][1])))
    boxes.sort(key=lambda b: -b[4])
    conn_d = cv2.dilate(conn, np.ones((module_dilate, module_dilate), np.uint8))

    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          np.ones((blackhat_kernel, blackhat_kernel), np.uint8))
    valid = board & (conn_d == 0)
    if valid.sum() < 100:
        raise RuntimeError("almost no board left after masking connectors")
    thr = max(wire_min_response, float(np.percentile(bh[valid], wire_percentile)))
    wires = (bh > thr) & valid
    wires = cv2.morphologyEx(wires.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8)).astype(bool)

    return Segmentation(wires=wires, connectors=conn_d.astype(bool),
                        board=board, connector_boxes=boxes,
                        module_centres=module_centres)


def wire_components(seg: Segmentation, min_area: int = 1200):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        seg.wires.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out.append((a, lab == i))
    out.sort(key=lambda t: -t[0])
    return out
