
from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


def _skeleton(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (cv2.ximgproc.thinning(m * 255) > 0)
    if _HAVE_SKIMAGE:
        return skeletonize(m > 0)
    raise RuntimeError("need cv2.ximgproc.thinning or skimage for skeletonize")


_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_graph(skel: np.ndarray):
    ys, xs = np.nonzero(skel)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    nodes = np.stack([ys, xs], -1)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(ys))}
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in _NEIGH:
            j = idx.get((y + dy, x + dx))
            if j is not None:
                adj[i].append((j, float(np.hypot(dy, dx))))
    return nodes, adj


def _bfs_farthest(adj, src, n):
    import heapq
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    reach = np.where(np.isfinite(dist), dist, -1)
    return int(reach.argmax()), dist, parent


def _trace_path(parent, target):
    path = []
    v = int(target)
    while v != -1:
        path.append(v)
        v = int(parent[v])
    return path[::-1]


def longest_path_pixels(mask: np.ndarray) -> np.ndarray:
    skel = _skeleton(mask)
    if skel.sum() < 2:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("empty wire mask")
        return np.stack([xs, ys], -1)[:1].astype(float)

    nodes, adj = _skeleton_graph(skel)
    n = len(nodes)
    a, _, _ = _bfs_farthest(adj, 0, n)
    b, dist, parent = _bfs_farthest(adj, a, n)
    pix = nodes[_trace_path(parent, b)]
    return pix[:, ::-1].astype(float)


def anchored_path_pixels(mask: np.ndarray, src_xy, dst_xy) -> np.ndarray:
    skel = _skeleton(mask)
    if skel.sum() < 2:
        return longest_path_pixels(mask)
    nodes, adj = _skeleton_graph(skel)
    nodes_xy = nodes[:, ::-1].astype(float)

    def nearest(pt):
        return int(np.linalg.norm(nodes_xy - np.asarray(pt, float), axis=1).argmin())

    src, dst = nearest(src_xy), nearest(dst_xy)
    _, _, parent = _bfs_farthest(adj, src, len(nodes))
    if parent[dst] == -1 and dst != src:
        return longest_path_pixels(mask)
    return nodes_xy[_trace_path(parent, dst)]


def _resample_pixels(xy: np.ndarray, n_out: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(cum[-1], 1e-9)
    want = np.linspace(0.0, total, n_out)
    x = np.interp(want, cum, xy[:, 0])
    y = np.interp(want, cum, xy[:, 1])
    return np.stack([x, y], -1)


def _smooth(xy: np.ndarray, win: int = 5) -> np.ndarray:
    n = xy.shape[0]
    if win < 3 or n < win:
        return xy
    half = win // 2
    sm = xy.copy()
    for c in range(2):
        padded = np.pad(xy[:, c], half, mode="edge")
        kernel = np.ones(win) / win
        sm[:, c] = np.convolve(padded, kernel, mode="valid")
    sm[0], sm[-1] = xy[0], xy[-1]
    return sm


def strand_centerline(
    mask: np.ndarray,
    n_nodes: int,
    hub_xy: np.ndarray,
    module_xy: np.ndarray,
    smooth_win: int = 5,
    reach_ends: bool = True,
    n_strips: int = 40,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], -1).astype(float)
    axis = np.asarray(module_xy, float) - np.asarray(hub_xy, float)
    length = np.linalg.norm(axis)
    if length < 1.0 or len(pts) < 4:
        return _resample_pixels(np.stack([hub_xy, module_xy]), n_nodes)
    u = axis / length

    t = (pts - hub_xy) @ u
    edges = np.linspace(t.min(), t.max(), n_strips + 1)
    centres = []
    for i in range(n_strips):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        if int(sel.sum()) >= 2:
            centres.append(pts[sel].mean(0))
    raw = np.array([hub_xy] + centres + [module_xy])
    raw = _smooth(raw, smooth_win)
    if reach_ends:
        raw[0] = hub_xy
        raw[-1] = module_xy
    return _resample_pixels(raw, n_nodes)


def strand_quality(centerline_px: np.ndarray, tf=None) -> dict:
    cl = centerline_px if tf is None else tf.to_mm(centerline_px)
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    arc = float(seg.sum())
    chord = float(np.linalg.norm(cl[-1] - cl[0]))
    u = (cl[-1] - cl[0]) / max(chord, 1e-9)
    prog = np.diff((cl - cl[0]) @ u)
    return {"arc": arc, "chord": chord,
            "slack": chord / arc if arc > 0 else 0.0,
            "monotonic": bool((prog > -1e-6).all())}


def assign_strands_to_modules(components, hub_xy, module_centres):
    peripheral = [m for m in module_centres
                  if np.linalg.norm(m - hub_xy) > 1.0]
    out = []
    for area, mask in components:
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], -1).astype(float)
        far = pts[np.linalg.norm(pts - hub_xy, axis=1).argmax()]
        j = int(np.linalg.norm(np.array(peripheral) - far, axis=1).argmin())
        out.append((mask, peripheral[j]))
    return out
