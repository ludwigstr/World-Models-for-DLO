
from __future__ import annotations

import heapq
from itertools import product

import cv2

import numpy as np

from .centerline import _resample_pixels, strand_quality
from .harness import N_MOVERS, corridor_cable, wire_mask
from .tracer import build_graph

W_PRIOR = 1.0
W_COVER = 6.0
W_OVERLAP = 3.0
PRIOR_SCALE = 60.0
COVER_TOL = 14.0
FAN_FRAC = 0.45
MAX_FAR_OVERLAP = 0.5
CARRY_COST = 2.5

W_LENGTH = 5.0
LEN_TOL = 0.12


def _path_overlap(p1, p2, tol=6.0, stride=3, hub=None, min_r=0.0):
    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
        return 0.0
    a = np.asarray(p1, float)[::stride]
    b = np.asarray(p2, float)[::stride]
    if hub is not None and min_r > 0:
        a = a[np.linalg.norm(a - hub, axis=1) > min_r]
        b = b[np.linalg.norm(b - hub, axis=1) > min_r]
        if len(a) < 2 or len(b) < 2:
            return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return max(float((d.min(axis=1) < tol).mean()),
               float((d.min(axis=0) < tol).mean()))


def _coverage(centerline, dist, tol=COVER_TOL, samples=200):
    if centerline is None or len(centerline) < 2:
        return 0.0
    P = _resample_pixels(np.asarray(centerline, float), samples)
    h, w = dist.shape
    x = np.clip(P[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(P[:, 1].round().astype(int), 0, h - 1)
    return float((dist[y, x] <= tol).mean())


def _shape_distance(poly, prior, n=16):
    if poly is None or prior is None or len(poly) < 2 or len(prior) < 2:
        return None
    a = _resample_pixels(np.asarray(poly, float), n)
    b = _resample_pixels(np.asarray(prior, float), n)
    return float(np.linalg.norm(a - b, axis=1).mean())


def _search(graph, start_px, goal_px, seg_penalty=None, straightness=0.2,
            max_gap=60.0, start_radius=170.0, goal_radius=200.0,
            turn_weight=1.0, gap_weight=0.5):
    segs, jcent, inc = graph
    if not segs:
        return None, np.inf, set()
    pen = seg_penalty or {}
    start_px = np.asarray(start_px, float)
    goal_px = np.asarray(goal_px, float)
    to_goal = goal_px - start_px
    ng = np.linalg.norm(to_goal)
    to_goal = to_goal / ng if ng > 1e-9 else np.zeros(2)

    def exit_of(si, e):
        xe = 0 if e == -1 else -1
        return xe, segs[si].endpoint(xe), segs[si].direction_at(xe)

    def successors(si, e):
        _, xpt, xdir = exit_of(si, e)
        out, seen = [], set()
        if len(jcent):
            dj = np.linalg.norm(jcent - xpt, axis=1)
            j = int(dj.argmin())
            if dj[j] <= 15:
                for (s2, e2) in inc.get(j, []):
                    if s2 == si:
                        continue
                    align = float(-np.dot(xdir, segs[s2].direction_at(e2)))
                    if align >= straightness:
                        out.append((s2, e2, turn_weight * (1.0 - align)))
                        seen.add(s2)
        for s2, s2o in enumerate(segs):
            if s2 == si or s2 in seen:
                continue
            for e2 in (0, -1):
                p2 = s2o.endpoint(e2)
                gap = float(np.linalg.norm(p2 - xpt))
                if gap > max_gap or gap < 1e-6:
                    continue
                if float(np.dot((p2 - xpt) / gap, xdir)) < 0.5:
                    continue
                align = float(-np.dot(xdir, s2o.direction_at(e2)))
                if align < straightness:
                    continue
                out.append((s2, e2, turn_weight * (1.0 - align)
                            + gap_weight * gap / max_gap))
        return out

    pq, dist, prev = [], {}, {}
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - start_px))
            if d > start_radius:
                continue
            heading = -s.direction_at(e)
            c = (d / start_radius + 0.5 * (1.0 - float(np.dot(heading, to_goal)))
                 + pen.get(si, 0.0))
            st = (si, e)
            if c < dist.get(st, np.inf):
                dist[st] = c
                prev[st] = None
                heapq.heappush(pq, (c, st))
    if not pq:
        return None, np.inf, set()

    goal_state, best_goal = None, np.inf
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, np.inf):
            continue
        si, e = st
        _, xpt, _ = exit_of(si, e)
        if float(np.linalg.norm(xpt - goal_px)) <= goal_radius and c < best_goal:
            best_goal, goal_state = c, st
            continue
        if c > best_goal:
            continue
        for (s2, e2, w) in successors(si, e):
            nc = c + w + pen.get(s2, 0.0)
            st2 = (s2, e2)
            if nc < dist.get(st2, np.inf):
                dist[st2] = nc
                prev[st2] = st
                heapq.heappush(pq, (nc, st2))

    if goal_state is None:
        return None, np.inf, set()
    chain, used, st = [], set(), goal_state
    while st is not None:
        si, e = st
        used.add(si)
        pts = segs[si].pts
        chain.append(pts if e == 0 else pts[::-1])
        st = prev.get(st)
    poly = np.concatenate(chain[::-1], axis=0)
    if np.linalg.norm(poly[0] - start_px) > np.linalg.norm(poly[-1] - start_px):
        poly = poly[::-1]
    return poly, float(best_goal), used


def _grow_from_hub(graph, hub_px, start_radius=200.0, straightness=0.2,
                   max_gap=180.0, turn_weight=1.0, gap_weight=0.5,
                   near_gap=30.0):
    segs, jcent, inc = graph
    hub_px = np.asarray(hub_px, float)
    if not segs:
        return {}, {}, {}

    def exit_of(si, e):
        xe = 0 if e == -1 else -1
        return segs[si].endpoint(xe), segs[si].direction_at(xe)

    def successors(si, e):
        xpt, xdir = exit_of(si, e)
        out, seen = [], set()
        if len(jcent):
            dj = np.linalg.norm(jcent - xpt, axis=1)
            j = int(dj.argmin())
            if dj[j] <= 15:
                for (s2, e2) in inc.get(j, []):
                    if s2 == si:
                        continue
                    align = float(-np.dot(xdir, segs[s2].direction_at(e2)))
                    if align >= straightness:
                        out.append((s2, e2, turn_weight * (1.0 - align)))
                        seen.add(s2)
        for s2, s2o in enumerate(segs):
            if s2 == si or s2 in seen:
                continue
            for e2 in (0, -1):
                p2 = s2o.endpoint(e2)
                gap = float(np.linalg.norm(p2 - xpt))
                if gap > max_gap or gap < 1e-6:
                    continue
                fwd = float(np.dot((p2 - xpt) / gap, xdir))
                if fwd < (-0.2 if gap < near_gap else 0.5):
                    continue
                align = float(-np.dot(xdir, s2o.direction_at(e2)))
                if align < straightness:
                    continue
                out.append((s2, e2, turn_weight * (1.0 - align)
                            + gap_weight * gap / max_gap))
        return out

    pq, dist, prev = [], {}, {}
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - hub_px))
            if d > start_radius:
                continue
            outward = s.endpoint(0 if e == -1 else -1) - s.endpoint(e)
            no = np.linalg.norm(outward)
            if no < 1e-9:
                continue
            radial = (s.endpoint(e) - hub_px)
            nr = np.linalg.norm(radial)
            align = float(np.dot(outward / no, radial / nr)) if nr > 1e-9 else 1.0
            if align < 0.0:
                continue
            c = d / start_radius + 0.5 * (1.0 - align)
            st = (si, e)
            if c < dist.get(st, np.inf):
                dist[st] = c
                prev[st] = None
                heapq.heappush(pq, (c, st))

    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, np.inf):
            continue
        for (s2, e2, w) in successors(*st):
            nc = c + w
            st2 = (s2, e2)
            if nc < dist.get(st2, np.inf):
                dist[st2] = nc
                prev[st2] = st
                heapq.heappush(pq, (nc, st2))

    xpts = {st: exit_of(*st)[0] for st in dist}
    return dist, prev, xpts


def hub_candidates(graph, hub_px, mover_px, dist, prev, xpts, k=6,
                   radius=230.0):
    segs = graph[0]
    mover_px = np.asarray(mover_px, float)
    by_terminal = {}
    for st, c in dist.items():
        d = float(np.linalg.norm(xpts[st] - mover_px))
        if d > radius:
            continue
        score = c + d / radius
        cur = by_terminal.get(st[0])
        if cur is None or score < cur[0]:
            by_terminal[st[0]] = (score, st)

    out = []
    for score, st in sorted(by_terminal.values())[:k]:
        chain, used, cur = [], set(), st
        while cur is not None:
            si, e = cur
            used.add(si)
            pts = segs[si].pts
            chain.append(pts if e == 0 else pts[::-1])
            cur = prev.get(cur)
        poly = np.concatenate(chain[::-1], axis=0)
        out.append({"poly": poly, "cost": score, "segs": used})
    return out


def candidates(graph, start_px, goal_px, k=4, penalty=1.5, **kw):
    out, pen = [], {}
    for _ in range(k):
        poly, cost, used = _search(graph, start_px, goal_px, seg_penalty=pen, **kw)
        if poly is None or len(poly) < 20:
            break
        out.append({"poly": poly, "cost": cost, "segs": used})
        for si in used:
            pen[si] = pen.get(si, 0.0) + penalty
    return out


def _arc_length(poly):
    return float(np.linalg.norm(np.diff(np.asarray(poly, float), axis=0),
                                axis=1).sum())


def solve_frame(graph, hub_px, movers_px, dist=None, priors=None, k=6,
                n_nodes=16, lengths=None):
    n = len(movers_px)
    gdist, gprev, gxpts = _grow_from_hub(graph, hub_px)

    fin, base, kind = [], [], []
    for i in range(n):
        m = movers_px[i]
        frow, brow, krow = [], [], []
        cands = hub_candidates(graph, hub_px, m, gdist, gprev, gxpts, k=k)
        if not cands:
            cands = candidates(graph, m, hub_px, k=k)
        for c in cands:
            poly = c["poly"]
            if np.linalg.norm(poly[0] - hub_px) > np.linalg.norm(poly[-1] - hub_px):
                poly = poly[::-1]
            cl = _resample_pixels(
                np.vstack([hub_px[None, :], poly, m[None, :]]), n_nodes)
            s = c["cost"]
            if dist is not None:
                s += W_COVER * (1.0 - _coverage(cl, dist))
            if priors is not None and priors[i] is not None:
                d = _shape_distance(cl, priors[i])
                if d is not None:
                    s += W_PRIOR * d / PRIOR_SCALE
            if lengths is not None and lengths[i]:
                rel = abs(_arc_length(cl) - lengths[i]) / lengths[i]
                s += W_LENGTH * max(0.0, rel - LEN_TOL)
            frow.append(cl); brow.append(s); krow.append("trace")
        if priors is not None and priors[i] is not None:
            cl = np.asarray(priors[i], float).copy()
            cl[0], cl[-1] = hub_px, m
            s = CARRY_COST
            if dist is not None:
                s += W_COVER * (1.0 - _coverage(cl, dist))
            frow.append(cl); brow.append(s); krow.append("carry")
        fin.append(frow); base.append(brow); kind.append(krow)


    ov = {}
    for a in range(n):
        ra = FAN_FRAC * float(np.linalg.norm(movers_px[a] - hub_px))
        for b in range(a + 1, n):
            rb = FAN_FRAC * float(np.linalg.norm(movers_px[b] - hub_px))
            r = min(ra, rb)
            for ia, ca in enumerate(fin[a]):
                for ib, cb in enumerate(fin[b]):
                    ov[(a, ia, b, ib)] = _path_overlap(
                        ca, cb, tol=10.0, stride=1, hub=hub_px, min_r=r)

    choices = [list(range(len(fin[i]))) + [None] for i in range(n)]
    best, best_score = None, np.inf
    for combo in product(*choices):
        score, ok = 0.0, True
        for i, ci in enumerate(combo):
            if ci is None:
                score += 5.0
                continue
            score += base[i][ci]
        for a in range(n):
            for b in range(a + 1, n):
                if combo[a] is None or combo[b] is None:
                    continue
                f = ov[(a, combo[a], b, combo[b])]
                if f > MAX_FAR_OVERLAP:
                    ok = False
                    break
                score += W_OVERLAP * f
            if not ok:
                break
        if ok and score < best_score:
            best_score, best = score, combo

    if best is None:
        return [None] * n, [None] * n, np.inf
    return ([fin[i][ci] if ci is not None else None for i, ci in enumerate(best)],
            [kind[i][ci] if ci is not None else None for i, ci in enumerate(best)],
            best_score)


class HarnessTracker:

    def __init__(self, tf, clip=None, n_nodes=16, conn_radius=135, k=6,
                 max_age=4, jump_mm=70.0, solo_seed=True, no_guard=False):
        self.tf = tf
        self.clip = clip
        self.n_nodes = n_nodes
        self.conn_radius = conn_radius
        self.k = k
        self.max_age = max_age
        self.jump_mm = jump_mm
        self.solo_seed = solo_seed
        self.no_guard = no_guard

    def _graph(self, frame, idx):
        w, _ = wire_mask(frame, self.tf, self.conn_radius,
                         clip=self.clip, orig_idx=idx)
        dist = cv2.distanceTransform((~w).astype(np.uint8), cv2.DIST_L2, 3)
        return w, build_graph(w), dist

    def run(self, cap, read_fn, tracks, indices):
        indices = sorted(indices)
        prepared = {}
        for idx in indices:
            fr = read_fn(cap, idx)
            if fr is None or not np.isfinite(tracks[idx]).all():
                continue
            w, gr, dist = self._graph(fr, idx)
            prepared[idx] = {"wires": w, "graph": gr, "dist": dist,
                             "mods": tracks[idx]}
        if not prepared:
            return {}

        solo, obs = {}, [[] for _ in range(N_MOVERS)]
        for idx, p in prepared.items():
            routes, kinds, score = solve_frame(
                p["graph"], p["mods"][0], p["mods"][1:], dist=p["dist"],
                k=self.k, n_nodes=self.n_nodes)
            n_found = sum(1 for r in routes if r is not None)
            solo[idx] = (routes, score, n_found)
            for j, r in enumerate(routes):
                if r is not None and j < N_MOVERS:
                    obs[j].append((idx, _coverage(r, p["dist"]), _arc_length(r)))
        anchor = min(solo, key=lambda i: (-solo[i][2], solo[i][1]))

        LW = 8

        def local_lengths(center):
            out = [None] * N_MOVERS
            for j, o in enumerate(obs):
                near = [(c, L) for (fi, c, L) in o if abs(fi - center) <= LW]
                if len(near) < 3:
                    near = [(c, L) for (_, c, L) in o]
                if len(near) < 3:
                    continue
                cov_cut = np.median([c for c, _ in near])
                good = [L for c, L in near if c >= cov_cut]
                if good:
                    out[j] = float(np.median(good))
            return out

        pa = prepared[anchor]
        a_routes, _, _ = solve_frame(
            pa["graph"], pa["mods"][0], pa["mods"][1:], dist=pa["dist"],
            k=self.k, n_nodes=self.n_nodes, lengths=local_lengths(anchor))
        result = {anchor: [(r, 0 if r is not None else None) for r in a_routes]}
        if getattr(self, "solo_seed", False):
            result = {idx: [(r, 0 if r is not None else None)
                            for r in solo[idx][0]] for idx in prepared}
        else:
          order = ([i for i in indices if i > anchor],
                   [i for i in indices if i < anchor][::-1])
          for seq in order:
            prev_idx = anchor
            for idx in seq:
                if idx not in prepared:
                    continue
                p = prepared[idx]
                prev = result.get(prev_idx)
                priors, prev_age = None, [None] * N_MOVERS
                if prev is not None:
                    shift = p["mods"] - prepared[prev_idx]["mods"]
                    priors = []
                    for j, (r, age) in enumerate(prev):
                        stale = age is None or age >= self.max_age
                        priors.append(None if (r is None or stale)
                                      else r + shift[j + 1])
                        prev_age[j] = age
                routes, kinds, _ = solve_frame(
                    p["graph"], p["mods"][0], p["mods"][1:], dist=p["dist"],
                    priors=priors, k=self.k, n_nodes=self.n_nodes,
                    lengths=local_lengths(idx))

                merged = []
                for j, r in enumerate(routes):
                    if r is None:
                        merged.append((None, None))
                    elif kinds[j] == "carry":
                        merged.append((r, (prev_age[j] or 0) + 1))
                    else:
                        merged.append((r, 0))
                result[idx] = merged
                prev_idx = idx

        _run_guard = not getattr(self, "no_guard", False)

        def _dev(ref, cur):
            shift = prepared[cur]["mods"][j + 1] - prepared[ref]["mods"][j + 1]
            d = (self.tf.to_mm(result[cur][j][0])
                 - self.tf.to_mm(result[ref][j][0] + shift))
            mv = float(np.linalg.norm(
                self.tf.to_mm(prepared[cur]["mods"][j + 1][None])[0]
                - self.tf.to_mm(prepared[ref]["mods"][j + 1][None])[0]))
            return float(np.mean(np.linalg.norm(d, axis=1))), mv

        for j in range(N_MOVERS if _run_guard else 0):
            allf = sorted(result)
            have = [i for i in allf if result[i][j][0] is not None]
            if len(have) < 3:
                continue

            segs = [[have[0]]]
            for a, b in zip(have, have[1:]):
                ok = b == a + 1
                if ok:
                    dev, mv = _dev(a, b)
                    ok = dev <= max(self.jump_mm, 2.5 * mv)
                segs[-1].append(b) if ok else segs.append([b])
            anchor = max(segs, key=len)[len(max(segs, key=len)) // 2]

            def _carry(cur, ref):
                r = result[ref][j][0]
                h0, m0 = r[0], r[-1]
                d0 = m0 - h0
                L0 = float(np.hypot(d0[0], d0[1]))
                h1 = prepared[cur]["mods"][0]
                m1 = prepared[cur]["mods"][j + 1]
                if L0 < 1e-6:
                    result[cur][j] = (r + (m1 - m0), result[cur][j][1])
                    return
                a0 = np.arctan2(d0[1], d0[0])
                q = (r - h0) @ np.array([[np.cos(-a0), -np.sin(-a0)],
                                         [np.sin(-a0), np.cos(-a0)]]).T / L0
                d1 = m1 - h1
                L1 = float(np.hypot(d1[0], d1[1]))
                a1 = np.arctan2(d1[1], d1[0])
                out = (q * L1) @ np.array([[np.cos(a1), -np.sin(a1)],
                                           [np.sin(a1), np.cos(a1)]]).T + h1
                result[cur][j] = (out, result[cur][j][1])

            def _good(ref, cur):
                if result[cur][j][0] is None:
                    return False
                dev, mv = _dev(ref, cur)
                return dev <= max(self.jump_mm, 2.5 * mv)

            ref = anchor
            for cur in [f for f in allf if f > anchor]:
                if not _good(ref, cur):
                    _carry(cur, ref)
                ref = cur
            ref = anchor
            for cur in [f for f in allf if f < anchor][::-1]:
                if not _good(ref, cur):
                    _carry(cur, ref)
                ref = cur

        out = {}
        for idx, routes in result.items():
            p = prepared[idx]
            hub, mods = p["mods"][0], p["mods"]
            cables = []
            for j, (poly, age) in enumerate(routes):
                m = mods[j + 1]
                method = "tracker" if age == 0 else f"fortgeschrieben({age})"
                cl = None
                if poly is not None:
                    cl = np.asarray(poly, float).copy()
                    cl[0], cl[-1] = hub, m
                if cl is None:
                    cc = corridor_cable(p["wires"], hub, m)
                    if cc is not None:
                        cl, method = cc, "korridor"
                if cl is None:
                    cl = _resample_pixels(np.vstack([hub[None, :], m[None, :]]),
                                          self.n_nodes)
                    method = "gerade (Platzhalter)"
                q = strand_quality(cl, self.tf)
                cables.append({"shape_mm": self.tf.to_mm(cl) / 1000.0, "px": cl,
                               "mover_px": np.asarray(m, float),
                               "hub_px": np.asarray(hub, float),
                               "method": method, "slack": q["slack"]})
            out[idx] = cables
        return out
