"""Polar quadtree divide-and-merge (Section III of the ICCAD 2023 paper).

The local solver is a callable: solver(points) -> Tree-like dict
    points[0] is the source of this subproblem
    return {"nodes": [(x,y), ...], "edges": [(i,j), ...]}
      nodes[:len(points)] must match the input points in order
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

EPS = 1e-9
PI = math.pi
TWO_PI = 2.0 * math.pi
MAX_DEPTH = 16
MAX_LAYERS = 24
MODERATE_DEGREE = 32

Point = Tuple[float, float]
Edge = Tuple[int, int]
Solver = Callable[[List[Point]], dict]


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _manh(a: Point, b: Point) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _same(a: Point, b: Point) -> bool:
    return abs(a[0] - b[0]) <= EPS and abs(a[1] - b[1]) <= EPS


def _polar_r(p: Point) -> float:
    return math.hypot(p[0], p[1])


def _polar_theta(p: Point) -> float:
    if _polar_r(p) <= EPS:
        return 0.0
    t = math.atan2(p[1], p[0])
    if t < 0.0:
        t += TWO_PI
    return t


def _wrap(t: float) -> float:
    t = math.fmod(t, TWO_PI)
    if t < 0.0:
        t += TWO_PI
    return t


def _sector(theta: float, n_sectors: int) -> int:
    theta = _wrap(theta)
    idx = int(theta / TWO_PI * n_sectors)
    return min(max(idx, 0), n_sectors - 1)


def _polar_xy(r: float, theta: float) -> Point:
    return (r * math.cos(theta), r * math.sin(theta))


@dataclass
class Block:
    id: int = -1
    layer: int = -1
    index: int = 0
    r_inner: float = 0.0
    r_outer: float = 0.0
    theta_min: float = 0.0
    theta_max: float = 0.0
    pin_ids: List[int] = field(default_factory=list)
    parent_id: int = -1
    left_id: int = -1
    right_id: int = -1
    left_child_id: int = -1
    right_child_id: int = -1
    merge_to: int = -1
    local_source: int = -1

    def nonempty(self) -> bool:
        return bool(self.pin_ids)


@dataclass
class PolarQuadtree:
    blocks: List[Block] = field(default_factory=list)
    layers: List[List[int]] = field(default_factory=list)
    x0_id: int = 0
    rho_max: float = 0.0
    default_search_bound: float = 0.0


def _center(b: Block) -> Point:
    if b.layer < 0:
        return (0.0, 0.0)
    r = 0.5 * (b.r_inner + b.r_outer)
    t = 0.5 * (b.theta_min + b.theta_max)
    return _polar_xy(r, t)


def _add_block(qt: PolarQuadtree) -> int:
    b = Block(id=len(qt.blocks))
    qt.blocks.append(b)
    return b.id


def _link_layer(qt: PolarQuadtree, layer: int) -> None:
    ids = qt.layers[layer]
    n = len(ids)
    if n <= 0:
        return
    for i, bid in enumerate(ids):
        b = qt.blocks[bid]
        b.left_id = ids[(i - 1 + n) % n]
        b.right_id = ids[(i + 1) % n]
        if layer == 0:
            b.parent_id = qt.x0_id
        else:
            b.parent_id = qt.layers[layer - 1][i // 2]
    if layer == 0:
        qt.blocks[qt.x0_id].left_child_id = ids[0]
        if n >= 2:
            qt.blocks[qt.x0_id].right_child_id = ids[1]
    else:
        parents = qt.layers[layer - 1]
        for i, bid in enumerate(ids):
            parent = qt.blocks[parents[i // 2]]
            if i % 2 == 0:
                parent.left_child_id = bid
            else:
                parent.right_child_id = bid


def _split_dense(qt: PolarQuadtree, bid: int, capacity: int, depth: int,
                 translated: Sequence[Point]) -> None:
    b = qt.blocks[bid]
    if len(b.pin_ids) <= capacity or depth >= MAX_DEPTH:
        return
    pins = sorted(b.pin_ids, key=lambda pid: _polar_theta(translated[pid]))
    mid = len(pins) // 2
    if mid == 0 or mid == len(pins):
        return
    left_id = _add_block(qt)
    right_id = _add_block(qt)
    parent = qt.blocks[bid]
    left, right = qt.blocks[left_id], qt.blocks[right_id]
    for child in (left, right):
        child.layer = parent.layer
        child.index = parent.index
        child.r_inner = parent.r_inner
        child.r_outer = parent.r_outer
    left.theta_min = parent.theta_min
    left.theta_max = _polar_theta(translated[pins[mid]])
    right.theta_min = left.theta_max
    right.theta_max = parent.theta_max
    left.parent_id = right.parent_id = bid
    left.left_id, left.right_id = parent.left_id, right_id
    right.left_id, right.right_id = left_id, parent.right_id
    left.pin_ids = pins[:mid]
    right.pin_ids = pins[mid:]
    parent.pin_ids = []
    parent.left_child_id, parent.right_child_id = left_id, right_id
    _split_dense(qt, left_id, capacity, depth + 1, translated)
    _split_dense(qt, right_id, capacity, depth + 1, translated)


def build_polar_quadtree(translated: Sequence[Point], capacity: int) -> PolarQuadtree:
    qt = PolarQuadtree()
    n = len(translated)
    cap = max(capacity, 1)
    x0 = _add_block(qt)
    qt.x0_id = x0
    qt.blocks[x0].layer = -1
    qt.blocks[x0].pin_ids = [0]

    remaining: List[int] = []
    qt.rho_max = 0.0
    for i in range(1, n):
        r = _polar_r(translated[i])
        qt.rho_max = max(qt.rho_max, r)
        if r <= EPS:
            qt.blocks[x0].pin_ids.append(i)
        else:
            remaining.append(i)
    remaining.sort(key=lambda i: (_polar_r(translated[i]), i))

    r_inner = 0.0
    max_thickness = 0.0
    layer = 0
    cursor = 0
    while cursor < len(remaining) and layer < MAX_LAYERS:
        n_sectors = 4 << layer
        counts = [0] * n_sectors
        assigned: List[List[int]] = [[] for _ in range(n_sectors)]
        r_outer = r_inner
        while cursor < len(remaining):
            pid = remaining[cursor]
            s = _sector(_polar_theta(translated[pid]), n_sectors)
            if counts[s] >= cap:
                break
            assigned[s].append(pid)
            counts[s] += 1
            r_outer = max(r_outer, _polar_r(translated[pid]))
            cursor += 1
        if r_outer <= r_inner and cursor < len(remaining):
            r_outer = _polar_r(translated[remaining[cursor]])
        qt.layers.append([])
        for s in range(n_sectors):
            bid = _add_block(qt)
            b = qt.blocks[bid]
            b.layer, b.index = layer, s
            b.r_inner, b.r_outer = r_inner, r_outer
            b.theta_min = TWO_PI * s / n_sectors
            b.theta_max = TWO_PI * (s + 1) / n_sectors
            b.pin_ids = assigned[s]
            qt.layers[-1].append(bid)
        _link_layer(qt, layer)
        max_thickness = max(max_thickness, r_outer - r_inner)
        r_inner = r_outer
        layer += 1

    if cursor < len(remaining):
        last = qt.layers[-1]
        n_sectors = len(last)
        for pid in remaining[cursor:]:
            s = _sector(_polar_theta(translated[pid]), n_sectors)
            qt.blocks[last[s]].pin_ids.append(pid)

    n_blocks = len(qt.blocks)
    for bid in range(n_blocks):
        if bid == qt.x0_id:
            continue
        _split_dense(qt, bid, cap, 0, translated)

    qt.default_search_bound = 2.0 * max(max_thickness, qt.rho_max * 0.25 if qt.rho_max > 0 else 1.0)
    return qt


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def unite(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.p[a] = b
        return True


def _apply_nei(qt: PolarQuadtree, bid: int, times: int, left: bool) -> int:
    y = bid
    for _ in range(times):
        nxt = qt.blocks[y].left_id if left else qt.blocks[y].right_id
        if nxt < 0:
            return -1
        y = nxt
    return y


def _zigzag(qt: PolarQuadtree, x: int, bound: float) -> int:
    best, best_d = -1, float("inf")
    for k in range(1, 65):
        left = _apply_nei(qt, x, k, True)
        right = _apply_nei(qt, x, k, False)
        cands = [
            left,
            qt.blocks[left].parent_id if left >= 0 else -1,
            right,
            qt.blocks[right].parent_id if right >= 0 else -1,
        ]
        progressed = False
        for c in cands:
            if c < 0 or c == x:
                continue
            progressed = True
            d = _manh(_center(qt.blocks[x]), _center(qt.blocks[c]))
            if d > bound + EPS:
                continue
            if qt.blocks[c].nonempty() and d < best_d:
                best_d, best = d, c
        if best >= 0:
            return best
        if not progressed:
            break
    return best


def compute_merge_topology(qt: PolarQuadtree, search_bound: float) -> None:
    n = len(qt.blocks)
    for b in qt.blocks:
        b.merge_to = -1
    uf = _UF(n)
    nonempty = [i for i, b in enumerate(qt.blocks) if b.nonempty()]
    nonempty.sort(key=lambda i: (-qt.blocks[i].layer, i))
    bound = search_bound if search_bound > 0 else max(qt.default_search_bound, 1.0)

    for x in nonempty:
        if x == qt.x0_id:
            continue
        cur = x
        target = qt.x0_id
        for _ in range(64):
            par = qt.blocks[cur].parent_id
            if par >= 0 and qt.blocks[par].nonempty():
                target = par
                break
            found = _zigzag(qt, cur, bound)
            if found >= 0:
                target = found
                break
            if par < 0 or par == qt.x0_id:
                target = qt.x0_id
                break
            cur = par
        if target < 0 or not uf.unite(x, target):
            target = qt.x0_id
            uf.unite(x, qt.x0_id)
        qt.blocks[x].merge_to = target

    root = uf.find(qt.x0_id)
    for x in nonempty:
        if uf.find(x) != root:
            qt.blocks[x].merge_to = qt.x0_id
            uf.unite(x, qt.x0_id)


def _nearest_pin(target: Point, pin_ids: Sequence[int], pts: Sequence[Point]) -> int:
    best, best_d = -1, float("inf")
    for pid in pin_ids:
        d = _manh(pts[pid], target)
        if d < best_d:
            best_d, best = d, pid
    return best


def _nearest_pair(a: Sequence[int], b: Sequence[int], pts: Sequence[Point]):
    best_i, best_j, best_d = -1, -1, float("inf")
    for i in a:
        for j in b:
            d = _manh(pts[i], pts[j])
            if d < best_d:
                best_d, best_i, best_j = d, i, j
    return best_i, best_j


def assign_sources_and_links(qt: PolarQuadtree, translated: Sequence[Point], origin: Point):
    links = []
    for b in qt.blocks:
        b.local_source = -1
    qt.blocks[qt.x0_id].local_source = 0
    children: List[List[int]] = [[] for _ in qt.blocks]
    for i, b in enumerate(qt.blocks):
        if b.merge_to >= 0 and b.nonempty():
            children[b.merge_to].append(i)

    def set_src(bid: int, pin: int) -> None:
        if bid != qt.x0_id and pin >= 0:
            qt.blocks[bid].local_source = pin

    for parent_id, ch in enumerate(children):
        if not ch:
            continue
        parent = qt.blocks[parent_id]
        used = [False] * len(ch)
        for i, ca in enumerate(ch):
            if used[i]:
                continue
            mate = None
            for j in range(i + 1, len(ch)):
                if used[j]:
                    continue
                a, b = qt.blocks[ch[i]], qt.blocks[ch[j]]
                if a.parent_id == parent_id and b.parent_id == parent_id and (
                    (parent.left_child_id == a.id and parent.right_child_id == b.id)
                    or (parent.left_child_id == b.id and parent.right_child_id == a.id)
                ):
                    mate = j
                    break
            if mate is None:
                continue
            used[i] = used[mate] = True
            a, b = qt.blocks[ch[i]], qt.blocks[ch[mate]]
            t = a.theta_max
            p = _polar_xy(parent.r_outer, _wrap(t))
            pu = _nearest_pin(p, a.pin_ids, translated)
            pv = _nearest_pin(p, b.pin_ids, translated)
            pw = _nearest_pin(p, parent.pin_ids, translated)
            set_src(ch[i], pu)
            set_src(ch[mate], pv)
            pins = [x for x in (pu, pv, pw) if x >= 0]
            if len(pins) >= 2:
                links.append({
                    "pins": pins,
                    "has_steiner": True,
                    "steiner": (p[0] + origin[0], p[1] + origin[1]),
                })
        for i, cid in enumerate(ch):
            if used[i]:
                continue
            child = qt.blocks[cid]
            ps, pe = _nearest_pair(parent.pin_ids, child.pin_ids, translated)
            set_src(cid, pe)
            pins = [x for x in (ps, pe) if x >= 0]
            if len(set(pins)) >= 2:
                links.append({"pins": pins, "has_steiner": False, "steiner": (0.0, 0.0)})

    for b in qt.blocks:
        if not b.nonempty() or b.local_source >= 0:
            continue
        b.local_source = 0 if b.id == qt.x0_id else _nearest_pin((0.0, 0.0), b.pin_ids, translated)
    return links


def _append_edge(edges: List[Edge], a: int, b: int) -> None:
    if a < 0 or b < 0 or a == b:
        return
    u, v = (a, b) if a < b else (b, a)
    if (u, v) not in edges:
        edges.append((u, v))


def _add_steiner(nodes: List[Point], p: Point) -> int:
    for i, q in enumerate(nodes):
        if _same(q, p):
            return i
    nodes.append(p)
    return len(nodes) - 1


def _extract_tree(nodes: List[Point], edges: List[Edge], n_original: int) -> dict:
    n = len(nodes)
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n and a != b:
            adj[a].append(b)
            adj[b].append(a)
    parent = [-2] * n
    parent[0] = -1
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if parent[v] != -2:
                continue
            parent[v] = u
            stack.append(v)
    tree_edges: List[Edge] = []
    keep = [False] * n
    for i in range(n_original):
        keep[i] = True
        if parent[i] == -2 and i != 0:
            _append_edge(tree_edges, 0, i)
            parent[i] = 0
    for i in range(1, n):
        if parent[i] >= 0:
            _append_edge(tree_edges, parent[i], i)
            u = i
            while u > 0 and not keep[u]:
                keep[u] = True
                u = parent[u]
            keep[u] = True
    remap = [-1] * n
    out_nodes: List[Point] = []
    for i in range(n):
        if not keep[i]:
            continue
        remap[i] = len(out_nodes)
        out_nodes.append(nodes[i])
    out_edges: List[Edge] = []
    for a, b in tree_edges:
        if remap[a] >= 0 and remap[b] >= 0:
            _append_edge(out_edges, remap[a], remap[b])
    return {"nodes": out_nodes, "edges": out_edges}


def divide_and_merge(
    points: Sequence[Point],
    solver: Solver,
    capacity: int = 30,
    search_bound: float = -1.0,
) -> dict:
    """points[0] is the global source. solver(local) with local[0] as source."""
    n = len(points)
    if n == 0:
        return {"nodes": [], "edges": []}
    if n <= max(MODERATE_DEGREE, capacity + 1):
        tree = solver(list(points))
        if len(tree["nodes"]) < n:
            tree = {"nodes": list(points), "edges": tree.get("edges", [])}
        return tree

    origin = points[0]
    translated = [(p[0] - origin[0], p[1] - origin[1]) for p in points]
    qt = build_polar_quadtree(translated, capacity)
    bound = qt.default_search_bound if search_bound < 0 else search_bound
    compute_merge_topology(qt, bound)
    links = assign_sources_and_links(qt, translated, origin)

    global_nodes = list(points)
    global_edges: List[Edge] = []

    for b in qt.blocks:
        if not b.nonempty() or b.local_source < 0:
            continue
        local = [points[b.local_source]]
        local_to_orig = [b.local_source]
        for pid in b.pin_ids:
            if pid == b.local_source:
                continue
            local.append(points[pid])
            local_to_orig.append(pid)
        if len(local) <= 1:
            continue
        sub = solver(local)
        sub_nodes = sub.get("nodes", local)
        node_map = list(local_to_orig)
        for i in range(len(local), len(sub_nodes)):
            node_map.append(_add_steiner(global_nodes, tuple(sub_nodes[i])))
        for a, c in sub.get("edges", []):
            if 0 <= a < len(node_map) and 0 <= c < len(node_map):
                _append_edge(global_edges, node_map[a], node_map[c])

    for link in links:
        pins = link["pins"]
        if link["has_steiner"]:
            sid = _add_steiner(global_nodes, tuple(link["steiner"]))
            for pid in pins:
                _append_edge(global_edges, pid, sid)
        elif len(pins) >= 2:
            _append_edge(global_edges, pins[0], pins[1])

    return _extract_tree(global_nodes, global_edges, n)


def rectilinear_mst(points: Sequence[Point]) -> dict:
    pts = list(points)
    n = len(pts)
    nodes = pts
    edges: List[Edge] = []
    if n <= 1:
        return {"nodes": nodes, "edges": edges}
    used = [False] * n
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[0] = 0.0
    for _ in range(n):
        u, best = -1, float("inf")
        for i in range(n):
            if not used[i] and dist[i] < best:
                best, u = dist[i], i
        if u < 0:
            break
        used[u] = True
        if prev[u] >= 0:
            edges.append((prev[u], u))
        for v in range(n):
            if used[v]:
                continue
            d = _manh(pts[u], pts[v])
            if d < dist[v]:
                dist[v] = d
                prev[v] = u
    return {"nodes": nodes, "edges": edges}
