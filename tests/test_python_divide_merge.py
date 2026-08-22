"""Sanity checks for the Python polar quadtree divide-and-merge."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.divide_merge import divide_and_merge, rectilinear_mst


def _connected(tree, n_pins):
    nodes = tree["nodes"]
    adj = [[] for _ in nodes]
    for a, b in tree["edges"]:
        adj[a].append(b)
        adj[b].append(a)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return all(i in seen for i in range(n_pins))


def test_small_equals_direct():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.2)]
    direct = rectilinear_mst(pts)
    wrapped = divide_and_merge(pts, rectilinear_mst, capacity=30)
    assert wrapped["nodes"][: len(pts)] == list(pts)
    assert _connected(wrapped, len(pts))
    assert _connected(direct, len(pts))


def test_large_tree():
    rng_pts = [(0.0, 0.0)]
    x, y = 0.13, 0.97
    for i in range(1, 50):
        x = (x * 17 + 0.1) % 1.0
        y = (y * 13 + 0.3) % 1.0
        rng_pts.append((x * 1000.0, y * 1000.0))
    tree = divide_and_merge(rng_pts, rectilinear_mst, capacity=8)
    assert _connected(tree, len(rng_pts))
    assert len(tree["edges"]) >= len(rng_pts) - 1


if __name__ == "__main__":
    test_small_equals_direct()
    test_large_tree()
    print("ok")
