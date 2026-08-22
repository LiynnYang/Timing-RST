"""Wrap the Timing-RST Actor as a divide-and-merge local solver.

Convention: points[0] is the source (matches Actor: visited[:,0]=1).
Nets smaller than the trained degree are right-padded via Actor.pad_len.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from models.model_sorted import Actor

Point = Tuple[float, float]


def _normalize(points: Sequence[Point]) -> List[Point]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    return [((p[0] - xmin) / dx, (p[1] - ymin) / dy) for p in points]


def _adj_to_edges(adj: np.ndarray, real_n: int) -> List[Tuple[int, int]]:
    edges = []
    seen = set()
    for i in range(real_n):
        for j in range(real_n):
            if i == j or adj[i, j] < 0.5:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
    return edges


class TimingRSTSolver:
    """Callable solver for utils.divide_merge.divide_and_merge."""

    def __init__(
        self,
        checkpoint: str,
        degree: int = 30,
        device: str = "cpu",
        transform: int = 1,
    ):
        self.degree = degree
        self.device = torch.device(device)
        self.transform = transform
        self.actor = Actor()
        self.actor.to(self.device)
        ckp = torch.load(checkpoint, map_location=self.device)
        self.actor.load_state_dict(ckp["actor_state_dict"])
        self.actor.eval()

    def _infer_batch(self, arr: torch.Tensor, pad_len: Optional[torch.Tensor]):
        with torch.no_grad():
            adj, _, _ = self.actor(arr, deterministic=True, pad_len=pad_len)
        return adj

    def __call__(self, points: Sequence[Point]) -> dict:
        pts = [(float(p[0]), float(p[1])) for p in points]
        n = len(pts)
        if n <= 1:
            return {"nodes": pts, "edges": []}
        if n == 2:
            return {"nodes": pts, "edges": [(0, 1)]}

        if n > self.degree:
            from utils.divide_merge import rectilinear_mst
            return rectilinear_mst(pts)

        work = pts
        pad = 0
        if n < self.degree:
            pad = self.degree - n
            work = pts + [pts[0]] * pad

        norm = _normalize(work)
        arr = torch.tensor([norm], dtype=torch.float32, device=self.device)
        pad_len = torch.tensor([pad], device=self.device) if pad else None
        adj = self._infer_batch(arr, pad_len)[0].detach().cpu().numpy()
        edges = _adj_to_edges(adj, n)
        return {"nodes": pts, "edges": edges}
