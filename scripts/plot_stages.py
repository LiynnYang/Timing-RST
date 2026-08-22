"""Plot the four polar-quadtree pipeline stages from pipeline.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "viz" / "pipeline.json"
OUT_DIR = ROOT / "viz"


def l_shape(p, q):
    if abs(p["x"] - q["x"]) < 1e-9 or abs(p["y"] - q["y"]) < 1e-9:
        return [(p["x"], p["y"]), (q["x"], q["y"])]
    return [(p["x"], p["y"]), (q["x"], p["y"]), (q["x"], q["y"])]


def sector_polygon(origin, r_in, r_out, t0, t1, n=36):
    ox, oy = origin["x"], origin["y"]
    ts = np.linspace(t0, t1, n)
    outer = [(ox + r_out * math.cos(t), oy + r_out * math.sin(t)) for t in ts]
    inner = [(ox + r_in * math.cos(t), oy + r_in * math.sin(t)) for t in ts[::-1]]
    return outer + inner


def style_ax(ax, title):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def plot_points(ax, points, origin):
    xs = [p["x"] for p in points[1:]]
    ys = [p["y"] for p in points[1:]]
    ax.scatter(xs, ys, s=22, c="#4c78a8", zorder=3, label="sink")
    ax.scatter([origin["x"]], [origin["y"]], s=90, c="#d62728", marker="*",
               zorder=4, label="source")


def draw_tree(ax, nodes, edges, color, lw=1.4, alpha=0.95):
    for a, b in edges:
        path = l_shape(nodes[a], nodes[b])
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=2)


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    points = data["points"]
    origin = data["origin"]
    blocks = data["blocks"]
    subtrees = data["subtrees"]
    final = data["final"]

    nonempty = [b for b in blocks if b["nonempty"]]
    cmap = cm.get_cmap("tab20", max(len(nonempty), 1))
    block_color = {b["id"]: cmap(i) for i, b in enumerate(nonempty)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.2))
    fig.suptitle("Polar Quadtree Divide-and-Merge  (n=48, B=8)", fontsize=13, y=0.98)

    ax0 = axes[0, 0]
    plot_points(ax0, points, origin)
    ax0.legend(loc="upper right", frameon=False, fontsize=8)
    style_ax(ax0, "1. Initial point set")

    ax1 = axes[0, 1]
    for b in blocks:
        poly = sector_polygon(origin, b["r_inner"], b["r_outer"], b["theta_min"], b["theta_max"])
        facecolor = block_color.get(b["id"], (0.85, 0.85, 0.85, 0.15))
        if b["nonempty"]:
            fc = (*facecolor[:3], 0.28)
            ec = facecolor[:3]
            lw = 0.9
        else:
            fc = (0.9, 0.9, 0.9, 0.08)
            ec = (0.7, 0.7, 0.7)
            lw = 0.4
        ax1.add_patch(Polygon(poly, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw))
    plot_points(ax1, points, origin)
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    style_ax(ax1, "2. Polar partition (data-dependent rings)")

    ax2 = axes[1, 0]
    for b in blocks:
        poly = sector_polygon(origin, b["r_inner"], b["r_outer"], b["theta_min"], b["theta_max"])
        ax2.add_patch(Polygon(poly, closed=True, facecolor=(0, 0, 0, 0.02),
                              edgecolor=(0.75, 0.75, 0.75), linewidth=0.4))
    for sub in subtrees:
        color = block_color.get(sub["block"], (0.2, 0.2, 0.2, 1))
        draw_tree(ax2, sub["nodes"], sub["edges"], color, lw=1.6)
        src = sub["nodes"][0]
        ax2.scatter([src["x"]], [src["y"]], s=36, c=[color], marker="s", zorder=5)
    plot_points(ax2, points, origin)
    style_ax(ax2, "3. Local subtrees (one solver call per block)")

    ax3 = axes[1, 1]
    draw_tree(ax3, final["nodes"], final["edges"], "#222222", lw=1.5)
    plot_points(ax3, points, origin)
    ax3.legend(loc="upper right", frameon=False, fontsize=8)
    style_ax(ax3, "4. Merged global tree")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_DIR.mkdir(exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(exist_ok=True)
    combined = OUT_DIR / "four_stages.png"
    fig.savefig(combined, dpi=160)
    fig.savefig(img_dir / "four_stages.png", dpi=160)
    names = ["01_points.png", "02_partition.png", "03_subtrees.png", "04_final_tree.png"]
    titles = [
        "1. Initial point set",
        "2. Polar partition",
        "3. Local subtrees",
        "4. Merged global tree",
    ]
    for ax, name, title in zip(axes.ravel(), names, titles):
        fig_i, ax_i = plt.subplots(figsize=(6.2, 6.0))
        for artist in ax.get_children():
            pass
        # redraw independently by invoking the same drawing on a new axis is messy;
        # crop from combined instead via bbox of each axes.
        fig_i.clf()
        plt.close(fig_i)

    # Save each panel via its bounding box on the combined figure.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, name in zip(axes.ravel(), names):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(OUT_DIR / name, dpi=160, bbox_inches=bbox.expanded(1.08, 1.10))
    plt.close(fig)
    print(f"wrote {combined}")
    for name in names:
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
