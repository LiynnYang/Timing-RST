#include "polar_quadtree/solvers.h"

#include "polar_quadtree/geometry.h"

#include <limits>
#include <vector>

namespace pqt {

Tree rectilinear_mst(const std::vector<Point>& points) {
    Tree tree;
    tree.nodes = points;
    const int n = static_cast<int>(points.size());
    if (n <= 1) {
        return tree;
    }

    std::vector<char> used(static_cast<std::size_t>(n), 0);
    std::vector<double> dist(static_cast<std::size_t>(n),
                             std::numeric_limits<double>::infinity());
    std::vector<int> prev(static_cast<std::size_t>(n), -1);
    dist[0] = 0.0;

    for (int iter = 0; iter < n; ++iter) {
        int u = -1;
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            if (!used[static_cast<std::size_t>(i)] && dist[static_cast<std::size_t>(i)] < best) {
                best = dist[static_cast<std::size_t>(i)];
                u = i;
            }
        }
        if (u < 0) {
            break;
        }
        used[static_cast<std::size_t>(u)] = 1;
        if (prev[static_cast<std::size_t>(u)] >= 0) {
            tree.edges.emplace_back(prev[static_cast<std::size_t>(u)], u);
        }
        for (int v = 0; v < n; ++v) {
            if (used[static_cast<std::size_t>(v)]) {
                continue;
            }
            const double d = manhattan(points[static_cast<std::size_t>(u)],
                                       points[static_cast<std::size_t>(v)]);
            if (d < dist[static_cast<std::size_t>(v)]) {
                dist[static_cast<std::size_t>(v)] = d;
                prev[static_cast<std::size_t>(v)] = u;
            }
        }
    }
    return tree;
}

}  // namespace pqt
