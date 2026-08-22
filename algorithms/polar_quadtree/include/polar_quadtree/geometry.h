#pragma once

#include "polar_quadtree/types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace pqt {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kEps = 1e-9;

inline double sqr(double v) { return v * v; }

inline double euclidean(const Point& a, const Point& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

inline double manhattan(const Point& a, const Point& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

inline bool same_point(const Point& a, const Point& b, double eps = kEps) {
    return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps;
}

inline double polar_r(const Point& p) { return std::hypot(p.x, p.y); }

// Returns theta in [0, 2pi). Origin is defined as 0.
inline double polar_theta(const Point& p) {
    if (polar_r(p) <= kEps) {
        return 0.0;
    }
    double t = std::atan2(p.y, p.x);
    if (t < 0.0) {
        t += kTwoPi;
    }
    if (t >= kTwoPi) {
        t -= kTwoPi;
    }
    return t;
}

inline double wrap_theta(double t) {
    t = std::fmod(t, kTwoPi);
    if (t < 0.0) {
        t += kTwoPi;
    }
    return t;
}

inline int sector_index(double theta, int n_sectors) {
    if (n_sectors <= 0) {
        return 0;
    }
    theta = wrap_theta(theta);
    int idx = static_cast<int>(theta / kTwoPi * n_sectors);
    if (idx >= n_sectors) {
        idx = n_sectors - 1;
    }
    if (idx < 0) {
        idx = 0;
    }
    return idx;
}

inline Point polar_to_cartesian(double r, double theta) {
    return Point{r * std::cos(theta), r * std::sin(theta)};
}

inline Point block_center(const Block& b) {
    if (b.layer < 0) {
        return Point{0.0, 0.0};
    }
    const double r = 0.5 * (b.r_inner + b.r_outer);
    double t = 0.5 * (b.theta_min + b.theta_max);
    if (b.theta_max < b.theta_min) {
        t = wrap_theta(t + kPi);
    }
    return polar_to_cartesian(r, t);
}

inline int nearest_pin(const Point& target, const std::vector<int>& pin_ids,
                       const std::vector<Point>& pts) {
    int best = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for (int id : pin_ids) {
        const double d = manhattan(pts[static_cast<std::size_t>(id)], target);
        if (d < best_d) {
            best_d = d;
            best = id;
        }
    }
    return best;
}

inline std::pair<int, int> nearest_pair(const std::vector<int>& a,
                                        const std::vector<int>& b,
                                        const std::vector<Point>& pts) {
    int best_i = -1;
    int best_j = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i : a) {
        for (int j : b) {
            const double d = manhattan(pts[static_cast<std::size_t>(i)],
                                       pts[static_cast<std::size_t>(j)]);
            if (d < best_d) {
                best_d = d;
                best_i = i;
                best_j = j;
            }
        }
    }
    return {best_i, best_j};
}

inline Point translate(const Point& p, const Point& origin) {
    return Point{p.x - origin.x, p.y - origin.y};
}

inline Point untranslate(const Point& p, const Point& origin) {
    return Point{p.x + origin.x, p.y + origin.y};
}

struct TreeMetrics {
    bool valid = false;
    double wirelength = 0.0;
    double max_path = 0.0;
    int n_original = 0;
};

inline TreeMetrics evaluate_tree(const Tree& tree, int n_original) {
    TreeMetrics m;
    m.n_original = n_original;
    if (tree.nodes.empty() || n_original <= 0 ||
        static_cast<int>(tree.nodes.size()) < n_original) {
        return m;
    }

    const int n = static_cast<int>(tree.nodes.size());
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
    for (const auto& e : tree.edges) {
        if (e.first < 0 || e.second < 0 || e.first >= n || e.second >= n ||
            e.first == e.second) {
            return m;
        }
        adj[static_cast<std::size_t>(e.first)].push_back(e.second);
        adj[static_cast<std::size_t>(e.second)].push_back(e.first);
        m.wirelength += manhattan(tree.nodes[static_cast<std::size_t>(e.first)],
                                  tree.nodes[static_cast<std::size_t>(e.second)]);
    }

    if (n_original == 1) {
        m.valid = tree.edges.empty();
        m.max_path = 0.0;
        return m;
    }

    if (static_cast<int>(tree.edges.size()) != n - 1) {
        return m;
    }

    std::vector<int> parent(static_cast<std::size_t>(n), -2);
    std::vector<double> dist(static_cast<std::size_t>(n), 0.0);
    std::vector<int> stack;
    stack.push_back(0);
    parent[0] = -1;
    int seen = 0;
    while (!stack.empty()) {
        const int u = stack.back();
        stack.pop_back();
        ++seen;
        for (int v : adj[static_cast<std::size_t>(u)]) {
            if (v == parent[static_cast<std::size_t>(u)]) {
                continue;
            }
            if (parent[static_cast<std::size_t>(v)] != -2) {
                return m;  // cycle
            }
            parent[static_cast<std::size_t>(v)] = u;
            dist[static_cast<std::size_t>(v)] =
                dist[static_cast<std::size_t>(u)] +
                manhattan(tree.nodes[static_cast<std::size_t>(u)],
                          tree.nodes[static_cast<std::size_t>(v)]);
            stack.push_back(v);
        }
    }
    if (seen != n) {
        return m;
    }
    for (int i = 1; i < n_original; ++i) {
        m.max_path = std::max(m.max_path, dist[static_cast<std::size_t>(i)]);
    }
    m.valid = true;
    return m;
}

}  // namespace pqt
