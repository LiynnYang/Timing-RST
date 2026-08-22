#include "polar_quadtree/framework.h"

#include "polar_quadtree/geometry.h"
#include "polar_quadtree/merge.h"
#include "polar_quadtree/quadtree.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pqt {
namespace {

constexpr int kModerateDegree = 32;

void append_unique_edge(Tree& tree, int a, int b) {
    if (a == b || a < 0 || b < 0) {
        return;
    }
    if (a > b) {
        std::swap(a, b);
    }
    for (const auto& e : tree.edges) {
        const int u = std::min(e.first, e.second);
        const int v = std::max(e.first, e.second);
        if (u == a && v == b) {
            return;
        }
    }
    tree.edges.emplace_back(a, b);
}

int add_steiner(Tree& tree, const Point& p) {
    for (int i = 0; i < static_cast<int>(tree.nodes.size()); ++i) {
        if (same_point(tree.nodes[static_cast<std::size_t>(i)], p)) {
            return i;
        }
    }
    tree.nodes.push_back(p);
    return static_cast<int>(tree.nodes.size()) - 1;
}

Tree extract_spanning_tree(const Tree& input, int n_original) {
    const int n = static_cast<int>(input.nodes.size());
    Tree out;
    out.nodes = input.nodes;
    if (n_original <= 1) {
        out.nodes.assign(input.nodes.begin(),
                         input.nodes.begin() + std::min(n, n_original));
        return out;
    }

    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
    for (const auto& e : input.edges) {
        if (e.first < 0 || e.second < 0 || e.first >= n || e.second >= n ||
            e.first == e.second) {
            continue;
        }
        adj[static_cast<std::size_t>(e.first)].push_back(e.second);
        adj[static_cast<std::size_t>(e.second)].push_back(e.first);
    }

    std::vector<int> parent(static_cast<std::size_t>(n), -2);
    std::vector<int> stack{0};
    parent[0] = -1;
    while (!stack.empty()) {
        const int u = stack.back();
        stack.pop_back();
        for (int v : adj[static_cast<std::size_t>(u)]) {
            if (parent[static_cast<std::size_t>(v)] != -2) {
                continue;
            }
            parent[static_cast<std::size_t>(v)] = u;
            stack.push_back(v);
        }
    }

    std::vector<char> keep(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n_original; ++i) {
        keep[static_cast<std::size_t>(i)] = 1;
        if (parent[static_cast<std::size_t>(i)] == -2 && i != 0) {
            append_unique_edge(out, 0, i);
            parent[static_cast<std::size_t>(i)] = 0;
        }
    }
    for (int i = 1; i < n; ++i) {
        if (parent[static_cast<std::size_t>(i)] >= 0) {
            append_unique_edge(out, parent[static_cast<std::size_t>(i)], i);
            int u = i;
            while (u > 0 && !keep[static_cast<std::size_t>(u)]) {
                keep[static_cast<std::size_t>(u)] = 1;
                u = parent[static_cast<std::size_t>(u)];
            }
            keep[static_cast<std::size_t>(u)] = 1;
        }
    }

    std::vector<int> remap(static_cast<std::size_t>(n), -1);
    Tree compact;
    for (int i = 0; i < n; ++i) {
        if (!keep[static_cast<std::size_t>(i)]) {
            continue;
        }
        remap[static_cast<std::size_t>(i)] = static_cast<int>(compact.nodes.size());
        compact.nodes.push_back(out.nodes[static_cast<std::size_t>(i)]);
    }
    for (const auto& e : out.edges) {
        const int a = remap[static_cast<std::size_t>(e.first)];
        const int b = remap[static_cast<std::size_t>(e.second)];
        if (a >= 0 && b >= 0) {
            append_unique_edge(compact, a, b);
        }
    }
    return compact;
}

}  // namespace

Tree divide_and_merge(const std::vector<Point>& points, const Solver& solver,
                      int capacity, double search_bound) {
    if (points.empty()) {
        return Tree{};
    }
    if (!solver) {
        throw std::invalid_argument("divide_and_merge: solver is empty");
    }

    const int n = static_cast<int>(points.size());
    if (n <= std::max(kModerateDegree, capacity + 1)) {
        Tree t = solver(points);
        if (static_cast<int>(t.nodes.size()) < n) {
            t.nodes = points;
        }
        return t;
    }

    const Point origin = points[0];
    std::vector<Point> translated;
    translated.reserve(points.size());
    for (const Point& p : points) {
        translated.push_back(translate(p, origin));
    }

    PolarQuadtree qt = build_polar_quadtree(translated, capacity);
    const double bound = search_bound < 0.0 ? qt.default_search_bound : search_bound;
    compute_merge_topology(qt, bound);

    std::vector<InterBlockLink> links;
    assign_sources_and_links(qt, translated, origin, links);

    Tree global;
    global.nodes = points;

    for (const Block& block : qt.blocks) {
        if (!block.nonempty() || block.local_source < 0) {
            continue;
        }
        std::vector<Point> local;
        std::vector<int> local_to_orig;
        local.push_back(points[static_cast<std::size_t>(block.local_source)]);
        local_to_orig.push_back(block.local_source);
        for (int pid : block.pin_ids) {
            if (pid == block.local_source) {
                continue;
            }
            local.push_back(points[static_cast<std::size_t>(pid)]);
            local_to_orig.push_back(pid);
        }

        Tree sub;
        if (local.size() <= 1) {
            sub.nodes = local;
        } else {
            sub = solver(local);
            if (sub.nodes.size() < local.size()) {
                sub.nodes = local;
            }
        }

        std::vector<int> node_map(sub.nodes.size(), -1);
        for (std::size_t i = 0; i < local.size() && i < sub.nodes.size(); ++i) {
            node_map[i] = local_to_orig[i];
        }
        for (std::size_t i = local.size(); i < sub.nodes.size(); ++i) {
            node_map[i] = add_steiner(global, sub.nodes[i]);
        }
        for (const auto& e : sub.edges) {
            if (e.first < 0 || e.second < 0 ||
                e.first >= static_cast<int>(node_map.size()) ||
                e.second >= static_cast<int>(node_map.size())) {
                continue;
            }
            append_unique_edge(global, node_map[static_cast<std::size_t>(e.first)],
                               node_map[static_cast<std::size_t>(e.second)]);
        }
    }

    for (const InterBlockLink& link : links) {
        std::vector<int> ids;
        ids.reserve(link.pins.size() + 1);
        for (int pid : link.pins) {
            ids.push_back(pid);
        }
        if (link.has_steiner) {
            const int sid = add_steiner(global, link.steiner);
            for (int pid : ids) {
                append_unique_edge(global, pid, sid);
            }
        } else if (ids.size() >= 2) {
            append_unique_edge(global, ids[0], ids[1]);
        }
    }

    // If the result is disconnected (should be rare), star-connect leftovers to source.
    {
        const int m = static_cast<int>(global.nodes.size());
        std::vector<std::vector<int>> adj(static_cast<std::size_t>(m));
        for (const auto& e : global.edges) {
            adj[static_cast<std::size_t>(e.first)].push_back(e.second);
            adj[static_cast<std::size_t>(e.second)].push_back(e.first);
        }
        std::vector<char> seen(static_cast<std::size_t>(m), 0);
        std::vector<int> stack{0};
        seen[0] = 1;
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            for (int v : adj[static_cast<std::size_t>(u)]) {
                if (!seen[static_cast<std::size_t>(v)]) {
                    seen[static_cast<std::size_t>(v)] = 1;
                    stack.push_back(v);
                }
            }
        }
        for (int i = 1; i < n; ++i) {
            if (!seen[static_cast<std::size_t>(i)]) {
                append_unique_edge(global, 0, i);
            }
        }
    }

    return extract_spanning_tree(global, n);
}

}  // namespace pqt
