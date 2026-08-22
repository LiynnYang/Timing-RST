#include "polar_quadtree/quadtree.h"

#include "polar_quadtree/geometry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace pqt {
namespace {

constexpr int kMaxDepth = 16;
constexpr int kMaxLayers = 24;

int add_block(PolarQuadtree& qt) {
    Block b;
    b.id = static_cast<int>(qt.blocks.size());
    qt.blocks.push_back(std::move(b));
    return qt.blocks.back().id;
}

void link_neighbors_on_layer(PolarQuadtree& qt, int layer) {
    auto& ids = qt.layers[static_cast<std::size_t>(layer)];
    const int n = static_cast<int>(ids.size());
    if (n <= 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        Block& b = qt.blocks[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])];
        b.left_id = ids[static_cast<std::size_t>((i - 1 + n) % n)];
        b.right_id = ids[static_cast<std::size_t>((i + 1) % n)];
        if (layer == 0) {
            b.parent_id = qt.x0_id;
        } else {
            b.parent_id = qt.layers[static_cast<std::size_t>(layer - 1)]
                                    [static_cast<std::size_t>(i / 2)];
        }
    }
    if (layer == 0) {
        Block& x0 = qt.blocks[static_cast<std::size_t>(qt.x0_id)];
        if (n >= 1) {
            x0.left_child_id = ids[0];
        }
        if (n >= 2) {
            x0.right_child_id = ids[1];
        }
    } else {
        auto& parents = qt.layers[static_cast<std::size_t>(layer - 1)];
        for (int i = 0; i < n; ++i) {
            Block& parent =
                qt.blocks[static_cast<std::size_t>(parents[static_cast<std::size_t>(i / 2)])];
            if (i % 2 == 0) {
                parent.left_child_id = ids[static_cast<std::size_t>(i)];
            } else {
                parent.right_child_id = ids[static_cast<std::size_t>(i)];
            }
        }
    }
}

void split_block_recursive(PolarQuadtree& qt, int bid, int capacity, int depth,
                           const std::vector<Point>& translated) {
    Block& b = qt.blocks[static_cast<std::size_t>(bid)];
    if (static_cast<int>(b.pin_ids.size()) <= capacity || depth >= kMaxDepth) {
        return;
    }

    std::vector<int> pins = b.pin_ids;
    std::sort(pins.begin(), pins.end(), [&](int a, int b_id) {
        return polar_theta(translated[static_cast<std::size_t>(a)]) <
               polar_theta(translated[static_cast<std::size_t>(b_id)]);
    });
    const std::size_t mid = pins.size() / 2;
    if (mid == 0 || mid == pins.size()) {
        return;
    }

    const int left_id = add_block(qt);
    const int right_id = add_block(qt);
    Block& parent = qt.blocks[static_cast<std::size_t>(bid)];
    Block& left = qt.blocks[static_cast<std::size_t>(left_id)];
    Block& right = qt.blocks[static_cast<std::size_t>(right_id)];

    left.layer = parent.layer;
    right.layer = parent.layer;
    left.index = parent.index;
    right.index = parent.index;
    left.r_inner = parent.r_inner;
    right.r_inner = parent.r_inner;
    left.r_outer = parent.r_outer;
    right.r_outer = parent.r_outer;
    left.theta_min = parent.theta_min;
    left.theta_max = polar_theta(translated[static_cast<std::size_t>(pins[mid])]);
    right.theta_min = left.theta_max;
    right.theta_max = parent.theta_max;
    left.parent_id = bid;
    right.parent_id = bid;
    left.left_id = parent.left_id;
    left.right_id = right_id;
    right.left_id = left_id;
    right.right_id = parent.right_id;
    left.pin_ids.assign(pins.begin(), pins.begin() + static_cast<std::ptrdiff_t>(mid));
    right.pin_ids.assign(pins.begin() + static_cast<std::ptrdiff_t>(mid), pins.end());

    parent.pin_ids.clear();
    parent.left_child_id = left_id;
    parent.right_child_id = right_id;

    split_block_recursive(qt, left_id, capacity, depth + 1, translated);
    split_block_recursive(qt, right_id, capacity, depth + 1, translated);
}

}  // namespace

PolarQuadtree build_polar_quadtree(const std::vector<Point>& translated,
                                   int capacity) {
    PolarQuadtree qt;
    const int n = static_cast<int>(translated.size());
    const int cap = std::max(capacity, 1);

    const int x0 = add_block(qt);
    qt.x0_id = x0;
    qt.blocks[static_cast<std::size_t>(x0)].layer = -1;
    qt.blocks[static_cast<std::size_t>(x0)].pin_ids.push_back(0);

    std::vector<int> remaining;
    remaining.reserve(static_cast<std::size_t>(std::max(n - 1, 0)));
    qt.rho_max = 0.0;
    for (int i = 1; i < n; ++i) {
        const double r = polar_r(translated[static_cast<std::size_t>(i)]);
        qt.rho_max = std::max(qt.rho_max, r);
        if (r <= kEps) {
            qt.blocks[static_cast<std::size_t>(x0)].pin_ids.push_back(i);
        } else {
            remaining.push_back(i);
        }
    }
    std::sort(remaining.begin(), remaining.end(), [&](int a, int b) {
        const double ra = polar_r(translated[static_cast<std::size_t>(a)]);
        const double rb = polar_r(translated[static_cast<std::size_t>(b)]);
        if (std::abs(ra - rb) > kEps) {
            return ra < rb;
        }
        return a < b;
    });

    double r_inner = 0.0;
    double max_thickness = 0.0;
    int layer = 0;
    std::size_t cursor = 0;
    while (cursor < remaining.size() && layer < kMaxLayers) {
        const int n_sectors = 4 << layer;
        std::vector<int> counts(static_cast<std::size_t>(n_sectors), 0);
        std::vector<std::vector<int>> assigned(static_cast<std::size_t>(n_sectors));
        double r_outer = r_inner;

        while (cursor < remaining.size()) {
            const int pid = remaining[cursor];
            const Point& p = translated[static_cast<std::size_t>(pid)];
            const int s = sector_index(polar_theta(p), n_sectors);
            if (counts[static_cast<std::size_t>(s)] >= cap) {
                break;
            }
            assigned[static_cast<std::size_t>(s)].push_back(pid);
            ++counts[static_cast<std::size_t>(s)];
            r_outer = std::max(r_outer, polar_r(p));
            ++cursor;
        }

        if (r_outer <= r_inner) {
            r_outer = r_inner;
            if (cursor < remaining.size()) {
                r_outer = polar_r(translated[static_cast<std::size_t>(remaining[cursor])]);
            }
        }

        qt.layers.emplace_back();
        auto& layer_ids = qt.layers.back();
        layer_ids.reserve(static_cast<std::size_t>(n_sectors));
        for (int s = 0; s < n_sectors; ++s) {
            const int id = add_block(qt);
            Block& b = qt.blocks[static_cast<std::size_t>(id)];
            b.layer = layer;
            b.index = s;
            b.r_inner = r_inner;
            b.r_outer = r_outer;
            b.theta_min = kTwoPi * s / n_sectors;
            b.theta_max = kTwoPi * (s + 1) / n_sectors;
            b.pin_ids = std::move(assigned[static_cast<std::size_t>(s)]);
            layer_ids.push_back(id);
        }
        link_neighbors_on_layer(qt, layer);
        max_thickness = std::max(max_thickness, r_outer - r_inner);
        r_inner = r_outer;
        ++layer;

        if (cursor < remaining.size()) {
            const double r_next =
                polar_r(translated[static_cast<std::size_t>(remaining[cursor])]);
            if (r_next <= r_inner + kEps && layer > 1) {
                // Zero-thickness leftover in a full sector: force angular split later.
            }
        }
    }

    // Any leftover pins (identical polar location) go into the last matching sector.
    if (cursor < remaining.size()) {
        if (qt.layers.empty()) {
            qt.layers.emplace_back();
            const int n_sectors = 4;
            for (int s = 0; s < n_sectors; ++s) {
                const int id = add_block(qt);
                Block& b = qt.blocks[static_cast<std::size_t>(id)];
                b.layer = 0;
                b.index = s;
                b.r_inner = 0.0;
                b.r_outer = qt.rho_max;
                b.theta_min = kTwoPi * s / n_sectors;
                b.theta_max = kTwoPi * (s + 1) / n_sectors;
                qt.layers[0].push_back(id);
            }
            link_neighbors_on_layer(qt, 0);
        }
        const auto& last = qt.layers.back();
        const int n_sectors = static_cast<int>(last.size());
        for (; cursor < remaining.size(); ++cursor) {
            const int pid = remaining[cursor];
            const int s =
                sector_index(polar_theta(translated[static_cast<std::size_t>(pid)]),
                             n_sectors);
            qt.blocks[static_cast<std::size_t>(last[static_cast<std::size_t>(s)])]
                .pin_ids.push_back(pid);
        }
    }

    const int n_blocks = static_cast<int>(qt.blocks.size());
    for (int id = 0; id < n_blocks; ++id) {
        if (id == qt.x0_id) {
            continue;
        }
        split_block_recursive(qt, id, cap, 0, translated);
    }

    qt.default_search_bound =
        2.0 * std::max(max_thickness, qt.rho_max > 0.0 ? qt.rho_max * 0.25 : 1.0);
    return qt;
}

}  // namespace pqt
