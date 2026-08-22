#include "polar_quadtree/merge.h"

#include "polar_quadtree/geometry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace pqt {
namespace {

struct UnionFind {
    std::vector<int> p;
    explicit UnionFind(int n) : p(static_cast<std::size_t>(n)) {
        for (int i = 0; i < n; ++i) {
            p[static_cast<std::size_t>(i)] = i;
        }
    }
    int find(int x) {
        if (p[static_cast<std::size_t>(x)] != x) {
            p[static_cast<std::size_t>(x)] = find(p[static_cast<std::size_t>(x)]);
        }
        return p[static_cast<std::size_t>(x)];
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return false;
        }
        p[static_cast<std::size_t>(a)] = b;
        return true;
    }
};

int left_nei(const PolarQuadtree& qt, int id) {
    return qt.blocks[static_cast<std::size_t>(id)].left_id;
}

int right_nei(const PolarQuadtree& qt, int id) {
    return qt.blocks[static_cast<std::size_t>(id)].right_id;
}

int parent_of(const PolarQuadtree& qt, int id) {
    return qt.blocks[static_cast<std::size_t>(id)].parent_id;
}

int apply_nei(const PolarQuadtree& qt, int id, int times, bool left) {
    int y = id;
    for (int i = 0; i < times; ++i) {
        const int nxt = left ? left_nei(qt, y) : right_nei(qt, y);
        if (nxt < 0) {
            return -1;
        }
        y = nxt;
    }
    return y;
}

bool is_nonempty(const PolarQuadtree& qt, int id) {
    return id >= 0 &&
           qt.blocks[static_cast<std::size_t>(id)].nonempty();
}

double center_dist(const PolarQuadtree& qt, int a, int b) {
    return manhattan(block_center(qt.blocks[static_cast<std::size_t>(a)]),
                     block_center(qt.blocks[static_cast<std::size_t>(b)]));
}

// Simultaneous left/right zigzag. Returns the first nonempty block within bound.
int zigzag_search(const PolarQuadtree& qt, int x, double bound) {
    int best = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for (int k = 1; k <= 64; ++k) {
        const int left = apply_nei(qt, x, k, true);
        const int right = apply_nei(qt, x, k, false);
        const int cand[4] = {left, left >= 0 ? parent_of(qt, left) : -1, right,
                             right >= 0 ? parent_of(qt, right) : -1};
        bool progressed = false;
        for (int c : cand) {
            if (c < 0 || c == x) {
                continue;
            }
            progressed = true;
            const double d = center_dist(qt, x, c);
            if (d > bound + kEps) {
                continue;
            }
            if (is_nonempty(qt, c) && d < best_d) {
                best_d = d;
                best = c;
            }
        }
        if (best >= 0) {
            return best;
        }
        if (!progressed) {
            break;
        }
    }
    return best;
}

int find_merge_target(const PolarQuadtree& qt, int x, double bound) {
    if (x == qt.x0_id) {
        return -1;
    }
    int cur = x;
    for (int hop = 0; hop < 64 && cur >= 0; ++hop) {
        const int par = parent_of(qt, cur);
        if (is_nonempty(qt, par)) {
            return par;
        }
        const int found = zigzag_search(qt, cur, bound);
        if (found >= 0) {
            return found;
        }
        if (par < 0 || par == qt.x0_id) {
            return qt.x0_id;
        }
        cur = par;
    }
    return qt.x0_id;
}

Point three_block_intersection(const Block& parent, const Block& child_a,
                               const Block& child_b) {
    const double r = parent.r_outer;
    double t = 0.5 * (child_a.theta_min + child_a.theta_max);
    if (child_a.right_id == child_b.id || parent.left_child_id == child_a.id) {
        t = child_a.theta_max;
    } else if (child_b.right_id == child_a.id || parent.right_child_id == child_a.id) {
        t = child_b.theta_max;
    } else {
        t = 0.5 * (parent.theta_min + parent.theta_max);
    }
    return polar_to_cartesian(r, wrap_theta(t));
}

}  // namespace

void compute_merge_topology(PolarQuadtree& qt, double search_bound) {
    const int n = static_cast<int>(qt.blocks.size());
    for (int i = 0; i < n; ++i) {
        qt.blocks[static_cast<std::size_t>(i)].merge_to = -1;
    }

    UnionFind uf(n);
    std::vector<int> nonempty;
    nonempty.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (qt.blocks[static_cast<std::size_t>(i)].nonempty()) {
            nonempty.push_back(i);
        }
    }
    std::sort(nonempty.begin(), nonempty.end(), [&](int a, int b) {
        const int la = qt.blocks[static_cast<std::size_t>(a)].layer;
        const int lb = qt.blocks[static_cast<std::size_t>(b)].layer;
        if (la != lb) {
            return la > lb;  // outer first
        }
        return a < b;
    });

    const double bound =
        search_bound > 0.0 ? search_bound : std::max(qt.default_search_bound, 1.0);

    for (int x : nonempty) {
        if (x == qt.x0_id) {
            continue;
        }
        int target = find_merge_target(qt, x, bound);
        if (target < 0 || !uf.unite(x, target)) {
            target = qt.x0_id;
            uf.unite(x, target);
        }
        qt.blocks[static_cast<std::size_t>(x)].merge_to = target;
    }

    // Attach leftover components to x0.
    const int root = uf.find(qt.x0_id);
    for (int x : nonempty) {
        if (uf.find(x) == root) {
            continue;
        }
        qt.blocks[static_cast<std::size_t>(x)].merge_to = qt.x0_id;
        uf.unite(x, qt.x0_id);
    }
}

void assign_sources_and_links(PolarQuadtree& qt,
                              const std::vector<Point>& translated,
                              const Point& origin,
                              std::vector<InterBlockLink>& links) {
    links.clear();
    const int n = static_cast<int>(qt.blocks.size());
    for (int i = 0; i < n; ++i) {
        qt.blocks[static_cast<std::size_t>(i)].local_source = -1;
    }
    qt.blocks[static_cast<std::size_t>(qt.x0_id)].local_source = 0;

    std::vector<std::vector<int>> children(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int p = qt.blocks[static_cast<std::size_t>(i)].merge_to;
        if (p >= 0 && qt.blocks[static_cast<std::size_t>(i)].nonempty()) {
            children[static_cast<std::size_t>(p)].push_back(i);
        }
    }

    auto set_source = [&](int bid, int pin) {
        if (bid == qt.x0_id) {
            return;
        }
        if (pin >= 0) {
            qt.blocks[static_cast<std::size_t>(bid)].local_source = pin;
        }
    };

    for (int parent_id = 0; parent_id < n; ++parent_id) {
        auto& ch = children[static_cast<std::size_t>(parent_id)];
        if (ch.empty()) {
            continue;
        }
        Block& parent = qt.blocks[static_cast<std::size_t>(parent_id)];

        // Pair adjacent binary children as Case 1; remaining children as Case 2.
        std::vector<char> used(ch.size(), 0);
        for (std::size_t i = 0; i < ch.size(); ++i) {
            if (used[i]) {
                continue;
            }
            std::size_t mate = ch.size();
            for (std::size_t j = i + 1; j < ch.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                const Block& a = qt.blocks[static_cast<std::size_t>(ch[i])];
                const Block& b = qt.blocks[static_cast<std::size_t>(ch[j])];
                if (a.parent_id == parent_id && b.parent_id == parent_id &&
                    ((parent.left_child_id == a.id && parent.right_child_id == b.id) ||
                     (parent.left_child_id == b.id && parent.right_child_id == a.id))) {
                    mate = j;
                    break;
                }
            }
            if (mate < ch.size()) {
                used[i] = 1;
                used[mate] = 1;
                const int ca = ch[i];
                const int cb = ch[mate];
                Block& a = qt.blocks[static_cast<std::size_t>(ca)];
                Block& b = qt.blocks[static_cast<std::size_t>(cb)];
                const Point p = three_block_intersection(parent, a, b);
                const int pu = nearest_pin(p, a.pin_ids, translated);
                const int pv = nearest_pin(p, b.pin_ids, translated);
                const int pw = nearest_pin(p, parent.pin_ids, translated);
                set_source(ca, pu);
                set_source(cb, pv);
                InterBlockLink link;
                link.has_steiner = true;
                link.steiner = untranslate(p, origin);
                if (pu >= 0) {
                    link.pins.push_back(pu);
                }
                if (pv >= 0) {
                    link.pins.push_back(pv);
                }
                if (pw >= 0) {
                    link.pins.push_back(pw);
                }
                if (link.pins.size() >= 2) {
                    links.push_back(std::move(link));
                }
            }
        }
        for (std::size_t i = 0; i < ch.size(); ++i) {
            if (used[i]) {
                continue;
            }
            const int cid = ch[i];
            Block& child = qt.blocks[static_cast<std::size_t>(cid)];
            const auto pair = nearest_pair(parent.pin_ids, child.pin_ids, translated);
            const int ps = pair.first;
            const int pe = pair.second;
            set_source(cid, pe);
            InterBlockLink link;
            link.has_steiner = false;
            if (ps >= 0) {
                link.pins.push_back(ps);
            }
            if (pe >= 0 && pe != ps) {
                link.pins.push_back(pe);
            }
            if (link.pins.size() >= 2) {
                links.push_back(std::move(link));
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        Block& b = qt.blocks[static_cast<std::size_t>(i)];
        if (!b.nonempty()) {
            continue;
        }
        if (b.local_source < 0) {
            if (i == qt.x0_id) {
                b.local_source = 0;
            } else {
                b.local_source = nearest_pin(Point{0.0, 0.0}, b.pin_ids, translated);
            }
        }
    }
}

}  // namespace pqt
