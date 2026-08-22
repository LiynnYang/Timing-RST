#include "polar_quadtree/framework.h"
#include "polar_quadtree/geometry.h"
#include "polar_quadtree/quadtree.h"
#include "polar_quadtree/solvers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

std::set<std::pair<int, int>> edge_set(const pqt::Tree& tree) {
    std::set<std::pair<int, int>> s;
    for (auto e : tree.edges) {
        if (e.first > e.second) {
            std::swap(e.first, e.second);
        }
        s.insert(e);
    }
    return s;
}

int degree_of_source(const pqt::Tree& tree) {
    int d = 0;
    for (const auto& e : tree.edges) {
        if (e.first == 0 || e.second == 0) {
            ++d;
        }
    }
    return d;
}

std::vector<pqt::Point> random_points(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);
    std::vector<pqt::Point> pts(static_cast<std::size_t>(n));
    pts[0] = {500.0, 500.0};
    for (int i = 1; i < n; ++i) {
        pts[static_cast<std::size_t>(i)] = {dist(rng), dist(rng)};
    }
    return pts;
}

void test_small_equals_direct_solver() {
    auto pts = random_points(10, 1);
    const pqt::Tree direct = pqt::rectilinear_mst(pts);
    const pqt::Tree wrapped = pqt::divide_and_merge(pts, pqt::rectilinear_mst, 30);
    expect(edge_set(direct) == edge_set(wrapped),
           "n=10 should call solver on the whole set");
    expect(pqt::evaluate_tree(wrapped, 10).valid, "n=10 tree should be valid");
}

void test_solver_source_is_local_first() {
    auto pts = random_points(50, 7);
    int calls = 0;
    bool ok = true;
    pqt::Solver wrapped = [&](const std::vector<pqt::Point>& local) {
        ++calls;
        if (local.empty()) {
            ok = false;
            return pqt::Tree{};
        }
        bool found = false;
        for (const auto& p : pts) {
            if (pqt::same_point(p, local[0])) {
                found = true;
                break;
            }
        }
        if (!found) {
            ok = false;
        }
        for (const auto& q : local) {
            bool in_set = false;
            for (const auto& p : pts) {
                if (pqt::same_point(p, q)) {
                    in_set = true;
                    break;
                }
            }
            if (!in_set) {
                ok = false;
            }
        }
        return pqt::rectilinear_mst(local);
    };
    const pqt::Tree tree = pqt::divide_and_merge(pts, wrapped, 8);
    expect(ok, "solver local[0] must be the block source (an original pin)");
    expect(calls >= 1, "solver should be invoked");
    expect(pqt::evaluate_tree(tree, 50).valid, "n=50 wrapped solver tree");
}

void test_block_capacity() {
    auto pts = random_points(80, 3);
    const pqt::Point origin = pts[0];
    std::vector<pqt::Point> translated;
    for (const auto& p : pts) {
        translated.push_back(pqt::translate(p, origin));
    }
    const int cap = 10;
    const pqt::PolarQuadtree qt = pqt::build_polar_quadtree(translated, cap);
    bool bounded = true;
    int nonempty = 0;
    for (const auto& b : qt.blocks) {
        if (b.layer >= 0 && static_cast<int>(b.pin_ids.size()) > cap) {
            bounded = false;
        }
        if (b.nonempty()) {
            ++nonempty;
        }
    }
    expect(bounded, "each polar block should have at most capacity pins");
    expect(nonempty >= 2, "large net should be partitioned into multiple blocks");
    expect(qt.blocks[static_cast<std::size_t>(qt.x0_id)].pin_ids[0] == 0,
           "x0 must contain the global source");
}

void test_random_valid_tree(int n, uint32_t seed) {
    auto pts = random_points(n, seed);
    const pqt::Tree tree = pqt::divide_and_merge(pts, pqt::rectilinear_mst, 12);
    const pqt::TreeMetrics m = pqt::evaluate_tree(tree, n);
    expect(m.valid, "random n=" + std::to_string(n) + " should yield a valid tree");
    expect(static_cast<int>(tree.nodes.size()) >= n, "all original pins kept");
}

void test_fig7_like_outer_ring() {
    const int n = 41;
    std::vector<pqt::Point> pts(static_cast<std::size_t>(n));
    pts[0] = {0.0, 0.0};
    for (int i = 1; i < n; ++i) {
        const double t = pqt::kTwoPi * (i - 1) / (n - 1);
        pts[static_cast<std::size_t>(i)] = {1000.0 * std::cos(t), 1000.0 * std::sin(t)};
    }
    const pqt::Tree tree = pqt::divide_and_merge(pts, pqt::rectilinear_mst, 6);
    const pqt::TreeMetrics m = pqt::evaluate_tree(tree, n);
    expect(m.valid, "Fig.7-like instance should be a valid tree");

    double star = 0.0;
    for (int i = 1; i < n; ++i) {
        star += pqt::manhattan(pts[0], pts[static_cast<std::size_t>(i)]);
    }
    expect(m.wirelength + 1e-6 < star,
           "Fig.7-like merge should not star-connect every pin to source");
    expect(degree_of_source(tree) < n - 1,
           "source degree should be less than n-1 on an outer ring");
}

}  // namespace

int main() {
    test_small_equals_direct_solver();
    test_solver_source_is_local_first();
    test_block_capacity();
    test_random_valid_tree(50, 11);
    test_random_valid_tree(100, 13);
    test_fig7_like_outer_ring();

    if (g_failures == 0) {
        std::cout << "All tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " test(s) failed.\n";
    return 1;
}
