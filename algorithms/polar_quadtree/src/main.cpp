#include "polar_quadtree/framework.h"
#include "polar_quadtree/geometry.h"
#include "polar_quadtree/solvers.h"

#include <iostream>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    const int n = 80;
    std::vector<pqt::Point> points(static_cast<std::size_t>(n));
    points[0] = {500.0, 500.0};
    for (int i = 1; i < n; ++i) {
        points[static_cast<std::size_t>(i)] = {dist(rng), dist(rng)};
    }

    const pqt::Tree tree = pqt::divide_and_merge(points, pqt::rectilinear_mst, 30);
    const pqt::TreeMetrics metrics = pqt::evaluate_tree(tree, n);

    std::cout << "pins=" << n << " nodes=" << tree.nodes.size()
              << " edges=" << tree.edges.size() << " valid=" << metrics.valid
              << " wirelength=" << metrics.wirelength
              << " max_path=" << metrics.max_path << "\n";
    for (const auto& e : tree.edges) {
        std::cout << e.first << " " << e.second << "\n";
    }
    return metrics.valid ? 0 : 1;
}
