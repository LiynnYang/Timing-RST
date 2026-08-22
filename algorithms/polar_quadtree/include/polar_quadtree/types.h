#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace pqt {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Tree {
    std::vector<Point> nodes;                 // original pins first, then Steiner points
    std::vector<std::pair<int, int>> edges;   // undirected rectilinear edges
};

// Local solver: input[0] is the source of this subproblem.
// The first input.size() nodes of the returned tree must equal the input
// points in the same order; extra nodes are Steiner points.
using Solver = std::function<Tree(const std::vector<Point>&)>;

struct Block {
    int id = -1;
    int layer = -1;  // -1 for the dedicated source block x0
    int index = 0;   // index within the layer
    double r_inner = 0.0;
    double r_outer = 0.0;
    double theta_min = 0.0;  // [0, 2pi)
    double theta_max = 0.0;
    std::vector<int> pin_ids;

    int parent_id = -1;
    int left_id = -1;
    int right_id = -1;
    int left_child_id = -1;
    int right_child_id = -1;

    int merge_to = -1;      // parent in the block-level merge tree
    int local_source = -1;  // original pin index used as solver source

    bool nonempty() const { return !pin_ids.empty(); }
};

struct InterBlockLink {
    std::vector<int> pins;  // 2 (case 2) or 3 (case 1) original pin ids
    bool has_steiner = false;
    Point steiner;
};

struct PolarQuadtree {
    std::vector<Block> blocks;  // blocks[0] is always x0
    std::vector<std::vector<int>> layers;
    int x0_id = 0;
    double rho_max = 0.0;
    double default_search_bound = 0.0;
};

}  // namespace pqt
