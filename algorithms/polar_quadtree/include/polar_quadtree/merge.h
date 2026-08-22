#pragma once

#include "polar_quadtree/types.h"

#include <vector>

namespace pqt {

// Fill Block::merge_to so nonempty blocks form a tree rooted at x0.
void compute_merge_topology(PolarQuadtree& qt, double search_bound);

// Assign local sources and inter-block links (Fig. 9).
// `translated` is used for nearest-pin queries; `origin` maps Steiner points back.
void assign_sources_and_links(PolarQuadtree& qt,
                              const std::vector<Point>& translated,
                              const Point& origin,
                              std::vector<InterBlockLink>& links);

}  // namespace pqt
