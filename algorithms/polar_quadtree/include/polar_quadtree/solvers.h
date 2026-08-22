#pragma once

#include "polar_quadtree/types.h"

namespace pqt {

// Placeholder local solver: rectilinear minimum spanning tree (Manhattan).
// Input points are kept as the first nodes of the returned tree.
Tree rectilinear_mst(const std::vector<Point>& points);

}  // namespace pqt
