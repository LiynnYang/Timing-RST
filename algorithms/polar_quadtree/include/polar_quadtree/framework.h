#pragma once

#include "polar_quadtree/types.h"

namespace pqt {

// Divide-and-merge wrapper around a local solver.
// points[0] is the global source.
// If search_bound < 0, a default of 2 * max ring thickness is used.
Tree divide_and_merge(const std::vector<Point>& points, const Solver& solver,
                      int capacity = 30, double search_bound = -1.0);

}  // namespace pqt
