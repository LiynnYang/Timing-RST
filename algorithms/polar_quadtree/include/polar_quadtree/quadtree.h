#pragma once

#include "polar_quadtree/types.h"

namespace pqt {

// Build a data-dependent polar quadtree. `translated` has the global source
// at the origin (index 0). Each leaf block contains at most `capacity` pins.
PolarQuadtree build_polar_quadtree(const std::vector<Point>& translated,
                                   int capacity);

}  // namespace pqt
