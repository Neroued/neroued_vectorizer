#pragma once

/// \file shape_merge.h
/// \brief Shape bounding-box computation and Z-order-preserving same-color merging.

#include "detail/vectorized_shape.h"

#include <neroued/vectorizer/vec2.h>

#include <limits>
#include <vector>

namespace neroued::vectorizer::detail {

struct BBox {
    float xmin = std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymax = std::numeric_limits<float>::lowest();

    void Expand(const Vec2f& p) {
        xmin = std::min(xmin, p.x);
        ymin = std::min(ymin, p.y);
        xmax = std::max(xmax, p.x);
        ymax = std::max(ymax, p.y);
    }

    bool Overlaps(const BBox& o) const {
        return xmin < o.xmax && xmax > o.xmin && ymin < o.ymax && ymax > o.ymin;
    }
};

BBox ComputeShapeBBox(const VectorizedShape& s);

/// Merge consecutive runs of same-color fill shapes whose bboxes are
/// pairwise non-overlapping. Must be called AFTER shapes are sorted by area
/// (painter's algorithm) so that Z-order is preserved exactly.
void MergeAdjacentSameColorShapes(std::vector<VectorizedShape>& shapes);

} // namespace neroued::vectorizer::detail
