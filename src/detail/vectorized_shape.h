#pragma once

/// \file vectorized_shape.h
/// \brief Core VectorizedShape type used across vectorization modules.

#include "curve/bezier.h"
#include <neroued/vectorizer/color.h>

#include <vector>

namespace neroued::vectorizer::detail {

struct VectorizedShape {
    std::vector<BezierContour> contours;
    Rgb color;
    double area        = 0.0;
    bool is_stroke     = false;
    float stroke_width = 0.0f;

    bool operator==(const VectorizedShape& o) const = delete;
};

} // namespace neroued::vectorizer::detail
