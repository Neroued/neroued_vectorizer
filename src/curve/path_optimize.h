#pragma once

/// \file path_optimize.h
/// \brief Two-pass Bezier path optimization: near-linear merging and adjacent segment re-fitting.

#include "bezier.h"
#include "detail/vectorized_shape.h"

#include <vector>

namespace neroued::vectorizer::detail {

/// Optimize a single BezierContour in-place.
///
/// Pass 1: Near-linear segments (where control points are close to the chord)
///         are merged/collapsed to reduce node count.
/// Pass 2: Adjacent cubic segments are tentatively merged by least-squares
///         re-fitting; if the error is below \p merge_eps the merge is kept.
///
/// \param contour    The contour to optimize (modified in-place).
/// \param linear_eps Max deviation from chord to qualify as near-linear (pixels).
/// \param merge_eps  Max re-fit error to accept a two-segment merge (pixels).
void OptimizeBezierContour(BezierContour& contour, float linear_eps, float merge_eps);

/// Optimize all paths in a set of shapes.
void OptimizeShapePaths(std::vector<VectorizedShape>& shapes, float linear_eps, float merge_eps);

} // namespace neroued::vectorizer::detail
