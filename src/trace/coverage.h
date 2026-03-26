#pragma once

/// \file coverage.h
/// \brief Coverage validation and patching for vectorized output.

#include "detail/vectorized_shape.h"

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

void ApplyCoverageGuard(std::vector<VectorizedShape>& shapes, const cv::Mat& labels,
                        const std::vector<Rgb>& palette, float min_ratio, float tracing_epsilon,
                        float min_patch_area);

} // namespace neroued::vectorizer::detail
