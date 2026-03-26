#pragma once

/// \file potrace.h
/// \brief Mask-to-polygon tracing adapter (Potrace-style interface).

#include "curve/bezier.h"
#include <neroued/vectorizer/vec2.h>

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

struct TracedPolygonGroup {
    std::vector<Vec2f> outer;
    std::vector<std::vector<Vec2f>> holes;
    double area = 0.0;
};

/// Trace a binary mask into polygon groups with Potrace (legacy polygon path).
std::vector<TracedPolygonGroup> TraceMaskWithPotrace(const cv::Mat& mask, float simplify_epsilon);

double SignedArea(const std::vector<Vec2f>& ring);

struct TracedBezierGroup {
    BezierContour outer;
    std::vector<BezierContour> holes;
    double area = 0.0;
};

/// Trace a binary mask preserving Potrace's native cubic Bezier curves and path hierarchy.
std::vector<TracedBezierGroup> TraceMaskWithPotraceBezier(const cv::Mat& mask, int turdsize = 2,
                                                          double opttolerance = 0.2);

/// Convert a polygon ring to a degenerate linear BezierContour.
BezierContour RingToBezier(const std::vector<Vec2f>& ring);

} // namespace neroued::vectorizer::detail
