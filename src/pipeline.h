#pragma once

#include "detail/vectorized_shape.h"

#include <neroued/vectorizer/vectorizer.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace neroued::vectorizer::detail {

VectorizerResult RunPipeline(const cv::Mat& bgr, const VectorizerConfig& cfg,
                             const cv::Mat& opaque_mask = cv::Mat());

VectorizerResult RunPipelineV2(const cv::Mat& bgr, const VectorizerConfig& cfg,
                               const cv::Mat& opaque_mask = cv::Mat());

struct TraceParams {
    float trace_eps;
    int turdsize;
    double opttolerance;
};

inline TraceParams DeriveTraceParams(float contour_simplify) {
    TraceParams tp;
    tp.trace_eps    = std::max(0.2f, std::clamp(contour_simplify * 0.45f + 0.2f, 0.2f, 2.0f));
    tp.turdsize     = std::max(0, static_cast<int>(std::lround(tp.trace_eps * 0.5f)));
    tp.opttolerance = std::clamp(static_cast<double>(tp.trace_eps), 0.2, 2.0);
    return tp;
}

inline void RescaleShapes(std::vector<VectorizedShape>& shapes, float inv_scale) {
    for (auto& shape : shapes) {
        for (auto& contour : shape.contours) {
            for (auto& s : contour.segments) {
                s.p0 = s.p0 * inv_scale;
                s.p1 = s.p1 * inv_scale;
                s.p2 = s.p2 * inv_scale;
                s.p3 = s.p3 * inv_scale;
            }
        }
    }
}

inline void ClampShapesToBounds(std::vector<VectorizedShape>& shapes, float fw, float fh,
                                bool clamp_control_points) {
    for (auto& shape : shapes) {
        for (auto& contour : shape.contours) {
            for (auto& s : contour.segments) {
                s.p0 = {std::clamp(s.p0.x, 0.f, fw), std::clamp(s.p0.y, 0.f, fh)};
                s.p3 = {std::clamp(s.p3.x, 0.f, fw), std::clamp(s.p3.y, 0.f, fh)};
                if (clamp_control_points) {
                    s.p1 = {std::clamp(s.p1.x, 0.f, fw), std::clamp(s.p1.y, 0.f, fh)};
                    s.p2 = {std::clamp(s.p2.x, 0.f, fw), std::clamp(s.p2.y, 0.f, fh)};
                }
            }
        }
    }
}

} // namespace neroued::vectorizer::detail
