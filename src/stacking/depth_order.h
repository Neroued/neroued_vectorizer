#pragma once

/// \file depth_order.h
/// \brief Shape layer extraction and depth ordering for the stacking vectorization model.

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

/// A single shape layer: one connected component of a single color label.
/// The mask is cropped to the bounding box to minimize memory usage.
struct ShapeLayer {
    int label = -1;    ///< Original quantized color label.
    int cc_id = -1;    ///< Connected-component id within that label.
    cv::Rect bbox;     ///< Bounding rectangle in full-image coordinates.
    cv::Mat mask;      ///< CV_8UC1 binary mask cropped to bbox (255 = shape, 0 = background).
    double area = 0.0; ///< Pixel area.
};

/// Reconstruct a full-size mask from a ROI-cropped ShapeLayer.
inline cv::Mat FullSizeMask(const ShapeLayer& layer, cv::Size img_size) {
    cv::Mat full = cv::Mat::zeros(img_size, CV_8UC1);
    if (!layer.mask.empty() && layer.bbox.area() > 0) { layer.mask.copyTo(full(layer.bbox)); }
    return full;
}

/// Extract shape layers from a quantized label map.
/// Each connected component of each label becomes one ShapeLayer.
/// Labels with value < 0 (e.g. transparent) are skipped.
/// Connected components with area < \p min_area are discarded early to avoid
/// feeding thousands of tiny fragments into O(N²) depth ordering.
std::vector<ShapeLayer> ExtractShapeLayers(const cv::Mat& labels, int num_labels,
                                           double min_area = 1.0);

/// Compute a bottom-to-top depth ordering of shape layers.
///
/// Uses a hybrid approach:
///   1. Background identification (border-touching + largest area)
///   2. Covered-area energy D(i,j) for adjacent layer pairs
///   3. Directed graph + cycle removal + topological sort
///
/// \return Indices into \p layers, ordered from bottom (first) to top (last).
std::vector<int> ComputeDepthOrder(const std::vector<ShapeLayer>& layers, int img_rows,
                                   int img_cols);

} // namespace neroued::vectorizer::detail
