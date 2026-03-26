#include "shape_extend.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace neroued::vectorizer::detail {

void ExtendShapeMasks(std::vector<ShapeLayer>& layers, const std::vector<int>& depth_order,
                      cv::Size img_size, int dilate_iterations) {
    if (depth_order.size() <= 1 || dilate_iterations <= 0) return;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // Pre-build union of all layers; then subtract layer-by-layer as we descend.
    // Layers are disjoint before extension, so subtraction is pixel-exact.
    cv::Mat above_union = cv::Mat::zeros(img_size, CV_8UC1);
    for (int idx : depth_order) {
        const auto& layer = layers[idx];
        cv::Mat roi       = above_union(layer.bbox);
        cv::bitwise_or(roi, layer.mask, roi);
    }

    int total_extended_pixels = 0;

    for (int rank = 0; rank < static_cast<int>(depth_order.size()); ++rank) {
        int idx = depth_order[rank];

        {
            const auto& cur = layers[idx];
            cv::Mat roi     = above_union(cur.bbox);
            cv::Mat inv;
            cv::bitwise_not(cur.mask, inv);
            cv::bitwise_and(roi, inv, roi);
        }

        if (cv::countNonZero(above_union) == 0) continue;
        cv::Mat& occluder_union = above_union;

        cv::Mat full_mask = FullSizeMask(layers[idx], img_size);

        cv::Mat dilated;
        cv::dilate(full_mask, dilated, kernel, cv::Point(-1, -1), dilate_iterations);

        cv::Mat extension;
        cv::bitwise_and(dilated, occluder_union, extension);
        cv::Mat not_original;
        cv::bitwise_not(full_mask, not_original);
        cv::bitwise_and(extension, not_original, extension);

        int ext_pixels = cv::countNonZero(extension);
        if (ext_pixels > 0) {
            cv::bitwise_or(full_mask, extension, full_mask);
            cv::Rect new_bbox = cv::boundingRect(full_mask);
            layers[idx].bbox  = new_bbox;
            layers[idx].mask  = full_mask(new_bbox).clone();
            layers[idx].area  = cv::countNonZero(layers[idx].mask);
            total_extended_pixels += ext_pixels;
        }
    }

    spdlog::info("ExtendShapeMasks: dilate_iterations={}, total_extended_pixels={}",
                 dilate_iterations, total_extended_pixels);
}

} // namespace neroued::vectorizer::detail
