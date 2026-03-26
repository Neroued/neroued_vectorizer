#include "shape_extend.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace neroued::vectorizer::detail {

void ExtendShapeMasks(std::vector<ShapeLayer>& layers, const std::vector<int>& depth_order,
                      cv::Size img_size, int dilate_iterations) {
    if (depth_order.size() <= 1 || dilate_iterations <= 0) return;

    cv::Mat kernel     = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    const int img_rows = img_size.height;
    const int img_cols = img_size.width;

    cv::Mat above_union   = cv::Mat::zeros(img_size, CV_8UC1);
    int above_union_count = 0;
    for (int idx : depth_order) {
        const auto& layer = layers[idx];
        cv::Mat roi       = above_union(layer.bbox);
        int before        = cv::countNonZero(roi);
        cv::bitwise_or(roi, layer.mask, roi);
        above_union_count += cv::countNonZero(roi) - before;
    }

    int total_extended_pixels = 0;
    const int pad             = dilate_iterations;

    for (int rank = 0; rank < static_cast<int>(depth_order.size()); ++rank) {
        int idx = depth_order[rank];

        {
            const auto& cur = layers[idx];
            cv::Mat roi     = above_union(cur.bbox);
            int before      = cv::countNonZero(roi);
            cv::Mat inv;
            cv::bitwise_not(cur.mask, inv);
            cv::bitwise_and(roi, inv, roi);
            above_union_count -= before - cv::countNonZero(roi);
        }

        if (above_union_count <= 0) continue;

        const auto& bbox = layers[idx].bbox;
        int ex0          = std::max(0, bbox.x - pad);
        int ey0          = std::max(0, bbox.y - pad);
        int ex1          = std::min(img_cols, bbox.x + bbox.width + pad);
        int ey1          = std::min(img_rows, bbox.y + bbox.height + pad);
        cv::Rect exp_roi(ex0, ey0, ex1 - ex0, ey1 - ey0);

        cv::Mat local_mask = cv::Mat::zeros(exp_roi.size(), CV_8UC1);
        {
            cv::Rect src_in_exp(bbox.x - ex0, bbox.y - ey0, bbox.width, bbox.height);
            layers[idx].mask.copyTo(local_mask(src_in_exp));
        }

        cv::Mat dilated;
        cv::dilate(local_mask, dilated, kernel, cv::Point(-1, -1), dilate_iterations);

        cv::Mat au_roi = above_union(exp_roi);
        cv::Mat extension;
        cv::bitwise_and(dilated, au_roi, extension);
        cv::Mat not_original;
        cv::bitwise_not(local_mask, not_original);
        cv::bitwise_and(extension, not_original, extension);

        int ext_pixels = cv::countNonZero(extension);
        if (ext_pixels > 0) {
            cv::bitwise_or(local_mask, extension, local_mask);
            cv::Rect local_nz = cv::boundingRect(local_mask);
            cv::Rect new_bbox(local_nz.x + ex0, local_nz.y + ey0, local_nz.width, local_nz.height);
            layers[idx].bbox = new_bbox;
            layers[idx].mask = local_mask(local_nz).clone();
            layers[idx].area = cv::countNonZero(layers[idx].mask);
            total_extended_pixels += ext_pixels;
        }
    }

    spdlog::info("ExtendShapeMasks: dilate_iterations={}, total_extended_pixels={}",
                 dilate_iterations, total_extended_pixels);
}

} // namespace neroued::vectorizer::detail
