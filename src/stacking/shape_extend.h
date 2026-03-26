#pragma once

/// \file shape_extend.h
/// \brief Morphological shape extension into occluded regions for gap-free stacking SVG.

#include "depth_order.h"

#include <vector>

namespace neroued::vectorizer::detail {

/// Extend shape masks into regions occluded by higher layers.
///
/// For each layer (processed bottom-to-top), the mask is dilated and the
/// extension is clipped to the union of all layers above it.  This ensures
/// that lower shapes extend under upper shapes, eliminating gaps without
/// altering the visible result.
///
/// \param layers           Shape layers whose masks will be modified in-place.
/// \param depth_order      Bottom-to-top index order (from ComputeDepthOrder).
/// \param img_size         Full image dimensions (needed to reconstruct full-size masks).
/// \param dilate_iterations Number of 3x3 disk-kernel dilation iterations.
void ExtendShapeMasks(std::vector<ShapeLayer>& layers, const std::vector<int>& depth_order,
                      cv::Size img_size, int dilate_iterations = 3);

} // namespace neroued::vectorizer::detail
