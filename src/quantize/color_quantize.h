#pragma once

/// \file color_quantize.h
/// \brief Modified Median Cut Quantization (MMCQ) in OKLab color space.

#include <neroued/vectorizer/color.h>

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

struct QuantizeResult {
    cv::Mat labels;                     ///< CV_32SC1 per-pixel label map.
    std::vector<Rgb> palette;           ///< Quantized palette (linear sRGB).
    std::vector<cv::Vec3f> centers_lab; ///< Palette centers in CIE-Lab (for downstream merge).
};

/// Quantize image colors using Modified Median Cut in OKLab space.
///
/// \param bgr        Input BGR uint8 image.
/// \param num_colors Target palette size. If 0, automatically determine a good K
///                   via variance-based elbow detection.
/// \return Labels map, palette, and Lab-space centers.
QuantizeResult QuantizeColors(const cv::Mat& bgr, int num_colors);

} // namespace neroued::vectorizer::detail
