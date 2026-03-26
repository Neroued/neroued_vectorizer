#pragma once

#include <neroued/vectorizer/vectorizer.h>

#include <opencv2/core.hpp>

namespace neroued::vectorizer::detail {

VectorizerResult RunPipeline(const cv::Mat& bgr, const VectorizerConfig& cfg,
                             const cv::Mat& opaque_mask = cv::Mat());

VectorizerResult RunPipelineV2(const cv::Mat& bgr, const VectorizerConfig& cfg,
                               const cv::Mat& opaque_mask = cv::Mat());

} // namespace neroued::vectorizer::detail
