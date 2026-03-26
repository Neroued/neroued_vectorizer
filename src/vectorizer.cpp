#include <neroued/vectorizer/vectorizer.h>

#include "detail/cv_utils.h"
#include "detail/icc_utils.h"
#include "pipeline.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <stdexcept>

namespace neroued::vectorizer {

namespace {

struct PreparedVectorizeInput {
    cv::Mat bgr;
    cv::Mat opaque_mask;
};

PreparedVectorizeInput PrepareVectorizeInput(const cv::Mat& input) {
    PreparedVectorizeInput prepared;
    if (input.empty()) {
        spdlog::warn("Vectorize input prepare: empty input");
        return prepared;
    }

    if (input.channels() == 4) {
        prepared.opaque_mask    = detail::ExtractOpaqueMask(input, 0);
        prepared.bgr            = detail::EnsureBgr(input, detail::BgraPolicy::DropAlpha);
        const int opaque_pixels = cv::countNonZero(prepared.opaque_mask);
        const int total_pixels  = prepared.opaque_mask.rows * prepared.opaque_mask.cols;
        spdlog::debug("Vectorize input prepare: BGRA {}x{}, opaque_pixels={}, opaque_ratio={:.4f}",
                      input.cols, input.rows, opaque_pixels,
                      total_pixels > 0 ? static_cast<double>(opaque_pixels) / total_pixels : 0.0);
        return prepared;
    }

    prepared.bgr = detail::EnsureBgr(input);
    spdlog::debug("Vectorize input prepare: channels={} -> BGR {}x{}", input.channels(),
                  prepared.bgr.cols, prepared.bgr.rows);
    return prepared;
}

VectorizerResult VectorizeImpl(PreparedVectorizeInput& prepared, const VectorizerConfig& config,
                               const char* tag) {
    if (prepared.bgr.empty()) {
        spdlog::error("Vectorize({}) failed: prepared image empty", tag);
        throw std::runtime_error(std::string("Vectorize(") + tag + "): empty input");
    }
    return (config.pipeline_mode == PipelineMode::V2)
               ? detail::RunPipelineV2(prepared.bgr, config, prepared.opaque_mask)
               : detail::RunPipeline(prepared.bgr, config, prepared.opaque_mask);
}

} // namespace

VectorizerResult Vectorize(const std::string& image_path, const VectorizerConfig& config) {
    const auto start = std::chrono::steady_clock::now();
    spdlog::info("Vectorize(file) start: path='{}'", image_path);
    try {
        cv::Mat img   = detail::LoadImageIcc(image_path);
        auto prepared = PrepareVectorizeInput(img);
        auto result   = VectorizeImpl(prepared, config, "file");
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        spdlog::info(
            "Vectorize(file) completed: path='{}', elapsed_ms={:.2f}, width={}, height={}, "
            "num_shapes={}, svg_bytes={}",
            image_path, elapsed_ms, result.width, result.height, result.num_shapes,
            result.svg_content.size());
        return result;
    } catch (const std::exception& e) {
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        spdlog::error("Vectorize(file) error: path='{}', elapsed_ms={:.2f}, error={}", image_path,
                      elapsed_ms, e.what());
        throw;
    }
}

VectorizerResult Vectorize(const uint8_t* image_data, size_t image_size,
                           const VectorizerConfig& config) {
    const auto start = std::chrono::steady_clock::now();
    spdlog::info("Vectorize(buffer) start: bytes={}", image_size);
    try {
        cv::Mat img   = detail::LoadImageIcc(image_data, image_size);
        auto prepared = PrepareVectorizeInput(img);
        auto result   = VectorizeImpl(prepared, config, "buffer");
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        spdlog::info(
            "Vectorize(buffer) completed: bytes={}, elapsed_ms={:.2f}, width={}, height={}, "
            "num_shapes={}, svg_bytes={}",
            image_size, elapsed_ms, result.width, result.height, result.num_shapes,
            result.svg_content.size());
        return result;
    } catch (const std::exception& e) {
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        spdlog::error("Vectorize(buffer) error: bytes={}, elapsed_ms={:.2f}, error={}", image_size,
                      elapsed_ms, e.what());
        throw;
    }
}

VectorizerResult Vectorize(const cv::Mat& bgr_image, const VectorizerConfig& config) {
    const auto start = std::chrono::steady_clock::now();
    spdlog::info("Vectorize(mat) start: width={}, height={}, channels={}", bgr_image.cols,
                 bgr_image.rows, bgr_image.channels());
    try {
        auto prepared = PrepareVectorizeInput(bgr_image);
        auto result   = VectorizeImpl(prepared, config, "mat");
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        spdlog::info(
            "Vectorize(mat) completed: elapsed_ms={:.2f}, width={}, height={}, num_shapes={}, "
            "svg_bytes={}",
            elapsed_ms, result.width, result.height, result.num_shapes, result.svg_content.size());
        return result;
    } catch (const std::exception& e) {
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        spdlog::error("Vectorize(mat) error: elapsed_ms={:.2f}, error={}", elapsed_ms, e.what());
        throw;
    }
}

} // namespace neroued::vectorizer
