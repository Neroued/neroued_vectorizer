#include "pipeline.h"

#include "curve/bezier.h"
#include "curve/path_optimize.h"
#include "detail/cv_utils.h"
#include "output/svg_writer.h"
#include "preprocess/preprocess.h"
#include "quantize/color_quantize.h"
#include "segment/color_segment.h"
#include "stacking/depth_order.h"
#include "stacking/shape_extend.h"
#include "trace/coverage.h"
#include "trace/potrace.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace neroued::vectorizer::detail {

namespace {

/// z-order safe same-color shape merging.
/// Shapes that share the same color AND appear consecutively in the z-order
/// can be safely merged without altering the visual result.
void MergeSameColorShapesV2(std::vector<VectorizedShape>& shapes, float min_area) {
    if (shapes.size() <= 1) return;

    int before = static_cast<int>(shapes.size());

    // Filter out tiny fragments first.
    if (min_area > 0.f) {
        shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                                    [min_area](const VectorizedShape& s) {
                                        return s.area < static_cast<double>(min_area);
                                    }),
                     shapes.end());
    }

    // Merge consecutive same-color shapes.
    std::vector<VectorizedShape> merged;
    merged.reserve(shapes.size());

    for (size_t i = 0; i < shapes.size();) {
        merged.push_back(std::move(shapes[i]));
        auto& cur = merged.back();
        size_t j  = i + 1;

        auto color_key = [](const Rgb& c) -> uint64_t {
            uint8_t r8, g8, b8;
            c.ToRgb255(r8, g8, b8);
            return (static_cast<uint64_t>(r8) << 16) | (static_cast<uint64_t>(g8) << 8) | b8;
        };

        uint64_t cur_key = color_key(cur.color);
        while (j < shapes.size() && color_key(shapes[j].color) == cur_key) {
            for (auto& contour : shapes[j].contours) { cur.contours.push_back(std::move(contour)); }
            cur.area += shapes[j].area;
            ++j;
        }
        i = j;
    }

    shapes = std::move(merged);
    spdlog::info("MergeSameColorShapesV2: {} -> {} shapes (min_area={:.1f})", before, shapes.size(),
                 min_area);
}

} // namespace

VectorizerResult RunPipelineV2(const cv::Mat& bgr, const VectorizerConfig& cfg,
                               const cv::Mat& opaque_mask) {
    const auto pipeline_start = std::chrono::steady_clock::now();

    spdlog::info("RunPipelineV2 start: input={}x{}, num_colors={}{}", bgr.cols, bgr.rows,
                 cfg.num_colors, cfg.num_colors == 0 ? " (auto)" : "");

    // ── 1. Preprocess ───────────────────────────────────────────────────────
    auto preproc    = PreprocessForVectorize(bgr, true, cfg.smoothing_spatial, cfg.smoothing_color,
                                             cfg.upscale_short_edge, cfg.max_working_pixels);
    cv::Mat working = preproc.bgr;
    const float scale = preproc.scale;
    const bool scaled = std::abs(scale - 1.0f) > 1e-6f;

    cv::Mat working_mask = opaque_mask;
    if (scaled && !opaque_mask.empty()) {
        cv::resize(opaque_mask, working_mask, working.size(), 0, 0, cv::INTER_NEAREST);
    }

    spdlog::debug("V2 preprocess: working={}x{}, scale={:.3f}", working.cols, working.rows, scale);

    // ── 2. OKLab MMCQ color quantization ────────────────────────────────────
    auto qr                            = QuantizeColors(working, cfg.num_colors);
    cv::Mat labels                     = std::move(qr.labels);
    std::vector<cv::Vec3f> centers_lab = std::move(qr.centers_lab);
    int resolved_colors                = static_cast<int>(qr.palette.size());

    cv::Mat lab = BgrToLab(working);

    spdlog::debug("V2 MMCQ quantization: {} colors", resolved_colors);

    // ── 3. Apply transparency mask ──────────────────────────────────────────
    if (!working_mask.empty()) {
        cv::Mat transparent;
        cv::compare(working_mask, 0, transparent, cv::CMP_EQ);
        labels.setTo(cv::Scalar(-1), transparent);
    }
    working_mask.release();

    // ── 4. Small-region merge + compact labels ──────────────────────────────
    const cv::Size working_size(working.cols, working.rows);
    int effective_min_region = std::max(
        cfg.min_region_area, std::min(200, static_cast<int>(working_size.area() * 0.0005f)));

    MergeSmallComponents(labels, lab, centers_lab, std::max(2, effective_min_region),
                         cfg.max_merge_color_dist);
    lab.release();

    int num_labels = CompactLabels(labels, centers_lab);
    auto palette   = ComputePalette(working, labels, num_labels);
    working.release();

    spdlog::info("V2 labels compacted: num_labels={}, palette_size={}", num_labels, palette.size());

    // ── 5. Extract shape layers (connected components per label) ────────────
    auto layers = ExtractShapeLayers(labels, num_labels, cfg.min_contour_area);
    // labels kept alive for ApplyCoverageGuard; released after step 10b.
    spdlog::info("V2 shape layers: {}", layers.size());

    if (layers.empty()) {
        spdlog::error("RunPipelineV2: no shape layers extracted");
        throw std::runtime_error("RunPipelineV2: no shape layers");
    }

    // ── 6. Depth ordering ───────────────────────────────────────────────────
    auto depth_order = ComputeDepthOrder(layers, working_size.height, working_size.width);

    // ── 7. Shape extension (dilate into occluded regions) ───────────────────
    cv::Mat gt_labels;
    if (cfg.enable_depth_validation) {
        gt_labels.create(working_size, CV_32SC1);
        gt_labels.setTo(cv::Scalar(-1));
        for (const auto& layer : layers) {
            const auto& bbox = layer.bbox;
            const auto& mask = layer.mask;
            for (int r = 0; r < bbox.height; ++r) {
                const auto* mrow = mask.ptr<uint8_t>(r);
                auto* lrow       = gt_labels.ptr<int>(r + bbox.y);
                for (int c = 0; c < bbox.width; ++c) {
                    if (mrow[c] > 0) lrow[c + bbox.x] = layer.label;
                }
            }
        }
    }

    ExtendShapeMasks(layers, depth_order, working_size, 3);

    // ── 7b. Depth order validation (diagnostic, opt-in) ─────────────────────
    if (cfg.enable_depth_validation) {
        cv::Mat rendered(working_size, CV_32SC1, cv::Scalar(-1));
        for (int idx : depth_order) {
            const auto& layer = layers[idx];
            const auto& bbox  = layer.bbox;
            const auto& mask  = layer.mask;
            for (int r = 0; r < bbox.height; ++r) {
                const auto* mrow = mask.ptr<uint8_t>(r);
                auto* lrow       = rendered.ptr<int>(r + bbox.y);
                for (int c = 0; c < bbox.width; ++c) {
                    if (mrow[c] > 0) lrow[c + bbox.x] = layer.label;
                }
            }
        }

        int total_opaque = 0, mismatch = 0;
        const int val_rows = working_size.height;
        const int val_cols = working_size.width;
#pragma omp parallel for reduction(+ : total_opaque, mismatch) schedule(static)
        for (int r = 0; r < val_rows; ++r) {
            const int* gt_row = gt_labels.ptr<int>(r);
            const int* rd_row = rendered.ptr<int>(r);
            for (int c = 0; c < val_cols; ++c) {
                if (gt_row[c] < 0) continue;
                ++total_opaque;
                if (rd_row[c] != gt_row[c]) ++mismatch;
            }
        }

        float mismatch_rate = total_opaque > 0 ? static_cast<float>(mismatch) / total_opaque : 0.0f;
        if (mismatch_rate > 0.05f) {
            spdlog::warn("V2 depth order validation: mismatch_rate={:.4f} ({}/{} opaque pixels)",
                         mismatch_rate, mismatch, total_opaque);
        } else {
            spdlog::debug("V2 depth order validation: mismatch_rate={:.4f} ({}/{} opaque pixels)",
                          mismatch_rate, mismatch, total_opaque);
        }
        gt_labels.release();
    }

    // ── 8. Per-layer Potrace tracing ────────────────────────────────────────
    auto tp                   = DeriveTraceParams(cfg.contour_simplify);
    const float trace_eps     = tp.trace_eps;
    const int turdsize        = tp.turdsize;
    const double opttolerance = tp.opttolerance;

    const int depth_count = static_cast<int>(depth_order.size());
    std::vector<std::vector<VectorizedShape>> per_rank_shapes(depth_count);

#pragma omp parallel for schedule(dynamic)
    for (int rank = 0; rank < depth_count; ++rank) {
        int idx           = depth_order[rank];
        const auto& layer = layers[idx];
        if (layer.area < cfg.min_contour_area) continue;

        auto traced = TraceMaskWithPotraceBezier(layer.mask, turdsize, opttolerance);
        const Vec2f roi_offset(static_cast<float>(layer.bbox.x), static_cast<float>(layer.bbox.y));

        for (auto& g : traced) {
            if (g.area < static_cast<double>(cfg.min_contour_area)) continue;

            for (auto& seg : g.outer.segments) {
                seg.p0 = seg.p0 + roi_offset;
                seg.p1 = seg.p1 + roi_offset;
                seg.p2 = seg.p2 + roi_offset;
                seg.p3 = seg.p3 + roi_offset;
            }

            VectorizedShape shape;
            shape.color = palette[layer.label];
            shape.area  = g.area;
            shape.contours.push_back(std::move(g.outer));

            for (auto& hole : g.holes) {
                double hole_area = std::abs(BezierContourSignedArea(hole));
                if (hole_area < static_cast<double>(cfg.min_hole_area)) continue;
                for (auto& seg : hole.segments) {
                    seg.p0 = seg.p0 + roi_offset;
                    seg.p1 = seg.p1 + roi_offset;
                    seg.p2 = seg.p2 + roi_offset;
                    seg.p3 = seg.p3 + roi_offset;
                }
                hole.is_hole = true;
                shape.contours.push_back(std::move(hole));
            }

            if (!shape.contours.empty()) { per_rank_shapes[rank].push_back(std::move(shape)); }
        }
    }

    std::vector<VectorizedShape> shapes;
    shapes.reserve(layers.size());
    for (int rank = 0; rank < depth_count; ++rank) {
        for (auto& s : per_rank_shapes[rank]) shapes.push_back(std::move(s));
    }

    spdlog::info("V2 tracing done: {} shapes", shapes.size());

    // ── 9. Path optimization ───────────────────────────────────────────────
    {
        float linear_eps = std::max(0.3f, cfg.merge_segment_tolerance * 5.f);
        float merge_eps  = std::max(0.3f, cfg.curve_fit_error * 0.5f);
        OptimizeShapePaths(shapes, linear_eps, merge_eps);
    }

    // ── 10. z-order safe same-color merge + fragment filtering ──────────────
    MergeSameColorShapesV2(shapes, cfg.min_contour_area);

    // ── 10b. Coverage guard — patch uncovered pixels ────────────────────────
    if (cfg.enable_coverage_fix) {
        float min_ratio =
            (num_labels > 2) ? std::min(cfg.min_coverage_ratio, 0.995f) : cfg.min_coverage_ratio;
        float min_patch_area = std::max(1.f, cfg.min_contour_area * 0.5f);
        size_t pre_patch     = shapes.size();
        ApplyCoverageGuard(shapes, labels, palette, min_ratio, trace_eps, min_patch_area);
        size_t num_patches = shapes.size() - pre_patch;
        if (num_patches > 0) {
            // Rotate patches to bottom of z-order so they only show through
            // uncovered gaps, never painting over correct existing content.
            std::rotate(shapes.begin(), shapes.begin() + static_cast<ptrdiff_t>(pre_patch),
                        shapes.end());
        }
    }
    labels.release();

    if (scaled) { RescaleShapes(shapes, 1.0f / scale); }
    ClampShapesToBounds(shapes, static_cast<float>(bgr.cols), static_cast<float>(bgr.rows), false);

    // ── 12. Build result ────────────────────────────────────────────────────
    VectorizerResult result;
    result.width               = bgr.cols;
    result.height              = bgr.rows;
    result.num_shapes          = static_cast<int>(shapes.size());
    result.resolved_num_colors = resolved_colors;
    result.palette             = std::move(palette);
    result.svg_content =
        WriteSvg(shapes, bgr.cols, bgr.rows, cfg.svg_enable_stroke, cfg.svg_stroke_width);

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pipeline_start)
            .count();
    spdlog::info("RunPipelineV2 completed: elapsed_ms={:.2f}, shapes={}, svg_bytes={}", elapsed_ms,
                 result.num_shapes, result.svg_content.size());
    return result;
}

} // namespace neroued::vectorizer::detail
