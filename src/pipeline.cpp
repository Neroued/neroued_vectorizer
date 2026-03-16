#include "pipeline.h"

#include "boundary/aa_detector.h"
#include "boundary/boundary_graph.h"
#include "boundary/subpixel_refine.h"
#include "contour/assembly.h"
#include "contour/thin_line.h"
#include "curve/bezier.h"
#include "detail/cv_utils.h"
#include "output/shape_merge.h"
#include "output/svg_writer.h"
#include "preprocess/preprocess.h"
#include "segment/color_segment.h"
#include "segment/morphology.h"
#include "segment/slic.h"
#include "trace/coverage.h"
#include "trace/potrace.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace neroued::vectorizer::detail {

VectorizerResult RunPipeline(const cv::Mat& bgr, const VectorizerConfig& cfg,
                             const cv::Mat& opaque_mask) {
    const auto pipeline_start = std::chrono::steady_clock::now();
    int resolved_colors       = (cfg.num_colors == 0) ? EstimateOptimalColors(bgr) : cfg.num_colors;

    const int short_edge = std::min(bgr.cols, bgr.rows);
    const bool is_small  = short_edge <= 128;

    float adaptive_smoothing_spatial = cfg.smoothing_spatial;
    float adaptive_smoothing_color   = cfg.smoothing_color;
    int adaptive_min_region          = cfg.min_region_area;
    int adaptive_upscale_short_edge  = cfg.upscale_short_edge;

    if (is_small) {
        if (cfg.num_colors == 0) resolved_colors = std::min(resolved_colors, 8);
        adaptive_smoothing_spatial  = std::min(cfg.smoothing_spatial, 8.0f);
        adaptive_smoothing_color    = std::min(cfg.smoothing_color, 15.0f);
        adaptive_min_region         = std::max(cfg.min_region_area, short_edge * short_edge / 50);
        adaptive_upscale_short_edge = std::min(cfg.upscale_short_edge, 400);
        spdlog::info("Small image adaptation: short_edge={}, colors={}, smoothing=({:.0f},{:.0f}), "
                     "min_region={}, upscale_target={}",
                     short_edge, resolved_colors, adaptive_smoothing_spatial,
                     adaptive_smoothing_color, adaptive_min_region, adaptive_upscale_short_edge);
    }

    spdlog::info("RunPipeline start: input={}x{}, num_colors={}{}, "
                 "min_region_area={}, curve_fit_error={:.2f}, contour_simplify={:.2f}, "
                 "svg_stroke={}, coverage_fix={}, max_working_pixels={}",
                 bgr.cols, bgr.rows, resolved_colors, cfg.num_colors == 0 ? " (auto)" : "",
                 adaptive_min_region, cfg.curve_fit_error, cfg.contour_simplify,
                 cfg.svg_enable_stroke, cfg.enable_coverage_fix, cfg.max_working_pixels);
    const bool multicolor = resolved_colors > 2;
    auto preproc          = PreprocessForVectorize(bgr, multicolor, adaptive_smoothing_spatial,
                                                   adaptive_smoothing_color, adaptive_upscale_short_edge,
                                                   cfg.max_working_pixels);
    cv::Mat working       = preproc.bgr;
    cv::Mat unsmoothed    = preproc.unsmoothed_bgr;
    const float scale     = preproc.scale;
    const bool scaled     = std::abs(scale - 1.0f) > 1e-6f;

    cv::Mat working_mask = opaque_mask;
    if (scaled && !opaque_mask.empty()) {
        cv::resize(opaque_mask, working_mask, working.size(), 0, 0, cv::INTER_NEAREST);
    }
    spdlog::debug("Vectorize preprocess done: working={}x{}, scale={:.3f}, mask_present={}",
                  working.cols, working.rows, scale, !working_mask.empty());

    cv::Mat edge_map;
    if (multicolor && !unsmoothed.empty()) {
        edge_map = ComputeEdgeMap(unsmoothed);
        spdlog::debug("Vectorize edge map computed: size={}x{}", edge_map.cols, edge_map.rows);
    }

    cv::Mat lab = BgrToLab(working);
    SegmentationResult seg =
        multicolor ? SegmentMultiColor(lab, resolved_colors, cfg.slic_region_size,
                                       cfg.slic_compactness, edge_map, cfg.edge_sensitivity)
                   : SegmentBinary(working, lab);
    if (seg.labels.empty()) {
        spdlog::error("Vectorize segmentation failed: empty labels");
        throw std::runtime_error("RunPipeline: segmentation failed");
    }
    spdlog::info("Vectorize segmentation completed: mode={}, centers={}, label_map={}x{}",
                 multicolor ? "multicolor" : "binary", seg.centers_lab.size(), seg.labels.cols,
                 seg.labels.rows);

    if (!working_mask.empty()) {
        if (working_mask.type() != CV_8UC1 || working_mask.size() != seg.labels.size()) {
            spdlog::error(
                "Vectorize mask invalid: expected type=CV_8UC1 size={}x{}, got type={} size={}x{}",
                seg.labels.cols, seg.labels.rows, working_mask.type(), working_mask.cols,
                working_mask.rows);
            throw std::runtime_error("RunPipeline: invalid opaque mask");
        }
        cv::Mat transparent;
        cv::compare(working_mask, 0, transparent, cv::CMP_EQ);
        const int transparent_px = cv::countNonZero(transparent);
        seg.labels.setTo(cv::Scalar(-1), transparent);
        spdlog::debug("Vectorize transparent mask applied: transparent_pixels={}", transparent_px);
    }

    cv::Mat unsmoothed_lab;
    if (multicolor && !unsmoothed.empty()) { unsmoothed_lab = BgrToLab(unsmoothed); }

    if (multicolor && cfg.refine_passes > 0 && !unsmoothed_lab.empty() &&
        !seg.centers_lab.empty()) {
        RefineLabelsBoundary(seg.labels, unsmoothed_lab, seg.centers_lab, cfg.refine_passes);
        spdlog::info("Vectorize label refinement applied: passes={}", cfg.refine_passes);
    }

    int area_proportional_min =
        std::min(200, static_cast<int>(working.rows * working.cols * 0.0005f));
    int effective_min_region = std::max(adaptive_min_region, area_proportional_min);
    spdlog::debug("MergeSmallComponents min_region: cfg={}, adaptive={}, proportional={}, "
                  "effective={}",
                  cfg.min_region_area, adaptive_min_region, area_proportional_min,
                  effective_min_region);
    MergeSmallComponents(seg.labels, seg.lab, seg.centers_lab, std::max(2, effective_min_region),
                         cfg.max_merge_color_dist);
    if (multicolor) {
        MorphologicalCleanup(seg.labels, static_cast<int>(seg.centers_lab.size()), 1);
    }
    int num_labels = CompactLabels(seg.labels, seg.centers_lab);
    auto palette   = ComputePalette(working, seg.labels, num_labels);
    spdlog::info("Vectorize labels compacted: num_labels={}, palette_size={}", num_labels,
                 palette.size());

    float effective_curve_fit_error  = cfg.curve_fit_error;
    float effective_contour_simplify = cfg.contour_simplify;
    if (cfg.detail_level >= 0.0f) {
        static const VectorizerConfig kDefaults;
        float dl = std::clamp(cfg.detail_level, 0.0f, 1.0f);
        if (cfg.curve_fit_error == kDefaults.curve_fit_error)
            effective_curve_fit_error = 2.0f - 1.7f * dl;
        if (cfg.contour_simplify == kDefaults.contour_simplify)
            effective_contour_simplify = 0.8f - 0.6f * dl;
        spdlog::info("detail_level={:.2f}: derived curve_fit_error={:.2f}, "
                     "contour_simplify={:.2f}",
                     dl, effective_curve_fit_error, effective_contour_simplify);
    }

    const float trace_eps =
        std::max(0.2f, std::clamp(effective_contour_simplify * 0.45f + 0.2f, 0.2f, 2.0f));
    const int turdsize        = std::max(0, static_cast<int>(std::lround(trace_eps * 0.5f)));
    const double opttolerance = std::clamp(static_cast<double>(trace_eps), 0.2, 2.0);
    spdlog::debug("Vectorize trace params: trace_eps={:.3f}, turdsize={}, opttolerance={:.3f}",
                  trace_eps, turdsize, opttolerance);

    std::vector<VectorizedShape> shapes;

    if (multicolor && num_labels > 2) {
        spdlog::info("Vectorize contour mode: BoundaryGraph+CurveFit");
        auto boundary_graph = BuildBoundaryGraph(seg.labels);
        spdlog::debug("BoundaryGraph built: nodes={}, edges={}", boundary_graph.nodes.size(),
                      boundary_graph.edges.size());

        if (cfg.enable_subpixel_refine) {
            const cv::Mat& refine_lab =
                unsmoothed_lab.empty() ? (unsmoothed_lab = BgrToLab(working)) : unsmoothed_lab;
            SubpixelRefineConfig sp_cfg;
            sp_cfg.max_displacement = cfg.subpixel_max_displacement;

            if (cfg.enable_antialias_detect) {
                AADetectConfig aa_cfg;
                aa_cfg.tolerance = cfg.aa_tolerance;
                auto aa_map      = DetectAAPixels(refine_lab, seg.labels, seg.centers_lab, aa_cfg);
                RefineEdgesSubpixelAA(boundary_graph, refine_lab, aa_map.is_aa, aa_map.alpha,
                                      sp_cfg);
            } else {
                RefineEdgesSubpixel(boundary_graph, refine_lab, sp_cfg);
            }
        }

        CurveFitConfig fit_cfg;
        fit_cfg.error_threshold            = std::clamp(effective_curve_fit_error, 0.05f, 10.0f);
        fit_cfg.corner_angle_threshold_deg = std::clamp(cfg.corner_angle_threshold, 60.0f, 179.0f);
        auto smooth_cfg                    = ContourSmoothFromLevel(cfg.smoothness);
        shapes = AssembleContoursFromGraph(boundary_graph, num_labels, palette,
                                           cfg.min_contour_area, cfg.min_hole_area, &fit_cfg,
                                           smooth_cfg, cfg.merge_segment_tolerance);
        spdlog::info("BoundaryGraph contour assembly done: shapes={}", shapes.size());

        std::vector<double> label_pixel_count(num_labels, 0.0);
        std::vector<double> label_shape_area(num_labels, 0.0);
        for (int r = 0; r < seg.labels.rows; ++r) {
            const int* lrow = seg.labels.ptr<int>(r);
            for (int c = 0; c < seg.labels.cols; ++c) {
                int lid = lrow[c];
                if (lid >= 0 && lid < num_labels) label_pixel_count[lid] += 1.0;
            }
        }
        constexpr float kCoverageColorThreshold = 5.0f;
        std::vector<Lab> palette_lab(num_labels);
        for (int rid = 0; rid < num_labels; ++rid) palette_lab[rid] = palette[rid].ToLab();

        for (const auto& s : shapes) {
            if (s.is_stroke) continue;
            Lab shape_lab = s.color.ToLab();
            for (int rid = 0; rid < num_labels; ++rid) {
                if (Lab::DeltaE76(shape_lab, palette_lab[rid]) < kCoverageColorThreshold) {
                    label_shape_area[rid] += s.area;
                }
            }
        }
        std::vector<bool> label_covered(num_labels, false);
        for (int rid = 0; rid < num_labels; ++rid) {
            if (label_pixel_count[rid] < 1.0) {
                label_covered[rid] = true;
            } else if (label_shape_area[rid] > label_pixel_count[rid] * 0.3) {
                label_covered[rid] = true;
            }
        }
        int uncovered_labels = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            if (!label_covered[rid]) ++uncovered_labels;
        }
        if (uncovered_labels > 0) {
            spdlog::warn(
                "Vectorize fallback triggered: uncovered_labels={} (BoundaryGraph -> Potrace)",
                uncovered_labels);
        }
        int fallback_labels     = 0;
        int fallback_shapes_add = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            if (label_covered[rid]) continue;
            ++fallback_labels;
            cv::Mat mask = (seg.labels == rid);
            mask.convertTo(mask, CV_8UC1, 255);
            if (cv::countNonZero(mask) <= 0) continue;

            auto traced = TraceMaskWithPotraceBezier(mask, turdsize, opttolerance);
            for (auto& g : traced) {
                if (g.area < static_cast<double>(cfg.min_contour_area)) continue;
                VectorizedShape shape;
                shape.color = palette[rid];
                shape.area  = g.area;
                shape.contours.push_back(std::move(g.outer));
                for (auto& hole : g.holes) {
                    double hole_area = std::abs(BezierContourSignedArea(hole));
                    if (hole_area < static_cast<double>(cfg.min_hole_area)) continue;
                    shape.contours.push_back(std::move(hole));
                }
                if (!shape.contours.empty()) {
                    shapes.push_back(std::move(shape));
                    ++fallback_shapes_add;
                }
            }
        }
        if (fallback_labels > 0) {
            spdlog::info("Vectorize fallback completed: labels={}, shapes_added={}",
                         fallback_labels, fallback_shapes_add);
        }
    } else {
        spdlog::info("Vectorize contour mode: per-label Potrace");
        int labels_traced       = 0;
        int dilate_retry_count  = 0;
        int direct_shapes_added = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            cv::Mat mask = (seg.labels == rid);
            mask.convertTo(mask, CV_8UC1, 255);
            int px = cv::countNonZero(mask);
            if (px <= 0) continue;
            ++labels_traced;

            auto traced = TraceMaskWithPotraceBezier(mask, turdsize, opttolerance);
            if (traced.empty()) {
                ++dilate_retry_count;
                cv::Mat dilated;
                cv::dilate(mask, dilated,
                           cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
                traced = TraceMaskWithPotraceBezier(dilated, std::max(0, turdsize - 1),
                                                    std::max(0.2, opttolerance * 0.8));
            }

            for (auto& g : traced) {
                if (g.area < static_cast<double>(cfg.min_contour_area)) continue;
                VectorizedShape shape;
                shape.color = palette[rid];
                shape.area  = g.area;
                shape.contours.push_back(std::move(g.outer));
                for (auto& hole : g.holes) {
                    double hole_area = std::abs(BezierContourSignedArea(hole));
                    if (hole_area < static_cast<double>(cfg.min_hole_area)) continue;
                    shape.contours.push_back(std::move(hole));
                }
                if (!shape.contours.empty()) {
                    shapes.push_back(std::move(shape));
                    ++direct_shapes_added;
                }
            }
        }
        if (dilate_retry_count > 0) {
            spdlog::warn("Vectorize Potrace retry with dilation: labels_retried={}",
                         dilate_retry_count);
        }
        spdlog::info("Vectorize per-label Potrace done: labels_traced={}, shapes_added={}",
                     labels_traced, direct_shapes_added);
    }

    if (cfg.svg_enable_stroke && multicolor && num_labels > 1) {
        const int fill_count       = static_cast<int>(shapes.size());
        const int max_stroke_count = fill_count * 2;

        const int working_short_edge = std::min(working.cols, working.rows);
        const float adaptive_thin_radius =
            (working_short_edge <= 400) ? std::clamp(cfg.thin_line_max_radius * 0.6f, 0.1f, 50.0f)
                                        : std::clamp(cfg.thin_line_max_radius, 0.1f, 50.0f);

        int labels_with_thin = 0;
        int stroke_added     = 0;
        for (int rid = 0; rid < num_labels && stroke_added < max_stroke_count; ++rid) {
            cv::Mat mask = (seg.labels == rid);
            mask.convertTo(mask, CV_8UC1, 255);
            if (cv::countNonZero(mask) <= 0) continue;

            cv::Mat thin = DetectThinRegion(mask, adaptive_thin_radius);
            if (cv::countNonZero(thin) < 3) continue;
            ++labels_with_thin;

            cv::Mat dist;
            cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
            cv::Mat skel = ZhangSuenThinning(thin);
            if (cv::countNonZero(skel) < 3) continue;

            auto strokes = ExtractStrokePaths(skel, dist, palette[rid], 3.0f);
            for (auto& s : strokes) {
                if (stroke_added >= max_stroke_count) break;
                shapes.push_back(std::move(s));
                ++stroke_added;
            }
        }
        spdlog::info("Vectorize thin-line enhancement: labels={}, strokes_added={} (cap={})",
                     labels_with_thin, stroke_added, max_stroke_count);
    } else if (cfg.svg_enable_stroke) {
        spdlog::debug("Vectorize thin-line enhancement skipped: multicolor={}, num_labels={}",
                      multicolor, num_labels);
    }

    std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) {
        if (a.is_stroke != b.is_stroke) return !a.is_stroke;
        return a.area > b.area;
    });

    MergeAdjacentSameColorShapes(shapes);

    if (cfg.enable_coverage_fix) {
        float effective_coverage_ratio = cfg.min_coverage_ratio;
        if (multicolor && num_labels > 2) {
            effective_coverage_ratio = std::min(cfg.min_coverage_ratio, 0.995f);
        }
        const auto before = shapes.size();
        ApplyCoverageGuard(shapes, seg.labels, palette, effective_coverage_ratio, trace_eps,
                           std::max(1.0f, cfg.min_contour_area * 0.5f));
        const auto added = shapes.size() >= before ? (shapes.size() - before) : 0;
        spdlog::info("Vectorize coverage guard applied: added_shapes={}", added);
    }

    if (scaled) {
        const float inv = 1.0f / scale;
        for (auto& shape : shapes) {
            for (auto& contour : shape.contours) {
                for (auto& s : contour.segments) {
                    s.p0 = s.p0 * inv;
                    s.p1 = s.p1 * inv;
                    s.p2 = s.p2 * inv;
                    s.p3 = s.p3 * inv;
                }
            }
        }
        spdlog::debug("Vectorize output rescaled by inverse factor={:.4f}", inv);
    }

    const float fw = static_cast<float>(bgr.cols);
    const float fh = static_cast<float>(bgr.rows);
    for (auto& shape : shapes) {
        for (auto& contour : shape.contours) {
            for (auto& s : contour.segments) {
                s.p0 = {std::clamp(s.p0.x, 0.f, fw), std::clamp(s.p0.y, 0.f, fh)};
                s.p1 = {std::clamp(s.p1.x, 0.f, fw), std::clamp(s.p1.y, 0.f, fh)};
                s.p2 = {std::clamp(s.p2.x, 0.f, fw), std::clamp(s.p2.y, 0.f, fh)};
                s.p3 = {std::clamp(s.p3.x, 0.f, fw), std::clamp(s.p3.y, 0.f, fh)};
            }
        }
    }

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
    spdlog::info("RunPipeline completed: elapsed_ms={:.2f}, width={}, height={}, "
                 "num_shapes={}, palette_size={}, svg_bytes={}",
                 elapsed_ms, result.width, result.height, result.num_shapes, result.palette.size(),
                 result.svg_content.size());
    return result;
}

} // namespace neroued::vectorizer::detail
