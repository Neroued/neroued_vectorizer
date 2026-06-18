#include "coverage.h"

#include "curve/bezier.h"
#include "potrace.h"
#include "topology.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace neroued::vectorizer::detail {

namespace {

std::vector<cv::Point> FlattenContour(const BezierContour& contour, int width, int height) {
    std::vector<cv::Point> poly;
    if (contour.segments.empty()) return poly;
    std::vector<Vec2f> pts;
    pts.reserve(contour.segments.size() * 8 + 1);
    pts.push_back(contour.segments.front().p0);
    for (const auto& seg : contour.segments) FlattenCubicBezier(seg, 0.45f, pts);

    poly.reserve(pts.size());
    for (const auto& p : pts) {
        int x = std::clamp(static_cast<int>(std::lround(p.x)), 0, width - 1);
        int y = std::clamp(static_cast<int>(std::lround(p.y)), 0, height - 1);
        poly.emplace_back(x, y);
    }
    if (poly.size() > 1 && poly.front() == poly.back()) poly.pop_back();
    return poly;
}

cv::Mat RasterizeCoverage(const std::vector<VectorizedShape>& shapes, int width, int height) {
    cv::Mat coverage(height, width, CV_8UC1, cv::Scalar(0));
    for (const auto& shape : shapes) {
        std::vector<std::vector<cv::Point>> polys;
        for (const auto& contour : shape.contours) {
            auto poly = FlattenContour(contour, width, height);
            if (poly.size() >= 3) polys.push_back(std::move(poly));
        }
        if (!polys.empty()) cv::fillPoly(coverage, polys, cv::Scalar(255));
    }
    return coverage;
}

struct GapInfo {
    cv::Mat coverage;
    cv::Mat cc_labels;
    int ncc                   = 0;
    int source_px             = 0;
    int covered_px            = 0;
    int max_missing_component = 0;
    float ratio               = 0.0f;
};

bool FindCoverageGaps(const std::vector<VectorizedShape>& shapes, const cv::Mat& labels,
                      float min_ratio, float max_unpatched_gap_area, int w, int h, GapInfo& out) {
    cv::Mat source_mask(h, w, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < h; ++r) {
        const int* row = labels.ptr<int>(r);
        uint8_t* mout  = source_mask.ptr<uint8_t>(r);
        for (int c = 0; c < w; ++c) mout[c] = (row[c] >= 0) ? 255 : 0;
    }

    out.coverage = RasterizeCoverage(shapes, w, h);
    cv::Mat covered;
    cv::bitwise_and(source_mask, out.coverage, covered);

    out.source_px  = cv::countNonZero(source_mask);
    out.covered_px = cv::countNonZero(covered);
    if (out.source_px <= 0) {
        spdlog::debug("CoverageGuard skipped: source pixels are zero");
        return false;
    }

    out.ratio = static_cast<float>(out.covered_px) / static_cast<float>(out.source_px);

    cv::Mat missing;
    cv::bitwise_not(out.coverage, missing);
    cv::bitwise_and(missing, source_mask, missing);

    cv::Mat stats;
    cv::Mat centroids;
    out.ncc = cv::connectedComponentsWithStats(missing, out.cc_labels, stats, centroids, 8, CV_32S);
    if (out.ncc <= 1) {
        spdlog::debug("CoverageGuard no missing connected components");
        return false;
    }

    for (int cid = 1; cid < out.ncc; ++cid) {
        out.max_missing_component =
            std::max(out.max_missing_component, stats.at<int>(cid, cv::CC_STAT_AREA));
    }

    const bool global_trigger = out.ratio < min_ratio;
    const bool local_trigger =
        max_unpatched_gap_area >= 0.0f &&
        out.max_missing_component > static_cast<int>(std::floor(max_unpatched_gap_area));
    if (!global_trigger && !local_trigger) {
        spdlog::debug("CoverageGuard skipped: coverage_ratio={:.4f} >= min_ratio={:.4f}, "
                      "max_missing_component={} <= max_unpatched_gap_area={:.1f}",
                      out.ratio, min_ratio, out.max_missing_component, max_unpatched_gap_area);
        return false;
    }

    if (global_trigger) {
        spdlog::warn("CoverageGuard triggered: coverage_ratio={:.4f} < min_ratio={:.4f}", out.ratio,
                     min_ratio);
    } else {
        spdlog::debug(
            "CoverageGuard local gap triggered: coverage_ratio={:.4f} >= min_ratio={:.4f}, "
            "max_missing_component={} > max_unpatched_gap_area={:.1f}",
            out.ratio, min_ratio, out.max_missing_component, max_unpatched_gap_area);
    }
    return true;
}

struct PatchStats {
    int eligible                  = 0;
    int patched                   = 0;
    int added                     = 0;
    int bad_labels                = 0;
    int underpaint_added          = 0;
    int boundary_underpaint_added = 0;
};

void AccumulatePatchStats(PatchStats& total, const PatchStats& next) {
    total.eligible += next.eligible;
    total.patched += next.patched;
    total.added += next.added;
    total.bad_labels += next.bad_labels;
    total.underpaint_added += next.underpaint_added;
    total.boundary_underpaint_added += next.boundary_underpaint_added;
}

VectorizedShape BuildPatchShape(const TracedPolygonGroup& group, const Rgb& color,
                                const Vec2f& offset) {
    VectorizedShape patch;
    patch.color = color;
    patch.area  = group.area;

    auto shift_contour = [&](BezierContour& bc) {
        for (auto& seg : bc.segments) {
            seg.p0 = seg.p0 + offset;
            seg.p1 = seg.p1 + offset;
            seg.p2 = seg.p2 + offset;
            seg.p3 = seg.p3 + offset;
        }
    };

    auto outer_bc = RingToBezier(group.outer);
    shift_contour(outer_bc);
    patch.contours.push_back(std::move(outer_bc));
    for (const auto& hole : group.holes) {
        auto hc = RingToBezier(hole);
        shift_contour(hc);
        hc.is_hole = true;
        patch.contours.push_back(std::move(hc));
    }
    return patch;
}

VectorizedShape BuildRectPatchShape(float x0, float y0, float x1, float y1, const Rgb& color) {
    VectorizedShape patch;
    patch.color = color;
    patch.area  = static_cast<double>((x1 - x0) * (y1 - y0));

    BezierContour contour;
    contour.segments = {
        MakeLinearBezier({x0, y0}, {x1, y0}),
        MakeLinearBezier({x1, y0}, {x1, y1}),
        MakeLinearBezier({x1, y1}, {x0, y1}),
        MakeLinearBezier({x0, y1}, {x0, y0}),
    };
    contour.closed = true;
    patch.contours.push_back(std::move(contour));
    return patch;
}

std::vector<TracedPolygonGroup> BuildTinyMaskFallbackGroups(const cv::Mat& mask,
                                                            float min_patch_area) {
    constexpr int kTinyFallbackMaxPx = 32;
    cv::Mat cc_labels;
    cv::Mat stats;
    cv::Mat centroids;
    int ncc = cv::connectedComponentsWithStats(mask, cc_labels, stats, centroids, 8, CV_32S);

    std::vector<TracedPolygonGroup> groups;
    for (int cid = 1; cid < ncc; ++cid) {
        int area = stats.at<int>(cid, cv::CC_STAT_AREA);
        if (area < static_cast<int>(std::max(1.0f, min_patch_area)) || area > kTinyFallbackMaxPx) {
            continue;
        }
        float x = static_cast<float>(stats.at<int>(cid, cv::CC_STAT_LEFT));
        float y = static_cast<float>(stats.at<int>(cid, cv::CC_STAT_TOP));
        float w = static_cast<float>(stats.at<int>(cid, cv::CC_STAT_WIDTH));
        float h = static_cast<float>(stats.at<int>(cid, cv::CC_STAT_HEIGHT));

        TracedPolygonGroup group;
        group.outer = {{x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}};
        group.area  = static_cast<double>(w * h);
        groups.push_back(std::move(group));
    }
    return groups;
}

void AddHorizontalBoundaryRuns(std::vector<VectorizedShape>& shapes, PatchStats& stats,
                               const cv::Mat& labels, const std::vector<Rgb>& palette, int row,
                               float y0, float y1, const cv::Mat& cc_labels) {
    const int width = labels.cols;
    if (row < 0 || row >= labels.rows || y1 <= y0) return;

    const int* label_row = labels.ptr<int>(row);
    const int* cc_row    = cc_labels.ptr<int>(row);
    int run_label        = (width > 0 && cc_row[0] > 0) ? label_row[0] : -1;
    int run_start        = 0;
    for (int x = 1; x <= width; ++x) {
        int label = (x < width && cc_row[x] > 0) ? label_row[x] : -1;
        if (label == run_label) continue;

        if (run_label >= 0 && run_label < static_cast<int>(palette.size()) && x > run_start) {
            shapes.push_back(BuildRectPatchShape(static_cast<float>(run_start), y0,
                                                 static_cast<float>(x), y1, palette[run_label]));
            ++stats.added;
            ++stats.underpaint_added;
            ++stats.boundary_underpaint_added;
        }
        run_label = label;
        run_start = x;
    }
}

void AddVerticalBoundaryRuns(std::vector<VectorizedShape>& shapes, PatchStats& stats,
                             const cv::Mat& labels, const std::vector<Rgb>& palette, int col,
                             int y_begin, int y_end, float x0, float x1, const cv::Mat& cc_labels) {
    if (col < 0 || col >= labels.cols || y_begin >= y_end || x1 <= x0) return;

    int run_label = cc_labels.ptr<int>(y_begin)[col] > 0 ? labels.ptr<int>(y_begin)[col] : -1;
    int run_start = y_begin;
    for (int y = y_begin + 1; y <= y_end; ++y) {
        int label = (y < y_end && cc_labels.ptr<int>(y)[col] > 0) ? labels.ptr<int>(y)[col] : -1;
        if (label == run_label) continue;

        if (run_label >= 0 && run_label < static_cast<int>(palette.size()) && y > run_start) {
            shapes.push_back(BuildRectPatchShape(x0, static_cast<float>(run_start), x1,
                                                 static_cast<float>(y), palette[run_label]));
            ++stats.added;
            ++stats.underpaint_added;
            ++stats.boundary_underpaint_added;
        }
        run_label = label;
        run_start = y;
    }
}

PatchStats AddBoundaryUnderpaintShapes(std::vector<VectorizedShape>& shapes, const cv::Mat& labels,
                                       const std::vector<Rgb>& palette, const GapInfo& gaps) {
    PatchStats stats;
    if (labels.empty() || palette.empty() || gaps.cc_labels.empty()) return stats;

    constexpr int kBoundaryUnderpaintPx = 4;
    const int width                     = labels.cols;
    const int height                    = labels.rows;
    const int band = std::max(1, std::min({kBoundaryUnderpaintPx, width, height}));

    for (int y = 0; y < band; ++y) {
        AddHorizontalBoundaryRuns(shapes, stats, labels, palette, y, static_cast<float>(y),
                                  static_cast<float>(y + 1), gaps.cc_labels);
    }
    for (int y = height - band; y < height; ++y) {
        if (y >= band) {
            AddHorizontalBoundaryRuns(shapes, stats, labels, palette, y, static_cast<float>(y),
                                      static_cast<float>(y + 1), gaps.cc_labels);
        }
    }

    const int vertical_y_begin = band;
    const int vertical_y_end   = height - band;
    if (vertical_y_begin < vertical_y_end) {
        for (int x = 0; x < band; ++x) {
            AddVerticalBoundaryRuns(shapes, stats, labels, palette, x, vertical_y_begin,
                                    vertical_y_end, static_cast<float>(x),
                                    static_cast<float>(x + 1), gaps.cc_labels);
        }
        for (int x = width - band; x < width; ++x) {
            if (x >= band) {
                AddVerticalBoundaryRuns(shapes, stats, labels, palette, x, vertical_y_begin,
                                        vertical_y_end, static_cast<float>(x),
                                        static_cast<float>(x + 1), gaps.cc_labels);
            }
        }
    }

    return stats;
}

PatchStats PatchMissingRegions(std::vector<VectorizedShape>& shapes, const GapInfo& gaps,
                               const cv::Mat& labels, const std::vector<Rgb>& palette,
                               float tracing_epsilon, float min_patch_area, int w, int h) {
    const int ncc = gaps.ncc;
    std::vector<std::vector<VectorizedShape>> per_cid_patches(ncc);
    std::vector<int> per_cid_eligible(ncc, 0);
    std::vector<int> per_cid_patched(ncc, 0);
    std::vector<int> per_cid_bad(ncc, 0);

#pragma omp parallel for schedule(dynamic)
    for (int cid = 1; cid < ncc; ++cid) {
        cv::Rect roi;
        {
            int rmin = h, rmax = 0, cmin = w, cmax = 0;
            for (int r = 0; r < h; ++r) {
                const int* cc_row = gaps.cc_labels.ptr<int>(r);
                for (int c = 0; c < w; ++c) {
                    if (cc_row[c] != cid) continue;
                    rmin = std::min(rmin, r);
                    rmax = std::max(rmax, r);
                    cmin = std::min(cmin, c);
                    cmax = std::max(cmax, c);
                }
            }
            if (rmin > rmax) continue;
            roi = cv::Rect(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1);
        }

        std::unordered_map<int, int> label_hist;

        for (int r = roi.y; r < roi.y + roi.height; ++r) {
            const int* cc_row = gaps.cc_labels.ptr<int>(r);
            const int* lb_row = labels.ptr<int>(r);
            for (int c = roi.x; c < roi.x + roi.width; ++c) {
                if (cc_row[c] != cid) continue;
                label_hist[lb_row[c]]++;
            }
        }

        if (label_hist.empty()) continue;

        for (const auto& kv : label_hist) {
            const int label = kv.first;
            const int area  = kv.second;
            if (area < static_cast<int>(std::max(1.0f, min_patch_area))) continue;
            ++per_cid_eligible[cid];

            if (label < 0 || label >= static_cast<int>(palette.size())) {
                ++per_cid_bad[cid];
                continue;
            }

            cv::Mat comp_mask(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
            cv::Mat label_mask(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
            for (int r = roi.y; r < roi.y + roi.height; ++r) {
                const int* cc_row  = gaps.cc_labels.ptr<int>(r);
                const int* lb_row  = labels.ptr<int>(r);
                uint8_t* comp_out  = comp_mask.ptr<uint8_t>(r - roi.y);
                uint8_t* label_out = label_mask.ptr<uint8_t>(r - roi.y);
                for (int c = roi.x; c < roi.x + roi.width; ++c) {
                    if (lb_row[c] == label) label_out[c - roi.x] = 255;
                    if (cc_row[c] == cid && lb_row[c] == label) comp_out[c - roi.x] = 255;
                }
            }

            cv::Mat trace_mask;
            {
                cv::Mat dilated;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::dilate(comp_mask, dilated, kernel, cv::Point(-1, -1), 1);
                cv::bitwise_and(dilated, label_mask, trace_mask);
            }

            auto traced = TraceMaskWithPotrace(trace_mask, tracing_epsilon * 0.8f);
            auto fixed =
                RepairTopology(traced, tracing_epsilon * 0.6f, min_patch_area, min_patch_area);
            if (fixed.empty()) { fixed = BuildTinyMaskFallbackGroups(trace_mask, min_patch_area); }
            if (!fixed.empty()) ++per_cid_patched[cid];

            for (auto& g : fixed) {
                auto patch = BuildPatchShape(
                    g, palette[label], Vec2f(static_cast<float>(roi.x), static_cast<float>(roi.y)));
                if (patch.contours.empty()) continue;

                cv::Mat patch_raster(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
                {
                    std::vector<std::vector<cv::Point>> polys;
                    for (const auto& cnt : patch.contours) {
                        auto poly = FlattenContour(cnt, w, h);
                        std::vector<cv::Point> local_poly;
                        local_poly.reserve(poly.size());
                        for (const auto& pt : poly) {
                            local_poly.emplace_back(pt.x - roi.x, pt.y - roi.y);
                        }
                        if (local_poly.size() >= 3) polys.push_back(std::move(local_poly));
                    }
                    if (!polys.empty()) cv::fillPoly(patch_raster, polys, cv::Scalar(255));
                }
                cv::Mat overlap_mask;
                cv::bitwise_and(patch_raster, gaps.coverage(roi), overlap_mask);
                int patch_px   = cv::countNonZero(patch_raster);
                int overlap_px = cv::countNonZero(overlap_mask);
                int new_px     = patch_px - overlap_px;
                if (new_px <= 0) continue;

                per_cid_patches[cid].push_back(std::move(patch));
            }
        }
    }

    PatchStats stats;
    for (int cid = 1; cid < ncc; ++cid) {
        stats.eligible += per_cid_eligible[cid];
        stats.patched += per_cid_patched[cid];
        stats.bad_labels += per_cid_bad[cid];
        for (auto& p : per_cid_patches[cid]) {
            shapes.push_back(std::move(p));
            ++stats.added;
        }
    }
    return stats;
}

} // namespace

void ApplyCoverageGuard(std::vector<VectorizedShape>& shapes, const cv::Mat& labels,
                        const std::vector<Rgb>& palette, float min_ratio, float tracing_epsilon,
                        float min_patch_area, float max_unpatched_gap_area) {
    if (labels.empty() || labels.type() != CV_32SC1) {
        spdlog::warn("CoverageGuard skipped: invalid labels (empty={} type={})", labels.empty(),
                     labels.empty() ? -1 : labels.type());
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const int h      = labels.rows;
    const int w      = labels.cols;
    spdlog::debug("CoverageGuard start: labels={}x{}, min_ratio={:.4f}, tracing_eps={:.3f}", w, h,
                  min_ratio, tracing_epsilon);

    constexpr int kMaxPatchPasses = 3;
    PatchStats total_stats;
    GapInfo last_gaps;
    GapInfo initial_gaps;
    bool has_initial_gaps = false;
    int passes            = 0;

    for (int pass = 0; pass < kMaxPatchPasses; ++pass) {
        GapInfo gaps;
        if (!FindCoverageGaps(shapes, labels, min_ratio, max_unpatched_gap_area, w, h, gaps)) {
            last_gaps = std::move(gaps);
            break;
        }
        if (!has_initial_gaps) {
            initial_gaps     = gaps;
            has_initial_gaps = true;
        }
        last_gaps = gaps;

        auto stats = PatchMissingRegions(shapes, gaps, labels, palette, tracing_epsilon,
                                         min_patch_area, w, h);
        AccumulatePatchStats(total_stats, stats);
        ++passes;
        if (stats.added <= 0) break;
    }
    if (has_initial_gaps) {
        auto boundary_stats = AddBoundaryUnderpaintShapes(shapes, labels, palette, initial_gaps);
        AccumulatePatchStats(total_stats, boundary_stats);
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info(
        "CoverageGuard done: source_px={}, covered_px={}, ratio={:.4f}, ncc={}, "
        "max_missing_component={}, max_unpatched_gap_area={:.1f}, passes={}, "
        "eligible={}, patched_components={}, patch_shapes_added={}, underpaint_shapes_added={}, "
        "boundary_underpaint_shapes_added={}, invalid_label_skips={}, elapsed_ms={:.2f}",
        last_gaps.source_px, last_gaps.covered_px, last_gaps.ratio, last_gaps.ncc,
        last_gaps.max_missing_component, max_unpatched_gap_area, passes, total_stats.eligible,
        total_stats.patched, total_stats.added - total_stats.underpaint_added,
        total_stats.underpaint_added, total_stats.boundary_underpaint_added, total_stats.bad_labels,
        elapsed_ms);
}

} // namespace neroued::vectorizer::detail
