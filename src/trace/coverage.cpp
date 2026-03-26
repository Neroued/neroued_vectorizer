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
    int ncc;
    int source_px;
    int covered_px;
    float ratio;
};

bool FindCoverageGaps(const std::vector<VectorizedShape>& shapes, const cv::Mat& labels,
                      float min_ratio, int w, int h, GapInfo& out) {
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
    if (out.ratio >= min_ratio) {
        spdlog::debug("CoverageGuard skipped: coverage_ratio={:.4f} >= min_ratio={:.4f}", out.ratio,
                      min_ratio);
        return false;
    }
    spdlog::warn("CoverageGuard triggered: coverage_ratio={:.4f} < min_ratio={:.4f}", out.ratio,
                 min_ratio);

    cv::Mat missing;
    cv::bitwise_not(out.coverage, missing);
    cv::bitwise_and(missing, source_mask, missing);

    out.ncc = cv::connectedComponents(missing, out.cc_labels, 8, CV_32S);
    if (out.ncc <= 1) {
        spdlog::debug("CoverageGuard no missing connected components");
        return false;
    }
    return true;
}

struct PatchStats {
    int eligible   = 0;
    int patched    = 0;
    int added      = 0;
    int bad_labels = 0;
};

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

        cv::Mat comp_mask(roi.height, roi.width, CV_8UC1, cv::Scalar(0));
        std::unordered_map<int, int> label_hist;
        int area = 0;

        for (int r = roi.y; r < roi.y + roi.height; ++r) {
            const int* cc_row = gaps.cc_labels.ptr<int>(r);
            const int* lb_row = labels.ptr<int>(r);
            uint8_t* out      = comp_mask.ptr<uint8_t>(r - roi.y);
            for (int c = roi.x; c < roi.x + roi.width; ++c) {
                if (cc_row[c] != cid) continue;
                out[c - roi.x] = 255;
                ++area;
                label_hist[lb_row[c]]++;
            }
        }

        if (area < static_cast<int>(std::max(1.0f, min_patch_area))) continue;
        if (label_hist.empty()) continue;
        per_cid_eligible[cid] = 1;

        int best_label = -1;
        int best_count = -1;
        for (const auto& kv : label_hist) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_label = kv.first;
            }
        }
        if (best_label < 0 || best_label >= static_cast<int>(palette.size())) {
            per_cid_bad[cid] = 1;
            continue;
        }

        auto traced = TraceMaskWithPotrace(comp_mask, tracing_epsilon * 0.8f);
        auto fixed = RepairTopology(traced, tracing_epsilon * 0.6f, min_patch_area, min_patch_area);
        if (!fixed.empty()) per_cid_patched[cid] = 1;

        for (auto& g : fixed) {
            VectorizedShape patch;
            patch.color = palette[best_label];
            patch.area  = g.area;

            auto shift_contour = [&](BezierContour& bc) {
                Vec2f offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
                for (auto& seg : bc.segments) {
                    seg.p0 = seg.p0 + offset;
                    seg.p1 = seg.p1 + offset;
                    seg.p2 = seg.p2 + offset;
                    seg.p3 = seg.p3 + offset;
                }
            };

            auto outer_bc = RingToBezier(g.outer);
            shift_contour(outer_bc);
            patch.contours.push_back(std::move(outer_bc));
            for (const auto& hole : g.holes) {
                auto hc = RingToBezier(hole);
                shift_contour(hc);
                hc.is_hole = true;
                patch.contours.push_back(std::move(hc));
            }
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
            if (patch_px > 0 &&
                static_cast<float>(overlap_px) / static_cast<float>(patch_px) > 0.5f) {
                continue;
            }

            per_cid_patches[cid].push_back(std::move(patch));
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
                        float min_patch_area) {
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

    GapInfo gaps;
    if (!FindCoverageGaps(shapes, labels, min_ratio, w, h, gaps)) return;

    auto stats =
        PatchMissingRegions(shapes, gaps, labels, palette, tracing_epsilon, min_patch_area, w, h);

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info(
        "CoverageGuard done: source_px={}, covered_px={}, ratio={:.4f}, ncc={}, eligible={}, "
        "patched_components={}, patch_shapes_added={}, invalid_label_skips={}, elapsed_ms={:.2f}",
        gaps.source_px, gaps.covered_px, gaps.ratio, gaps.ncc, stats.eligible, stats.patched,
        stats.added, stats.bad_labels, elapsed_ms);
}

} // namespace neroued::vectorizer::detail
