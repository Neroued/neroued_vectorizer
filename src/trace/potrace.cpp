#include "potrace.h"

#include "curve/bezier.h"

#include <potracelib.h>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace neroued::vectorizer::detail {

namespace {

constexpr int kWordBits = static_cast<int>(8 * sizeof(potrace_word));

Vec2f EvalCubic(const Vec2f& p0, const Vec2f& p1, const Vec2f& p2, const Vec2f& p3, float t) {
    float u = 1.0f - t;
    return p0 * (u * u * u) + p1 * (3.0f * u * u * t) + p2 * (3.0f * u * t * t) + p3 * (t * t * t);
}

Vec2f ToVec2(const potrace_dpoint_t& p) {
    return {static_cast<float>(p.x), static_cast<float>(p.y)};
}

void AppendCubicSamples(const Vec2f& p0, const Vec2f& p1, const Vec2f& p2, const Vec2f& p3,
                        int sample_count, std::vector<Vec2f>& out) {
    sample_count = std::max(3, sample_count);
    for (int i = 1; i <= sample_count; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_count);
        out.push_back(EvalCubic(p0, p1, p2, p3, t));
    }
}

std::vector<Vec2f> SimplifyRing(const std::vector<Vec2f>& ring, float eps) {
    if (ring.size() < 4) return ring;

    std::vector<cv::Point2f> in;
    in.reserve(ring.size());
    for (const auto& p : ring) in.push_back({p.x, p.y});

    std::vector<cv::Point2f> approx;
    cv::approxPolyDP(in, approx, std::max(0.2f, eps), true);
    if (approx.size() < 3) return {};

    std::vector<Vec2f> out;
    out.reserve(approx.size());
    for (const auto& p : approx) out.push_back({p.x, p.y});
    if (out.size() > 1 && (out.front() - out.back()).LengthSquared() < 1e-6f) out.pop_back();
    return out;
}

std::vector<Vec2f> ConvertCvContour(const std::vector<cv::Point>& contour, float eps) {
    std::vector<Vec2f> ring;
    ring.reserve(contour.size());
    for (const auto& p : contour)
        ring.push_back({static_cast<float>(p.x), static_cast<float>(p.y)});
    return SimplifyRing(ring, eps);
}

std::vector<TracedPolygonGroup> TraceMaskFallbackContours(const cv::Mat& mask,
                                                          float simplify_epsilon) {
    std::vector<TracedPolygonGroup> groups;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return groups;

    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        if (hierarchy[i][3] != -1) continue;
        auto outer = ConvertCvContour(contours[i], simplify_epsilon);
        if (outer.size() < 3) continue;
        if (SignedArea(outer) < 0.0) std::reverse(outer.begin(), outer.end());

        TracedPolygonGroup g;
        g.outer = std::move(outer);
        g.area  = std::abs(SignedArea(g.outer));

        int child = hierarchy[i][2];
        while (child != -1) {
            auto hole = ConvertCvContour(contours[child], simplify_epsilon);
            if (hole.size() >= 3) {
                if (SignedArea(hole) > 0.0) std::reverse(hole.begin(), hole.end());
                g.holes.push_back(std::move(hole));
            }
            child = hierarchy[child][0];
        }
        groups.push_back(std::move(g));
    }
    return groups;
}

std::vector<Vec2f> ConvertPathToRing(const potrace_path_t* path, float simplify_epsilon) {
    if (!path) return {};
    const potrace_curve_t& curve = path->curve;
    if (curve.n <= 0 || !curve.tag || !curve.c) return {};

    std::vector<Vec2f> ring;
    ring.reserve(static_cast<size_t>(curve.n) * 6);

    Vec2f current = ToVec2(curve.c[curve.n - 1][2]);
    ring.push_back(current);

    int cubic_samples =
        std::clamp(static_cast<int>(std::lround(4.0f / std::max(0.2f, simplify_epsilon))), 3, 12);

    for (int i = 0; i < curve.n; ++i) {
        if (curve.tag[i] == POTRACE_CURVETO) {
            Vec2f c1  = ToVec2(curve.c[i][0]);
            Vec2f c2  = ToVec2(curve.c[i][1]);
            Vec2f end = ToVec2(curve.c[i][2]);
            AppendCubicSamples(current, c1, c2, end, cubic_samples, ring);
            current = end;
        } else {
            Vec2f corner = ToVec2(curve.c[i][1]);
            Vec2f end    = ToVec2(curve.c[i][2]);
            ring.push_back(corner);
            ring.push_back(end);
            current = end;
        }
    }

    if (ring.size() > 1 && (ring.front() - ring.back()).LengthSquared() < 1e-6f) ring.pop_back();
    if (ring.size() < 3) return {};
    return SimplifyRing(ring, simplify_epsilon);
}

potrace_bitmap_t BuildPotraceBitmap(const cv::Mat& mask, std::vector<potrace_word>& storage) {
    potrace_bitmap_t bm{};
    bm.w  = mask.cols;
    bm.h  = mask.rows;
    bm.dy = (bm.w + kWordBits - 1) / kWordBits;
    storage.assign(static_cast<size_t>(bm.dy) * static_cast<size_t>(bm.h), 0);

    for (int y = 0; y < bm.h; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < bm.w; ++x) {
            if (row[x] == 0) continue;
            int word_idx = y * bm.dy + (x / kWordBits);
            int bit_idx  = kWordBits - 1 - (x % kWordBits);
            storage[static_cast<size_t>(word_idx)] |= (static_cast<potrace_word>(1) << bit_idx);
        }
    }

    bm.map = storage.data();
    return bm;
}

BezierContour ConvertPathToBezierContour(const potrace_path_t* path) {
    BezierContour contour;
    contour.closed = true;
    if (!path) return contour;
    const potrace_curve_t& curve = path->curve;
    if (curve.n <= 0 || !curve.tag || !curve.c) return contour;

    contour.segments.reserve(static_cast<size_t>(curve.n) * 2);
    Vec2f prev = ToVec2(curve.c[curve.n - 1][2]);

    for (int i = 0; i < curve.n; ++i) {
        if (curve.tag[i] == POTRACE_CURVETO) {
            Vec2f c1  = ToVec2(curve.c[i][0]);
            Vec2f c2  = ToVec2(curve.c[i][1]);
            Vec2f end = ToVec2(curve.c[i][2]);
            contour.segments.push_back({prev, c1, c2, end});
            prev = end;
        } else {
            Vec2f corner = ToVec2(curve.c[i][1]);
            Vec2f end    = ToVec2(curve.c[i][2]);
            if ((corner - prev).LengthSquared() > 1e-8f) {
                Vec2f d = corner - prev;
                contour.segments.push_back(
                    {prev, prev + d * (1.0f / 3.0f), prev + d * (2.0f / 3.0f), corner});
            }
            if ((end - corner).LengthSquared() > 1e-8f) {
                Vec2f d = end - corner;
                contour.segments.push_back(
                    {corner, corner + d * (1.0f / 3.0f), corner + d * (2.0f / 3.0f), end});
            }
            prev = end;
        }
    }
    return contour;
}

void CollectBezierGroupsFromTree(const potrace_path_t* path_list,
                                 std::vector<TracedBezierGroup>& groups) {
    for (const potrace_path_t* p = path_list; p; p = p->sibling) {
        auto contour = ConvertPathToBezierContour(p);
        if (contour.segments.empty()) continue;
        double area = BezierContourSignedArea(contour);

        if (p->sign == '+') {
            if (area < 0) {
                ReverseBezierContour(contour);
                area = -area;
            }
            if (area < std::numeric_limits<double>::epsilon()) continue;

            TracedBezierGroup g;
            g.outer = std::move(contour);
            g.area  = area;

            for (const potrace_path_t* child = p->childlist; child; child = child->sibling) {
                auto hole = ConvertPathToBezierContour(child);
                if (hole.segments.empty()) continue;
                double ha = BezierContourSignedArea(hole);
                if (std::abs(ha) < std::numeric_limits<double>::epsilon()) continue;
                if (ha > 0) ReverseBezierContour(hole);
                g.holes.push_back(std::move(hole));

                if (child->childlist) { CollectBezierGroupsFromTree(child->childlist, groups); }
            }
            groups.push_back(std::move(g));
        } else {
            if (p->childlist) { CollectBezierGroupsFromTree(p->childlist, groups); }
        }
    }
}

BezierContour PointsToBezierContour(const std::vector<cv::Point>& pts) {
    std::vector<Vec2f> ring;
    ring.reserve(pts.size());
    for (const auto& p : pts) ring.push_back({static_cast<float>(p.x), static_cast<float>(p.y)});
    return RingToBezier(ring);
}

std::vector<TracedBezierGroup> FallbackBezierContours(const cv::Mat& mask) {
    std::vector<TracedBezierGroup> groups;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return groups;

    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        if (hierarchy[i][3] != -1) continue;
        auto outer = PointsToBezierContour(contours[i]);
        if (outer.segments.empty()) continue;
        double area = BezierContourSignedArea(outer);
        if (area < 0) {
            ReverseBezierContour(outer);
            area = -area;
        }
        if (area < std::numeric_limits<double>::epsilon()) continue;

        TracedBezierGroup g;
        g.outer = std::move(outer);
        g.area  = area;

        int child = hierarchy[i][2];
        while (child != -1) {
            auto hole = PointsToBezierContour(contours[child]);
            if (!hole.segments.empty()) {
                double ha = BezierContourSignedArea(hole);
                if (ha > 0) ReverseBezierContour(hole);
                g.holes.push_back(std::move(hole));
            }
            child = hierarchy[child][0];
        }
        groups.push_back(std::move(g));
    }
    return groups;
}

struct PotraceTraceResult {
    potrace_state_t* state = nullptr;
    bool incomplete        = false;
    int path_count         = 0;
};

PotraceTraceResult RunPotraceTrace(const cv::Mat& mask, int turdsize, double opttolerance,
                                   std::vector<potrace_word>& bitmap_storage) {
    potrace_bitmap_t bm = BuildPotraceBitmap(mask, bitmap_storage);

    potrace_param_t* params = potrace_param_default();
    if (!params) throw std::runtime_error("potrace_param_default failed");
    params->turdsize     = std::max(0, turdsize);
    params->turnpolicy   = POTRACE_TURNPOLICY_MAJORITY;
    params->alphamax     = 1.0;
    params->opticurve    = 1;
    params->opttolerance = std::clamp(opttolerance, 0.2, 2.0);

    potrace_state_t* state = potrace_trace(params, &bm);
    potrace_param_free(params);
    if (!state) throw std::runtime_error("potrace_trace failed");

    if (state->status != POTRACE_STATUS_OK && state->status != POTRACE_STATUS_INCOMPLETE) {
        potrace_state_free(state);
        throw std::runtime_error("potrace_trace returned invalid status");
    }

    PotraceTraceResult result;
    result.state      = state;
    result.incomplete = (state->status == POTRACE_STATUS_INCOMPLETE);
    for (const potrace_path_t* p = state->plist; p; p = p->next) ++result.path_count;
    return result;
}

} // namespace

BezierContour RingToBezier(const std::vector<Vec2f>& ring) {
    BezierContour contour;
    contour.closed = true;
    if (ring.size() < 3) return contour;
    contour.segments.reserve(ring.size());
    for (size_t i = 0; i < ring.size(); ++i) {
        const Vec2f& a = ring[i];
        const Vec2f& b = ring[(i + 1) % ring.size()];
        Vec2f d        = b - a;
        if (d.LengthSquared() < 1e-8f) continue;
        contour.segments.push_back({a, a + d * (1.0f / 3.0f), a + d * (2.0f / 3.0f), b});
    }
    return contour;
}

double SignedArea(const std::vector<Vec2f>& ring) {
    if (ring.size() < 3) return 0.0;
    double acc = 0.0;
    for (size_t i = 0; i < ring.size(); ++i) {
        const Vec2f& a = ring[i];
        const Vec2f& b = ring[(i + 1) % ring.size()];
        acc += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return 0.5 * acc;
}

std::vector<TracedPolygonGroup> TraceMaskWithPotrace(const cv::Mat& mask, float simplify_epsilon) {
    std::vector<TracedPolygonGroup> groups;
    if (mask.empty() || mask.type() != CV_8UC1) {
        spdlog::debug("TraceMaskWithPotrace skipped: invalid mask (empty={} type={})", mask.empty(),
                      mask.empty() ? -1 : mask.type());
        return groups;
    }
    const auto start       = std::chrono::steady_clock::now();
    const int mask_nonzero = cv::countNonZero(mask);
    spdlog::debug("TraceMaskWithPotrace start: mask={}x{}, nonzero={}, epsilon={:.3f}", mask.cols,
                  mask.rows, mask_nonzero, simplify_epsilon);

    int turdsize        = std::max(0, static_cast<int>(std::lround(simplify_epsilon * 0.5f)));
    double opttolerance = static_cast<double>(simplify_epsilon);

    std::vector<potrace_word> bitmap_storage;
    auto tr = RunPotraceTrace(mask, turdsize, opttolerance, bitmap_storage);
    if (tr.incomplete) {
        spdlog::warn("TraceMaskWithPotrace status incomplete: mask={}x{}", mask.cols, mask.rows);
    }

    for (const potrace_path_t* p = tr.state->plist; p; p = p->next) {
        auto ring = ConvertPathToRing(p, simplify_epsilon);
        if (ring.size() < 3) continue;
        double area = SignedArea(ring);
        if (area < 0.0) std::reverse(ring.begin(), ring.end());

        TracedPolygonGroup g;
        g.outer = std::move(ring);
        g.area  = std::abs(SignedArea(g.outer));
        if (g.area > std::numeric_limits<double>::epsilon()) groups.push_back(std::move(g));
    }
    potrace_state_free(tr.state);

    bool fallback_used = false;
    if (groups.empty()) {
        fallback_used = true;
        spdlog::warn("TraceMaskWithPotrace fallback to findContours: mask={}x{}", mask.cols,
                     mask.rows);
        groups = TraceMaskFallbackContours(mask, simplify_epsilon * 0.8f);
    }

    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.area > b.area; });
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::debug("TraceMaskWithPotrace done: mask={}x{}, paths={}, groups={}, fallback_used={}, "
                  "elapsed_ms={:.2f}",
                  mask.cols, mask.rows, tr.path_count, groups.size(), fallback_used, elapsed_ms);
    return groups;
}

std::vector<TracedBezierGroup> TraceMaskWithPotraceBezier(const cv::Mat& mask, int turdsize,
                                                          double opttolerance) {
    std::vector<TracedBezierGroup> groups;
    if (mask.empty() || mask.type() != CV_8UC1) {
        spdlog::debug("TraceMaskWithPotraceBezier skipped: invalid mask (empty={} type={})",
                      mask.empty(), mask.empty() ? -1 : mask.type());
        return groups;
    }
    const auto start       = std::chrono::steady_clock::now();
    const int mask_nonzero = cv::countNonZero(mask);
    spdlog::debug("TraceMaskWithPotraceBezier start: mask={}x{}, nonzero={}, turdsize={}, "
                  "opttolerance={:.3f}",
                  mask.cols, mask.rows, mask_nonzero, turdsize, opttolerance);

    std::vector<potrace_word> bitmap_storage;
    auto tr = RunPotraceTrace(mask, turdsize, opttolerance, bitmap_storage);
    if (tr.incomplete) {
        spdlog::warn("TraceMaskWithPotraceBezier status incomplete: mask={}x{}", mask.cols,
                     mask.rows);
    }

    CollectBezierGroupsFromTree(tr.state->plist, groups);
    potrace_state_free(tr.state);

    bool fallback_used = false;
    if (groups.empty()) {
        fallback_used = true;
        spdlog::warn("TraceMaskWithPotraceBezier fallback to findContours: mask={}x{}", mask.cols,
                     mask.rows);
        groups = FallbackBezierContours(mask);
    }

    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.area > b.area; });
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::debug(
        "TraceMaskWithPotraceBezier done: mask={}x{}, paths={}, groups={}, fallback_used={}, "
        "elapsed_ms={:.2f}",
        mask.cols, mask.rows, tr.path_count, groups.size(), fallback_used, elapsed_ms);
    return groups;
}

} // namespace neroued::vectorizer::detail
