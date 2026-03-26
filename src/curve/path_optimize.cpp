#include "path_optimize.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace neroued::vectorizer::detail {

namespace {

float PointToSegmentDistance(const Vec2f& p, const Vec2f& a, const Vec2f& b) {
    Vec2f ab   = b - a;
    float len2 = ab.LengthSquared();
    if (len2 < 1e-12f) return Vec2f::Distance(p, a);
    float t    = std::clamp((p - a).Dot(ab) / len2, 0.f, 1.f);
    Vec2f proj = a + ab * t;
    return Vec2f::Distance(p, proj);
}

float MaxControlPointDeviation(const CubicBezier& seg) {
    float d1 = PointToSegmentDistance(seg.p1, seg.p0, seg.p3);
    float d2 = PointToSegmentDistance(seg.p2, seg.p0, seg.p3);
    return std::max(d1, d2);
}

bool IsNearLinear(const CubicBezier& seg, float eps) { return MaxControlPointDeviation(seg) < eps; }

std::vector<Vec2f> SampleBezier(const CubicBezier& b, int n) {
    std::vector<Vec2f> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / (n - 1);
        pts.push_back(EvalBezier(b, t));
    }
    return pts;
}

float MaxDeviationFromBezier(const CubicBezier& candidate, const std::vector<Vec2f>& samples) {
    float max_d = 0.f;
    for (const auto& p : samples) {
        float best            = 1e30f;
        constexpr int kProbes = 16;
        for (int i = 0; i <= kProbes; ++i) {
            float t = static_cast<float>(i) / kProbes;
            float d = Vec2f::Distance(p, EvalBezier(candidate, t));
            best    = std::min(best, d);
        }
        max_d = std::max(max_d, best);
    }
    return max_d;
}

CubicBezier FitCubicToPoints(const std::vector<Vec2f>& pts) {
    if (pts.size() < 2) return {pts[0], pts[0], pts[0], pts[0]};

    Vec2f p0 = pts.front();
    Vec2f p3 = pts.back();

    if (pts.size() == 2) return MakeLinearBezier(p0, p3);

    const int n = static_cast<int>(pts.size());
    std::vector<float> params(n);
    params[0]       = 0.f;
    float total_len = 0.f;
    for (int i = 1; i < n; ++i) {
        total_len += Vec2f::Distance(pts[i], pts[i - 1]);
        params[i] = total_len;
    }
    if (total_len > 1e-8f) {
        for (auto& p : params) p /= total_len;
    } else {
        for (int i = 0; i < n; ++i) params[i] = static_cast<float>(i) / (n - 1);
    }

    float a11 = 0, a12 = 0, a22 = 0;
    Vec2f c1{}, c2{};

    for (int i = 0; i < n; ++i) {
        float t  = params[i];
        float u  = 1.f - t;
        float b1 = 3.f * u * u * t;
        float b2 = 3.f * u * t * t;

        Vec2f rhs = pts[i] - p0 * (u * u * u) - p3 * (t * t * t);

        a11 += b1 * b1;
        a12 += b1 * b2;
        a22 += b2 * b2;
        c1 += rhs * b1;
        c2 += rhs * b2;
    }

    float det = a11 * a22 - a12 * a12;
    Vec2f cp1, cp2;
    if (std::abs(det) < 1e-10f) {
        cp1 = Vec2f::Lerp(p0, p3, 1.f / 3.f);
        cp2 = Vec2f::Lerp(p0, p3, 2.f / 3.f);
    } else {
        float inv_det = 1.f / det;
        cp1           = (c1 * a22 - c2 * a12) * inv_det;
        cp2           = (c2 * a11 - c1 * a12) * inv_det;
    }

    return {p0, cp1, cp2, p3};
}

bool TryMergeSegments(const CubicBezier& a, const CubicBezier& b, float merge_eps,
                      CubicBezier& result) {
    constexpr int kSamplesPerSeg = 12;
    auto sa                      = SampleBezier(a, kSamplesPerSeg);
    auto sb                      = SampleBezier(b, kSamplesPerSeg);

    std::vector<Vec2f> all_pts;
    all_pts.reserve(sa.size() + sb.size());
    all_pts.insert(all_pts.end(), sa.begin(), sa.end());
    for (size_t i = 1; i < sb.size(); ++i) all_pts.push_back(sb[i]);

    CubicBezier candidate = FitCubicToPoints(all_pts);
    float err             = MaxDeviationFromBezier(candidate, all_pts);

    if (err <= merge_eps) {
        result = candidate;
        return true;
    }
    return false;
}

} // namespace

void OptimizeBezierContour(BezierContour& contour, float linear_eps, float merge_eps) {
    if (contour.segments.size() <= 1) return;

    // Pass 1: Collapse near-linear segments to line segments.
    // Consecutive collinear segments are merged; corners are preserved.
    {
        std::vector<CubicBezier> pass1;
        pass1.reserve(contour.segments.size());
        size_t i = 0;
        while (i < contour.segments.size()) {
            if (IsNearLinear(contour.segments[i], linear_eps)) {
                Vec2f start = contour.segments[i].p0;
                Vec2f end   = contour.segments[i].p3;
                Vec2f dir   = (end - start).Normalized();
                size_t j    = i + 1;
                while (j < contour.segments.size() &&
                       IsNearLinear(contour.segments[j], linear_eps)) {
                    Vec2f next_end = contour.segments[j].p3;
                    Vec2f next_dir = (next_end - end).Normalized();
                    // Only merge if roughly collinear (dot product > 0.95).
                    if (dir.Length() > 1e-6f && next_dir.Length() > 1e-6f &&
                        dir.Dot(next_dir) > 0.95f) {
                        end = next_end;
                        dir = (end - start).Normalized();
                        ++j;
                    } else {
                        break;
                    }
                }
                float chord = Vec2f::Distance(start, end);
                if (chord < 1e-4f) {
                    // Degenerate: keep the individual segments as lines instead.
                    for (size_t k = i; k < j; ++k) {
                        pass1.push_back(
                            MakeLinearBezier(contour.segments[k].p0, contour.segments[k].p3));
                    }
                } else {
                    pass1.push_back(MakeLinearBezier(start, end));
                }
                i = j;
            } else {
                pass1.push_back(contour.segments[i]);
                ++i;
            }
        }
        contour.segments = std::move(pass1);
    }

    // Pass 2: Try to merge adjacent segments by re-fitting.
    // Chain-merge is capped to avoid a single cubic representing too many
    // original segments, which causes visible dents on smooth arcs.
    if (contour.segments.size() > 2 && merge_eps > 0.f) {
        constexpr int kMaxChainLength = 3;
        std::vector<CubicBezier> pass2;
        pass2.reserve(contour.segments.size());
        size_t i = 0;
        while (i < contour.segments.size()) {
            CubicBezier current = contour.segments[i];
            ++i;
            int chain_count = 1;
            while (i < contour.segments.size() && chain_count < kMaxChainLength) {
                CubicBezier merged;
                if (TryMergeSegments(current, contour.segments[i], merge_eps, merged)) {
                    current = merged;
                    ++i;
                    ++chain_count;
                } else {
                    break;
                }
            }
            pass2.push_back(current);
        }
        contour.segments = std::move(pass2);
    }
}

void OptimizeShapePaths(std::vector<VectorizedShape>& shapes, float linear_eps, float merge_eps) {
    int total_before = 0, total_after = 0;
    const int n = static_cast<int>(shapes.size());

#pragma omp parallel for schedule(dynamic) reduction(+ : total_before, total_after)
    for (int i = 0; i < n; ++i) {
        for (auto& contour : shapes[i].contours) {
            total_before += static_cast<int>(contour.segments.size());
            OptimizeBezierContour(contour, linear_eps, merge_eps);
            total_after += static_cast<int>(contour.segments.size());
        }
    }
    spdlog::info(
        "OptimizeShapePaths: segments {} -> {} ({:.1f}% reduction)", total_before, total_after,
        total_before > 0 ? 100.0 * (1.0 - static_cast<double>(total_after) / total_before) : 0.0);
}

} // namespace neroued::vectorizer::detail
