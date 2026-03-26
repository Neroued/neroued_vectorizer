#include "shape_merge.h"

#include <spdlog/spdlog.h>

namespace neroued::vectorizer::detail {

BBox ComputeShapeBBox(const VectorizedShape& s) {
    BBox bb;
    for (const auto& c : s.contours) {
        for (const auto& seg : c.segments) {
            bb.Expand(seg.p0);
            bb.Expand(seg.p1);
            bb.Expand(seg.p2);
            bb.Expand(seg.p3);
        }
    }
    return bb;
}

void MergeAdjacentSameColorShapes(std::vector<VectorizedShape>& shapes) {
    if (shapes.size() < 2) return;

    constexpr float kMergeColorThreshold = 3.0f;
    auto color_match                     = [](const Rgb& a, const Rgb& b) -> bool {
        Lab la = a.ToLab(), lb = b.ToLab();
        return Lab::DeltaE76(la, lb) < kMergeColorThreshold;
    };

    const size_t n = shapes.size();
    std::vector<VectorizedShape> merged;
    merged.reserve(n);

    size_t i = 0;
    while (i < n) {
        if (shapes[i].is_stroke || shapes[i].contours.empty()) {
            merged.push_back(std::move(shapes[i]));
            ++i;
            continue;
        }

        size_t run_start = i;
        std::vector<BBox> run_bboxes;
        run_bboxes.push_back(ComputeShapeBBox(shapes[i]));

        size_t j = i + 1;
        while (j < n && !shapes[j].is_stroke &&
               color_match(shapes[run_start].color, shapes[j].color)) {
            BBox jbb          = ComputeShapeBBox(shapes[j]);
            bool overlaps_any = false;
            for (const auto& rb : run_bboxes) {
                if (jbb.Overlaps(rb)) {
                    overlaps_any = true;
                    break;
                }
            }
            if (overlaps_any) break;
            run_bboxes.push_back(jbb);
            ++j;
        }

        if (j - run_start == 1) {
            merged.push_back(std::move(shapes[i]));
        } else {
            VectorizedShape combined;
            combined.color = shapes[run_start].color;
            combined.area  = shapes[run_start].area;
            for (size_t k = run_start; k < j; ++k) {
                for (auto& c : shapes[k].contours) combined.contours.push_back(std::move(c));
            }
            merged.push_back(std::move(combined));
        }
        i = j;
    }

    int reduced = static_cast<int>(n) - static_cast<int>(merged.size());
    if (reduced > 0) {
        spdlog::info("MergeAdjacentSameColorShapes: {} -> {} shapes (reduced by {})", n,
                     merged.size(), reduced);
    }
    shapes = std::move(merged);
}

} // namespace neroued::vectorizer::detail
