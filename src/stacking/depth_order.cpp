#include "stacking/depth_order.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace neroued::vectorizer::detail {

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<long long>()(static_cast<long long>(p.first) << 32 | p.second);
    }
};

using AdjSet = std::unordered_set<std::pair<int, int>, PairHash>;

AdjSet BuildAdjacency(const std::vector<ShapeLayer>& layers, int img_rows, int img_cols) {
    cv::Mat layer_map(img_rows, img_cols, CV_32SC1, cv::Scalar(-1));
    for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
        const auto& bbox = layers[i].bbox;
        const auto& mask = layers[i].mask;
        for (int r = 0; r < bbox.height; ++r) {
            const auto* mrow = mask.ptr<uint8_t>(r);
            auto* lrow       = layer_map.ptr<int>(r + bbox.y);
            for (int c = 0; c < bbox.width; ++c) {
                if (mrow[c] > 0) lrow[c + bbox.x] = i;
            }
        }
    }

    AdjSet adj;
    for (int r = 0; r < img_rows; ++r) {
        const auto* row = layer_map.ptr<int>(r);
        for (int c = 0; c < img_cols; ++c) {
            int cur = row[c];
            if (cur < 0) continue;
            if (c + 1 < img_cols) {
                int right = row[c + 1];
                if (right >= 0 && right != cur) {
                    adj.emplace(std::min(cur, right), std::max(cur, right));
                }
            }
            if (r + 1 < img_rows) {
                int down = layer_map.ptr<int>(r + 1)[c];
                if (down >= 0 && down != cur) {
                    adj.emplace(std::min(cur, down), std::max(cur, down));
                }
            }
        }
    }
    return adj;
}

int FindBackground(const std::vector<ShapeLayer>& layers, int img_rows, int img_cols) {
    const int N = static_cast<int>(layers.size());
    std::vector<bool> touches_top(N, false), touches_bottom(N, false);
    std::vector<bool> touches_left(N, false), touches_right(N, false);

    for (int i = 0; i < N; ++i) {
        const auto& bbox = layers[i].bbox;
        const auto& mask = layers[i].mask;

        if (bbox.y == 0) {
            const auto* row = mask.ptr<uint8_t>(0);
            for (int c = 0; c < bbox.width && !touches_top[i]; ++c)
                if (row[c] > 0) touches_top[i] = true;
        }
        if (bbox.y + bbox.height >= img_rows) {
            const auto* row = mask.ptr<uint8_t>(img_rows - 1 - bbox.y);
            for (int c = 0; c < bbox.width && !touches_bottom[i]; ++c)
                if (row[c] > 0) touches_bottom[i] = true;
        }
        if (bbox.x == 0) {
            for (int r = 0; r < bbox.height && !touches_left[i]; ++r)
                if (mask.at<uint8_t>(r, 0) > 0) touches_left[i] = true;
        }
        if (bbox.x + bbox.width >= img_cols) {
            int local_c = img_cols - 1 - bbox.x;
            for (int r = 0; r < bbox.height && !touches_right[i]; ++r)
                if (mask.at<uint8_t>(r, local_c) > 0) touches_right[i] = true;
        }
    }

    int best         = -1;
    double best_area = -1.0;
    for (int i = 0; i < N; ++i) {
        if (touches_top[i] && touches_bottom[i] && touches_left[i] && touches_right[i]) {
            if (layers[i].area > best_area) {
                best      = i;
                best_area = layers[i].area;
            }
        }
    }
    if (best >= 0) return best;

    for (int i = 0; i < N; ++i) {
        int sides = static_cast<int>(touches_top[i]) + static_cast<int>(touches_bottom[i]) +
                    static_cast<int>(touches_left[i]) + static_cast<int>(touches_right[i]);
        if (sides >= 2 && layers[i].area > best_area) {
            best      = i;
            best_area = layers[i].area;
        }
    }
    if (best >= 0) return best;

    best = 0;
    for (int i = 1; i < N; ++i) {
        if (layers[i].area > layers[best].area) best = i;
    }
    return best;
}

struct RoiMask {
    cv::Mat mask;
    cv::Rect bbox;
};

double ComputeRoiIntersectionArea(const cv::Rect& a_bbox, const cv::Mat& a_mask,
                                  const cv::Rect& b_bbox, const cv::Mat& b_mask) {
    cv::Rect overlap = a_bbox & b_bbox;
    if (overlap.area() <= 0) return 0.0;

    cv::Rect a_roi(overlap.x - a_bbox.x, overlap.y - a_bbox.y, overlap.width, overlap.height);
    cv::Rect b_roi(overlap.x - b_bbox.x, overlap.y - b_bbox.y, overlap.width, overlap.height);

    cv::Mat inter;
    cv::bitwise_and(a_mask(a_roi), b_mask(b_roi), inter);
    return cv::countNonZero(inter);
}

bool BboxStrictlyContains(const cv::Rect& outer, const cv::Rect& inner) {
    return inner.x >= outer.x && inner.y >= outer.y &&
           inner.x + inner.width <= outer.x + outer.width &&
           inner.y + inner.height <= outer.y + outer.height && outer.area() > inner.area();
}

RoiMask MakeConvexHullMask(const ShapeLayer& layer) {
    std::vector<cv::Point> pts;
    cv::findNonZero(layer.mask, pts);
    if (pts.empty()) return {cv::Mat::zeros(1, 1, CV_8UC1), layer.bbox};

    for (auto& p : pts) {
        p.x += layer.bbox.x;
        p.y += layer.bbox.y;
    }
    std::vector<cv::Point> hull;
    cv::convexHull(pts, hull);
    if (hull.size() < 3) return {layer.mask.clone(), layer.bbox};

    cv::Rect hull_bbox = cv::boundingRect(hull);
    for (auto& p : hull) {
        p.x -= hull_bbox.x;
        p.y -= hull_bbox.y;
    }
    cv::Mat hull_mask = cv::Mat::zeros(hull_bbox.size(), CV_8UC1);
    cv::fillConvexPoly(hull_mask, hull, cv::Scalar(255));
    return {hull_mask, hull_bbox};
}

} // namespace

std::vector<ShapeLayer> ExtractShapeLayers(const cv::Mat& labels, int num_labels, double min_area) {
    std::vector<ShapeLayer> layers;
    int skipped = 0;

    for (int lid = 0; lid < num_labels; ++lid) {
        cv::Mat label_mask = (labels == lid);
        label_mask.convertTo(label_mask, CV_8UC1, 255);
        if (cv::countNonZero(label_mask) == 0) continue;

        cv::Mat cc_labels;
        int num_cc = cv::connectedComponents(label_mask, cc_labels, 4, CV_32S);

        for (int cc = 1; cc < num_cc; ++cc) {
            cv::Mat cc_mask = (cc_labels == cc);
            cc_mask.convertTo(cc_mask, CV_8UC1, 255);
            double area = cv::countNonZero(cc_mask);
            if (area < min_area) {
                ++skipped;
                continue;
            }

            cv::Rect bbox = cv::boundingRect(cc_mask);

            ShapeLayer layer;
            layer.label = lid;
            layer.cc_id = cc;
            layer.bbox  = bbox;
            layer.mask  = cc_mask(bbox).clone();
            layer.area  = area;
            layers.push_back(std::move(layer));
        }
    }

    spdlog::info("ExtractShapeLayers: num_labels={}, shape_layers={}, skipped_small={}", num_labels,
                 layers.size(), skipped);
    return layers;
}

std::vector<int> ComputeDepthOrder(const std::vector<ShapeLayer>& layers, int img_rows,
                                   int img_cols) {
    const int N = static_cast<int>(layers.size());
    if (N <= 1) {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

    int bg_idx = FindBackground(layers, img_rows, img_cols);
    spdlog::debug("ComputeDepthOrder: N={}, background_idx={}, background_area={:.0f}", N, bg_idx,
                  bg_idx >= 0 ? layers[bg_idx].area : 0.0);

    auto adj = BuildAdjacency(layers, img_rows, img_cols);
    spdlog::debug("ComputeDepthOrder: adjacent_pairs={}", adj.size());

    constexpr double kDelta = 0.02;

    auto DirEdgeKey = [N](int from, int to) -> long long {
        return static_cast<long long>(from) * N + to;
    };

    std::vector<std::vector<int>> graph(N);
    std::unordered_map<long long, double> edge_v;

    std::unordered_map<int, RoiMask> hull_cache;
    auto GetHull = [&](int idx) -> const RoiMask& {
        auto it = hull_cache.find(idx);
        if (it != hull_cache.end()) return it->second;
        return hull_cache.emplace(idx, MakeConvexHullMask(layers[idx])).first->second;
    };

    for (auto& [i, j] : adj) {
        if (i == bg_idx || j == bg_idx) {
            int other = (i == bg_idx) ? j : i;
            graph[bg_idx].push_back(other);
            continue;
        }

        const auto& hull_j = GetHull(j);
        const auto& hull_i = GetHull(i);

        double area_i = layers[i].area;
        double area_j = layers[j].area;
        if (area_i < 1.0 || area_j < 1.0) continue;

        double inter_ij =
            ComputeRoiIntersectionArea(layers[i].bbox, layers[i].mask, hull_j.bbox, hull_j.mask);
        double inter_ji =
            ComputeRoiIntersectionArea(layers[j].bbox, layers[j].mask, hull_i.bbox, hull_i.mask);
        double a_ij = inter_ij / area_i;
        double a_ji = inter_ji / area_j;
        double d_ij = a_ij - a_ji;

        // D(i,j) > 0  →  j below i  (edge j → i)
        // D(i,j) < 0  →  i below j  (edge i → j)
        if (d_ij > kDelta) {
            graph[j].push_back(i);
            edge_v[DirEdgeKey(j, i)] = std::abs(d_ij);
        } else if (d_ij < -kDelta) {
            graph[i].push_back(j);
            edge_v[DirEdgeKey(i, j)] = std::abs(d_ij);
        } else {
            constexpr double kAreaRatioFallback = 3.0;
            double ratio = std::max(area_i, area_j) / std::min(area_i, area_j);
            if (ratio > kAreaRatioFallback) {
                int big   = (area_i > area_j) ? i : j;
                int small = (area_i > area_j) ? j : i;
                graph[big].push_back(small);
                edge_v[DirEdgeKey(big, small)] = kDelta * 0.1;
            }
        }
    }

    // --- Containment edges for non-adjacent nested shapes ---
    {
        constexpr double kContainAreaRatio = 4.0;
        constexpr int kMaxCandidates       = 50;

        std::vector<int> by_bbox_area(N);
        std::iota(by_bbox_area.begin(), by_bbox_area.end(), 0);
        std::sort(by_bbox_area.begin(), by_bbox_area.end(),
                  [&](int a, int b) { return layers[a].bbox.area() > layers[b].bbox.area(); });

        std::unordered_set<long long> existing_edges;
        for (int u = 0; u < N; ++u)
            for (int v : graph[u]) existing_edges.insert(DirEdgeKey(u, v));

        int M              = std::min(kMaxCandidates, N);
        int containment_ct = 0;
        for (int oi = 0; oi < M; ++oi) {
            int outer = by_bbox_area[oi];
            if (outer == bg_idx) continue;
            const auto& ob = layers[outer].bbox;
            for (int ii = oi + 1; ii < N; ++ii) {
                int inner = by_bbox_area[ii];
                if (inner == bg_idx) continue;
                if (layers[outer].area < kContainAreaRatio * layers[inner].area) continue;
                if (existing_edges.count(DirEdgeKey(outer, inner))) continue;
                if (existing_edges.count(DirEdgeKey(inner, outer))) continue;

                if (!BboxStrictlyContains(ob, layers[inner].bbox)) continue;

                graph[outer].push_back(inner);
                edge_v[DirEdgeKey(outer, inner)] = kDelta * 0.05;
                existing_edges.insert(DirEdgeKey(outer, inner));
                ++containment_ct;
            }
        }
        if (containment_ct > 0)
            spdlog::debug("ComputeDepthOrder: added {} containment edges", containment_ct);
    }

    // --- Topological sort with confidence-based cycle removal ---

    std::unordered_set<long long> removed_edges;

    auto area_cmp = [&](int a, int b) { return layers[a].area < layers[b].area; };
    using AreaPQ  = std::priority_queue<int, std::vector<int>, decltype(area_cmp)>;

    auto RunKahn = [&]() -> std::vector<int> {
        std::vector<int> deg(N, 0);
        for (int u = 0; u < N; ++u) {
            for (int v : graph[u]) {
                if (removed_edges.count(DirEdgeKey(u, v))) continue;
                deg[v]++;
            }
        }
        AreaPQ q(area_cmp);
        for (int i = 0; i < N; ++i) {
            if (deg[i] == 0) q.push(i);
        }
        std::vector<int> order;
        order.reserve(N);
        while (!q.empty()) {
            int u = q.top();
            q.pop();
            order.push_back(u);
            for (int v : graph[u]) {
                if (removed_edges.count(DirEdgeKey(u, v))) continue;
                if (--deg[v] == 0) q.push(v);
            }
        }
        return order;
    };

    std::vector<int> topo_order = RunKahn();
    int removed_count           = 0;

    while (static_cast<int>(topo_order.size()) < N) {
        std::unordered_set<int> placed(topo_order.begin(), topo_order.end());

        long long weakest_key = -1;
        double weakest_conf   = std::numeric_limits<double>::max();

        for (int u = 0; u < N; ++u) {
            if (placed.count(u)) continue;
            for (int v : graph[u]) {
                if (placed.count(v)) continue;
                long long key = DirEdgeKey(u, v);
                if (removed_edges.count(key)) continue;

                double conf = std::numeric_limits<double>::max();
                auto it     = edge_v.find(key);
                if (it != edge_v.end()) conf = it->second;

                if (conf < weakest_conf || weakest_key < 0) {
                    weakest_conf = conf;
                    weakest_key  = key;
                }
            }
        }

        if (weakest_key < 0) break;

        removed_edges.insert(weakest_key);
        ++removed_count;
        spdlog::debug("ComputeDepthOrder: removed edge {}->{} (conf={:.4f}) to break cycle",
                      static_cast<int>(weakest_key / N), static_cast<int>(weakest_key % N),
                      weakest_conf);

        topo_order = RunKahn();
    }

    if (removed_count > 0) {
        spdlog::warn("ComputeDepthOrder: removed {} edge(s) to break cycles", removed_count);
    }

    // Area-based fallback for any remaining unplaced nodes (e.g. isolated nodes).
    if (static_cast<int>(topo_order.size()) < N) {
        std::unordered_set<int> in_topo(topo_order.begin(), topo_order.end());
        std::vector<int> remaining;
        for (int i = 0; i < N; ++i) {
            if (!in_topo.count(i)) remaining.push_back(i);
        }
        std::sort(remaining.begin(), remaining.end(),
                  [&](int a, int b) { return layers[a].area > layers[b].area; });
        for (int idx : remaining) topo_order.push_back(idx);
    }

    if (bg_idx >= 0 && !topo_order.empty() && topo_order[0] != bg_idx) {
        auto it = std::find(topo_order.begin(), topo_order.end(), bg_idx);
        if (it != topo_order.end()) {
            topo_order.erase(it);
            topo_order.insert(topo_order.begin(), bg_idx);
        }
    }

    spdlog::info("ComputeDepthOrder: final ordering computed, {} layers", topo_order.size());
    return topo_order;
}

} // namespace neroued::vectorizer::detail
