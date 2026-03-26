#include "depth_order.h"

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

cv::Mat BuildLayerMap(const std::vector<ShapeLayer>& layers, int img_rows, int img_cols) {
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
    return layer_map;
}

AdjSet BuildAdjacency(const cv::Mat& layer_map, int img_rows, int img_cols) {
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

struct BorderInfo {
    std::vector<bool> touches_top, touches_bottom, touches_left, touches_right;
    std::vector<int> border_px;
    int total_border_px = 0;
};

BorderInfo CollectBorderInfo(const std::vector<ShapeLayer>& layers, int N, int img_rows,
                             int img_cols, const cv::Mat& layer_map) {
    BorderInfo info;
    info.touches_top.assign(N, false);
    info.touches_bottom.assign(N, false);
    info.touches_left.assign(N, false);
    info.touches_right.assign(N, false);

    for (int i = 0; i < N; ++i) {
        const auto& bbox = layers[i].bbox;
        const auto& mask = layers[i].mask;

        if (bbox.y == 0) {
            const auto* row = mask.ptr<uint8_t>(0);
            for (int c = 0; c < bbox.width && !info.touches_top[i]; ++c)
                if (row[c] > 0) info.touches_top[i] = true;
        }
        if (bbox.y + bbox.height >= img_rows) {
            const auto* row = mask.ptr<uint8_t>(img_rows - 1 - bbox.y);
            for (int c = 0; c < bbox.width && !info.touches_bottom[i]; ++c)
                if (row[c] > 0) info.touches_bottom[i] = true;
        }
        if (bbox.x == 0) {
            for (int r = 0; r < bbox.height && !info.touches_left[i]; ++r)
                if (mask.at<uint8_t>(r, 0) > 0) info.touches_left[i] = true;
        }
        if (bbox.x + bbox.width >= img_cols) {
            int local_c = img_cols - 1 - bbox.x;
            for (int r = 0; r < bbox.height && !info.touches_right[i]; ++r)
                if (mask.at<uint8_t>(r, local_c) > 0) info.touches_right[i] = true;
        }
    }

    constexpr int kBorderWidth = 3;
    info.border_px.assign(N, 0);
    for (int r = 0; r < img_rows; ++r) {
        if (r >= kBorderWidth && r < img_rows - kBorderWidth) continue;
        const auto* row = layer_map.ptr<int>(r);
        for (int c = 0; c < img_cols; ++c) {
            if (r >= kBorderWidth && c >= kBorderWidth && c < img_cols - kBorderWidth) continue;
            int idx = row[c];
            if (idx >= 0) {
                info.border_px[idx]++;
                info.total_border_px++;
            }
        }
    }

    return info;
}

int SelectBackground(const std::vector<ShapeLayer>& layers, const BorderInfo& info) {
    const int N = static_cast<int>(layers.size());

    int best         = -1;
    double best_area = -1.0;
    for (int i = 0; i < N; ++i) {
        if (info.touches_top[i] && info.touches_bottom[i] && info.touches_left[i] &&
            info.touches_right[i]) {
            if (layers[i].area > best_area) {
                best      = i;
                best_area = layers[i].area;
            }
        }
    }
    if (best >= 0) return best;

    double max_area = 0.0;
    for (int i = 0; i < N; ++i) max_area = std::max(max_area, layers[i].area);

    double best_score = -1.0;
    for (int i = 0; i < N; ++i) {
        int sides =
            static_cast<int>(info.touches_top[i]) + static_cast<int>(info.touches_bottom[i]) +
            static_cast<int>(info.touches_left[i]) + static_cast<int>(info.touches_right[i]);
        if (sides < 2) continue;
        double side_ratio   = sides / 4.0;
        double area_ratio   = (max_area > 0.0) ? layers[i].area / max_area : 0.0;
        double border_ratio = (info.total_border_px > 0)
                                  ? static_cast<double>(info.border_px[i]) / info.total_border_px
                                  : 0.0;

        double score = side_ratio * 0.3 + area_ratio * 0.3 + border_ratio * 0.4;
        if (score > best_score) {
            best_score = score;
            best       = i;
        }
    }
    if (best >= 0) return best;

    best = 0;
    for (int i = 1; i < N; ++i) {
        if (layers[i].area > layers[best].area) best = i;
    }
    return best;
}

int FindBackground(const std::vector<ShapeLayer>& layers, int img_rows, int img_cols,
                   const cv::Mat& layer_map) {
    const int N = static_cast<int>(layers.size());
    auto info   = CollectBorderInfo(layers, N, img_rows, img_cols, layer_map);
    return SelectBackground(layers, info);
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

RoiMask MakeDilatedMask(const ShapeLayer& layer, int img_rows, int img_cols, int radius) {
    int x0 = std::max(0, layer.bbox.x - radius);
    int y0 = std::max(0, layer.bbox.y - radius);
    int x1 = std::min(img_cols, layer.bbox.x + layer.bbox.width + radius);
    int y1 = std::min(img_rows, layer.bbox.y + layer.bbox.height + radius);
    cv::Rect expanded(x0, y0, x1 - x0, y1 - y0);

    cv::Mat local = cv::Mat::zeros(expanded.size(), CV_8UC1);
    cv::Rect src_roi(layer.bbox.x - x0, layer.bbox.y - y0, layer.bbox.width, layer.bbox.height);
    layer.mask.copyTo(local(src_roi));

    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1));
    cv::Mat dilated;
    cv::dilate(local, dilated, kernel);
    return {dilated, expanded};
}

struct TarjanSCC {
    const std::vector<std::vector<int>>& graph;
    const std::unordered_set<long long>& removed;
    const std::unordered_set<int>& active;
    int N;

    std::vector<int> disc, low;
    std::vector<bool> on_stack;
    std::vector<int> stk;
    std::vector<std::vector<int>> sccs;
    int timer = 0;

    void Run() {
        disc.assign(N, -1);
        low.assign(N, -1);
        on_stack.assign(N, false);
        for (int u : active) {
            if (disc[u] < 0) Dfs(u);
        }
    }

    void Dfs(int u) {
        disc[u] = low[u] = timer++;
        stk.push_back(u);
        on_stack[u] = true;

        for (int v : graph[u]) {
            long long key = static_cast<long long>(u) * N + v;
            if (!active.count(v) || removed.count(key)) continue;
            if (disc[v] < 0) {
                Dfs(v);
                low[u] = std::min(low[u], low[v]);
            } else if (on_stack[v]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }

        if (low[u] == disc[u]) {
            std::vector<int> scc;
            int w;
            do {
                w = stk.back();
                stk.pop_back();
                on_stack[w] = false;
                scc.push_back(w);
            } while (w != u);
            if (scc.size() > 1) sccs.push_back(std::move(scc));
        }
    }
};

long long DirEdgeKey(int from, int to, int N) { return static_cast<long long>(from) * N + to; }

struct DepthGraph {
    int N;
    std::vector<std::vector<int>> graph;
    std::unordered_map<long long, double> confidence;
};

DepthGraph BuildDepthGraph(const std::vector<ShapeLayer>& layers, const AdjSet& adj, int bg_idx,
                           int img_rows, int img_cols) {
    const int N = static_cast<int>(layers.size());
    DepthGraph dg;
    dg.N = N;
    dg.graph.resize(N);

    constexpr double kDeltaFloor        = 0.02;
    const double significance_threshold = static_cast<double>(img_rows) * img_cols * 0.0002;
    const int dilate_radius             = std::clamp(
        static_cast<int>(std::sqrt(static_cast<double>(img_rows) * img_cols) * 0.008), 5, 20);

    std::unordered_map<int, RoiMask> dilated_cache;
    auto get_dilated = [&](int idx) -> const RoiMask& {
        auto it = dilated_cache.find(idx);
        if (it != dilated_cache.end()) return it->second;
        return dilated_cache
            .emplace(idx, MakeDilatedMask(layers[idx], img_rows, img_cols, dilate_radius))
            .first->second;
    };

    struct PairD {
        int i, j;
        double d_ij, area_i, area_j;
    };

    std::vector<PairD> pair_data;
    std::vector<double> all_d_abs;
    int skipped_small_pairs = 0;

    for (auto& [i, j] : adj) {
        if (i == bg_idx || j == bg_idx) {
            int other = (i == bg_idx) ? j : i;
            dg.graph[bg_idx].push_back(other);
            continue;
        }

        double area_i = layers[i].area;
        double area_j = layers[j].area;
        if (area_i < 1.0 || area_j < 1.0) continue;

        if (area_i < significance_threshold && area_j < significance_threshold) {
            ++skipped_small_pairs;
            continue;
        }

        const auto& ext_j = get_dilated(j);
        const auto& ext_i = get_dilated(i);

        double inter_ij =
            ComputeRoiIntersectionArea(layers[i].bbox, layers[i].mask, ext_j.bbox, ext_j.mask);
        double inter_ji =
            ComputeRoiIntersectionArea(layers[j].bbox, layers[j].mask, ext_i.bbox, ext_i.mask);
        double a_ij     = inter_ij / area_i;
        double a_ji     = inter_ji / area_j;
        double d_ij_val = a_ij - a_ji;

        pair_data.push_back({i, j, d_ij_val, area_i, area_j});
        if (std::abs(d_ij_val) > 1e-6) all_d_abs.push_back(std::abs(d_ij_val));
    }

    double adaptive_delta = kDeltaFloor;
    if (all_d_abs.size() > 10) {
        std::sort(all_d_abs.begin(), all_d_abs.end());
        adaptive_delta = std::max(kDeltaFloor, all_d_abs[all_d_abs.size() / 4]);
    }

    for (const auto& p : pair_data) {
        double conf = std::abs(p.d_ij) * std::log2(std::max(p.area_i, p.area_j) + 1.0);

        if (p.d_ij > adaptive_delta) {
            dg.graph[p.j].push_back(p.i);
            dg.confidence[DirEdgeKey(p.j, p.i, N)] = conf;
        } else if (p.d_ij < -adaptive_delta) {
            dg.graph[p.i].push_back(p.j);
            dg.confidence[DirEdgeKey(p.i, p.j, N)] = conf;
        } else {
            constexpr double kAreaRatioFallback = 3.0;
            double ratio = std::max(p.area_i, p.area_j) / std::min(p.area_i, p.area_j);
            if (ratio > kAreaRatioFallback) {
                int big   = (p.area_i > p.area_j) ? p.i : p.j;
                int small = (p.area_i > p.area_j) ? p.j : p.i;
                dg.graph[big].push_back(small);
                dg.confidence[DirEdgeKey(big, small, N)] = kDeltaFloor * 0.1;
            }
        }
    }

    spdlog::debug(
        "ComputeDepthOrder: skipped_small_pairs={}, dilate_radius={}, adaptive_delta={:.4f}",
        skipped_small_pairs, dilate_radius, adaptive_delta);

    return dg;
}

void AddContainmentEdges(DepthGraph& dg, const std::vector<ShapeLayer>& layers, int bg_idx) {
    constexpr double kContainAreaRatio = 4.0;
    constexpr int kMaxCandidates       = 50;
    constexpr double kDeltaFloor       = 0.02;
    const int N                        = dg.N;

    std::vector<int> by_bbox_area(N);
    std::iota(by_bbox_area.begin(), by_bbox_area.end(), 0);
    std::sort(by_bbox_area.begin(), by_bbox_area.end(),
              [&](int a, int b) { return layers[a].bbox.area() > layers[b].bbox.area(); });

    std::unordered_set<long long> existing_edges;
    for (int u = 0; u < N; ++u)
        for (int v : dg.graph[u]) existing_edges.insert(DirEdgeKey(u, v, N));

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
            if (existing_edges.count(DirEdgeKey(outer, inner, N))) continue;
            if (existing_edges.count(DirEdgeKey(inner, outer, N))) continue;

            if (!BboxStrictlyContains(ob, layers[inner].bbox)) continue;

            dg.graph[outer].push_back(inner);
            dg.confidence[DirEdgeKey(outer, inner, N)] = kDeltaFloor * 0.05;
            existing_edges.insert(DirEdgeKey(outer, inner, N));
            ++containment_ct;
        }
    }
    if (containment_ct > 0)
        spdlog::debug("ComputeDepthOrder: added {} containment edges", containment_ct);
}

std::vector<int> TopologicalSortWithCycleBreaking(const DepthGraph& dg,
                                                  const std::vector<ShapeLayer>& layers) {
    const int N = dg.N;
    std::unordered_set<long long> removed_edges;

    auto area_cmp = [&](int a, int b) { return layers[a].area < layers[b].area; };
    using AreaPQ  = std::priority_queue<int, std::vector<int>, decltype(area_cmp)>;

    auto run_kahn = [&]() -> std::vector<int> {
        std::vector<int> deg(N, 0);
        for (int u = 0; u < N; ++u) {
            for (int v : dg.graph[u]) {
                if (removed_edges.count(DirEdgeKey(u, v, N))) continue;
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
            for (int v : dg.graph[u]) {
                if (removed_edges.count(DirEdgeKey(u, v, N))) continue;
                if (--deg[v] == 0) q.push(v);
            }
        }
        return order;
    };

    std::vector<int> topo_order = run_kahn();
    int removed_count           = 0;
    int scc_rounds              = 0;

    while (static_cast<int>(topo_order.size()) < N) {
        ++scc_rounds;
        std::unordered_set<int> placed(topo_order.begin(), topo_order.end());
        std::unordered_set<int> active;
        for (int i = 0; i < N; ++i)
            if (!placed.count(i)) active.insert(i);

        TarjanSCC tarjan{dg.graph, removed_edges, active, N, {}, {}, {}, {}, {}, 0};
        tarjan.Run();

        if (tarjan.sccs.empty()) break;

        int batch = 0;
        for (const auto& scc : tarjan.sccs) {
            std::unordered_set<int> scc_set(scc.begin(), scc.end());
            long long weakest_key = -1;
            double weakest_conf   = std::numeric_limits<double>::max();

            for (int u : scc) {
                for (int v : dg.graph[u]) {
                    if (!scc_set.count(v)) continue;
                    long long key = DirEdgeKey(u, v, N);
                    if (removed_edges.count(key)) continue;
                    double conf = std::numeric_limits<double>::max();
                    auto it     = dg.confidence.find(key);
                    if (it != dg.confidence.end()) conf = it->second;
                    if (conf < weakest_conf) {
                        weakest_conf = conf;
                        weakest_key  = key;
                    }
                }
            }

            if (weakest_key >= 0) {
                removed_edges.insert(weakest_key);
                ++batch;
            }
        }

        removed_count += batch;
        spdlog::debug("ComputeDepthOrder: SCC round {}: {} SCCs, removed {} edges", scc_rounds,
                      tarjan.sccs.size(), batch);

        topo_order = run_kahn();
    }

    if (removed_count > 0) {
        spdlog::warn("ComputeDepthOrder: removed {} edge(s) in {} SCC round(s)", removed_count,
                     scc_rounds);
    }

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

    return topo_order;
}

} // namespace

std::vector<ShapeLayer> ExtractShapeLayers(const cv::Mat& labels, int num_labels, double min_area) {
    const int rows = labels.rows;
    const int cols = labels.cols;

    cv::Mat cc_labels(rows, cols, CV_32SC1, cv::Scalar(0));
    int next_cc = 1;

    struct CCInfo {
        int label;
        int min_r, min_c, max_r, max_c;
        int area;
    };

    std::vector<CCInfo> cc_infos;
    cc_infos.reserve(256);

    std::vector<std::pair<int, int>> stack;
    stack.reserve(std::max(rows, cols) * 4);

    for (int r = 0; r < rows; ++r) {
        const int* lrow  = labels.ptr<int>(r);
        const int* ccrow = cc_labels.ptr<int>(r);
        for (int c = 0; c < cols; ++c) {
            if (lrow[c] < 0 || ccrow[c] != 0) continue;

            int lid   = lrow[c];
            int cc_id = next_cc++;
            CCInfo info{lid, r, c, r, c, 0};

            stack.clear();
            stack.push_back({r, c});
            cc_labels.at<int>(r, c) = cc_id;

            while (!stack.empty()) {
                auto [cr, cc_] = stack.back();
                stack.pop_back();
                info.area++;
                info.min_r = std::min(info.min_r, cr);
                info.max_r = std::max(info.max_r, cr);
                info.min_c = std::min(info.min_c, cc_);
                info.max_c = std::max(info.max_c, cc_);

                constexpr int dr[] = {-1, 1, 0, 0};
                constexpr int dc[] = {0, 0, -1, 1};
                for (int d = 0; d < 4; ++d) {
                    int nr = cr + dr[d], nc = cc_ + dc[d];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                    if (cc_labels.at<int>(nr, nc) != 0) continue;
                    if (labels.at<int>(nr, nc) != lid) continue;
                    cc_labels.at<int>(nr, nc) = cc_id;
                    stack.push_back({nr, nc});
                }
            }

            cc_infos.push_back(info);
        }
    }

    std::vector<ShapeLayer> layers;
    int skipped = 0;

    for (int i = 0; i < static_cast<int>(cc_infos.size()); ++i) {
        const auto& info = cc_infos[i];
        if (info.area < min_area) {
            ++skipped;
            continue;
        }

        int cc_id = i + 1;
        cv::Rect bbox(info.min_c, info.min_r, info.max_c - info.min_c + 1,
                      info.max_r - info.min_r + 1);
        cv::Mat mask(bbox.height, bbox.width, CV_8UC1, cv::Scalar(0));

        for (int r = bbox.y; r < bbox.y + bbox.height; ++r) {
            const int* ccrow = cc_labels.ptr<int>(r);
            auto* mrow       = mask.ptr<uint8_t>(r - bbox.y);
            for (int c = bbox.x; c < bbox.x + bbox.width; ++c) {
                if (ccrow[c] == cc_id) mrow[c - bbox.x] = 255;
            }
        }

        ShapeLayer layer;
        layer.label = info.label;
        layer.cc_id = cc_id;
        layer.bbox  = bbox;
        layer.mask  = std::move(mask);
        layer.area  = info.area;
        layers.push_back(std::move(layer));
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

    cv::Mat layer_map = BuildLayerMap(layers, img_rows, img_cols);
    int bg_idx        = FindBackground(layers, img_rows, img_cols, layer_map);
    spdlog::debug("ComputeDepthOrder: N={}, background_idx={}, background_area={:.0f}", N, bg_idx,
                  bg_idx >= 0 ? layers[bg_idx].area : 0.0);

    auto adj = BuildAdjacency(layer_map, img_rows, img_cols);
    spdlog::debug("ComputeDepthOrder: adjacent_pairs={}", adj.size());

    auto dg = BuildDepthGraph(layers, adj, bg_idx, img_rows, img_cols);
    AddContainmentEdges(dg, layers, bg_idx);
    auto topo_order = TopologicalSortWithCycleBreaking(dg, layers);

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
