#include "boundary_graph.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <unordered_map>
#include <vector>

namespace neroued::vectorizer::detail {

namespace {

struct IVec2 {
    int x, y;

    bool operator==(const IVec2& o) const { return x == o.x && y == o.y; }
};

struct IVec2Hash {
    std::size_t operator()(const IVec2& v) const {
        return std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 16);
    }
};

using LabelPair = std::pair<int, int>;

LabelPair MakeOrderedPair(int a, int b) { return a < b ? LabelPair{a, b} : LabelPair{b, a}; }

struct LabelPairHash {
    std::size_t operator()(const LabelPair& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 16);
    }
};

int GetLabel(const cv::Mat& labels, int r, int c) {
    if (r < 0 || r >= labels.rows || c < 0 || c >= labels.cols) return -1;
    return labels.at<int>(r, c);
}

struct CrackEdge {
    IVec2 v0, v1;
    int label_a, label_b;
};

// Crack grid vertices are at integer positions on a (H+1) x (W+1) grid.
// Horizontal crack between pixel (r, c) and (r+1, c) connects
//   vertex (r+1, c) to vertex (r+1, c+1).
// Vertical crack between pixel (r, c) and (r, c+1) connects
//   vertex (r, c+1) to vertex (r+1, c+1).

void CollectCracks(const cv::Mat& labels, std::vector<CrackEdge>& cracks,
                   std::unordered_map<IVec2, std::vector<int>, IVec2Hash>& vertex_cracks) {
    const int H = labels.rows;
    const int W = labels.cols;

    auto addCrack = [&](IVec2 v0, IVec2 v1, int la, int lb) {
        int idx = static_cast<int>(cracks.size());
        cracks.push_back({v0, v1, la, lb});
        vertex_cracks[v0].push_back(idx);
        vertex_cracks[v1].push_back(idx);
    };

    // Horizontal cracks: between row r and row r+1
    for (int r = 0; r < H - 1; ++r) {
        const int* row0 = labels.ptr<int>(r);
        const int* row1 = labels.ptr<int>(r + 1);
        for (int c = 0; c < W; ++c) {
            int la = row0[c];
            int lb = row1[c];
            if (la == lb) continue;
            if (la < 0 && lb < 0) continue;
            IVec2 v0{c, r + 1};
            IVec2 v1{c + 1, r + 1};
            addCrack(v0, v1, la, lb);
        }
    }

    // Vertical cracks: between col c and col c+1
    for (int r = 0; r < H; ++r) {
        const int* row = labels.ptr<int>(r);
        for (int c = 0; c < W - 1; ++c) {
            int la = row[c];
            int lb = row[c + 1];
            if (la == lb) continue;
            if (la < 0 && lb < 0) continue;
            IVec2 v0{c + 1, r};
            IVec2 v1{c + 1, r + 1};
            addCrack(v0, v1, la, lb);
        }
    }

    // Image border cracks (boundary between label and "outside" = -1)
    // Top border: between virtual row -1 (label=-1) and row 0
    for (int c = 0; c < W; ++c) {
        int la = labels.at<int>(0, c);
        if (la < 0) continue;
        addCrack({c, 0}, {c + 1, 0}, -1, la);
    }
    // Bottom border
    for (int c = 0; c < W; ++c) {
        int la = labels.at<int>(H - 1, c);
        if (la < 0) continue;
        addCrack({c, H}, {c + 1, H}, la, -1);
    }
    // Left border
    for (int r = 0; r < H; ++r) {
        int la = labels.at<int>(r, 0);
        if (la < 0) continue;
        addCrack({0, r}, {0, r + 1}, -1, la);
    }
    // Right border
    for (int r = 0; r < H; ++r) {
        int la = labels.at<int>(r, W - 1);
        if (la < 0) continue;
        addCrack({W, r}, {W, r + 1}, la, -1);
    }
}

bool IsJunctionVertex(const std::unordered_map<IVec2, std::vector<int>, IVec2Hash>& vertex_cracks,
                      const std::vector<CrackEdge>& cracks, const IVec2& v) {
    auto it = vertex_cracks.find(v);
    if (it == vertex_cracks.end()) return false;
    const auto& ci = it->second;
    if (ci.size() != 2) return true;
    const auto& c0 = cracks[ci[0]];
    const auto& c1 = cracks[ci[1]];
    auto p0        = MakeOrderedPair(c0.label_a, c0.label_b);
    auto p1        = MakeOrderedPair(c1.label_a, c1.label_b);
    return p0 != p1;
}

// Determine correct label_left/label_right by sampling the label map on both
// sides of the edge's midpoint. This is robust against crack direction conventions.
void DetermineEdgeSides(BoundaryEdge& edge, const cv::Mat& labels) {
    if (edge.points.size() < 2) return;

    int mid = static_cast<int>(edge.points.size()) / 2;
    int i0  = std::max(0, mid - 1);
    int i1  = std::min(static_cast<int>(edge.points.size()) - 1, mid);
    if (i0 == i1 && edge.points.size() >= 2) {
        i0 = 0;
        i1 = 1;
    }

    Vec2f p0  = edge.points[i0];
    Vec2f p1  = edge.points[i1];
    Vec2f dir = p1 - p0;
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    if (len < 1e-6f) return;

    // "Left" in screen coords (y down): rotate direction 90° to the left = (dy, -dx)
    Vec2f n_left = {dir.y / len, -dir.x / len};
    Vec2f mid_pt = (p0 + p1) * 0.5f;

    Vec2f left_pt  = mid_pt + n_left * 0.5f;
    Vec2f right_pt = mid_pt - n_left * 0.5f;

    auto sample = [&](Vec2f pt) -> int {
        int r = static_cast<int>(std::floor(pt.y));
        int c = static_cast<int>(std::floor(pt.x));
        if (r < 0 || r >= labels.rows || c < 0 || c >= labels.cols) return -1;
        return labels.at<int>(r, c);
    };

    int ll = sample(left_pt);
    int lr = sample(right_pt);

    auto expected = MakeOrderedPair(edge.label_left, edge.label_right);
    auto sampled  = MakeOrderedPair(ll, lr);
    if (sampled == expected) {
        edge.label_left  = ll;
        edge.label_right = lr;
    }
}

} // namespace

BoundaryGraph BuildBoundaryGraph(const cv::Mat& labels) {
    BoundaryGraph graph;
    if (labels.empty() || labels.type() != CV_32SC1) {
        spdlog::warn("BuildBoundaryGraph skipped: invalid labels (empty={} type={})",
                     labels.empty(), labels.empty() ? -1 : labels.type());
        return graph;
    }
    const auto start = std::chrono::steady_clock::now();
    spdlog::debug("BuildBoundaryGraph start: labels={}x{}", labels.cols, labels.rows);

    std::vector<CrackEdge> cracks;
    std::unordered_map<IVec2, std::vector<int>, IVec2Hash> vertex_cracks;
    CollectCracks(labels, cracks, vertex_cracks);
    if (cracks.empty()) {
        spdlog::debug("BuildBoundaryGraph no cracks found: labels={}x{}", labels.cols, labels.rows);
        return graph;
    }

    // Identify junction vertices
    std::unordered_map<IVec2, int, IVec2Hash> junction_to_node;
    for (auto& [v, ci] : vertex_cracks) {
        if (IsJunctionVertex(vertex_cracks, cracks, v)) {
            int nid = static_cast<int>(graph.nodes.size());
            graph.nodes.push_back({{static_cast<float>(v.x), static_cast<float>(v.y)}, {}});
            junction_to_node[v] = nid;
        }
    }

    // Trace chains of cracks between junctions
    std::vector<bool> crack_used(cracks.size(), false);
    int closed_loop_count = 0;
    int dead_end_count    = 0;

    for (auto& [start_v, start_nid] : junction_to_node) {
        auto& incident = vertex_cracks[start_v];
        for (int ci : incident) {
            if (crack_used[ci]) continue;

            // Start a new edge chain from this junction through this crack
            std::vector<Vec2f> chain;
            chain.push_back({static_cast<float>(start_v.x), static_cast<float>(start_v.y)});

            int cur_crack     = ci;
            IVec2 cur_v       = start_v;
            int chain_label_a = cracks[ci].label_a;
            int chain_label_b = cracks[ci].label_b;

            int end_node = -1;

            while (true) {
                crack_used[cur_crack] = true;
                const auto& ck        = cracks[cur_crack];
                IVec2 next_v          = (ck.v0 == cur_v) ? ck.v1 : ck.v0;
                chain.push_back({static_cast<float>(next_v.x), static_cast<float>(next_v.y)});

                auto jit = junction_to_node.find(next_v);
                if (jit != junction_to_node.end()) {
                    end_node = jit->second;
                    break;
                }

                // Find the next crack from next_v with the same label pair
                auto pair_want            = MakeOrderedPair(chain_label_a, chain_label_b);
                const auto& next_incident = vertex_cracks[next_v];
                int found                 = -1;
                for (int ni : next_incident) {
                    if (ni == cur_crack || crack_used[ni]) continue;
                    auto pair_ni = MakeOrderedPair(cracks[ni].label_a, cracks[ni].label_b);
                    if (pair_ni == pair_want) {
                        found = ni;
                        break;
                    }
                }

                if (found < 0) {
                    // Dead end — treat as virtual junction
                    ++dead_end_count;
                    int nid = static_cast<int>(graph.nodes.size());
                    graph.nodes.push_back(
                        {{static_cast<float>(next_v.x), static_cast<float>(next_v.y)}, {}});
                    junction_to_node[next_v] = nid;
                    end_node                 = nid;
                    break;
                }

                cur_crack = found;
                cur_v     = next_v;
            }

            if (end_node < 0 || chain.size() < 2) continue;

            // Determine consistent left/right orientation.
            // Convention: walking from v0 to v1 of the first crack,
            // label_a is to the "top/left" and label_b to "bottom/right".
            int ll = chain_label_a;
            int lr = chain_label_b;

            int eid = static_cast<int>(graph.edges.size());
            graph.edges.push_back({start_nid, end_node, ll, lr, std::move(chain)});
            DetermineEdgeSides(graph.edges.back(), labels);
            graph.nodes[start_nid].edge_ids.push_back(eid);
            graph.nodes[end_node].edge_ids.push_back(eid);
        }
    }

    // Handle closed loops (no junction on the loop)
    for (int ci = 0; ci < static_cast<int>(cracks.size()); ++ci) {
        if (crack_used[ci]) continue;

        auto pair = MakeOrderedPair(cracks[ci].label_a, cracks[ci].label_b);
        std::vector<Vec2f> chain;
        IVec2 start_v = cracks[ci].v0;
        chain.push_back({static_cast<float>(start_v.x), static_cast<float>(start_v.y)});

        int cur_crack = ci;
        IVec2 cur_v   = start_v;
        bool closed   = false;

        while (true) {
            crack_used[cur_crack] = true;
            const auto& ck        = cracks[cur_crack];
            IVec2 next_v          = (ck.v0 == cur_v) ? ck.v1 : ck.v0;

            if (next_v == start_v) {
                closed = true;
                break;
            }

            chain.push_back({static_cast<float>(next_v.x), static_cast<float>(next_v.y)});

            const auto& next_incident = vertex_cracks[next_v];
            int found                 = -1;
            for (int ni : next_incident) {
                if (ni == cur_crack || crack_used[ni]) continue;
                if (MakeOrderedPair(cracks[ni].label_a, cracks[ni].label_b) == pair) {
                    found = ni;
                    break;
                }
            }
            if (found < 0) {
                ++dead_end_count;
                break;
            }
            cur_crack = found;
            cur_v     = next_v;
        }

        if (!closed || chain.size() < 3) continue;

        // Create a self-loop edge with a virtual node
        int nid = static_cast<int>(graph.nodes.size());
        graph.nodes.push_back({{static_cast<float>(start_v.x), static_cast<float>(start_v.y)}, {}});
        chain.push_back(chain.front());

        int eid = static_cast<int>(graph.edges.size());
        graph.edges.push_back({nid, nid, cracks[ci].label_a, cracks[ci].label_b, std::move(chain)});
        DetermineEdgeSides(graph.edges.back(), labels);
        graph.nodes[nid].edge_ids.push_back(eid);
        ++closed_loop_count;
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::debug(
        "BuildBoundaryGraph done: cracks={}, vertices={}, nodes={}, edges={}, closed_loops={}, "
        "dead_ends={}, elapsed_ms={:.2f}",
        cracks.size(), vertex_cracks.size(), graph.nodes.size(), graph.edges.size(),
        closed_loop_count, dead_end_count, elapsed_ms);
    return graph;
}

} // namespace neroued::vectorizer::detail
