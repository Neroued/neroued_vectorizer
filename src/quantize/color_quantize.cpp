#include "color_quantize.h"
#include "oklab.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace neroued::vectorizer::detail {

namespace {

constexpr int kGridDim   = 32;
constexpr int kGridTotal = kGridDim * kGridDim * kGridDim;

struct HistBin {
    int count     = 0;
    double sum_l  = 0;
    double sum_a  = 0;
    double sum_b  = 0;
    double sum2_l = 0;
    double sum2_a = 0;
    double sum2_b = 0;
};

struct ColorGrid {
    std::vector<HistBin> bins;
    float L_min = 0, a_min = 0, b_min = 0;
    float L_inv = 0, a_inv = 0, b_inv = 0;

    void Build(const cv::Mat& bgr, cv::Mat& oklab_cache) {
        oklab_cache.create(bgr.rows, bgr.cols, CV_32FC3);

        float lo_L = 1e9f, lo_a = 1e9f, lo_b = 1e9f;
        float hi_L = -1e9f, hi_a = -1e9f, hi_b = -1e9f;
        for (int r = 0; r < bgr.rows; ++r) {
            const auto* brow = bgr.ptr<cv::Vec3b>(r);
            auto* orow       = oklab_cache.ptr<cv::Vec3f>(r);
            for (int c = 0; c < bgr.cols; ++c) {
                auto ok = SrgbToOklab(brow[c][2], brow[c][1], brow[c][0]);
                orow[c] = cv::Vec3f(ok.L, ok.a, ok.b);
                lo_L    = std::min(lo_L, ok.L);
                hi_L    = std::max(hi_L, ok.L);
                lo_a    = std::min(lo_a, ok.a);
                hi_a    = std::max(hi_a, ok.a);
                lo_b    = std::min(lo_b, ok.b);
                hi_b    = std::max(hi_b, ok.b);
            }
        }
        constexpr float kEps = 1e-7f;
        L_min                = lo_L;
        a_min                = lo_a;
        b_min                = lo_b;
        L_inv                = static_cast<float>(kGridDim) / std::max(kEps, hi_L - lo_L);
        a_inv                = static_cast<float>(kGridDim) / std::max(kEps, hi_a - lo_a);
        b_inv                = static_cast<float>(kGridDim) / std::max(kEps, hi_b - lo_b);

        bins.assign(kGridTotal, HistBin{});
#ifdef _OPENMP
        const int nt = omp_get_max_threads();
#else
        const int nt = 1;
#endif
        std::vector<std::vector<HistBin>> tl_bins(nt, std::vector<HistBin>(kGridTotal));

#pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            auto& local = tl_bins[tid];
#pragma omp for schedule(static)
            for (int r = 0; r < bgr.rows; ++r) {
                const auto* orow = oklab_cache.ptr<cv::Vec3f>(r);
                for (int c = 0; c < bgr.cols; ++c) {
                    float L = orow[c][0], a = orow[c][1], b = orow[c][2];
                    int bi  = BinIndex(L, a, b);
                    auto& h = local[bi];
                    h.count++;
                    h.sum_l += L;
                    h.sum_a += a;
                    h.sum_b += b;
                    h.sum2_l += static_cast<double>(L) * L;
                    h.sum2_a += static_cast<double>(a) * a;
                    h.sum2_b += static_cast<double>(b) * b;
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            for (int bi = 0; bi < kGridTotal; ++bi) {
                bins[bi].count += tl_bins[t][bi].count;
                bins[bi].sum_l += tl_bins[t][bi].sum_l;
                bins[bi].sum_a += tl_bins[t][bi].sum_a;
                bins[bi].sum_b += tl_bins[t][bi].sum_b;
                bins[bi].sum2_l += tl_bins[t][bi].sum2_l;
                bins[bi].sum2_a += tl_bins[t][bi].sum2_a;
                bins[bi].sum2_b += tl_bins[t][bi].sum2_b;
            }
        }
    }

    int BinIndex(float L, float a, float b) const {
        int iL = std::clamp(static_cast<int>((L - L_min) * L_inv), 0, kGridDim - 1);
        int ia = std::clamp(static_cast<int>((a - a_min) * a_inv), 0, kGridDim - 1);
        int ib = std::clamp(static_cast<int>((b - b_min) * b_inv), 0, kGridDim - 1);
        return iL * kGridDim * kGridDim + ia * kGridDim + ib;
    }

    float BinChannel(int bi, int axis) const {
        const auto& h = bins[bi];
        if (h.count == 0) return 0.f;
        double inv = 1.0 / h.count;
        switch (axis) {
        case 0:
            return static_cast<float>(h.sum_l * inv);
        case 1:
            return static_cast<float>(h.sum_a * inv);
        default:
            return static_cast<float>(h.sum_b * inv);
        }
    }
};

struct OkLabPixel {
    float L, a, b;
};

struct ColorBox {
    std::vector<int> bin_ids;
    int total_count = 0;
    OkLabPixel min_corner{}, max_corner{};
    OkLabPixel mean{};
    double var_L = 0.0, var_a = 0.0, var_b = 0.0;
    double priority = 0.0;

    bool operator<(const ColorBox& rhs) const { return priority < rhs.priority; }
};

void ComputeBoxStats(ColorBox& box, const ColorGrid& grid) {
    if (box.bin_ids.empty()) return;

    box.min_corner  = {1e9f, 1e9f, 1e9f};
    box.max_corner  = {-1e9f, -1e9f, -1e9f};
    box.total_count = 0;
    double tL = 0, tA = 0, tB = 0;
    double t2L = 0, t2A = 0, t2B = 0;

    for (int bi : box.bin_ids) {
        const auto& h = grid.bins[bi];
        if (h.count == 0) continue;
        box.total_count += h.count;
        tL += h.sum_l;
        tA += h.sum_a;
        tB += h.sum_b;
        t2L += h.sum2_l;
        t2A += h.sum2_a;
        t2B += h.sum2_b;

        float mL         = static_cast<float>(h.sum_l / h.count);
        float ma         = static_cast<float>(h.sum_a / h.count);
        float mb         = static_cast<float>(h.sum_b / h.count);
        box.min_corner.L = std::min(box.min_corner.L, mL);
        box.min_corner.a = std::min(box.min_corner.a, ma);
        box.min_corner.b = std::min(box.min_corner.b, mb);
        box.max_corner.L = std::max(box.max_corner.L, mL);
        box.max_corner.a = std::max(box.max_corner.a, ma);
        box.max_corner.b = std::max(box.max_corner.b, mb);
    }

    if (box.total_count == 0) return;
    double n = static_cast<double>(box.total_count);
    box.mean = {static_cast<float>(tL / n), static_cast<float>(tA / n), static_cast<float>(tB / n)};

    box.var_L        = std::max(0.0, t2L - tL * tL / n);
    box.var_a        = std::max(0.0, t2A - tA * tA / n);
    box.var_b        = std::max(0.0, t2B - tB * tB / n);
    double total_var = box.var_L + box.var_a + box.var_b;
    box.priority     = total_var / std::max(1.0, std::sqrt(n));
}

int MaxVarianceAxis(const ColorBox& box) {
    if (box.var_L >= box.var_a && box.var_L >= box.var_b) return 0;
    if (box.var_a >= box.var_b) return 1;
    return 2;
}

std::pair<ColorBox, ColorBox> MedianCutSplit(ColorBox& box, const ColorGrid& grid) {
    int axis = MaxVarianceAxis(box);

    struct BinEntry {
        int bi;
        float val;
        int count;
    };

    std::vector<BinEntry> sorted;
    sorted.reserve(box.bin_ids.size());
    for (int bi : box.bin_ids) {
        if (grid.bins[bi].count == 0) continue;
        sorted.push_back({bi, grid.BinChannel(bi, axis), grid.bins[bi].count});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const BinEntry& a, const BinEntry& b) { return a.val < b.val; });

    int half_count    = box.total_count / 2;
    int cumulative    = 0;
    size_t median_idx = 0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        cumulative += sorted[i].count;
        if (cumulative >= half_count) {
            median_idx = i;
            break;
        }
    }
    if (median_idx >= sorted.size() - 1 && sorted.size() > 1) median_idx = sorted.size() - 2;

    ColorBox left, right;
    left.bin_ids.reserve(median_idx + 1);
    right.bin_ids.reserve(sorted.size() - median_idx - 1);
    for (size_t i = 0; i <= median_idx; ++i) left.bin_ids.push_back(sorted[i].bi);
    for (size_t i = median_idx + 1; i < sorted.size(); ++i) right.bin_ids.push_back(sorted[i].bi);

    if (left.bin_ids.empty()) std::swap(left.bin_ids, right.bin_ids);
    if (right.bin_ids.empty()) {
        size_t half = left.bin_ids.size() / 2;
        right.bin_ids.assign(left.bin_ids.begin() + static_cast<ptrdiff_t>(half),
                             left.bin_ids.end());
        left.bin_ids.resize(half);
    }

    ComputeBoxStats(left, grid);
    ComputeBoxStats(right, grid);
    return {std::move(left), std::move(right)};
}

std::pair<ColorBox, ColorBox> SplitBox(ColorBox& box, const ColorGrid& grid) {
    OkLabPixel seed_a = box.mean;
    OkLabPixel seed_b = seed_a;
    float max_d2      = 0;

    for (int bi : box.bin_ids) {
        const auto& h = grid.bins[bi];
        if (h.count == 0) continue;
        float mL = static_cast<float>(h.sum_l / h.count);
        float ma = static_cast<float>(h.sum_a / h.count);
        float mb = static_cast<float>(h.sum_b / h.count);
        float dL = mL - seed_a.L, da = ma - seed_a.a, db = mb - seed_a.b;
        float d2 = dL * dL + da * da + db * db;
        if (d2 > max_d2) {
            max_d2 = d2;
            seed_b = {mL, ma, mb};
        }
    }

    if (max_d2 < 1e-12f) return MedianCutSplit(box, grid);

    constexpr int kIters = 3;
    std::vector<int8_t> assign(box.bin_ids.size(), 0);

    for (int iter = 0; iter < kIters; ++iter) {
        double sL_a = 0, sA_a = 0, sB_a = 0;
        double sL_b = 0, sA_b = 0, sB_b = 0;
        int cnt_a = 0, cnt_b = 0;

        for (size_t idx = 0; idx < box.bin_ids.size(); ++idx) {
            const auto& h = grid.bins[box.bin_ids[idx]];
            if (h.count == 0) continue;
            float mL = static_cast<float>(h.sum_l / h.count);
            float ma = static_cast<float>(h.sum_a / h.count);
            float mb = static_cast<float>(h.sum_b / h.count);

            float daL = mL - seed_a.L, daa = ma - seed_a.a, dab = mb - seed_a.b;
            float dbL = mL - seed_b.L, dba = ma - seed_b.a, dbb = mb - seed_b.b;
            float da2 = daL * daL + daa * daa + dab * dab;
            float db2 = dbL * dbL + dba * dba + dbb * dbb;

            if (da2 <= db2) {
                assign[idx] = 0;
                sL_a += h.sum_l;
                sA_a += h.sum_a;
                sB_a += h.sum_b;
                cnt_a += h.count;
            } else {
                assign[idx] = 1;
                sL_b += h.sum_l;
                sA_b += h.sum_a;
                sB_b += h.sum_b;
                cnt_b += h.count;
            }
        }

        if (cnt_a > 0) {
            double inv = 1.0 / cnt_a;
            seed_a     = {static_cast<float>(sL_a * inv), static_cast<float>(sA_a * inv),
                          static_cast<float>(sB_a * inv)};
        }
        if (cnt_b > 0) {
            double inv = 1.0 / cnt_b;
            seed_b     = {static_cast<float>(sL_b * inv), static_cast<float>(sA_b * inv),
                          static_cast<float>(sB_b * inv)};
        }
    }

    ColorBox left, right;
    for (size_t idx = 0; idx < box.bin_ids.size(); ++idx) {
        if (grid.bins[box.bin_ids[idx]].count == 0) continue;
        if (assign[idx] == 0)
            left.bin_ids.push_back(box.bin_ids[idx]);
        else
            right.bin_ids.push_back(box.bin_ids[idx]);
    }

    if (left.bin_ids.empty() || right.bin_ids.empty()) return MedianCutSplit(box, grid);

    ComputeBoxStats(left, grid);
    ComputeBoxStats(right, grid);
    return {std::move(left), std::move(right)};
}

double BoxTotalVar(const ColorBox& b) { return b.var_L + b.var_a + b.var_b; }

int FindNearestCentroid(float L, float a, float b, const std::vector<OkLabPixel>& centroids) {
    float best_d = 1e30f;
    int best_j   = 0;
    for (int j = 0; j < static_cast<int>(centroids.size()); ++j) {
        float dL = L - centroids[j].L;
        float da = a - centroids[j].a;
        float db = b - centroids[j].b;
        float d  = dL * dL + da * da + db * db;
        if (d < best_d) {
            best_d = d;
            best_j = j;
        }
    }
    return best_j;
}

struct TotalVarGreater {
    bool operator()(const ColorBox& a, const ColorBox& b) const {
        return BoxTotalVar(a) < BoxTotalVar(b);
    }
};

int AutoDetectK(const ColorGrid& grid, const ColorBox& full_box) {
    if (full_box.total_count < 100) return 2;

    constexpr int kMaxK               = 64;
    constexpr double kTargetRemaining = 0.005;
    constexpr double kStallThresh     = 0.0003;
    constexpr int kStallLimit         = 3;
    constexpr double kElbowRatio      = 0.05;

    std::vector<ColorBox> pq;
    pq.push_back(full_box);

    double total_variance = BoxTotalVar(full_box);
    if (total_variance < 1e-12) return 2;

    double running_var   = total_variance;
    double prev_var      = total_variance;
    double prev_marginal = 0;
    int k                = 1;
    int stall_count      = 0;
    bool elbow_detected  = false;

    while (k < kMaxK && !pq.empty()) {
        std::pop_heap(pq.begin(), pq.end(), TotalVarGreater{});
        auto top = std::move(pq.back());
        pq.pop_back();

        if (top.bin_ids.size() < 2) {
            pq.push_back(std::move(top));
            std::push_heap(pq.begin(), pq.end(), TotalVarGreater{});
            break;
        }

        double removed_var = BoxTotalVar(top);
        auto [left, right] = MedianCutSplit(top, grid);
        double added_var   = BoxTotalVar(left) + BoxTotalVar(right);
        running_var += added_var - removed_var;
        pq.push_back(std::move(left));
        std::push_heap(pq.begin(), pq.end(), TotalVarGreater{});
        pq.push_back(std::move(right));
        std::push_heap(pq.begin(), pq.end(), TotalVarGreater{});
        k++;

        double remaining = running_var / total_variance;
        if (remaining <= kTargetRemaining && k >= 2) break;

        double marginal = (prev_var - running_var) / total_variance;

        if (k >= 3 && prev_marginal > 1e-12 && marginal / prev_marginal < kElbowRatio) {
            elbow_detected = true;
            spdlog::debug("AutoDetectK: elbow at k={}, marginal={:.6f}, prev={:.6f}", k, marginal,
                          prev_marginal);
            break;
        }

        if (marginal < kStallThresh)
            stall_count++;
        else
            stall_count = 0;
        if (stall_count >= kStallLimit && k >= 2) break;

        prev_marginal = marginal;
        prev_var      = running_var;
    }

    int final_k = elbow_detected ? k - 1 : k;
    spdlog::debug("AutoDetectK: k={}, remaining_var={:.4f}{}", final_k,
                  running_var / total_variance, elbow_detected ? " (elbow)" : "");
    return std::clamp(final_k, 2, kMaxK);
}

std::vector<OkLabPixel> RunMmcq(const ColorGrid& grid, int num_colors) {
    ColorBox initial;
    for (int i = 0; i < kGridTotal; ++i) {
        if (grid.bins[i].count > 0) initial.bin_ids.push_back(i);
    }
    ComputeBoxStats(initial, grid);

    if (num_colors <= 0) {
        num_colors = AutoDetectK(grid, initial);
        spdlog::info("QuantizeColors: auto-detected K={}", num_colors);
    }
    num_colors = std::max(2, num_colors);

    std::vector<ColorBox> heap;
    heap.push_back(std::move(initial));

    while (static_cast<int>(heap.size()) < num_colors) {
        std::pop_heap(heap.begin(), heap.end());
        auto top = std::move(heap.back());
        heap.pop_back();
        if (top.bin_ids.size() < 2) {
            heap.push_back(std::move(top));
            std::push_heap(heap.begin(), heap.end());
            break;
        }
        auto [left, right] = SplitBox(top, grid);
        heap.push_back(std::move(left));
        std::push_heap(heap.begin(), heap.end());
        heap.push_back(std::move(right));
        std::push_heap(heap.begin(), heap.end());
    }

    std::vector<OkLabPixel> centroids;
    centroids.reserve(heap.size());
    for (const auto& box : heap) { centroids.push_back(box.mean); }
    spdlog::debug("QuantizeColors: MMCQ produced {} centroids from {} bins", centroids.size(),
                  kGridTotal);
    return centroids;
}

void RefineCentroidsKMeans(std::vector<OkLabPixel>& centroids, const ColorGrid& grid,
                           const std::vector<int>& active_bins) {
    const int K                   = static_cast<int>(centroids.size());
    constexpr int kRefineIters    = 8;
    constexpr float kConvergeEps2 = 1e-12f;
    int actual_iters              = 0;

    std::vector<double> sL(K), sA(K), sB(K);
    std::vector<int> cnt(K);

    for (int iter = 0; iter < kRefineIters; ++iter) {
        std::fill(sL.begin(), sL.end(), 0.0);
        std::fill(sA.begin(), sA.end(), 0.0);
        std::fill(sB.begin(), sB.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);

        for (int bi : active_bins) {
            const auto& h = grid.bins[bi];
            float mL      = static_cast<float>(h.sum_l / h.count);
            float ma      = static_cast<float>(h.sum_a / h.count);
            float mb      = static_cast<float>(h.sum_b / h.count);

            int best_j = FindNearestCentroid(mL, ma, mb, centroids);
            sL[best_j] += h.sum_l;
            sA[best_j] += h.sum_a;
            sB[best_j] += h.sum_b;
            cnt[best_j] += h.count;
        }

        float max_shift2 = 0;
        for (int j = 0; j < K; ++j) {
            if (cnt[j] == 0) continue;
            double inv     = 1.0 / cnt[j];
            float new_L    = static_cast<float>(sL[j] * inv);
            float new_a    = static_cast<float>(sA[j] * inv);
            float new_b    = static_cast<float>(sB[j] * inv);
            float dL       = new_L - centroids[j].L;
            float da       = new_a - centroids[j].a;
            float db       = new_b - centroids[j].b;
            max_shift2     = std::max(max_shift2, dL * dL + da * da + db * db);
            centroids[j].L = new_L;
            centroids[j].a = new_a;
            centroids[j].b = new_b;
        }
        ++actual_iters;
        if (max_shift2 < kConvergeEps2) break;
    }
    spdlog::debug("QuantizeColors: K-Means refinement done ({}/{} iters on {} bins)", actual_iters,
                  kRefineIters, active_bins.size());
}

void ConsolidatePalette(std::vector<OkLabPixel>& centroids, const ColorGrid& grid,
                        const std::vector<int>& active_bins) {
    constexpr float kMergeThreshold2 = 0.025f * 0.025f;
    int K                            = static_cast<int>(centroids.size());

    std::vector<int> pixel_count(K, 0);
    for (int bi : active_bins) {
        const auto& h = grid.bins[bi];
        float mL      = static_cast<float>(h.sum_l / h.count);
        float ma      = static_cast<float>(h.sum_a / h.count);
        float mb      = static_cast<float>(h.sum_b / h.count);
        pixel_count[FindNearestCentroid(mL, ma, mb, centroids)] += h.count;
    }

    struct PairDist {
        int i, j;
        float d2;
    };

    std::vector<PairDist> pairs;
    pairs.reserve(K * (K - 1) / 2);
    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            float dL = centroids[i].L - centroids[j].L;
            float da = centroids[i].a - centroids[j].a;
            float db = centroids[i].b - centroids[j].b;
            float d2 = dL * dL + da * da + db * db;
            if (d2 < kMergeThreshold2) pairs.push_back({i, j, d2});
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const PairDist& a, const PairDist& b) { return a.d2 < b.d2; });

    std::vector<int> root(K);
    std::iota(root.begin(), root.end(), 0);
    auto find_root = [&](int x) {
        while (root[x] != x) x = root[x] = root[root[x]];
        return x;
    };

    int merge_count = 0;
    for (const auto& p : pairs) {
        int ri = find_root(p.i);
        int rj = find_root(p.j);
        if (ri == rj) continue;

        float cur_dL = centroids[ri].L - centroids[rj].L;
        float cur_da = centroids[ri].a - centroids[rj].a;
        float cur_db = centroids[ri].b - centroids[rj].b;
        if (cur_dL * cur_dL + cur_da * cur_da + cur_db * cur_db >= kMergeThreshold2) continue;

        double w_i     = static_cast<double>(pixel_count[ri]);
        double w_j     = static_cast<double>(pixel_count[rj]);
        double w_total = w_i + w_j;
        if (w_total > 0) {
            centroids[ri].L =
                static_cast<float>((w_i * centroids[ri].L + w_j * centroids[rj].L) / w_total);
            centroids[ri].a =
                static_cast<float>((w_i * centroids[ri].a + w_j * centroids[rj].a) / w_total);
            centroids[ri].b =
                static_cast<float>((w_i * centroids[ri].b + w_j * centroids[rj].b) / w_total);
        }
        pixel_count[ri] += pixel_count[rj];
        root[rj] = ri;
        ++merge_count;
    }

    if (merge_count > 0) {
        std::vector<OkLabPixel> compacted;
        compacted.reserve(K - merge_count);
        for (int i = 0; i < K; ++i) {
            if (root[i] == i) compacted.push_back(centroids[i]);
        }
        centroids = std::move(compacted);
        spdlog::info("QuantizeColors: palette consolidation merged {} pairs, K={}", merge_count,
                     centroids.size());
    }
}

void SmoothLabels(cv::Mat& labels, const cv::Mat& bgr, const std::vector<OkLabPixel>& centroids) {
    constexpr int kRadius             = 2;
    constexpr float kMaxReassignDist2 = 0.04f * 0.04f;
    const int K                       = static_cast<int>(centroids.size());
    const int side                    = 2 * kRadius + 1;
    const int half_window             = side * side / 2;
    const int rows                    = bgr.rows;
    const int cols                    = bgr.cols;

    cv::Mat smoothed = labels.clone();
    int reassigned   = 0;

#pragma omp parallel reduction(+ : reassigned)
    {
        std::vector<int> freq(K);
#pragma omp for schedule(static)
        for (int r = kRadius; r < rows - kRadius; ++r) {
            const auto* brow = bgr.ptr<cv::Vec3b>(r);
            const int* lrow  = labels.ptr<int>(r);
            int* srow        = smoothed.ptr<int>(r);
            for (int c = kRadius; c < cols - kRadius; ++c) {
                int cur_label = lrow[c];

                std::fill(freq.begin(), freq.end(), 0);
                for (int dr = -kRadius; dr <= kRadius; ++dr) {
                    const int* nr = labels.ptr<int>(r + dr);
                    for (int dc = -kRadius; dc <= kRadius; ++dc) freq[nr[c + dc]]++;
                }

                int majority    = cur_label;
                int majority_ct = freq[cur_label];
                for (int k = 0; k < K; ++k) {
                    if (freq[k] > majority_ct) {
                        majority_ct = freq[k];
                        majority    = k;
                    }
                }

                if (majority == cur_label || majority_ct <= half_window) continue;

                auto ok  = SrgbToOklab(brow[c][2], brow[c][1], brow[c][0]);
                float dL = ok.L - centroids[majority].L;
                float da = ok.a - centroids[majority].a;
                float db = ok.b - centroids[majority].b;
                if (dL * dL + da * da + db * db < kMaxReassignDist2) {
                    srow[c] = majority;
                    ++reassigned;
                }
            }
        }
    }
    labels = smoothed;
    spdlog::debug("QuantizeColors: spatial smoothing reassigned {} pixels", reassigned);
}

} // namespace

QuantizeResult QuantizeColors(const cv::Mat& bgr, int num_colors) {
    ColorGrid grid;
    cv::Mat oklab_cache;
    grid.Build(bgr, oklab_cache);

    auto centroids = RunMmcq(grid, num_colors);

    std::vector<int> active_bins;
    active_bins.reserve(kGridTotal);
    for (int i = 0; i < kGridTotal; ++i) {
        if (grid.bins[i].count > 0) active_bins.push_back(i);
    }

    RefineCentroidsKMeans(centroids, grid, active_bins);
    ConsolidatePalette(centroids, grid, active_bins);

    const int K    = static_cast<int>(centroids.size());
    const int rows = bgr.rows;
    const int cols = bgr.cols;

    QuantizeResult result;
    result.labels = cv::Mat(rows, cols, CV_32SC1);
    result.palette.resize(K);
    result.centers_lab.resize(K);

#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const auto* orow = oklab_cache.ptr<cv::Vec3f>(r);
        auto* lrow       = result.labels.ptr<int>(r);
        for (int c = 0; c < cols; ++c) {
            lrow[c] = FindNearestCentroid(orow[c][0], orow[c][1], orow[c][2], centroids);
        }
    }
    oklab_cache.release();

    SmoothLabels(result.labels, bgr, centroids);

    for (int i = 0; i < K; ++i) {
        uint8_t r8, g8, b8;
        OkLab ok{centroids[i].L, centroids[i].a, centroids[i].b};
        OklabToSrgb(ok, r8, g8, b8);
        result.palette[i] = Rgb::FromRgb255(r8, g8, b8);

        Lab cie_lab           = result.palette[i].ToLab();
        result.centers_lab[i] = cv::Vec3f(cie_lab.l(), cie_lab.a(), cie_lab.b());
    }

    spdlog::info("QuantizeColors: {} colors, {} pixels", K, rows * cols);
    return result;
}

} // namespace neroued::vectorizer::detail
