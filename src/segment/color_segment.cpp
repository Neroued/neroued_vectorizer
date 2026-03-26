#include "color_segment.h"

#include "detail/cv_utils.h"
#include "slic.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace neroued::vectorizer::detail {

namespace {

constexpr int kSlicIterations       = 10;
constexpr float kSlicMinRegionRatio = 0.25f;

} // namespace

SegmentationResult SegmentBinary(const cv::Mat& bgr, const cv::Mat& lab) {
    SegmentationResult out;
    out.lab = lab;

    cv::Mat gray, bw;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    out.labels = cv::Mat(bgr.rows, bgr.cols, CV_32SC1);
    out.centers_lab.resize(2, cv::Vec3f(0, 0, 0));
    std::array<cv::Vec3f, 2> sums{cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0)};
    std::array<int, 2> counts{0, 0};

    for (int r = 0; r < bgr.rows; ++r) {
        const uint8_t* bw_row    = bw.ptr<uint8_t>(r);
        const cv::Vec3f* lab_row = lab.ptr<cv::Vec3f>(r);
        int* out_row             = out.labels.ptr<int>(r);
        for (int c = 0; c < bgr.cols; ++c) {
            int lid    = (bw_row[c] > 0) ? 1 : 0;
            out_row[c] = lid;
            sums[lid] += lab_row[c];
            counts[lid]++;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (counts[i] > 0) out.centers_lab[i] = sums[i] * (1.0f / static_cast<float>(counts[i]));
    }
    return out;
}

SegmentationResult SegmentMultiColor(const cv::Mat& lab, int num_colors, int slic_region_size,
                                     float slic_compactness, const cv::Mat& edge_map,
                                     float edge_sensitivity) {
    SegmentationResult out;
    out.lab    = lab;
    out.labels = cv::Mat(lab.rows, lab.cols, CV_32SC1, cv::Scalar(0));

    SlicConfig slic_cfg;
    slic_cfg.region_size      = std::max(0, slic_region_size);
    slic_cfg.compactness      = std::max(0.001f, slic_compactness);
    slic_cfg.iterations       = kSlicIterations;
    slic_cfg.min_region_ratio = kSlicMinRegionRatio;
    slic_cfg.edge_map         = edge_map;
    slic_cfg.edge_sensitivity = edge_sensitivity;

    auto slic  = SegmentBySlic(lab, cv::Mat(), slic_cfg);
    int num_sp = static_cast<int>(slic.centers.size());
    spdlog::debug(
        "Vectorize segmentation (SLIC): requested_colors={}, region_size={}, superpixels={}",
        num_colors, slic_region_size, num_sp);

    if (num_sp < num_colors) {
        spdlog::warn("Vectorize segmentation fallback: superpixels={} < requested_colors={}, "
                     "switching to pixel kmeans",
                     num_sp, num_colors);
        cv::Mat samples = lab.reshape(1, lab.rows * lab.cols);
        samples.convertTo(samples, CV_32F);
        int K = std::clamp(num_colors, 2, std::max(2, samples.rows));

        cv::Mat km_labels, km_centers;
        cv::kmeans(samples, K, km_labels,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.2), 5,
                   cv::KMEANS_PP_CENTERS, km_centers);

        out.centers_lab.resize(K);
        for (int k = 0; k < K; ++k) {
            out.centers_lab[k] = {km_centers.at<float>(k, 0), km_centers.at<float>(k, 1),
                                  km_centers.at<float>(k, 2)};
        }
        out.labels = km_labels.reshape(1, lab.rows).clone();
        spdlog::debug("Vectorize segmentation (pixel kmeans) done: K={}", K);
        return out;
    }

    cv::Mat sp_samples(num_sp, 3, CV_32FC1);
    for (int i = 0; i < num_sp; ++i) {
        sp_samples.at<float>(i, 0) = slic.centers[i][0];
        sp_samples.at<float>(i, 1) = slic.centers[i][1];
        sp_samples.at<float>(i, 2) = slic.centers[i][2];
    }
    int K = std::clamp(num_colors, 2, num_sp);

    cv::Mat km_labels, km_centers;
    cv::kmeans(sp_samples, K, km_labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.2), 5,
               cv::KMEANS_PP_CENTERS, km_centers);

    for (int r = 0; r < lab.rows; ++r) {
        const int* slic_row = slic.labels.ptr<int>(r);
        int* out_row        = out.labels.ptr<int>(r);
        for (int c = 0; c < lab.cols; ++c) {
            int sp_id = slic_row[c];
            if (sp_id >= 0 && sp_id < num_sp) {
                out_row[c] = km_labels.at<int>(sp_id, 0);
            } else {
                out_row[c] = -1;
            }
        }
    }

    out.centers_lab.resize(K);
    for (int k = 0; k < K; ++k) {
        out.centers_lab[k] = {km_centers.at<float>(k, 0), km_centers.at<float>(k, 1),
                              km_centers.at<float>(k, 2)};
    }
    spdlog::debug("Vectorize segmentation (SLIC+kmeans) done: K={}", K);
    return out;
}

cv::Mat ComputeEdgeMap(const cv::Mat& bgr) {
    cv::Mat bgr_float;
    bgr.convertTo(bgr_float, CV_32F, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(bgr_float, lab, cv::COLOR_BGR2Lab);

    cv::Mat channels[3];
    cv::split(lab, channels);

    cv::Mat mag_max = cv::Mat::zeros(bgr.size(), CV_32FC1);
    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat gx, gy, mag;
        cv::Sobel(channels[ch], gx, CV_32F, 1, 0, 3);
        cv::Sobel(channels[ch], gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        mag_max = cv::max(mag_max, mag);
    }

    double max_val = 0.0;
    cv::minMaxLoc(mag_max, nullptr, &max_val);
    if (max_val > 0.0) mag_max *= (1.0 / max_val);
    return mag_max;
}

void RefineLabelsBoundary(cv::Mat& labels, const cv::Mat& unsmoothed_lab,
                          const std::vector<cv::Vec3f>& centers_lab, int passes) {
    if (labels.empty() || unsmoothed_lab.empty() || centers_lab.empty()) return;

    const int h          = labels.rows;
    const int w          = labels.cols;
    const int num_labels = static_cast<int>(centers_lab.size());

    constexpr int kDr8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    constexpr int kDc8[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int pass = 0; pass < passes; ++pass) {
        cv::Mat snapshot = labels.clone();
        int changed      = 0;

        for (int r = 0; r < h; ++r) {
            const int* snap_row      = snapshot.ptr<int>(r);
            const cv::Vec3f* lab_row = unsmoothed_lab.ptr<cv::Vec3f>(r);
            int* out_row             = labels.ptr<int>(r);

            for (int c = 0; c < w; ++c) {
                const int lid = snap_row[c];
                if (lid < 0 || lid >= num_labels) continue;

                bool has_different_neighbor = false;
                int neighbor_set[8];
                int neighbor_count = 0;

                for (int k = 0; k < 8; ++k) {
                    int nr = r + kDr8[k];
                    int nc = c + kDc8[k];
                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                    int nl = snapshot.at<int>(nr, nc);
                    if (nl < 0 || nl >= num_labels || nl == lid) continue;
                    has_different_neighbor = true;
                    bool already           = false;
                    for (int j = 0; j < neighbor_count; ++j) {
                        if (neighbor_set[j] == nl) {
                            already = true;
                            break;
                        }
                    }
                    if (!already && neighbor_count < 8) { neighbor_set[neighbor_count++] = nl; }
                }
                if (!has_different_neighbor) continue;

                const cv::Vec3f& pixel = lab_row[c];
                float d_current        = LabDistSq(pixel, centers_lab[lid]);

                int best_label  = lid;
                float best_dist = d_current;
                for (int j = 0; j < neighbor_count; ++j) {
                    float d = LabDistSq(pixel, centers_lab[neighbor_set[j]]);
                    if (d < best_dist) {
                        best_dist  = d;
                        best_label = neighbor_set[j];
                    }
                }

                if (best_label != lid) {
                    out_row[c] = best_label;
                    ++changed;
                }
            }
        }

        spdlog::debug("RefineLabelsBoundary pass {}/{}: changed={}", pass + 1, passes, changed);
        if (changed == 0) break;
    }
}

void MergeSmallComponents(cv::Mat& labels, const cv::Mat& lab, std::vector<cv::Vec3f>& centers_lab,
                          int min_region_area, float max_merge_color_dist) {
    if (min_region_area <= 1 || labels.empty()) return;

    const int h                   = labels.rows;
    const int w                   = labels.cols;
    constexpr int kMaxMergeRounds = 3;
    constexpr int dr[4]           = {1, -1, 0, 0};
    constexpr int dc[4]           = {0, 0, 1, -1};

    std::queue<cv::Point> q;
    std::vector<cv::Point> component;
    component.reserve(1024);

    for (int round = 0; round < kMaxMergeRounds; ++round) {
        cv::Mat visited(h, w, CV_8UC1, cv::Scalar(0));
        int merged_count = 0;

        for (int sr = 0; sr < h; ++sr) {
            for (int sc = 0; sc < w; ++sc) {
                if (visited.at<uint8_t>(sr, sc) != 0) continue;

                const int label0            = labels.at<int>(sr, sc);
                visited.at<uint8_t>(sr, sc) = 1;
                if (label0 < 0) continue;
                q.push({sc, sr});
                component.clear();
                component.push_back({sc, sr});

                std::unordered_map<int, int> border_hist;
                cv::Vec3f mean_lab(0, 0, 0);

                while (!q.empty()) {
                    cv::Point p = q.front();
                    q.pop();
                    mean_lab += lab.at<cv::Vec3f>(p.y, p.x);

                    for (int k = 0; k < 4; ++k) {
                        int nr = p.y + dr[k];
                        int nc = p.x + dc[k];
                        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                        int nl = labels.at<int>(nr, nc);
                        if (nl == label0) {
                            if (visited.at<uint8_t>(nr, nc) == 0) {
                                visited.at<uint8_t>(nr, nc) = 1;
                                q.push({nc, nr});
                                component.push_back({nc, nr});
                            }
                        } else if (nl >= 0) {
                            border_hist[nl]++;
                        }
                    }
                }

                if (static_cast<int>(component.size()) >= min_region_area || border_hist.empty())
                    continue;
                mean_lab *= (1.0f / static_cast<float>(component.size()));

                int best_label       = label0;
                int best_border_vote = -1;
                float best_dist      = std::numeric_limits<float>::max();
                for (const auto& [candidate, vote] : border_hist) {
                    if (candidate < 0 || candidate >= static_cast<int>(centers_lab.size()))
                        continue;
                    float d2 = LabDistSq(mean_lab, centers_lab[candidate]);
                    if (d2 < best_dist || (d2 == best_dist && vote > best_border_vote)) {
                        best_border_vote = vote;
                        best_dist        = d2;
                        best_label       = candidate;
                    }
                }

                if (best_label == label0) continue;
                if (best_dist > max_merge_color_dist) continue;

                for (const auto& p : component) labels.at<int>(p.y, p.x) = best_label;
                ++merged_count;
            }
        }

        spdlog::debug("MergeSmallComponents round {}/{}: merged={}", round + 1, kMaxMergeRounds,
                      merged_count);
        if (merged_count == 0) break;

        for (int lid = 0; lid < static_cast<int>(centers_lab.size()); ++lid) {
            cv::Vec3d sum(0, 0, 0);
            int count = 0;
            for (int r = 0; r < h; ++r) {
                const int* lrow          = labels.ptr<int>(r);
                const cv::Vec3f* lab_row = lab.ptr<cv::Vec3f>(r);
                for (int c = 0; c < w; ++c) {
                    if (lrow[c] == lid) {
                        sum += cv::Vec3d(lab_row[c][0], lab_row[c][1], lab_row[c][2]);
                        ++count;
                    }
                }
            }
            if (count > 0) {
                double inv = 1.0 / static_cast<double>(count);
                centers_lab[lid] =
                    cv::Vec3f(static_cast<float>(sum[0] * inv), static_cast<float>(sum[1] * inv),
                              static_cast<float>(sum[2] * inv));
            }
        }
    }
}

void MorphologicalCleanup(cv::Mat& labels, int num_labels, int close_radius) {
    if (close_radius <= 0 || labels.empty() || num_labels <= 0) return;

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(2 * close_radius + 1, 2 * close_radius + 1));

    std::vector<std::pair<int, int>> label_areas;
    std::vector<int> counts(num_labels, 0);
    for (int r = 0; r < labels.rows; ++r) {
        const int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            int lid = row[c];
            if (lid >= 0 && lid < num_labels) counts[lid]++;
        }
    }
    for (int i = 0; i < num_labels; ++i) {
        if (counts[i] > 0) label_areas.push_back({counts[i], i});
    }
    std::sort(label_areas.begin(), label_areas.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    cv::Mat valid_mask = (labels >= 0);
    valid_mask.convertTo(valid_mask, CV_8UC1, 255);

    for (const auto& [area, lid] : label_areas) {
        cv::Mat mask = (labels == lid);
        mask.convertTo(mask, CV_8UC1, 255);
        cv::Mat closed_mask;
        cv::morphologyEx(mask, closed_mask, cv::MORPH_CLOSE, kernel);

        cv::Mat newly_claimed;
        cv::subtract(closed_mask, mask, newly_claimed);
        cv::bitwise_and(newly_claimed, valid_mask, newly_claimed);

        labels.setTo(cv::Scalar(lid), newly_claimed > 0);
    }
}

int CompactLabels(cv::Mat& labels, std::vector<cv::Vec3f>& centers_lab) {
    int max_label = static_cast<int>(centers_lab.size());
    if (labels.empty() || max_label <= 0) return 0;

    std::vector<int> remap(max_label, -1);
    int next = 0;
    for (int r = 0; r < labels.rows; ++r) {
        int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            int lid = row[c];
            if (lid < 0 || lid >= max_label) continue;
            if (remap[lid] < 0) remap[lid] = next++;
            row[c] = remap[lid];
        }
    }

    std::vector<cv::Vec3f> compact(next, cv::Vec3f(0, 0, 0));
    for (int i = 0; i < max_label; ++i) {
        if (remap[i] >= 0) compact[remap[i]] = centers_lab[i];
    }
    centers_lab.swap(compact);
    return next;
}

std::vector<Rgb> ComputePalette(const cv::Mat& bgr, const cv::Mat& labels, int num_labels) {
    std::vector<Rgb> palette(std::max(0, num_labels), Rgb(0, 0, 0));
    if (num_labels <= 0) return palette;

    std::vector<std::array<double, 3>> sums(num_labels, {0.0, 0.0, 0.0});
    std::vector<int> counts(num_labels, 0);

    for (int r = 0; r < bgr.rows; ++r) {
        const cv::Vec3b* brow = bgr.ptr<cv::Vec3b>(r);
        const int* lrow       = labels.ptr<int>(r);
        for (int c = 0; c < bgr.cols; ++c) {
            int lid = lrow[c];
            if (lid < 0 || lid >= num_labels) continue;
            sums[lid][0] += brow[c][2] / 255.0;
            sums[lid][1] += brow[c][1] / 255.0;
            sums[lid][2] += brow[c][0] / 255.0;
            counts[lid]++;
        }
    }

    for (int i = 0; i < num_labels; ++i) {
        if (counts[i] <= 0) continue;
        float r    = static_cast<float>(sums[i][0] / counts[i]);
        float g    = static_cast<float>(sums[i][1] / counts[i]);
        float b    = static_cast<float>(sums[i][2] / counts[i]);
        palette[i] = Rgb(SrgbToLinear(r), SrgbToLinear(g), SrgbToLinear(b));
    }
    return palette;
}

namespace {

struct ColorSampleData {
    cv::Mat km_samples;
    cv::Mat proxy_cielab;
    cv::Mat proxy;
    int n_samples;
    int ch;
    bool achromatic;
    int proxy_area;
    float tiny_px_threshold;
};

ColorSampleData BuildColorSamples(const cv::Mat& bgr) {
    constexpr int kTargetSamples            = 30000;
    constexpr int kProxyShortEdge           = 300;
    constexpr float kAchromaticP90Threshold = 8.0f;
    constexpr float kTinyAreaFrac           = 0.002f;

    const int total_px    = bgr.rows * bgr.cols;
    const float grid_step = std::sqrt(static_cast<float>(std::max(1, total_px)) / kTargetSamples);
    const int row_step    = std::max(1, static_cast<int>(grid_step));
    const int col_step    = std::max(1, static_cast<int>(grid_step));

    int n_samples = 0;
    for (int r = 0; r < bgr.rows; r += row_step)
        for (int c = 0; c < bgr.cols; c += col_step) ++n_samples;

    cv::Mat sample_bgr(n_samples, 1, CV_8UC3);
    {
        int idx = 0;
        for (int r = 0; r < bgr.rows; r += row_step) {
            const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(r);
            for (int c = 0; c < bgr.cols; c += col_step)
                sample_bgr.at<cv::Vec3b>(idx++, 0) = row[c];
        }
    }
    cv::Mat sample_lab8;
    cv::cvtColor(sample_bgr, sample_lab8, cv::COLOR_BGR2Lab);

    const int short_edge = std::min(bgr.rows, bgr.cols);
    cv::Mat proxy;
    if (short_edge > kProxyShortEdge) {
        const float s = static_cast<float>(kProxyShortEdge) / short_edge;
        cv::resize(bgr, proxy, cv::Size(), s, s, cv::INTER_AREA);
    } else {
        proxy = bgr;
    }
    cv::Mat proxy_lab8;
    cv::cvtColor(proxy, proxy_lab8, cv::COLOR_BGR2Lab);
    const int proxy_area = proxy.rows * proxy.cols;

    std::vector<float> chromas(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const cv::Vec3b p = sample_lab8.at<cv::Vec3b>(i, 0);
        const float a     = static_cast<float>(p[1]) - 128.0f;
        const float b     = static_cast<float>(p[2]) - 128.0f;
        chromas[i]        = std::sqrt(a * a + b * b);
    }
    const int p90_idx = std::max(0, static_cast<int>(0.90f * (n_samples - 1)));
    std::nth_element(chromas.begin(), chromas.begin() + p90_idx, chromas.end());
    const bool achromatic = (chromas[p90_idx] < kAchromaticP90Threshold);

    const int ch = achromatic ? 1 : 3;

    cv::Mat km_samples(n_samples, ch, CV_32F);
    for (int i = 0; i < n_samples; ++i) {
        const cv::Vec3b p          = sample_lab8.at<cv::Vec3b>(i, 0);
        km_samples.at<float>(i, 0) = p[0] * (100.0f / 255.0f);
        if (!achromatic) {
            km_samples.at<float>(i, 1) = static_cast<float>(p[1]) - 128.0f;
            km_samples.at<float>(i, 2) = static_cast<float>(p[2]) - 128.0f;
        }
    }

    cv::Mat proxy_cielab(proxy_area, ch, CV_32F);
    for (int r = 0; r < proxy.rows; ++r) {
        const cv::Vec3b* row = proxy_lab8.ptr<cv::Vec3b>(r);
        for (int c = 0; c < proxy.cols; ++c) {
            const int idx                  = r * proxy.cols + c;
            const cv::Vec3b p              = row[c];
            proxy_cielab.at<float>(idx, 0) = p[0] * (100.0f / 255.0f);
            if (!achromatic) {
                proxy_cielab.at<float>(idx, 1) = static_cast<float>(p[1]) - 128.0f;
                proxy_cielab.at<float>(idx, 2) = static_cast<float>(p[2]) - 128.0f;
            }
        }
    }

    return {km_samples, proxy_cielab, proxy,      n_samples,
            ch,         achromatic,   proxy_area, proxy_area * kTinyAreaFrac};
}

float ScoreCandidateK(int K, const ColorSampleData& data) {
    constexpr float kW_meanDE   = 1.5f;
    constexpr float kW_p95DE    = 0.5f;
    constexpr float kW_tinyRate = 20.0f;
    constexpr float kW_compDens = 20.0f;
    constexpr float kW_logK     = 3.0f;

    cv::Mat km_labels, km_centers;
    cv::kmeans(data.km_samples, K, km_labels,
               cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.5), 3,
               cv::KMEANS_PP_CENTERS, km_centers);

    cv::Mat proxy_labels(data.proxy.rows, data.proxy.cols, CV_32SC1);
    std::vector<float> dE_vec(data.proxy_area);

    for (int i = 0; i < data.proxy_area; ++i) {
        float min_sq = std::numeric_limits<float>::max();
        int lbl      = 0;
        for (int k = 0; k < km_centers.rows; ++k) {
            float sq = 0.0f;
            for (int d = 0; d < data.ch; ++d) {
                const float diff = data.proxy_cielab.at<float>(i, d) - km_centers.at<float>(k, d);
                sq += diff * diff;
            }
            if (sq < min_sq) {
                min_sq = sq;
                lbl    = k;
            }
        }
        proxy_labels.at<int>(i / data.proxy.cols, i % data.proxy.cols) = lbl;
        dE_vec[i]                                                      = std::sqrt(min_sq);
    }

    float sum_dE = 0.0f;
    for (float d : dE_vec) sum_dE += d;
    const float mean_dE = sum_dE / std::max(1, data.proxy_area);

    const int p95 = std::max(0, static_cast<int>(0.95f * (data.proxy_area - 1)));
    std::nth_element(dE_vec.begin(), dE_vec.begin() + p95, dE_vec.end());
    const float p95_dE = dE_vec[p95];

    int total_comp = 0;
    int tiny_comp  = 0;
    for (int label = 0; label < km_centers.rows; ++label) {
        cv::Mat mask;
        cv::compare(proxy_labels, label, mask, cv::CMP_EQ);
        if (cv::countNonZero(mask) == 0) continue;

        cv::Mat cc_labels;
        int n_cc = cv::connectedComponents(mask, cc_labels, 8, CV_32S) - 1;
        total_comp += n_cc;

        std::vector<int> areas(n_cc + 1, 0);
        for (int r = 0; r < cc_labels.rows; ++r) {
            const int* row = cc_labels.ptr<int>(r);
            for (int c2 = 0; c2 < cc_labels.cols; ++c2)
                if (row[c2] > 0) areas[row[c2]]++;
        }
        for (int cc_id = 1; cc_id <= n_cc; ++cc_id)
            if (static_cast<float>(areas[cc_id]) < data.tiny_px_threshold) ++tiny_comp;
    }

    const float tiny_rate    = (total_comp > 0) ? static_cast<float>(tiny_comp) / total_comp : 0.0f;
    const float comp_density = static_cast<float>(total_comp) / std::max(1, data.proxy_area);
    const float log2K        = std::log2(static_cast<float>(K));

    const float score = kW_meanDE * mean_dE + kW_p95DE * p95_dE + kW_tinyRate * tiny_rate +
                        kW_compDens * comp_density + kW_logK * log2K;

    spdlog::debug("AutoColor K={:2d}: dE_mean={:.2f} dE_p95={:.2f} tiny={:.3f} "
                  "comp_dens={:.5f} log2K={:.2f} => score={:.2f}",
                  K, mean_dE, p95_dE, tiny_rate, comp_density, log2K, score);
    return score;
}

} // namespace

int EstimateOptimalColors(const cv::Mat& bgr) {
    constexpr int kCandidates[]  = {2, 3, 4, 6, 8, 12, 16, 24};
    constexpr int kNumCandidates = 8;

    auto data = BuildColorSamples(bgr);

    int actual_candidates = kNumCandidates;
    for (int ci = 0; ci < kNumCandidates; ++ci) {
        if (kCandidates[ci] > data.n_samples) {
            actual_candidates = ci;
            break;
        }
    }

    std::vector<float> scores(actual_candidates, std::numeric_limits<float>::max());
#pragma omp parallel for schedule(dynamic)
    for (int ci = 0; ci < actual_candidates; ++ci) {
        scores[ci] = ScoreCandidateK(kCandidates[ci], data);
    }

    int best_k       = 16;
    float best_score = std::numeric_limits<float>::max();
    for (int ci = 0; ci < actual_candidates; ++ci) {
        if (scores[ci] < best_score) {
            best_score = scores[ci];
            best_k     = kCandidates[ci];
        }
    }

    spdlog::info("AutoColor: achromatic={}, selected K={} (score={:.2f})", data.achromatic, best_k,
                 best_score);
    return best_k;
}

} // namespace neroued::vectorizer::detail
