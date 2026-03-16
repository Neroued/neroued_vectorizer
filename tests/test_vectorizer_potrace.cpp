#include <gtest/gtest.h>

#include <neroued/vectorizer/vectorizer.h>
#include "curve/bezier.h"

#include <nanosvg/nanosvg.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace neroued::vectorizer;

namespace {

struct RasterizedSvg {
    cv::Mat bgr;
    cv::Mat coverage;
};

cv::Scalar NsvgColorToBgr(unsigned int color) {
    int r = static_cast<int>((color >> 0) & 0xFF);
    int g = static_cast<int>((color >> 8) & 0xFF);
    int b = static_cast<int>((color >> 16) & 0xFF);
    return cv::Scalar(b, g, r);
}

std::vector<cv::Point> FlattenPathToPixels(const NSVGpath* path, int width, int height) {
    std::vector<cv::Point> out;
    if (!path || path->npts < 4 || width <= 0 || height <= 0) return out;

    std::vector<Vec2f> contour;
    contour.reserve(static_cast<size_t>(path->npts) * 2);
    contour.push_back({path->pts[0], path->pts[1]});

    for (int i = 0; i < path->npts - 1; i += 3) {
        const float* p = &path->pts[i * 2];
        Vec2f p0{p[0], p[1]};
        Vec2f p1{p[2], p[3]};
        Vec2f p2{p[4], p[5]};
        Vec2f p3{p[6], p[7]};
        detail::FlattenCubicBezier(p0, p1, p2, p3, 0.4f, contour);
    }

    if (path->closed && contour.size() > 1) {
        const Vec2f& a = contour.front();
        const Vec2f& b = contour.back();
        if (std::abs(a.x - b.x) < 1e-3f && std::abs(a.y - b.y) < 1e-3f) contour.pop_back();
    }

    out.reserve(contour.size());
    for (const Vec2f& p : contour) {
        int x = std::clamp(static_cast<int>(std::lround(p.x)), 0, width - 1);
        int y = std::clamp(static_cast<int>(std::lround(p.y)), 0, height - 1);
        out.emplace_back(x, y);
    }
    return out;
}

RasterizedSvg RasterizeSvg(const std::string& svg, int width, int height) {
    RasterizedSvg result;
    result.bgr      = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    result.coverage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    std::vector<char> buf(svg.begin(), svg.end());
    buf.push_back('\0');

    NSVGimage* image = nsvgParse(buf.data(), "px", 96.0f);
    if (!image) return result;

    for (const NSVGshape* shape = image->shapes; shape != nullptr; shape = shape->next) {
        if (!(shape->flags & NSVG_FLAGS_VISIBLE)) continue;
        if (shape->fill.type != NSVG_PAINT_COLOR) continue;

        std::vector<std::vector<cv::Point>> contours;
        for (const NSVGpath* path = shape->paths; path != nullptr; path = path->next) {
            auto poly = FlattenPathToPixels(path, width, height);
            if (poly.size() >= 3) contours.push_back(std::move(poly));
        }
        if (contours.empty()) continue;

        cv::Scalar color = NsvgColorToBgr(shape->fill.color);
        cv::fillPoly(result.coverage, contours, cv::Scalar(255));
        cv::fillPoly(result.bgr, contours, color);
        cv::polylines(result.coverage, contours, true, cv::Scalar(255), 1, cv::LINE_8);
        cv::polylines(result.bgr, contours, true, color, 1, cv::LINE_8);
    }

    nsvgDelete(image);
    return result;
}

double MeanDeltaE76(const cv::Mat& a_bgr, const cv::Mat& b_bgr) {
    cv::Mat a32, b32;
    a_bgr.convertTo(a32, CV_32F, 1.0 / 255.0);
    b_bgr.convertTo(b32, CV_32F, 1.0 / 255.0);

    cv::Mat a_lab, b_lab;
    cv::cvtColor(a32, a_lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(b32, b_lab, cv::COLOR_BGR2Lab);

    cv::Mat diff = a_lab - b_lab;
    std::vector<cv::Mat> ch(3);
    cv::split(diff, ch);
    cv::Mat de;
    cv::sqrt(ch[0].mul(ch[0]) + ch[1].mul(ch[1]) + ch[2].mul(ch[2]), de);
    return cv::mean(de)[0];
}

cv::Mat ExtractDarkMask(const cv::Mat& bgr, int threshold = 70) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::threshold(gray, mask, threshold, 255, cv::THRESH_BINARY_INV);
    return mask;
}

double MaskIoU(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    cv::Mat i, u;
    cv::bitwise_and(a, b, i);
    cv::bitwise_or(a, b, u);
    int union_px = cv::countNonZero(u);
    if (union_px <= 0) return 0.0;
    return static_cast<double>(cv::countNonZero(i)) / static_cast<double>(union_px);
}

VectorizerConfig PotraceCfg() {
    VectorizerConfig cfg;
    cfg.num_colors          = 8;
    cfg.min_region_area     = 2;
    cfg.min_contour_area    = 1.0f;
    cfg.contour_simplify    = 0.85f;
    cfg.enable_coverage_fix = true;
    cfg.min_coverage_ratio  = 0.995f;
    cfg.svg_enable_stroke   = false;
    return cfg;
}

} // namespace

TEST(VectorizerPotrace, BaselineMetricsStructuredImage) {
    cv::Mat img(96, 128, CV_8UC3, cv::Scalar(245, 245, 245));
    cv::rectangle(img, cv::Rect(6, 8, 40, 75), cv::Scalar(20, 30, 200), cv::FILLED);
    cv::circle(img, cv::Point(86, 34), 22, cv::Scalar(25, 180, 40), cv::FILLED);
    cv::ellipse(img, cv::Point(72, 72), cv::Size(30, 12), 20.0, 0.0, 360.0, cv::Scalar(200, 70, 30),
                cv::FILLED);
    cv::line(img, cv::Point(0, 95), cv::Point(127, 0), cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    VectorizerConfig cfg = PotraceCfg();
    auto out             = Vectorize(img, cfg);
    auto raster          = RasterizeSvg(out.svg_content, out.width, out.height);

    double coverage_ratio = static_cast<double>(cv::countNonZero(raster.coverage)) /
                            static_cast<double>(out.width * out.height);
    double mean_de76 = MeanDeltaE76(img, raster.bgr);
    int cubic_count =
        static_cast<int>(std::count(out.svg_content.begin(), out.svg_content.end(), 'C'));

    EXPECT_GT(coverage_ratio, 0.99);
    EXPECT_LT(mean_de76, 17.0);
    EXPECT_LT(cubic_count, 8000);
}

TEST(VectorizerPotrace, PreservesHoleTopologyForRing) {
    cv::Mat img(80, 80, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::circle(img, cv::Point(40, 40), 24, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_8);
    cv::circle(img, cv::Point(40, 40), 11, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_8);

    VectorizerConfig cfg = PotraceCfg();
    cfg.num_colors       = 2;
    auto out             = Vectorize(img, cfg);
    auto raster          = RasterizeSvg(out.svg_content, out.width, out.height);

    cv::Mat src_mask = ExtractDarkMask(img);
    cv::Mat out_mask = ExtractDarkMask(raster.bgr);
    double iou       = MaskIoU(src_mask, out_mask);

    cv::Vec3b center = raster.bgr.at<cv::Vec3b>(40, 40);
    int center_gray =
        (static_cast<int>(center[0]) + static_cast<int>(center[1]) + static_cast<int>(center[2])) /
        3;

    EXPECT_GT(iou, 0.74);
    EXPECT_GT(center_gray, 180);
}

TEST(VectorizerPotrace, PotracePipelineAvailable) {
    cv::Mat img(48, 48, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::line(img, cv::Point(2, 6), cv::Point(45, 40), cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    VectorizerConfig cfg = PotraceCfg();
    auto out             = Vectorize(img, cfg);
    EXPECT_EQ(out.width, 48);
    EXPECT_EQ(out.height, 48);
    EXPECT_GT(out.num_shapes, 0);
    EXPECT_FALSE(out.svg_content.empty());
}

TEST(VectorizerPotrace, MultiColorCoverageNoMissingBlocks) {
    cv::Mat img(80, 80, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(img, cv::Rect(0, 0, 40, 40), cv::Scalar(0, 0, 200), cv::FILLED);
    cv::rectangle(img, cv::Rect(40, 0, 40, 40), cv::Scalar(0, 200, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, 40, 40, 40), cv::Scalar(200, 0, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(40, 40, 40, 40), cv::Scalar(200, 200, 0), cv::FILLED);

    VectorizerConfig cfg = PotraceCfg();
    cfg.num_colors       = 4;
    auto out             = Vectorize(img, cfg);
    auto raster          = RasterizeSvg(out.svg_content, out.width, out.height);

    int filled   = cv::countNonZero(raster.coverage);
    int total    = out.width * out.height;
    double ratio = (total > 0) ? static_cast<double>(filled) / static_cast<double>(total) : 0.0;
    EXPECT_GT(ratio, 0.98) << "Coverage too low: large blocks likely missing";

    double mean_de76 = MeanDeltaE76(img, raster.bgr);
    EXPECT_LT(mean_de76, 20.0) << "Color accuracy too poor: likely label direction bug";
}

TEST(VectorizerPotrace, MultiColorNoLargeHoles) {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::circle(img, cv::Point(50, 50), 40, cv::Scalar(0, 0, 180), cv::FILLED);
    cv::circle(img, cv::Point(50, 50), 25, cv::Scalar(0, 180, 0), cv::FILLED);
    cv::circle(img, cv::Point(50, 50), 12, cv::Scalar(180, 0, 0), cv::FILLED);

    VectorizerConfig cfg = PotraceCfg();
    cfg.num_colors       = 4;
    auto out             = Vectorize(img, cfg);
    auto raster          = RasterizeSvg(out.svg_content, out.width, out.height);

    int filled   = cv::countNonZero(raster.coverage);
    int total    = out.width * out.height;
    double ratio = (total > 0) ? static_cast<double>(filled) / static_cast<double>(total) : 0.0;
    EXPECT_GT(ratio, 0.95) << "Coverage too low: concentric shapes likely missing";

    cv::Mat uncovered;
    cv::bitwise_not(raster.coverage, uncovered);
    cv::Mat cc_labels;
    int cc = cv::connectedComponents(uncovered, cc_labels, 8, CV_32S);
    for (int id = 1; id < cc; ++id) {
        int area = cv::countNonZero(cc_labels == id);
        EXPECT_LT(area, total / 4) << "Large uncovered hole detected (area=" << area << ")";
    }
}

TEST(VectorizerPotrace, ClosedCurveFitDoesNotCutShape) {
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::circle(img, cv::Point(32, 32), 20, cv::Scalar(40, 40, 200), cv::FILLED);

    VectorizerConfig cfg = PotraceCfg();
    cfg.num_colors       = 3;
    auto out             = Vectorize(img, cfg);
    auto raster          = RasterizeSvg(out.svg_content, out.width, out.height);

    cv::Vec3b center = raster.bgr.at<cv::Vec3b>(32, 32);
    int center_gray =
        (static_cast<int>(center[0]) + static_cast<int>(center[1]) + static_cast<int>(center[2])) /
        3;
    EXPECT_LT(center_gray, 180) << "Circle center is white: closed curve fitting cut the shape";

    int filled   = cv::countNonZero(raster.coverage);
    int total    = out.width * out.height;
    double ratio = (total > 0) ? static_cast<double>(filled) / static_cast<double>(total) : 0.0;
    EXPECT_GT(ratio, 0.95);
}

TEST(VectorizerPotrace, ColorAccuracyFourQuadrants) {
    cv::Mat img(60, 60, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(img, cv::Rect(0, 0, 30, 30), cv::Scalar(0, 0, 220), cv::FILLED);
    cv::rectangle(img, cv::Rect(30, 0, 30, 30), cv::Scalar(0, 220, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, 30, 30, 30), cv::Scalar(220, 0, 0), cv::FILLED);
    cv::rectangle(img, cv::Rect(30, 30, 30, 30), cv::Scalar(220, 220, 0), cv::FILLED);

    VectorizerConfig cfg = PotraceCfg();
    cfg.num_colors       = 4;
    auto out             = Vectorize(img, cfg);
    auto raster          = RasterizeSvg(out.svg_content, out.width, out.height);

    auto checkQuadrant = [&](int cx, int cy, int expect_b, int expect_g, int expect_r) {
        cv::Vec3b px = raster.bgr.at<cv::Vec3b>(cy, cx);
        int db       = std::abs(static_cast<int>(px[0]) - expect_b);
        int dg       = std::abs(static_cast<int>(px[1]) - expect_g);
        int dr       = std::abs(static_cast<int>(px[2]) - expect_r);
        EXPECT_LT(db + dg + dr, 180)
            << "Wrong color at (" << cx << "," << cy << "): " << "got BGR=(" << (int)px[0] << ","
            << (int)px[1] << "," << (int)px[2] << ")";
    };
    checkQuadrant(15, 15, 0, 0, 220);
    checkQuadrant(45, 15, 0, 220, 0);
    checkQuadrant(15, 45, 220, 0, 0);
    checkQuadrant(45, 45, 220, 220, 0);
}
