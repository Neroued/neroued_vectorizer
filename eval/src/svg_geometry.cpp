#include "svg_geometry.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

namespace neroued::vectorizer::eval {

double PolylineSignedArea(const std::vector<cv::Point>& pts) {
    if (pts.size() < 3) return 0;
    double area = 0;
    size_t n    = pts.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += static_cast<double>(pts[i].x) * pts[j].y;
        area -= static_cast<double>(pts[j].x) * pts[i].y;
    }
    return area * 0.5;
}

double PolylinePerimeter(const std::vector<cv::Point>& pts) {
    if (pts.size() < 2) return 0;
    double len = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        size_t j  = (i + 1) % pts.size();
        double dx = pts[j].x - pts[i].x;
        double dy = pts[j].y - pts[i].y;
        len += std::sqrt(dx * dx + dy * dy);
    }
    return len;
}

bool PointInPolyline(const std::vector<cv::Point>& poly, cv::Point pt) {
    bool inside = false;
    size_t n    = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
            (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) /
                            static_cast<double>(poly[j].y - poly[i].y) +
                        poly[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

cv::Mat FillShapeWithHoles(const std::vector<std::vector<cv::Point>>& contours, int width,
                           int height) {
    if (contours.empty() || width <= 0 || height <= 0)
        return cv::Mat::zeros(height, width, CV_8UC1);

    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
    for (auto& contour : contours) {
        if (contour.size() < 3) continue;
        cv::Mat single                           = cv::Mat::zeros(height, width, CV_8UC1);
        std::vector<std::vector<cv::Point>> wrap = {contour};
        cv::fillPoly(single, wrap, cv::Scalar(255));
        cv::bitwise_xor(result, single, result);
    }
    return result;
}

} // namespace neroued::vectorizer::eval
