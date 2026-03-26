#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::eval {

double PolylineSignedArea(const std::vector<cv::Point>& pts);
double PolylinePerimeter(const std::vector<cv::Point>& pts);
bool PointInPolyline(const std::vector<cv::Point>& poly, cv::Point pt);

/// Fill contours from a single SVG shape onto a binary mask using the even-odd
/// rule: each sub-path toggles the fill state via XOR.
cv::Mat FillShapeWithHoles(const std::vector<std::vector<cv::Point>>& contours, int width,
                           int height);

} // namespace neroued::vectorizer::eval
