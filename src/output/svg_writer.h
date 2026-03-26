#pragma once

/// \file svg_writer.h
/// \brief SVG document generation from vectorized shapes.

#include "detail/vectorized_shape.h"

#include <string>
#include <vector>

namespace neroued::vectorizer::detail {

/// Generate a complete SVG document string from vectorized shapes.
///
/// Shapes are rendered in the order given (first = bottom layer).
/// Uses fill-rule="evenodd" for correct hole handling.
///
/// \param shapes  Ordered list of shapes (background first).
/// \param width   Image width in pixels (becomes SVG viewBox width).
/// \param height  Image height in pixels (becomes SVG viewBox height).
/// \return        Complete SVG document as a string.
std::string WriteSvg(const std::vector<VectorizedShape>& shapes, int width, int height,
                     bool enable_stroke = false, float stroke_width = 0.5f);

/// Convert a BezierContour to an SVG path `d` attribute string.
/// Uses M (moveto), C (cubic bezier), and Z (close) commands.
std::string BezierToSvgPath(const BezierContour& contour);

/// Convert multiple contours (outer + holes) to a single SVG path `d` string.
/// Holes are drawn with reversed winding for evenodd fill-rule.
std::string ContoursToSvgPath(const std::vector<BezierContour>& contours);

} // namespace neroued::vectorizer::detail
