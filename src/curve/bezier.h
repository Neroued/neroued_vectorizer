#pragma once

/// \file bezier.h
/// \brief Cubic Bezier types, evaluation, and flattening.

#include <neroued/vectorizer/vec2.h>

#include <vector>

namespace neroued::vectorizer::detail {

struct CubicBezier {
    Vec2f p0, p1, p2, p3;
};

struct CurveSegment {
    enum Type { BEZIER, CORNER };

    Type type;
    Vec2f p0, p1, p2, p3;
};

struct BezierContour {
    std::vector<CubicBezier> segments;
    bool closed  = true;
    bool is_hole = false;
};

/// Evaluate a cubic Bezier curve at parameter t in [0,1].
Vec2f EvalBezier(const CubicBezier& b, float t);

/// Evaluate the first derivative of a cubic Bezier at parameter t.
Vec2f EvalBezierDeriv(const CubicBezier& b, float t);

/// Evaluate the second derivative of a cubic Bezier at parameter t.
Vec2f EvalBezierSecondDeriv(const CubicBezier& b, float t);

/// Flatten a cubic Bezier into a polyline (does NOT add p0).
void FlattenCubicBezier(Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3, float tolerance,
                        std::vector<Vec2f>& out);

/// Overload accepting a CubicBezier struct.
inline void FlattenCubicBezier(const CubicBezier& b, float tolerance, std::vector<Vec2f>& out) {
    FlattenCubicBezier(b.p0, b.p1, b.p2, b.p3, tolerance, out);
}

/// Compute signed area of a closed BezierContour (positive = CCW).
double BezierContourSignedArea(const BezierContour& contour);

/// Reverse the direction of a BezierContour in-place.
void ReverseBezierContour(BezierContour& contour);

} // namespace neroued::vectorizer::detail
