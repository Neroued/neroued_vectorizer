#include "svg_writer.h"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <string>

namespace neroued::vectorizer::detail {

namespace {

// Writes a float to buf with 2 decimal places and strips trailing zeros.
// Returns the number of characters written (not including null terminator).
int FmtFloatBuf(char* buf, float v) {
    int len = std::snprintf(buf, 16, "%.2f", static_cast<double>(v));
    // Strip trailing zeros after decimal point
    char* dot = nullptr;
    for (int i = 0; i < len; ++i) {
        if (buf[i] == '.') {
            dot = buf + i;
            break;
        }
    }
    if (dot) {
        char* end = buf + len - 1;
        while (end > dot && *end == '0') --end;
        if (end == dot) --end;
        len      = static_cast<int>(end - buf + 1);
        buf[len] = '\0';
    }
    return len;
}

std::string FmtFloat(float v) {
    char buf[16];
    FmtFloatBuf(buf, v);
    return buf;
}

std::string RgbToHex(const Rgb& c) {
    uint8_t r8, g8, b8;
    c.ToRgb255(r8, g8, b8);
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", r8, g8, b8);
    return buf;
}

void AppendBezierCmd(std::string& str, const CubicBezier& seg) {
    char buf[16];
    str += 'C';
    str.append(buf, FmtFloatBuf(buf, seg.p1.x));
    str += ',';
    str.append(buf, FmtFloatBuf(buf, seg.p1.y));
    str += ' ';
    str.append(buf, FmtFloatBuf(buf, seg.p2.x));
    str += ',';
    str.append(buf, FmtFloatBuf(buf, seg.p2.y));
    str += ' ';
    str.append(buf, FmtFloatBuf(buf, seg.p3.x));
    str += ',';
    str.append(buf, FmtFloatBuf(buf, seg.p3.y));
}

void AppendMoveTo(std::string& str, float x, float y) {
    char buf[16];
    str += 'M';
    str.append(buf, FmtFloatBuf(buf, x));
    str += ',';
    str.append(buf, FmtFloatBuf(buf, y));
}

} // namespace

std::string BezierToSvgPath(const BezierContour& contour) {
    if (contour.segments.empty()) return "";

    // ~50 chars per segment is a conservative estimate
    std::string s;
    s.reserve(contour.segments.size() * 50 + 16);

    AppendMoveTo(s, contour.segments[0].p0.x, contour.segments[0].p0.y);
    for (auto& seg : contour.segments) AppendBezierCmd(s, seg);
    if (contour.closed) s += 'Z';
    return s;
}

std::string ContoursToSvgPath(const std::vector<BezierContour>& contours) {
    std::string s;
    size_t total_segs = 0;
    for (auto& c : contours) total_segs += c.segments.size();
    s.reserve(total_segs * 50 + contours.size() * 20);

    for (size_t ci = 0; ci < contours.size(); ++ci) {
        auto& contour = contours[ci];
        if (contour.segments.empty()) continue;

        if (!contour.is_hole) {
            s += BezierToSvgPath(contour);
        } else {
            auto& segs = contour.segments;
            int n      = static_cast<int>(segs.size());
            AppendMoveTo(s, segs[n - 1].p3.x, segs[n - 1].p3.y);
            for (int i = n - 1; i >= 0; --i) {
                char buf[16];
                s += 'C';
                s.append(buf, FmtFloatBuf(buf, segs[i].p2.x));
                s += ',';
                s.append(buf, FmtFloatBuf(buf, segs[i].p2.y));
                s += ' ';
                s.append(buf, FmtFloatBuf(buf, segs[i].p1.x));
                s += ',';
                s.append(buf, FmtFloatBuf(buf, segs[i].p1.y));
                s += ' ';
                s.append(buf, FmtFloatBuf(buf, segs[i].p0.x));
                s += ',';
                s.append(buf, FmtFloatBuf(buf, segs[i].p0.y));
            }
            s += 'Z';
        }
    }
    return s;
}

std::string WriteSvg(const std::vector<VectorizedShape>& shapes, int width, int height,
                     bool enable_stroke, float stroke_width) {
    std::string svg;
    svg.reserve(shapes.size() * 1024 + 256);
    const std::size_t input_shape_count = shapes.size();
    std::size_t fill_shape_count        = 0;
    std::size_t stroke_shape_count      = 0;
    std::size_t empty_shape_skipped     = 0;
    std::size_t empty_path_skipped      = 0;

    svg += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ";
    svg += std::to_string(width);
    svg += ' ';
    svg += std::to_string(height);
    svg += "\" width=\"";
    svg += std::to_string(width);
    svg += "\" height=\"";
    svg += std::to_string(height);
    svg += "\">\n";

    for (auto& shape : shapes) {
        if (shape.contours.empty()) {
            ++empty_shape_skipped;
            continue;
        }
        std::string hex = RgbToHex(shape.color);

        if (shape.is_stroke && shape.stroke_width > 0.0f) {
            for (auto& contour : shape.contours) {
                std::string d = BezierToSvgPath(contour);
                if (d.empty()) {
                    ++empty_path_skipped;
                    continue;
                }
                svg += "  <path d=\"";
                svg += d;
                svg += "\" fill=\"none\" stroke=\"";
                svg += hex;
                svg += "\" stroke-width=\"";
                svg += FmtFloat(shape.stroke_width);
                svg += "\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n";
            }
            ++stroke_shape_count;
            continue;
        }

        std::string d = ContoursToSvgPath(shape.contours);
        if (d.empty()) {
            ++empty_path_skipped;
            continue;
        }

        svg += "  <path d=\"";
        svg += d;
        svg += "\" fill=\"";
        svg += hex;
        svg += "\" fill-rule=\"evenodd\"";
        if (enable_stroke && stroke_width > 0.0f) {
            svg += " stroke=\"";
            svg += hex;
            svg += "\" stroke-width=\"";
            svg += FmtFloat(stroke_width);
            svg += "\" stroke-linejoin=\"round\" paint-order=\"stroke fill\"";
        }
        svg += "/>\n";
        ++fill_shape_count;
    }

    svg += "</svg>\n";
    spdlog::debug(
        "WriteSvg done: width={}, height={}, input_shapes={}, fill_shapes={}, stroke_shapes={}, "
        "empty_shapes_skipped={}, empty_paths_skipped={}, svg_bytes={}",
        width, height, input_shape_count, fill_shape_count, stroke_shape_count, empty_shape_skipped,
        empty_path_skipped, svg.size());
    return svg;
}

} // namespace neroued::vectorizer::detail
