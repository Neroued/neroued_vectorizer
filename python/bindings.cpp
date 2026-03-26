#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <neroued/vectorizer/error.h>
#include <neroued/vectorizer/logging.h>
#include <neroued/vectorizer/vectorizer.h>

#include <opencv2/core.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

namespace py = pybind11;
using namespace neroued::vectorizer;

namespace {

cv::Mat numpy_to_mat(const py::array& arr) {
    auto buf = py::array_t<uint8_t, py::array::c_style>::ensure(arr);
    if (!buf) throw InputError("image array must be C-contiguous uint8");

    auto r = buf.request();
    if (r.ndim == 2) {
        return cv::Mat(static_cast<int>(r.shape[0]), static_cast<int>(r.shape[1]), CV_8UC1, r.ptr)
            .clone();
    }
    if (r.ndim == 3 && (r.shape[2] == 3 || r.shape[2] == 4)) {
        int type = r.shape[2] == 3 ? CV_8UC3 : CV_8UC4;
        return cv::Mat(static_cast<int>(r.shape[0]), static_cast<int>(r.shape[1]), type, r.ptr)
            .clone();
    }
    throw InputError("image array shape must be (H,W), (H,W,3), or (H,W,4)");
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "neroued_vectorizer: high-quality raster-to-SVG vectorization.\n\n"
              "This module exposes the core C++ vectorization engine to Python.\n"
              "Use :func:`vectorize` to convert raster images to SVG.";

    InitLogging(spdlog::level::warn);

    // ── Logging ─────────────────────────────────────────────────────────────
    m.def(
        "set_log_level", [](const std::string& level) { InitLogging(ParseLogLevel(level)); },
        py::arg("level"),
        "Set C++ log verbosity: 'trace', 'debug', 'info', 'warn', 'error', 'off'.");

    // ── Rgb ──────────────────────────────────────────────────────────────────
    py::class_<Rgb>(m, "Rgb",
                    "Linear sRGB color with components in [0, 1].\n\n"
                    "Internal representation uses linear (pre-gamma) sRGB values.\n"
                    "Use :meth:`from_rgb255` / :meth:`to_rgb255` for 0-255 sRGB conversion.")
        .def(py::init<>(), "Create a black color (0, 0, 0).")
        .def(py::init<float, float, float>(), py::arg("r"), py::arg("g"), py::arg("b"),
             "Create from linear sRGB components in [0, 1].")
        .def_property(
            "r", [](const Rgb& self) { return self.r(); }, [](Rgb& self, float v) { self.r() = v; },
            "Red component (linear sRGB, [0, 1]).")
        .def_property(
            "g", [](const Rgb& self) { return self.g(); }, [](Rgb& self, float v) { self.g() = v; },
            "Green component (linear sRGB, [0, 1]).")
        .def_property(
            "b", [](const Rgb& self) { return self.b(); }, [](Rgb& self, float v) { self.b() = v; },
            "Blue component (linear sRGB, [0, 1]).")
        .def_static("from_rgb255", &Rgb::FromRgb255, py::arg("r"), py::arg("g"), py::arg("b"),
                    "Create from 0-255 sRGB values (applies gamma decoding).")
        .def(
            "to_rgb255",
            [](const Rgb& self) -> std::tuple<int, int, int> {
                uint8_t r, g, b;
                self.ToRgb255(r, g, b);
                return {r, g, b};
            },
            "Convert to 0-255 sRGB tuple ``(r, g, b)`` (applies gamma encoding).")
        .def(
            "__eq__", [](const Rgb& a, const Rgb& b) { return a.NearlyEqual(b); }, py::arg("other"))
        .def("__repr__", [](const Rgb& self) {
            uint8_t r8, g8, b8;
            self.ToRgb255(r8, g8, b8);
            std::ostringstream os;
            os << "Rgb(r=" << self.r() << ", g=" << self.g() << ", b=" << self.b() << ")  # sRGB("
               << int(r8) << ", " << int(g8) << ", " << int(b8) << ")";
            return os.str();
        });

    // ── PipelineMode ──────────────────────────────────────────────────────
    py::enum_<PipelineMode>(m, "PipelineMode",
                            "Pipeline implementation selector.\n\n"
                            "V1: Original boundary-graph + cutout pipeline.\n"
                            "V2: Stacking model with per-layer Potrace and depth ordering.")
        .value("V1", PipelineMode::V1, "Original boundary-graph + cutout pipeline.")
        .value("V2", PipelineMode::V2, "Stacking model: per-layer Potrace with depth ordering.");

    // ── VectorizerConfig ─────────────────────────────────────────────────────
    auto cfg = py::class_<VectorizerConfig>(
        m, "VectorizerConfig",
        "Configuration for the vectorization pipeline.\n\n"
        "All fields have sensible defaults. Create an instance and override\n"
        "only the parameters you need::\n\n"
        "    cfg = VectorizerConfig()\n"
        "    cfg.num_colors = 8\n"
        "    cfg.smoothness = 0.7\n"
        "    result = vectorize('image.png', cfg)\n");

    cfg.def(py::init<>(), "Create a config with default values.");

    // Pipeline mode
    cfg.def_readwrite("pipeline_mode", &VectorizerConfig::pipeline_mode,
                      "Pipeline implementation: PipelineMode.V1 (default) or PipelineMode.V2.");

    // Color segmentation
    cfg.def_readwrite("num_colors", &VectorizerConfig::num_colors,
                      "K-Means palette size. 0 = auto-detect optimal count.");
    cfg.def_readwrite("min_region_area", &VectorizerConfig::min_region_area,
                      "Force-merge regions smaller than this (pixels squared).");

    // Curve fitting
    cfg.def_readwrite("curve_fit_error", &VectorizerConfig::curve_fit_error,
                      "Schneider curve fitting error threshold (pixels).");
    cfg.def_readwrite("corner_angle_threshold", &VectorizerConfig::corner_angle_threshold,
                      "Corner detection angle threshold in degrees.");
    cfg.def_readwrite("smoothness", &VectorizerConfig::smoothness,
                      "Contour smoothness [0, 1]. 0 = preserve detail, 1 = max smoothing.");

    // Preprocessing
    cfg.def_readwrite("smoothing_spatial", &VectorizerConfig::smoothing_spatial,
                      "Mean Shift spatial window radius.");
    cfg.def_readwrite("smoothing_color", &VectorizerConfig::smoothing_color,
                      "Mean Shift color window radius.");
    cfg.def_readwrite("upscale_short_edge", &VectorizerConfig::upscale_short_edge,
                      "Auto-upscale when short edge is below this (0 disables).");
    cfg.def_readwrite("max_working_pixels", &VectorizerConfig::max_working_pixels,
                      "Auto-downscale when total pixels exceed this (0 disables).");

    // Segmentation
    cfg.def_readwrite("slic_region_size", &VectorizerConfig::slic_region_size,
                      "SLIC target region size for multicolor mode.");
    cfg.def_readwrite("slic_compactness", &VectorizerConfig::slic_compactness,
                      "SLIC compactness (lower = follow color edges more).");
    cfg.def_readwrite("edge_sensitivity", &VectorizerConfig::edge_sensitivity,
                      "Edge-aware SLIC spatial weight reduction [0, 1].");
    cfg.def_readwrite("refine_passes", &VectorizerConfig::refine_passes,
                      "Boundary label refinement iterations (0 disables).");
    cfg.def_readwrite("max_merge_color_dist", &VectorizerConfig::max_merge_color_dist,
                      "Max LAB delta-E squared for small-region merging.");

    // Subpixel boundary
    cfg.def_readwrite("enable_subpixel_refine", &VectorizerConfig::enable_subpixel_refine,
                      "Enable gradient-guided sub-pixel boundary refinement.");
    cfg.def_readwrite("subpixel_max_displacement", &VectorizerConfig::subpixel_max_displacement,
                      "Max normal displacement for sub-pixel refine (px).");

    // Anti-aliasing
    cfg.def_readwrite("enable_antialias_detect", &VectorizerConfig::enable_antialias_detect,
                      "Detect AA mixed-edge pixels for better boundaries.");
    cfg.def_readwrite("aa_tolerance", &VectorizerConfig::aa_tolerance,
                      "Max LAB delta-E for AA blend pixel detection.");

    // Thin-line
    cfg.def_readwrite("thin_line_max_radius", &VectorizerConfig::thin_line_max_radius,
                      "Distance-transform radius for thin-line extraction.");

    // SVG output
    cfg.def_readwrite("svg_enable_stroke", &VectorizerConfig::svg_enable_stroke,
                      "Enable stroke output in SVG.");
    cfg.def_readwrite("svg_stroke_width", &VectorizerConfig::svg_stroke_width,
                      "Stroke width when svg_enable_stroke is True.");

    // Detail control
    cfg.def_readwrite("detail_level", &VectorizerConfig::detail_level,
                      "Unified detail control [0, 1]. -1 = disabled (use explicit params).");
    cfg.def_readwrite("merge_segment_tolerance", &VectorizerConfig::merge_segment_tolerance,
                      "Max control-point deviation to merge near-linear Bezier segments.");

    // Potrace knobs
    cfg.def_readwrite("min_contour_area", &VectorizerConfig::min_contour_area,
                      "Discard shapes smaller than this (pixels squared).");
    cfg.def_readwrite("min_hole_area", &VectorizerConfig::min_hole_area,
                      "Minimum hole area retained in final paths.");
    cfg.def_readwrite("contour_simplify", &VectorizerConfig::contour_simplify,
                      "Contour simplification strength (larger = fewer nodes).");
    cfg.def_readwrite("enable_coverage_fix", &VectorizerConfig::enable_coverage_fix,
                      "Patch uncovered pixels after vectorization.");
    cfg.def_readwrite("min_coverage_ratio", &VectorizerConfig::min_coverage_ratio,
                      "Minimum coverage ratio before patching kicks in.");
    cfg.def_readwrite("enable_depth_validation", &VectorizerConfig::enable_depth_validation,
                      "V2 only: run depth order validation (diagnostic).");

    cfg.def("__repr__", [](const VectorizerConfig& c) {
        std::ostringstream os;
        os << "VectorizerConfig(pipeline_mode="
           << (c.pipeline_mode == PipelineMode::V2 ? "V2" : "V1") << ", num_colors=" << c.num_colors
           << ", smoothness=" << c.smoothness << ", curve_fit_error=" << c.curve_fit_error
           << ", detail_level=" << c.detail_level << ", ...)";
        return os.str();
    });

    // ── VectorizerResult ─────────────────────────────────────────────────────
    py::class_<VectorizerResult>(m, "VectorizerResult",
                                 "Result of the vectorization pipeline (read-only).\n\n"
                                 "Contains the SVG output and associated metadata.")
        .def_readonly("svg_content", &VectorizerResult::svg_content,
                      "Complete SVG document as a string.")
        .def_readonly("width", &VectorizerResult::width, "Image width in pixels.")
        .def_readonly("height", &VectorizerResult::height, "Image height in pixels.")
        .def_readonly("num_shapes", &VectorizerResult::num_shapes, "Number of shapes in the SVG.")
        .def_readonly("resolved_num_colors", &VectorizerResult::resolved_num_colors,
                      "Actual color count used (from auto-detection or config).")
        .def_readonly("palette", &VectorizerResult::palette, "Color palette used (list of Rgb).")
        .def(
            "save",
            [](const VectorizerResult& self, const std::string& path) {
                py::gil_scoped_release release;
                std::ofstream f(path);
                if (!f) throw IOError("Cannot write to: " + path);
                f << self.svg_content;
            },
            py::arg("path"), "Save SVG content to a file.")
        .def("__repr__",
             [](const VectorizerResult& self) {
                 std::ostringstream os;
                 os << "VectorizerResult(width=" << self.width << ", height=" << self.height
                    << ", num_shapes=" << self.num_shapes << ", colors=" << self.resolved_num_colors
                    << ")";
                 return os.str();
             })
        .def(
            "__len__", [](const VectorizerResult& self) { return self.svg_content.size(); },
            "Length of the SVG content string.");

    // ── Exception translation ────────────────────────────────────────────────
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const InputError& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const IOError& e) {
            PyErr_SetString(PyExc_OSError, e.what());
        } catch (const InternalError& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const Error& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); }
    });

    // ── vectorize ────────────────────────────────────────────────────────────
    m.def(
        "vectorize",
        [](py::object input, const VectorizerConfig& config) -> VectorizerResult {
            if (py::isinstance<py::str>(input)) {
                std::string path = input.cast<std::string>();
                py::gil_scoped_release release;
                return Vectorize(path, config);
            }
            if (py::isinstance<py::bytes>(input)) {
                std::string buf = input.cast<std::string>();
                py::gil_scoped_release release;
                return Vectorize(reinterpret_cast<const uint8_t*>(buf.data()), buf.size(), config);
            }
            if (py::isinstance<py::array>(input)) {
                cv::Mat mat = numpy_to_mat(input);
                py::gil_scoped_release release;
                return Vectorize(mat, config);
            }
            throw InputError(
                "input must be str (file path), bytes (encoded image), or numpy.ndarray");
        },
        py::arg("input"), py::arg("config") = VectorizerConfig{},
        "Vectorize a raster image to SVG.\n\n"
        "Args:\n"
        "    input: File path (str), encoded image bytes (bytes),\n"
        "           or BGR/BGRA/GRAY uint8 numpy array.\n"
        "    config: Pipeline configuration (VectorizerConfig).\n\n"
        "Returns:\n"
        "    VectorizerResult with SVG content and metadata.\n\n"
        "Raises:\n"
        "    ValueError: Invalid input data.\n"
        "    OSError: File I/O error.\n"
        "    RuntimeError: Internal processing error.\n\n"
        "Examples:\n"
        "    >>> result = vectorize('photo.png')\n"
        "    >>> result = vectorize(open('photo.png', 'rb').read())\n"
        "    >>> result = vectorize(numpy_bgr_array)\n"
        "    >>> result = vectorize('photo.png', config)");
}
