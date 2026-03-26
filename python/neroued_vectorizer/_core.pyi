"""Type stubs for the neroued_vectorizer C++ extension module."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class Rgb:
    """Linear sRGB color with components in [0, 1].

    Internal representation uses linear (pre-gamma) sRGB values.
    Use :meth:`from_rgb255` / :meth:`to_rgb255` for 0--255 sRGB conversion.
    """

    r: float
    """Red component (linear sRGB, [0, 1])."""
    g: float
    """Green component (linear sRGB, [0, 1])."""
    b: float
    """Blue component (linear sRGB, [0, 1])."""

    def __init__(self, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> None:
        """Create from linear sRGB components in [0, 1]."""
        ...
    @staticmethod
    def from_rgb255(r: int, g: int, b: int) -> Rgb:
        """Create from 0--255 sRGB values (applies gamma decoding)."""
        ...
    def to_rgb255(self) -> tuple[int, int, int]:
        """Convert to 0--255 sRGB tuple ``(r, g, b)`` (applies gamma encoding)."""
        ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class PipelineMode:
    """Pipeline implementation selector."""

    V1: PipelineMode
    """Original boundary-graph + cutout pipeline."""
    V2: PipelineMode
    """Stacking model: per-layer Potrace with depth ordering."""

class VectorizerConfig:
    """Configuration for the vectorization pipeline.

    All fields have sensible defaults. Create an instance and override
    only the parameters you need::

        cfg = VectorizerConfig()
        cfg.num_colors = 8
        cfg.smoothness = 0.7
        result = vectorize("image.png", cfg)
    """

    # ── Pipeline mode ─────────────────────────────────────────────────────
    pipeline_mode: PipelineMode
    """Pipeline implementation: PipelineMode.V1 (default) or PipelineMode.V2."""

    # ── Color segmentation ───────────────────────────────────────────────
    num_colors: int
    """K-Means palette size. 0 = auto-detect optimal count."""
    min_region_area: int
    """Force-merge regions smaller than this (pixels squared)."""

    # ── Curve fitting ────────────────────────────────────────────────────
    curve_fit_error: float
    """Schneider curve fitting error threshold (pixels)."""
    corner_angle_threshold: float
    """Corner detection angle threshold in degrees."""
    smoothness: float
    """Contour smoothness [0, 1]. 0 = preserve detail, 1 = max smoothing."""

    # ── Preprocessing ────────────────────────────────────────────────────
    smoothing_spatial: float
    """Mean Shift spatial window radius."""
    smoothing_color: float
    """Mean Shift color window radius."""
    upscale_short_edge: int
    """Auto-upscale when short edge is below this (0 disables)."""
    max_working_pixels: int
    """Auto-downscale when total pixels exceed this (0 disables)."""

    # ── Segmentation ─────────────────────────────────────────────────────
    slic_region_size: int
    """SLIC target region size for multicolor mode."""
    slic_compactness: float
    """SLIC compactness (lower = follow color edges more)."""
    edge_sensitivity: float
    """Edge-aware SLIC spatial weight reduction [0, 1]."""
    refine_passes: int
    """Boundary label refinement iterations (0 disables)."""
    max_merge_color_dist: float
    """Max LAB delta-E squared for small-region merging."""

    # ── Subpixel boundary ────────────────────────────────────────────────
    enable_subpixel_refine: bool
    """Enable gradient-guided sub-pixel boundary refinement."""
    subpixel_max_displacement: float
    """Max normal displacement for sub-pixel refine (px)."""

    # ── Anti-aliasing detection ──────────────────────────────────────────
    enable_antialias_detect: bool
    """Detect AA mixed-edge pixels for better boundaries."""
    aa_tolerance: float
    """Max LAB delta-E for AA blend pixel detection."""

    # ── Thin-line enhancement ────────────────────────────────────────────
    thin_line_max_radius: float
    """Distance-transform radius for thin-line extraction."""

    # ── SVG output ───────────────────────────────────────────────────────
    svg_enable_stroke: bool
    """Enable stroke output in SVG."""
    svg_stroke_width: float
    """Stroke width when ``svg_enable_stroke`` is True."""

    # ── Detail control ───────────────────────────────────────────────────
    detail_level: float
    """Unified detail control [0, 1]. -1 = disabled (use explicit params)."""
    merge_segment_tolerance: float
    """Max control-point deviation to merge near-linear Bezier segments."""

    # ── Potrace pipeline ─────────────────────────────────────────────────
    min_contour_area: float
    """Discard shapes smaller than this (pixels squared)."""
    min_hole_area: float
    """Minimum hole area retained in final paths."""
    contour_simplify: float
    """Contour simplification strength (larger = fewer nodes)."""
    enable_coverage_fix: bool
    """Patch uncovered pixels after vectorization."""
    min_coverage_ratio: float
    """Minimum coverage ratio before patching kicks in."""
    enable_depth_validation: bool
    """V2 only: run depth order validation (diagnostic)."""

    def __init__(self) -> None:
        """Create a config with default values."""
        ...
    def __repr__(self) -> str: ...

class VectorizerResult:
    """Result of the vectorization pipeline (read-only).

    Contains the SVG output and associated metadata.
    """

    @property
    def svg_content(self) -> str:
        """Complete SVG document as a string."""
        ...
    @property
    def width(self) -> int:
        """Image width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Image height in pixels."""
        ...
    @property
    def num_shapes(self) -> int:
        """Number of shapes in the SVG."""
        ...
    @property
    def resolved_num_colors(self) -> int:
        """Actual color count used (from auto-detection or config)."""
        ...
    @property
    def palette(self) -> list[Rgb]:
        """Color palette used."""
        ...
    def save(self, path: str) -> None:
        """Save SVG content to a file.

        Args:
            path: Output file path.

        Raises:
            OSError: Cannot write to the path.
        """
        ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int:
        """Length of the SVG content string."""
        ...

def set_log_level(level: str) -> None:
    """Set C++ log verbosity.

    Args:
        level: One of ``'trace'``, ``'debug'``, ``'info'``, ``'warn'``,
               ``'error'``, ``'off'``. Default is ``'warn'``.
    """
    ...

def vectorize(
    input: str | bytes | NDArray[np.uint8],
    config: VectorizerConfig = ...,
) -> VectorizerResult:
    """Vectorize a raster image to SVG.

    Args:
        input: One of:

            - ``str`` -- file path to a raster image (PNG, JPG, BMP, etc.)
            - ``bytes`` -- encoded image data in memory
            - ``numpy.ndarray`` -- BGR/BGRA/GRAY ``uint8`` array (H, W[, C])

        config: Pipeline configuration. If omitted, sensible defaults are used.

    Returns:
        :class:`VectorizerResult` containing the SVG document and metadata.

    Raises:
        ValueError: Invalid input data or unsupported array format.
        OSError: File I/O error (file not found, permission denied, etc.).
        RuntimeError: Internal processing error.

    Examples:
        >>> result = vectorize("photo.png")
        >>> result = vectorize(open("photo.png", "rb").read())
        >>> result = vectorize(numpy_bgr_array)
        >>> result = vectorize("photo.png", config)
    """
    ...
