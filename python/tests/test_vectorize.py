"""Basic tests for neroued_vectorizer Python bindings."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import neroued_vectorizer as nv


# ── Import / version ─────────────────────────────────────────────────────────


def test_version_exists():
    assert isinstance(nv.__version__, str)


# ── VectorizerConfig ─────────────────────────────────────────────────────────


def test_config_defaults():
    cfg = nv.VectorizerConfig()
    assert cfg.num_colors == 0
    assert cfg.min_region_area == 50
    assert isinstance(cfg.curve_fit_error, float)
    assert cfg.enable_coverage_fix is True


def test_config_readwrite():
    cfg = nv.VectorizerConfig()
    cfg.num_colors = 8
    cfg.curve_fit_error = 1.5
    cfg.enable_coverage_fix = False
    assert cfg.num_colors == 8
    assert cfg.curve_fit_error == pytest.approx(1.5)
    assert cfg.enable_coverage_fix is False


# ── Rgb ──────────────────────────────────────────────────────────────────────


def test_rgb_default():
    c = nv.Rgb()
    assert c.r == pytest.approx(0.0)
    assert c.g == pytest.approx(0.0)
    assert c.b == pytest.approx(0.0)


def test_rgb_from_rgb255():
    c = nv.Rgb.from_rgb255(255, 0, 0)
    assert c.r > 0.9
    r, g, b = c.to_rgb255()
    assert r == 255
    assert g == 0
    assert b == 0


def test_rgb_repr():
    c = nv.Rgb(0.5, 0.5, 0.5)
    s = repr(c)
    assert "Rgb" in s
    assert "sRGB" in s


def test_rgb_equality():
    a = nv.Rgb(0.5, 0.5, 0.5)
    b = nv.Rgb(0.5, 0.5, 0.5)
    c = nv.Rgb(1.0, 0.0, 0.0)
    assert a == b
    assert not (a == c)


# ── vectorize with numpy array ───────────────────────────────────────────────


def test_vectorize_solid_color():
    """A solid-color image should vectorize without error."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = nv.vectorize(img)
    assert isinstance(result, nv.VectorizerResult)
    assert result.width > 0
    assert result.height > 0
    assert result.svg_content.startswith("<svg") or result.svg_content.startswith("<?xml")
    assert result.num_shapes >= 0


def test_vectorize_grayscale():
    img = np.full((64, 64), 200, dtype=np.uint8)
    result = nv.vectorize(img)
    assert result.width > 0
    assert result.svg_content


def test_vectorize_rgba():
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    img[:, :, :3] = 100
    img[:, :, 3] = 255
    result = nv.vectorize(img)
    assert result.width > 0


def test_vectorize_with_config():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    cfg = nv.VectorizerConfig()
    cfg.num_colors = 2
    cfg.curve_fit_error = 2.0
    result = nv.vectorize(img, cfg)
    assert result.resolved_num_colors <= 2


def test_result_palette():
    img = np.full((64, 64, 3), 50, dtype=np.uint8)
    result = nv.vectorize(img)
    assert isinstance(result.palette, list)
    if result.palette:
        assert isinstance(result.palette[0], nv.Rgb)


def test_result_repr():
    img = np.full((32, 48, 3), 128, dtype=np.uint8)
    result = nv.vectorize(img)
    s = repr(result)
    assert "VectorizerResult" in s


def test_result_len():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = nv.vectorize(img)
    assert len(result) == len(result.svg_content)
    assert len(result) > 0


def test_result_save(tmp_path):
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = nv.vectorize(img)
    out = tmp_path / "output.svg"
    result.save(str(out))
    assert out.exists()
    content = out.read_text()
    assert content == result.svg_content


def test_config_repr():
    cfg = nv.VectorizerConfig()
    s = repr(cfg)
    assert "VectorizerConfig" in s
    assert "num_colors" in s


# ── vectorize with bytes ─────────────────────────────────────────────────────


def test_vectorize_from_png_bytes():
    """Create a minimal valid PNG in memory and vectorize it."""
    try:
        import cv2

        img = np.full((32, 32, 3), 180, dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        result = nv.vectorize(bytes(buf))
        assert result.width == 32
    except ImportError:
        pytest.skip("cv2 not available for encoding test PNG")


# ── vectorize with file path ─────────────────────────────────────────────────


def test_vectorize_from_file():
    try:
        import cv2

        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        cv2.imwrite(path, img)
        result = nv.vectorize(path)
        assert result.width == 32
        Path(path).unlink(missing_ok=True)
    except ImportError:
        pytest.skip("cv2 not available for writing test image")


# ── Error handling ───────────────────────────────────────────────────────────


def test_bad_file_path():
    with pytest.raises((OSError, RuntimeError)):
        nv.vectorize("/nonexistent/image.png")


def test_bad_input_type():
    with pytest.raises((ValueError, TypeError)):
        nv.vectorize(12345)  # type: ignore[arg-type]


def test_bad_array_dtype():
    arr = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        nv.vectorize(arr)


def test_bad_array_shape():
    arr = np.zeros((32,), dtype=np.uint8)
    with pytest.raises(ValueError):
        nv.vectorize(arr)
