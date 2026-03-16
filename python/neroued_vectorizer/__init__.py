"""neroued_vectorizer -- High-quality raster-to-SVG vectorization.

Quick start::

    import neroued_vectorizer as nv

    result = nv.vectorize("photo.png")
    result.save("output.svg")

See :func:`vectorize` for full documentation.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from neroued_vectorizer._core import (
    Rgb,
    VectorizerConfig,
    VectorizerResult,
    vectorize,
)

try:
    __version__ = version("neroued-vectorizer")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "Rgb",
    "VectorizerConfig",
    "VectorizerResult",
    "vectorize",
]
