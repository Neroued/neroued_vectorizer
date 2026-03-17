"""neroued_vectorizer -- High-quality raster-to-SVG vectorization.

Quick start::

    import neroued_vectorizer as nv

    result = nv.vectorize("photo.png")
    result.save("output.svg")

See :func:`vectorize` for full documentation.
"""

from __future__ import annotations

import os
import sys

# Python 3.8+ on Windows no longer searches PATH for DLLs.  When the package
# is installed from source via ``pip install .`` (not from a pre-built wheel
# that already bundles DLLs via delvewheel), dynamically-linked libraries like
# OpenCV must be made visible through ``os.add_dll_directory()``.
# Set the ``NV_DLL_DIR`` environment variable to the directory containing
# those DLLs (e.g. ``C:/vcpkg/installed/x64-windows-release/bin``).
# Pre-built wheels ship with all required DLLs and do not need this.
if sys.platform == "win32":
    _dll_dir = os.environ.get("NV_DLL_DIR", "")
    if _dll_dir and os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)

from importlib.metadata import PackageNotFoundError, version

from neroued_vectorizer._core import (
    Rgb,
    VectorizerConfig,
    VectorizerResult,
    set_log_level,
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
    "set_log_level",
    "vectorize",
]
