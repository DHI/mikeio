"""Deprecation warnings attributed to the caller's code, not MIKE IO internals."""

from __future__ import annotations

import os
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _warn_deprecated(message: str) -> None:
    """Emit a FutureWarning pointing at the first frame outside MIKE IO."""
    warnings.warn(message, FutureWarning, skip_file_prefixes=(_PACKAGE_DIR,))


_PLOT_TYPE_METHOD = {
    "patch": "pcolormesh",
    "contour": "contour",
    "contourf": "contourf",
    "mesh_only": "mesh",
    "outline_only": "outline",
}


def _warn_plot_type(plot_type: object) -> None:
    """Warn that the plot_type keyword is deprecated, naming its replacement method."""
    method = _PLOT_TYPE_METHOD.get(str(plot_type))
    replacement = (
        f"Use .plot.{method}() instead."
        if method
        else "It has no replacement; customise the returned Axes with matplotlib."
    )
    _warn_deprecated(
        f"plot_type={plot_type!r} is deprecated and will be removed in MIKE IO 4.0. "
        + replacement
    )
