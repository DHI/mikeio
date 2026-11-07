"""Deprecation warnings attributed to the caller's code, not MIKE IO internals."""

from __future__ import annotations

import os
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _warn_deprecated(message: str) -> None:
    """Emit a FutureWarning pointing at the first frame outside MIKE IO."""
    warnings.warn(message, FutureWarning, skip_file_prefixes=(_PACKAGE_DIR,))
