"""Helpers for importing optional runtime dependencies (matplotlib, scipy)."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_optional(name: str, extra: str) -> ModuleType:
    """Import an optional dependency, raising a helpful error if it is missing.

    Parameters
    ----------
    name: str
        module to import, e.g. "matplotlib.pyplot"
    extra: str
        name of the mikeio optional-dependency extra that provides it,
        e.g. "plot" or "interp"

    """
    try:
        return import_module(name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{name} is required for this feature. "
            f"Install it with: pip install mikeio[{extra}]"
        ) from e
