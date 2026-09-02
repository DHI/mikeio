"""Helpers for importing optional runtime dependencies (matplotlib, scipy)."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_optional(name: str, extra: str) -> ModuleType:
    """Import an optional dependency, raising a helpful error if it is missing.

    Catches ImportError (not just ModuleNotFoundError) so a broken install
    (e.g. a matplotlib backend failing to load a C extension) also gets the
    install-hint message, while preserving the original exception type.

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
    except ImportError as e:
        raise type(e)(
            f"{name} is required for this feature. "
            f"Install it with: pip install mikeio[{extra}]"
        ) from e


def require_matplotlib(name: str = "matplotlib.pyplot") -> ModuleType:
    """Import a matplotlib module, raising a helpful error if unavailable."""
    return import_optional(name, "plot")


def require_scipy(name: str) -> ModuleType:
    """Import a scipy module, raising a helpful error if unavailable."""
    return import_optional(name, "interp")
