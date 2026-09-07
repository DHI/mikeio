"""Path handling utilities."""

from __future__ import annotations

import os
from pathlib import Path


def normalize_path(filename: str | Path) -> str:
    """Convert a path-like object to a string, expanding a leading ``~``.

    MIKE IO accepts both strings and :class:`pathlib.Path` objects, but the
    underlying *mikecore* library only understands strings and does not expand
    ``~`` to the user's home directory. Normalizing paths at the public API
    boundary makes MIKE IO behave like pandas and xarray, which expand ``~``
    for both reading and writing.

    Parameters
    ----------
    filename
        Path to normalize.

    Returns
    -------
    str
        The path as a string, with a leading ``~`` expanded.

    Examples
    --------
    >>> normalize_path("~/data/wl.dfs0")  # doctest: +SKIP
    '/home/user/data/wl.dfs0'

    """
    return os.path.expanduser(os.fspath(filename))
