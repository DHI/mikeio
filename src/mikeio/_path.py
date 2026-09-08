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


def normalize_output_path(filename: str | Path) -> str:
    """Convert a path to write to a string, expanding a leading `~`.

    Same as normalize_path, but also checks that the file can be created,
    creating missing parent directories if necessary. The underlying
    *mikecore* library does not check this: writing to a path that cannot be
    created crashes the interpreter.

    Parameters
    ----------
    filename
        Path to normalize.

    Returns
    -------
    str
        The path as a string, with a leading `~` expanded.

    Raises
    ------
    IsADirectoryError
        If the path is an existing directory.
    OSError
        If the file cannot be created, e.g. in a read-only directory.

    """
    path = normalize_path(filename)

    if os.path.isdir(path):
        raise IsADirectoryError(f"Cannot write to {path}, it is a directory")

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)

    # mikecore fails silently on a path it cannot create, so create it here
    with open(path, "ab"):
        pass

    return path
