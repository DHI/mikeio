"""Shared pytest fixtures."""

from __future__ import annotations

import gc
import os
from collections import Counter
from pathlib import Path
from typing import Callable

import pytest

_PROC_FD = Path("/proc/self/fd")


def _open_targets() -> Counter[str]:
    """Count this process's open descriptors by the path they point at."""
    targets: Counter[str] = Counter()
    for fd in os.listdir(_PROC_FD):
        try:
            targets[os.readlink(_PROC_FD / fd)] += 1
        except OSError:  # closed between listdir and readlink
            continue
    return targets


@pytest.fixture
def count_fds() -> Callable[[str | Path], int]:
    """Count descriptors this test left open on a path, relative to its start.

    Garbage is collected before every count, so a handle only counts while
    something still references it. Linux only: the test is skipped where
    /proc/self/fd does not exist.
    """
    if not _PROC_FD.is_dir():
        pytest.skip("counting open descriptors needs /proc/self/fd (Linux)")

    gc.collect()
    baseline = _open_targets()

    def count(path: str | Path) -> int:
        gc.collect()
        target = os.path.realpath(path)
        return _open_targets()[target] - baseline[target]

    return count
