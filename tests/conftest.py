"""Shared pytest fixtures."""

from __future__ import annotations

import gc
import os
from collections import Counter
from pathlib import Path
from typing import Callable

import pytest
from hypothesis import HealthCheck, settings

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


# Hypothesis: few, fixed examples by default, for a fast and stable suite;
# HYPOTHESIS_PROFILE=deep searches far wider, e.g. after changing geometry code
settings.register_profile(
    "fast",
    max_examples=30,
    derandomize=True,
    deadline=None,  # the first matplotlib call is slow
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))
