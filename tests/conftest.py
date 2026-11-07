"""Shared test fixtures and utilities."""

import gc
import os
import platform
from collections.abc import Callable
from pathlib import Path

import pytest


def _count_fds_for_file(filename: str) -> int:
    """Count open file descriptors pointing to a specific filename on Linux.

    Uses /proc/self/fd readlinks to identify descriptors matching the target file.
    Returns 0 on non-Linux platforms.
    """
    if platform.system() != "Linux":
        return 0

    gc.collect()  # release handles held only by unreachable objects, e.g. old tracebacks
    target = str(Path(filename).resolve())
    count = 0
    fd_dir = "/proc/self/fd"
    for entry in os.listdir(fd_dir):
        try:
            link = os.readlink(os.path.join(fd_dir, entry))
            if link == target:
                count += 1
        except OSError:
            continue
    return count


@pytest.fixture
def count_fds() -> Callable[[str], int]:
    """Return a function counting this process's open descriptors on a file."""
    return _count_fds_for_file
