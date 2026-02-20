"""Tests verifying that file handles are properly closed after use.

Before the fix, file handles opened via mikecore were stored on objects
(e.g. self._dfs) and never closed, so creating many instances while
holding references would leak one file descriptor per instance.
On Windows this would hit the 512 open-file limit; on Linux we detect
it via /proc/self/fd.

See: https://github.com/DHI/mikecore-python/issues/41
"""

import gc
import os
import platform

import pytest

import mikeio
from mikecore.DfsFileFactory import DfsFileFactory


def _count_open_fds() -> int:
    """Count open file descriptors on Linux via /proc/self/fd."""
    return len(os.listdir("/proc/self/fd"))


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="File descriptor counting via /proc only works on Linux",
)


def test_count_open_fds_sanity_check() -> None:
    """Verify _count_open_fds detects mikecore file handles."""
    gc.collect()
    baseline = _count_open_fds()

    dfs = DfsFileFactory.DfsGenericOpen("tests/testdata/random.dfs1")
    try:
        assert _count_open_fds() > baseline, "opening a file must increase FD count"
    finally:
        dfs.Close()

    gc.collect()
    assert _count_open_fds() == baseline, "closing a file must restore FD count"


def test_dfs1_init_closes_file_handle() -> None:
    """Dfs1.__init__ must not leak a file handle.

    Before the fix, Dfs1.__init__ stored the open handle in self._dfs
    without closing it, so each live instance held one file descriptor.
    """
    gc.collect()
    baseline = _count_open_fds()

    instances = []
    for _ in range(50):
        instances.append(mikeio.Dfs1("tests/testdata/random.dfs1"))

    assert _count_open_fds() - baseline == 0


def test_dfs3_init_closes_file_handle() -> None:
    """Dfs3._read_dfs3_header must not leak a file handle.

    Before the fix, _read_dfs3_header stored the open handle in
    self._dfs without closing it.
    """
    gc.collect()
    baseline = _count_open_fds()

    instances = []
    for _ in range(50):
        instances.append(mikeio.Dfs3("tests/testdata/Grid1.dfs3"))

    assert _count_open_fds() - baseline == 0


def test_dfsu_read_closes_file_handle() -> None:
    """Dfsu read must not leak file handles.

    Each read() opens a DfsuFile; it must be closed before returning.
    """
    gc.collect()
    baseline = _count_open_fds()

    results = []
    for _ in range(50):
        dfs = mikeio.open("tests/testdata/HD2D.dfsu")
        results.append(dfs.read())

    assert _count_open_fds() - baseline == 0
