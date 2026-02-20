"""Tests verifying that file handles are properly closed after use."""

import gc
import os
import platform

import pytest

import mikeio


def _count_open_fds() -> int:
    """Count open file descriptors on Linux via /proc/self/fd."""
    return len(os.listdir("/proc/self/fd"))


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="File descriptor counting via /proc only works on Linux",
)


def test_dfs1_init_closes_file_handle() -> None:
    gc.collect()
    baseline = _count_open_fds()
    for _ in range(50):
        mikeio.Dfs1("tests/testdata/random.dfs1")
    gc.collect()
    assert _count_open_fds() - baseline == 0


def test_dfs3_init_closes_file_handle() -> None:
    gc.collect()
    baseline = _count_open_fds()
    for _ in range(50):
        mikeio.Dfs3("tests/testdata/Grid1.dfs3")
    gc.collect()
    assert _count_open_fds() - baseline == 0


def test_dfsu_read_closes_file_handle() -> None:
    gc.collect()
    baseline = _count_open_fds()
    for _ in range(50):
        dfs = mikeio.open("tests/testdata/HD2D.dfsu")
        dfs.read()
    gc.collect()
    assert _count_open_fds() - baseline == 0
