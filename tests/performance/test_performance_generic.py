"""Benchmarks for the generic functions; run with `just perftest`."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

import mikeio
import mikeio.generic

Keep = Literal["first", "last", "average"]


def _write_overlapping_files(
    folder: Path,
    geometry: mikeio.Grid1D | mikeio.Grid2D,
    geometry_shape: tuple[int, ...],
    n_files: int,
    n_steps: int,
    overlap: int,
    suffix: str,
) -> list[Path]:
    rng = np.random.default_rng(0)
    files = []
    for i in range(n_files):
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i * (n_steps - overlap))
        time = pd.date_range(start, periods=n_steps, freq="h")
        shape = (n_steps, *geometry_shape)
        ds = mikeio.Dataset(
            [
                mikeio.DataArray(
                    rng.random(shape, dtype=np.float32),
                    time=time,
                    geometry=geometry,
                    item=mikeio.ItemInfo(f"Item {j + 1}"),
                )
                for j in range(3)
            ]
        )
        filename = folder / f"part{i}{suffix}"
        ds.to_dfs(filename)
        files.append(filename)
    return files


@pytest.fixture(scope="module")
def large_dfs2_files(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Three dfs2 files of about 110 MB each, overlapping by 20 steps."""
    folder = tmp_path_factory.mktemp("large_dfs2")
    grid = mikeio.Grid2D(nx=300, ny=300, dx=1.0)
    return _write_overlapping_files(folder, grid, (300, 300), 3, 100, 20, ".dfs2")


@pytest.fixture(scope="module")
def many_steps_dfs1_files(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Three dfs1 files with many small steps, overlapping by 1000 steps."""
    folder = tmp_path_factory.mktemp("many_steps_dfs1")
    grid = mikeio.Grid1D(nx=10, dx=1.0)
    return _write_overlapping_files(folder, grid, (10,), 3, 20_000, 1000, ".dfs1")


@pytest.mark.parametrize("keep", ["last", "first", "average"])
def test_concat_large_equidistant_files(
    large_dfs2_files: list[Path], tmp_path: Path, keep: Keep
) -> None:
    fp = tmp_path / "concat.dfs2"

    mikeio.generic.concat(large_dfs2_files, fp, keep=keep)

    dfs = mikeio.open(fp)
    assert isinstance(dfs, mikeio.Dfs2)
    assert dfs.n_timesteps == 3 * 100 - 2 * 20


@pytest.mark.parametrize("keep", ["last", "first", "average"])
def test_concat_equidistant_files_with_many_steps(
    many_steps_dfs1_files: list[Path], tmp_path: Path, keep: Keep
) -> None:
    fp = tmp_path / "concat.dfs1"

    mikeio.generic.concat(many_steps_dfs1_files, fp, keep=keep)

    dfs = mikeio.open(fp)
    assert isinstance(dfs, mikeio.Dfs1)
    assert dfs.n_timesteps == 3 * 20_000 - 2 * 1000
