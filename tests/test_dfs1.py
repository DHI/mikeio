from pathlib import Path
import numpy as np
import pytest
import pandas as pd

import mikeio

from mikeio import EUMType, EUMUnit


def test_filenotexist() -> None:
    with pytest.raises(FileNotFoundError):
        mikeio.Dfs1("file_that_does_not_exist.dfs1")


def test_repr() -> None:
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(filename)

    text = repr(dfs)

    assert "Dfs1" in text
    assert "items" in text
    assert "dx" in text


def test_properties() -> None:
    filename = r"tests/testdata/tide1.dfs1"
    dfs = mikeio.Dfs1(filename)

    assert dfs.dx == 0.06666692346334457
    assert dfs.x0 == 0.0
    assert dfs.nx == 10
    assert dfs.projection_string == "LONG/LAT"
    assert dfs.longitude == -5.0
    assert dfs.latitude == 51.20000076293945
    assert dfs.orientation == 180

    g = dfs.geometry
    assert isinstance(g, mikeio.Grid1D)
    assert g.dx == 0.06666692346334457
    assert g._x0 == 0.0
    assert g.nx == 10
    assert g.projection == "LONG/LAT"
    assert g.origin == (-5.0, 51.20000076293945)
    assert g.orientation == 180


def test_read_write_properties(tmp_path: Path) -> None:
    # test that properties are the same after read-write
    filename = r"tests/testdata/tide1.dfs1"
    ds1 = mikeio.read(filename)

    fp = tmp_path / "tide1.dfs1"
    ds1.to_dfs(fp)
    ds2 = mikeio.read(fp)

    assert ds1.geometry == ds2.geometry


def test_read() -> None:
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(filename)

    ds = dfs.read(items=[0])
    data = ds[0].to_numpy()
    assert data.shape == (100, 3)  # time, x


def test_read_item_names() -> None:
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(filename)

    ds = dfs.read(items=["testing water level"])
    data = ds[0].to_numpy()
    assert data.shape == (100, 3)  # time, x


def test_read_time_steps() -> None:
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(filename)

    ds = dfs.read(time=[3, 5])
    data = ds[0].to_numpy()
    assert data.shape == (2, 3)  # time, x


def test_write_some_time_steps_new_file(tmp_path: Path) -> None:
    fp = tmp_path / "random.dfs1"
    ds = mikeio.read("tests/testdata/random.dfs1", time=[0, 1, 2, 3, 4, 5])

    data = ds[0].to_numpy()
    assert data.shape == (6, 3)  # time, x

    ds.to_dfs(fp)

    dfsnew = mikeio.Dfs1(fp)

    dsnew = dfsnew.read()

    assert dsnew["testing water level"].shape == (6, 3)


def test_read_item_names_not_in_dataset_fails() -> None:
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(filename)

    with pytest.raises(KeyError):
        dfs.read(items=["NOTAREALVARIABLE"])


def test_read_names_access() -> None:
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(filename)

    res = dfs.read(items=[0])
    item_data = res[0].to_numpy()
    time = res.time
    assert item_data.shape == (100, 3)  # time, x
    assert len(time) == 100
    assert res.items[0].name == "testing water level"
    assert res.items[0].type == EUMType.Water_Level
    assert res.items[0].unit == EUMUnit.meter


def test_read_start_end_time() -> None:
    dfs = mikeio.Dfs1("tests/testdata/random.dfs1")
    ds = dfs.read()

    assert dfs.start_time == ds.start_time
    assert dfs.end_time == ds.end_time


def test_read_start_end_time_relative_time() -> None:
    dfs = mikeio.Dfs1("tests/testdata/physical_basin_wave_maker_signal.dfs1")
    ds = dfs.read()

    assert dfs.start_time is None
    assert ds.start_time.total_seconds() == 0.0
    assert ds.end_time.total_seconds() == 9.95


def test_get_time_axis_without_reading_data() -> None:
    dfs0file = r"tests/testdata/random.dfs1"
    dfs = mikeio.Dfs1(dfs0file)
    assert isinstance(dfs.time, pd.DatetimeIndex)
    assert len(dfs.time) == 100


def test_get_time_axis_without_reading_data_relative() -> None:
    dfs0file = r"tests/testdata/physical_basin_wave_maker_signal.dfs1"
    dfs = mikeio.Dfs1(dfs0file)
    assert isinstance(dfs.time, pd.TimedeltaIndex)
    assert len(dfs.time) == 200


def test_select_point_and_single_step_dfs1_to_dfs0(tmp_path: Path) -> None:
    outfilename = tmp_path / "vu_tide_hourly_p0.dfs0"

    ds = mikeio.read("tests/testdata/vu_tide_hourly.dfs1")

    ds_0 = ds.isel(0, axis="space")
    ds_0_0 = ds_0.isel(0)
    assert ds_0_0.n_timesteps == 1
    ds_0_0.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == 1


def test_interp_dfs1() -> None:
    ds = mikeio.read("tests/testdata/waterlevel_north.dfs1")

    da = ds["North WL"]

    assert da.geometry.x[-1] == 8800

    dai = da.interp(x=0)
    assert dai[0].values == pytest.approx(-0.33)

    dai = da.interp(x=4000)
    assert dai[0].values == pytest.approx(-0.3022272830659693)

    dai = da.interp(x=8800)
    assert dai[-1].values == pytest.approx(-0.0814)

    dai = da.interp(x=8900)  # outside the domain
    assert np.isnan(dai[-1].values)

    dai = da.interp(x=-10)  # outside the domain
    assert np.isnan(dai[-1].values)


def test_interp_onepoint_dfs1() -> None:
    ds = mikeio.read("tests/testdata/nx1.dfs1")
    assert ds.geometry.nx == 1

    with pytest.raises(AssertionError, match="not possible for Grid1D with one point"):
        ds[0].interp(x=0)
