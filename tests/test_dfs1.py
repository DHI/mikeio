import os

import numpy as np
import pytest
import pandas as pd

import mikeio

from mikeio import Dfs1
from mikeio.eum import EUMType, EUMUnit


def test_filenotexist():
    with pytest.raises(FileNotFoundError):
        mikeio.open("file_that_does_not_exist.dfs1")


def test_repr():

    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    text = repr(dfs)

    assert "Dfs1" in text
    assert "items" in text
    assert "dx" in text


def test_repr_empty():

    dfs = Dfs1()

    text = repr(dfs)

    assert "Dfs1" in text


def test_properties():
    filename = r"tests/testdata/tide1.dfs1"
    dfs = mikeio.open(filename)

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


def test_read_write_properties(tmpdir):
    # test that properties are the same after read-write
    filename = r"tests/testdata/tide1.dfs1"
    ds1 = mikeio.read(filename)

    outfilename = os.path.join(tmpdir.dirname, "tide1.dfs1")
    ds1.to_dfs(outfilename)
    ds2 = mikeio.read(outfilename)

    assert ds1.geometry == ds2.geometry


def test_read():

    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    ds = dfs.read(items=[0])
    data = ds[0].to_numpy()
    assert data.shape == (100, 3)  # time, x


def test_read_item_names():

    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    ds = dfs.read(items=["testing water level"])
    data = ds[0].to_numpy()
    assert data.shape == (100, 3)  # time, x


def test_read_time_steps():

    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    ds = dfs.read(time=[3, 5])
    data = ds[0].to_numpy()
    assert data.shape == (2, 3)  # time, x


def test_write_some_time_steps_new_file(tmpdir):

    outfilename = os.path.join(tmpdir.dirname, "subset.dfs1")
    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    ds = dfs.read(time=[0, 1, 2, 3, 4, 5])
    data = ds[0].to_numpy()
    assert data.shape == (6, 3)  # time, x

    dfs.write(outfilename, ds)

    dfsnew = mikeio.open(outfilename)

    dsnew = dfsnew.read()

    assert dsnew["testing water level"].shape == (6, 3)


def test_read_item_names_not_in_dataset_fails():

    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    with pytest.raises(Exception):
        dfs.read(["NOTAREALVARIABLE"])


def test_read_names_access():

    filename = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(filename)

    res = dfs.read(items=[0])
    item_data = res[0].to_numpy()
    time = res.time
    assert item_data.shape == (100, 3)  # time, x
    assert len(time) == 100
    assert res.items[0].name == "testing water level"
    assert res.items[0].type == EUMType.Water_Level
    assert res.items[0].unit == EUMUnit.meter


def test_read_start_end_time():

    dfs = mikeio.open("tests/testdata/random.dfs1")
    ds = dfs.read()

    assert dfs.start_time == ds.start_time
    assert dfs.end_time == ds.end_time


def test_read_start_end_time_relative_time():

    dfs = mikeio.open("tests/testdata/physical_basin_wave_maker_signal.dfs1")
    ds = dfs.read()

    assert dfs.start_time == ds.start_time
    assert dfs.end_time == ds.end_time


def test_get_time_axis_without_reading_data():

    dfs0file = r"tests/testdata/random.dfs1"
    dfs = mikeio.open(dfs0file)
    assert isinstance(dfs.time, pd.DatetimeIndex)
    assert len(dfs.time) == 100


def test_get_time_axis_without_reading_data_relative():
    dfs0file = r"tests/testdata/physical_basin_wave_maker_signal.dfs1"
    dfs = mikeio.open(dfs0file)
    assert isinstance(dfs.time, pd.DatetimeIndex)  # start time is not correct !
    assert len(dfs.time) == 200


def test_select_point_dfs1_to_dfs0(tmp_path):

    outfilename = tmp_path / "vu_tide_hourly_p0.dfs0"

    ds = mikeio.read("tests/testdata/vu_tide_hourly.dfs1")

    assert ds.n_elements > 1
    ds_0 = ds.isel(0, axis="space")
    assert ds_0.n_elements == 1
    ds_0.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == ds.n_timesteps


def test_select_point_and_single_step_dfs1_to_dfs0(tmp_path):

    outfilename = tmp_path / "vu_tide_hourly_p0.dfs0"

    ds = mikeio.read("tests/testdata/vu_tide_hourly.dfs1")

    assert ds.n_elements > 1
    ds_0 = ds.isel(0, axis="space")
    assert ds_0.n_elements == 1
    ds_0_0 = ds_0.isel(0)
    assert ds_0_0.n_timesteps == 1
    ds_0_0.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == 1


def test_select_point_dfs1_to_dfs0_double(tmp_path):

    outfilename = tmp_path / "vu_tide_hourly_p0_dbl.dfs0"

    ds = mikeio.read("tests/testdata/vu_tide_hourly.dfs1")

    assert ds.n_elements > 1
    ds_0 = ds.isel(0, axis="space")
    assert ds_0.n_elements == 1
    ds_0.to_dfs(outfilename, dtype=np.float64)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == ds.n_timesteps

def test_interp_dfs1():

    ds = mikeio.read("tests/testdata/waterlevel_north.dfs1")

    da: mikeio.DataArray = ds.North_WL

    assert da.geometry.x[-1] == 8800

    dai = da.interp(x=0)
    assert dai[0].values == pytest.approx(-0.33)

    dai = da.interp(x=4000)
    assert dai[0].values == pytest.approx(-0.3022272830659693)

    dai = da.interp(x=8800)
    assert dai[-1].values == pytest.approx(-0.0814)

    dai = da.interp(x=8900) # outside the domain
    assert np.isnan(dai[-1].values)

    dai = da.interp(x=-10) # outside the domain
    assert np.isnan(dai[-1].values)


def test_interp_onepoint_dfs1():

    ds = mikeio.read("tests/testdata/nx1.dfs1")    
    assert ds.geometry.nx == 1

    with pytest.raises(AssertionError, match="not possible for Grid1D with one point"):
        ds[0].interp(x=0)

