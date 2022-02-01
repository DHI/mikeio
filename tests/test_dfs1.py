import datetime
import os

import numpy as np
import pandas as pd
import pytest

import mikeio

from mikeio import Dfs1, Dataset
from mikeio.eum import EUMType, EUMUnit, ItemInfo


def test_filenotexist():
    with pytest.raises(FileNotFoundError):
        Dfs1("file_that_does_not_exist.dfs1")


def test_repr():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    text = repr(dfs)

    assert "Dfs1" in text
    assert "Items" in text
    assert "dx" in text


def test_repr_empty():

    dfs = Dfs1()

    text = repr(dfs)

    assert "Dfs1" in text


def test_simple_write(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs1")

    nt = 100
    nx = 20
    d = np.random.random([nt, nx])

    ds = Dataset(
        data=[d], time=pd.date_range("2000", freq="H", periods=nt), items=["My item"]
    )

    dfs = Dfs1()

    dfs.write(filename=filename, data=ds)

    assert True


def test_write_single_item(tmpdir):

    filename = os.path.join(tmpdir.dirname, "random.dfs1")

    d = np.random.random([100, 3])

    ds = Dataset(
        data=[d],
        time=pd.date_range("2012-1-1", periods=100, freq="12s"),
        items=[ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)],
    )

    dfs = Dfs1()

    dfs.write(
        filename=filename,
        data=ds,
        dx=100,
        title="test dfs1",
    )

    assert True


def test_read():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    ds = dfs.read([0])
    data = ds.data[0]
    assert data.shape == (100, 3)  # time, x


def test_read_item_names():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    ds = dfs.read(["testing water level"])
    data = ds.data[0]
    assert data.shape == (100, 3)  # time, x


def test_read_time_steps():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    ds = dfs.read(time_steps=[3, 5])
    data = ds.data[0]
    assert data.shape == (2, 3)  # time, x


def test_write_some_time_steps_new_file(tmpdir):

    outfilename = os.path.join(tmpdir.dirname, "subset.dfs1")
    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    ds = dfs.read(time_steps=[0, 1, 2, 3, 4, 5])
    data = ds.data[0]
    assert data.shape == (6, 3)  # time, x

    dfs.write(outfilename, ds)

    dfsnew = Dfs1(outfilename)

    dsnew = dfsnew.read()

    assert dsnew["testing water level"].shape == (6, 3)


def test_read_item_names_not_in_dataset_fails():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    with pytest.raises(Exception):
        dfs.read(["NOTAREALVARIABLE"])


def test_read_names_access():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    res = dfs.read([0])
    data = res.data
    item = data[0]
    time = res.time
    assert item.shape == (100, 3)  # time, x
    assert len(time) == 100
    assert res.items[0].name == "testing water level"
    assert res.items[0].type == EUMType.Water_Level
    assert res.items[0].unit == EUMUnit.meter


def test_read_start_end_time():

    dfs = Dfs1("tests/testdata/random.dfs1")
    ds = dfs.read()

    assert dfs.start_time == ds.start_time
    assert dfs.end_time == ds.end_time


def test_read_start_end_time_relative_time():

    dfs = Dfs1("tests/testdata/physical_basin_wave_maker_signal.dfs1")
    ds = dfs.read()

    assert dfs.start_time == ds.start_time
    assert dfs.end_time == ds.end_time


def test_select_point_dfs1_to_dfs0(tmp_path):

    outfilename = tmp_path / "vu_tide_hourly_p0.dfs0"

    ds = mikeio.read("tests/testdata/vu_tide_hourly.dfs1")

    assert ds.n_elements > 1
    ds_0 = ds.isel(0, axis="space")
    assert ds_0.n_elements == 1
    ds_0.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == ds.n_timesteps
