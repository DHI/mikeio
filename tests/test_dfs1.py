import datetime
import os

import numpy as np
import pytest

from mikeio.custom_exceptions import FileDoesNotExist
from mikeio.dfs1 import Dfs1
from mikeio.eum import EUMType, EUMUnit, ItemInfo


def test_filenotexist():
    with pytest.raises(FileDoesNotExist):
        Dfs1("file_that_does_not_exist.dfs1")


def test_repr():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1(filename)

    text = repr(dfs)

    assert "Dfs1" in text
    assert "Items" in text
    assert "dx" in text


def test_simple_write(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs1")

    data = []

    nt = 100
    nx = 20
    d = np.random.random([nt, nx])

    data.append(d)

    dfs = Dfs1()

    # write a file, without specifying dates, names, units etc.
    # Proably not so useful
    dfs.write(filename=filename, data=data)

    assert True


def test_write_single_item(tmpdir):

    filename = os.path.join(tmpdir.dirname, "random.dfs1")

    data = []
    d = np.random.random([100, 3])

    data.append(d)

    items = [ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)]
    title = "test dfs1"

    dfs = Dfs1()

    dfs.write(
        filename=filename,
        data=data,
        start_time=datetime.datetime(2012, 1, 1),
        dt=12,
        dx=100,
        items=items,
        title=title,
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
