import os
import numpy as np
import datetime
import pytest
from shutil import copyfile

from mikeio.dfs1 import Dfs1
from mikeio.eum import Item


def test_simple_create():

    filename = r"simple.dfs1"

    data = []

    nt = 100
    nx = 20
    d = np.random.random([nt, nx])

    data.append(d)

    dfs = Dfs1()

    dfs.create(filename=filename, data=data)

    assert True
    os.remove(filename)


def test_create_single_item():

    start_time = datetime.datetime(2012, 1, 1)

    dt = 12

    variable_type = [Item.Water_Level]  # [100000]
    unit = [Item.Water_Level.units["meter"]]  # [1000]

    filename = r"random.dfs1"

    data = []
    d = np.random.random([100, 3])

    data.append(d)
    length_x = 100

    names = ["testing water level"]
    title = "test dfs1"

    dfs = Dfs1()

    dfs.create(
        filename=filename,
        data=data,
        start_time=start_time,
        dt=dt,
        variable_type=variable_type,
        unit=unit,
        length_x=length_x,
        names=names,
        title=title,
    )

    assert True
    os.remove(filename)


def test_read():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1()

    data = dfs.read(filename, [0])[0]
    data = data[0]
    assert data.shape == (100, 3)  # time, x


def test_read_item_names():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1()

    data = dfs.read(filename, item_names=["testing water level"])[0]
    data = data[0]
    assert data.shape == (100, 3)  # time, x


def test_read_item_names_not_in_dataset_fails():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1()

    with pytest.raises(Exception):
        dfs.read(filename, item_names=["NOTAREALVARIABLE"])


def test_read_names_access():

    filename = r"tests/testdata/random.dfs1"
    dfs = Dfs1()

    res = dfs.read(filename, [0])
    data = res.data
    item = data[0]
    time = res.time
    assert item.shape == (100, 3)  # time, x
    assert len(time) == 100
    assert res.items[0].name == "testing water level"
    assert res.items[0].item == Item.Water_Level
    assert res.items[0].unit == Item.Water_Level.units["meter"]


def test_write():

    filename1 = r"tests/testdata/random.dfs1"
    filename2 = r"tests/testdata/random_for_write.dfs1"
    copyfile(filename1, filename2)

    # read contents of original file
    dfs = Dfs1()
    res1 = dfs.read(filename1, [0])

    # overwrite
    res1.data[0] = -2 * res1.data[0]
    dfs.write(filename2, res1.data)

    # read contents of manipulated file
    res1 = dfs.read(filename1, [0])
    res2 = dfs.read(filename2, [0])

    data1 = res1.data[0]
    data2 = res2.data[0]
    assert data2[2, 1] == -2 * data1[2, 1]

    # clean
    os.remove(filename2)
