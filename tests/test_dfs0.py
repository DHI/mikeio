import os
import numpy as np
import datetime
import mikeio
from mikeio.dfs0 import Dfs0
from mikeio.eum import TimeStep, Item
from datetime import timedelta
from shutil import copyfile
import pytest


def test_simple_create():

    dfs0File = r"simple.dfs0"

    data = []

    nt = 100
    d = np.random.random([nt])
    data.append(d)

    dfs = Dfs0()

    dfs.create(filename=dfs0File, data=data)

    assert True
    os.remove(dfs0File)


def test_multiple_create():

    dfs0File = r"zeros_ones.dfs0"

    data = []

    nt = 10
    d1 = np.zeros(nt)
    data.append(d1)
    d2 = np.ones(nt)
    data.append(d2)

    names = ["Zeros", "Ones"]

    dfs = Dfs0()

    dfs.create(filename=dfs0File, data=data, names=names, title="Zeros and ones")

    assert True
    os.remove(dfs0File)


def test_create_timestep_7days():

    dfs0File = r"zeros_ones.dfs0"

    data = []

    nt = 10
    d1 = np.zeros(nt)
    data.append(d1)
    d2 = np.ones(nt)
    data.append(d2)

    names = ["Zeros", "Ones"]

    dfs = Dfs0()

    dfs.create(
        filename=dfs0File,
        data=data,
        names=names,
        title="Zeros and ones",
        timeseries_unit=TimeStep.DAY,
        dt=7,
    )

    assert True

    res = dfs.read(dfs0File)

    dt = res.time[1] - res.time[0]

    assert dt == timedelta(days=7)

    os.remove(dfs0File)


def test_create_equidistant_calendar():

    dfs0file = r"random.dfs0"
    d1 = np.random.random([1000])
    d2 = np.random.random([1000])
    data = []
    data.append(d1)
    data.append(d2)
    start_time = datetime.datetime(2017, 1, 1)
    timeseries_unit = 1402
    title = "Hello Test"
    names = ["VarFun01", "NotFun"]
    variable_type = [100000, 100000]
    unit = [1000, 1000]
    data_value_type = [0, 1]
    dt = 5
    dfs = Dfs0()
    dfs.create(
        filename=dfs0file,
        data=data,
        start_time=start_time,
        timeseries_unit=timeseries_unit,
        dt=dt,
        names=names,
        title=title,
        variable_type=variable_type,
        unit=unit,
        data_value_type=data_value_type,
    )

    os.remove(dfs0file)
    assert True


def test_create_non_equidistant_calendar():
    dfs0file = r"neq.dfs0"
    d1 = np.random.random([1000])
    d2 = np.random.random([1000])
    data = []
    data.append(d1)
    data.append(d2)
    start_time = datetime.datetime(2017, 1, 1)
    time_vector = []
    for i in range(1000):
        time_vector.append(start_time + datetime.timedelta(hours=i * 0.1))
    title = "Hello Test"
    names = ["VarFun01", "NotFun"]
    variable_type = [100000, 100000]
    unit = [1000, 1000]
    data_value_type = [0, 1]

    dfs = Dfs0()
    dfs.create(
        filename=dfs0file,
        data=data,
        datetimes=time_vector,
        names=names,
        title=title,
        variable_type=variable_type,
        unit=unit,
        data_value_type=data_value_type,
    )

    assert True
    os.remove(dfs0file)


def test_read_dfs0_to_pandas_single_item():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    df = dfs.read_to_pandas(dfs0file, item_numbers=[1])

    assert df.shape[1] == 1


def test_read_dfs0_single_item():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    (data, t, items) = dfs.read(dfs0file, item_numbers=[1])

    assert len(data) == 1


def test_read_dfs0_single_item_named_access():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    res = dfs.read(dfs0file, item_numbers=[1])
    data = res.data

    assert len(data) == 1


def test_read_dfs0_single_item_read_by_name():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    res = dfs.read(
        dfs0file, item_names=["NotFun", "VarFun01"]
    )  # reversed order compare to original file
    data = res.data

    assert len(data) == 2
    assert res.items[0].name == "NotFun"
    assert res.items[0].item == Item.Water_Level
    assert (
        res.items[0].unit == Item.Water_Level.units["meter"]
    )  # Not sure this is the most readable way to specify unit


def test_read_dfs0_to_pandas():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    pd = dfs.read_to_pandas(dfs0file)

    assert np.isnan(pd[pd.columns[0]][2])


def test_read_dfs0_to_matrix():
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    (data, t, items) = dfs.read(filename=dfs0file)

    assert len(data) == 2


def test_write(tmpdir):
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    copyfile(dfs0file, tmpfile)
    dfs = Dfs0()
    res = dfs.read(tmpfile)
    data = res.data

    # Do something with the data
    data[0] = np.zeros_like(data[0])
    data[1] = np.ones_like(data[0])

    # Overwrite the file
    dfs.write(tmpfile, data)


def test_write_wrong_n_items(tmpdir):
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    copyfile(dfs0file, tmpfile)
    dfs = Dfs0()
    res = dfs.read(tmpfile)
    data = res.data

    # One item too many...
    data[0] = np.zeros_like(data[0])
    data[1] = np.ones_like(data[0])
    data.append(np.ones_like(data[0]))

    # Overwrite the file
    with pytest.raises(Exception):
        dfs.write(tmpfile, data)


def test_write_no_existing_file():
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    res = dfs.read(dfs0file)
    data = res.data

    # Overwrite the file
    with pytest.raises(Exception):
        dfs.write("not_a_file", data)


def test_read_dfs0_main_module():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = mikeio.Dfs0()
    (data, t, items) = dfs.read(dfs0file, item_numbers=[1])

    assert len(data) == 1
