import os
import numpy as np
import datetime
from mikeio import dfs0 as dfs0
from mikeio.eum import TimeStep
from datetime import timedelta


def test_simple_create():

    dfs0File = r"simple.dfs0"

    data = []

    nt = 100
    d = np.random.random([nt])
    data.append(d)

    dfs = dfs0.dfs0()

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

    dfs = dfs0.dfs0()

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

    dfs = dfs0.dfs0()

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
    dfs = dfs0.dfs0()
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

    dfs = dfs0.dfs0()
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

    dfs = dfs0.dfs0()
    df = dfs.read_to_pandas(dfs0file, item_numbers=[1])

    assert df.shape[1] == 1


def test_read_dfs0_single_item():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = dfs0.dfs0()
    (data, t, names) = dfs.read(dfs0file, item_numbers=[1])

    assert len(data) == 1


def test_read_dfs0_single_item_named_access():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = dfs0.dfs0()
    res = dfs.read(dfs0file, item_numbers=[1])
    data = res.data

    assert len(data) == 1


def test_read_dfs0_single_item_read_by_name():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = dfs0.dfs0()
    res = dfs.read(
        dfs0file, item_names=["NotFun", "VarFun01"]
    )  # reversed order compare to original file
    data = res.data

    assert len(data) == 2
    assert res.names[0] == "NotFun"


def test_read_dfs0_to_pandas():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = dfs0.dfs0()
    pd = dfs.read_to_pandas(dfs0file)

    assert np.isnan(pd[pd.columns[0]][2])


def test_read_dfs0_to_matrix():
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = dfs0.dfs0()
    (data, t, names) = dfs.read(filename=dfs0file)

    assert len(data) == 2
