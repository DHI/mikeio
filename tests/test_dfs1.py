import os
import numpy as np
import datetime

from mikeio import dfs1 as dfs1


def test_simple_create():

    filename = r"simple.dfs1"

    data = []

    nt = 100
    nx = 20
    d = np.random.random([nt, nx])

    data.append(d)

    dfs = dfs1.dfs1()

    dfs.create(filename=filename, data=data)

    assert True
    os.remove(filename)


def test_create_single_item():

    start_time = datetime.datetime(2012, 1, 1)

    # timeseries_unit = second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
    timeseries_unit = 1402
    dt = 12

    variable_type = [100000]
    unit = [1000]

    filename = r"random.dfs1"

    data = []
    d = np.random.random([100, 3])

    data.append(d)
    length_x = 100

    names = ['testing water level']
    title = 'test dfs1'

    dfs = dfs1.dfs1()

    dfs.create(filename=filename, data=data,
               start_time=start_time,
               dt=dt, variable_type=variable_type,
               unit=unit,
               length_x=length_x,
               names=names, title=title)

    assert True
    os.remove(filename)


def test_read():

    filename = r"tests/testdata/random.dfs1"
    dfs = dfs1.dfs1()

    data = dfs.read(filename, [0])[0]
    data = data[0]
    assert data.shape == (100, 3) # time, x

def test_read_names_access():

    filename = r"tests/testdata/random.dfs1"
    dfs = dfs1.dfs1()

    res = dfs.read(filename, [0])
    data = res.data
    item = data[0]
    time = res.time
    assert item.shape == (100, 3) # time, x
    assert len(time) == 100


def test_write():

    pass



