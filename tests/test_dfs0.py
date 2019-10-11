import os
import numpy as np
import datetime
from pydhi import dfs0 as dfs0


def test_simple_create():

    dfs0File = r"simple.dfs0"

    data = []

    nt = 100
    d = np.random.random([nt])
    data.append(d)

    dfs = dfs0.dfs0()

    dfs.create(dfs0file=dfs0File, data=data)

    assert True
    os.remove(dfs0File)


def test_create_equidistant_calendar():

    dfs0file = r'random.dfs0'
    d1 = np.random.random([1000])
    d2 = np.random.random([1000])
    data = []
    data.append(d1)
    data.append(d2)
    start_time = datetime.datetime(2017, 1, 1)
    timeseries_unit = 1402
    title = 'Hello Test'
    names = ['VarFun01', 'NotFun']
    variable_type = [100000, 100000]
    unit = [1000, 1000]
    data_value_type = [0, 1]
    dt = 5
    dfs = dfs0.dfs0()
    dfs.create(dfs0file=dfs0file, data=data,
               start_time=start_time,
               timeseries_unit=timeseries_unit, dt=dt,
               names=names, title=title,
               variable_type=variable_type,
               unit=unit, data_value_type=data_value_type)

    os.remove(dfs0file)
    assert True


def test_create_non_equidistant_calendar():
    dfs0file = r'neq.dfs0'
    d1 = np.random.random([1000])
    d2 = np.random.random([1000])
    data = []
    data.append(d1)
    data.append(d2)
    start_time = datetime.datetime(2017, 1, 1)
    time_vector = []
    for i in range(1000):
        time_vector.append(start_time + datetime.timedelta(hours=i*0.1))
    title = 'Hello Test'
    names = ['VarFun01', 'NotFun']
    variable_type = [100000, 100000]
    unit = [1000, 1000]
    data_value_type = [0, 1]

    dfs = dfs0.dfs0()
    dfs.create(dfs0file=dfs0file, data=data,
               datetimes=time_vector,
               names=names, title=title,
               variable_type=variable_type, unit=unit,
               data_value_type=data_value_type)

    assert True
    os.remove(dfs0file)

def test_read_dfs0_to_pandas():

    dfs0file = r'tests/testdata/random.dfs0'

    dfs = dfs0.dfs0()
    pd = dfs.read_to_pandas(dfs0file)

    assert np.isnan(pd[pd.columns[0]][2])

def test_read_dfs0_to_matrix():
    dfs0file = r'tests/testdata/random.dfs0'

    dfs = dfs0.dfs0()
    mat = dfs.read(dfs0file, indices=[0])[0]

    assert np.isnan(mat[2, 0])
