import os
import numpy as np
import datetime
from shutil import copyfile

from mikeio.dfs2 import Dfs2
from mikeio.eum import EUMType, ItemInfo, EUMUnit


def test_simple_create():

    filename = r"simple.dfs2"

    data = []

    nt = 100
    nx = 20
    ny = 5
    d = np.random.random([nt, ny, nx])

    data.append(d)

    dfs = Dfs2()

    dfs.create(filename=filename, data=data)

    assert True
    os.remove(filename)


def test_create_single_item():

    start_time = datetime.datetime(2012, 1, 1)

    # timeseries_unit = second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
    timeseries_unit = 1402
    dt = 12

    items = [ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)]

    filename = r"random.dfs2"

    data = []
    d = np.random.random([100, 2, 3])
    d[10, :, :] = np.nan
    d[11, :, :] = 0
    d[12, :, :] = 1e-10
    d[13, :, :] = 1e10

    data.append(d)
    # >>> from pyproj import Proj
    # >>> utm = Proj(32633)
    # >>> utm(12.0, 55.0)
    east = 308124
    north = 6098907
    orientation = 0

    coordinate = ["UTM-33", east, north, orientation]
    length_x = 100
    length_y = 100

    title = "test dfs2"

    dfs = Dfs2()

    dfs.create(
        filename=filename,
        data=data,
        start_time=start_time,
        timeseries_unit=timeseries_unit,
        dt=dt,
        items=items,
        coordinate=coordinate,
        length_x=length_x,
        length_y=length_y,
        title=title,
    )

    assert True
    os.remove(filename)


def test_create_multiple_item():

    start_time = datetime.datetime(2012, 1, 1)

    # timeseries_unit = second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
    timeseries_unit = 1402
    dt = 12

    # TODO change int to enum
    items = [
        ItemInfo("testing water level", 100000, 1000),
        ItemInfo("testing rainfall", 100004, 1002),
        ItemInfo("testing drain time constant", 100362, 2605),
    ]

    filename = r"multiple.dfs2"

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    d = np.zeros([100, 100, 30]) + 2.0
    data.append(d)
    d = np.zeros([100, 100, 30]) + 3.0
    data.append(d)

    coordinate = ["LONG/LAT", 12.4387, 55.2257, 0]
    length_x = 0.1
    length_y = 0.1

    title = "test dfs2"

    dfs = Dfs2()

    dfs.create(
        filename=filename,
        data=data,
        start_time=start_time,
        timeseries_unit=timeseries_unit,
        dt=dt,
        items=items,
        coordinate=coordinate,
        length_x=length_x,
        length_y=length_y,
        title=title,
    )

    assert True
    os.remove(filename)


def test_non_equidistant_calendar():

    filename = r"neq.dfs2"

    data = []

    datetimes = [datetime.datetime(2012, 1, 1), datetime.datetime(2012, 2, 1)]

    nt = len(datetimes)
    nx = 20
    ny = 5
    d = np.random.random([nt, ny, nx])

    data.append(d)

    dfs = Dfs2()

    dfs.create(filename=filename, data=data, datetimes=datetimes)

    assert True
    os.remove(filename)


def test_read():

    filename = r"tests/testdata/random.dfs2"
    dfs = Dfs2()
    data = dfs.read(filename, [0])[0]
    data = dfs.read(filename, item_names=["testing water level"])[0]
    data = data[0]
    assert data[0, 11, 0] == 0
    assert np.isnan(data[0, 10, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_item_names():

    filename = r"tests/testdata/random.dfs2"
    dfs = Dfs2()

    data = dfs.read(filename, item_names=["testing water level"])[0]
    data = data[0]
    assert data[0, 11, 0] == 0
    assert np.isnan(data[0, 10, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_numbered_access():

    filename = r"tests/testdata/random_two_item.dfs2"
    dfs = Dfs2()

    res = dfs.read(filename, item_numbers=[1])

    assert np.isnan(res.data[0][0, 0, 0])
    assert res.time is not None
    assert res.items[0].name == "Untitled"


def test_read_some_time_step():

    filename = r"tests/testdata/random_two_item.dfs2"
    dfs = Dfs2()

    res = dfs.read(filename, time_steps=[1, 2])

    assert res.data[0].shape[0] == 2
    assert len(res.time) == 2


def test_write():

    filename1 = r"tests/testdata/random.dfs2"
    filename2 = r"tests/testdata/random_for_write.dfs2"
    copyfile(filename1, filename2)

    # read contents of original file
    dfs = Dfs2()
    res1 = dfs.read(filename1, [0])

    # overwrite
    res1.data[0] = -2 * res1.data[0]
    dfs.write(filename2, res1.data)

    # read contents of manipulated file
    res1 = dfs.read(filename1, [0])
    res2 = dfs.read(filename2, [0])

    data1 = res1.data[0]
    data2 = res2.data[0]
    assert data2[0, 11, 0] == -2 * data1[0, 11, 0]

    # clean
    os.remove(filename2)
