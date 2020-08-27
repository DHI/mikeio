import os
import numpy as np
import datetime
from shutil import copyfile

from mikeio.dfs2 import Dfs2
from mikeio.eum import EUMType, ItemInfo, EUMUnit


def test_simple_write():

    filename = r"simple.dfs2"

    data = []

    nt = 100
    nx = 20
    ny = 5
    d = np.random.random([nt, ny, nx])

    data.append(d)

    dfs = Dfs2()

    dfs.write(filename=filename, data=data)

    assert True
    os.remove(filename)


def test_write_single_item():

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

    dfs.write(
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


def test_write_multiple_item():

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

    dfs.write(
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

    dfs.write(filename=filename, data=data, datetimes=datetimes)

    newdfs = Dfs2(filename)
    ds = newdfs.read()

    assert ds.time[1] == datetimes[1]

    assert True
    os.remove(filename)


def test_read():

    filename = r"tests/testdata/random.dfs2"
    dfs = Dfs2(filename)
    ds = dfs.read(["testing water level"])
    data = ds.data[0]
    assert data[0, 11, 0] == 0
    assert np.isnan(data[0, 10, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_item_names():

    filename = r"tests/testdata/random.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read(["testing water level"])
    data = ds.data[0]
    assert data[0, 11, 0] == 0
    assert np.isnan(data[0, 10, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_numbered_access():

    filename = r"tests/testdata/random_two_item.dfs2"
    dfs = Dfs2(filename)

    res = dfs.read([1])

    assert np.isnan(res.data[0][0, 0, 0])
    assert res.time is not None
    assert res.items[0].name == "Untitled"


def test_read_some_time_step():

    filename = r"tests/testdata/random_two_item.dfs2"
    dfs = Dfs2(filename)

    res = dfs.read(time_steps=[1, 2])

    assert res.data[0].shape[0] == 2
    assert len(res.time) == 2


def test_find_index_from_coordinate():

    filename = "tests/testdata/gebco_sound.dfs2"

    dfs = Dfs2(filename)

    # TODO it should not be necessary to read the data to get coordinates
    ds = dfs.read()

    i, j = dfs.find_closest_element_index(lon=12.74792, lat=55.865)

    assert i == 104
    assert j == 131

    assert ds.data[0][0, i, j] == -43.0

    # try some outside the domain
    i, j = dfs.find_closest_element_index(lon=11.0, lat=57.0)

    assert i == 0
    assert j == 0

    i, j = dfs.find_closest_element_index(lon=15.0, lat=50.0)

    assert i == 263
    assert j == 215
