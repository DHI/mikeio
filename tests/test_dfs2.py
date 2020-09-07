import os
import datetime
import numpy as np
from mikeio.dfs2 import Dfs2
from mikeio.eum import EUMType, ItemInfo, EUMUnit
import pytest


def test_simple_write(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

    data = []

    nt = 100
    nx = 20
    ny = 5
    d = np.random.random([nt, ny, nx])

    data.append(d)

    dfs = Dfs2()

    dfs.write(filename=filename, data=data)

    newdfs = Dfs2(filename)

    ds = newdfs.read()

    assert len(ds) == 1
    assert ds.items[0].type == EUMType.Undefined


def test_write_single_item(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

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

    dfs = Dfs2()

    dfs.write(
        filename=filename,
        data=data,
        start_time=datetime.datetime(2012, 1, 1),
        dt=12,
        items=[ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)],
        coordinate=["UTM-33", east, north, orientation],
        dx=100,
        dy=200,
        title="test dfs2",
    )

    newdfs = Dfs2(filename)
    assert newdfs.projection_string == "UTM-33"
    assert pytest.approx(newdfs.longitude) == 12.0
    assert pytest.approx(newdfs.latitude) == 55.0
    assert newdfs.dx == 100.0
    assert newdfs.dy == 200.0


def test_non_equidistant_calendar(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

    data = []

    datetimes = [
        datetime.datetime(2012, 1, 1),
        datetime.datetime(2012, 2, 1),
        datetime.datetime(2012, 2, 10),
    ]

    nt = len(datetimes)
    nx = 20
    ny = 5
    d = np.random.random([nt, ny, nx])

    data.append(d)

    dfs = Dfs2()

    dfs.write(filename=filename, data=data, datetimes=datetimes)

    newdfs = Dfs2(filename)
    ds = newdfs.read()

    assert not ds.is_equidistant


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


def test_write_selected_item_to_new_file(tmpdir):

    filename = r"tests/testdata/random_two_item.dfs2"
    dfs = Dfs2(filename)

    outfilename = os.path.join(tmpdir.dirname, "simple.dfs2")

    ds = dfs.read(["Untitled"])

    dfs.write(outfilename, ds)

    dfs2 = Dfs2(outfilename)

    ds2 = dfs2.read()

    assert len(ds2) == 1
    assert ds.items[0].name == "Untitled"
    assert dfs.start_time == dfs2.start_time
    assert dfs.projection_string == dfs2.projection_string
    assert dfs.longitude == dfs2.longitude
    assert dfs.latitude == dfs2.latitude
    assert dfs.orientation == dfs2.orientation


def test_write_modified_data_to_new_file(tmpdir):

    filename = r"tests/testdata/gebco_sound.dfs2"
    dfs = Dfs2(filename)

    outfilename = os.path.join(tmpdir.dirname, "mod.dfs2")

    ds = dfs.read()

    ds.data[0] = ds.data[0] + 10.0

    dfs.write(outfilename, ds)

    dfsmod = Dfs2(outfilename)

    assert dfs._longitude == dfsmod._longitude


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

    i, j = dfs.find_nearest_element(lon=12.74792, lat=55.865)

    assert i == 104
    assert j == 131

    assert ds.data[0][0, i, j] == -43.0

    # try some outside the domain
    i, j = dfs.find_nearest_element(lon=11.0, lat=57.0)

    assert i == 0
    assert j == 0

    i, j = dfs.find_nearest_element(lon=15.0, lat=50.0)

    assert i == 263
    assert j == 215
