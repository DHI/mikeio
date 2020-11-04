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


def test_read():

    filename = r"tests/testdata/random.dfs2"
    dfs = Dfs2(filename)
    ds = dfs.read(["testing water level"])
    data = ds.data[0]
    assert data[0, 11, 0] == 0
    assert np.isnan(data[0, 10, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_temporal_subset_slice():

    filename = r"tests/testdata/eq.dfs2"
    dfs = Dfs2(filename)
    ds = dfs.read(time_steps=slice("2000-01-01 00:00", "2000-01-01 12:00"))

    assert len(ds.time) == 13


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


def test_repr():

    filename = r"tests/testdata/gebco_sound.dfs2"
    dfs = Dfs2(filename)

    text = repr(dfs)

    assert "Dfs2" in text
    assert "Items" in text
    assert "dx" in text


def test_repr_empty():

    dfs = Dfs2()

    text = repr(dfs)

    assert "Dfs2" in text


def test_repr_time():

    filename = r"tests/testdata/random.dfs2"
    dfs = Dfs2(filename)

    text = repr(dfs)

    assert "Dfs2" in text
    assert "Items" in text
    assert "dx" in text
    assert "steps" in text


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


def test_interpolate_non_equidistant_data(tmpdir):

    filename = r"tests/testdata/eq.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read(time_steps=[0, 2, 3, 6])  # non-equidistant dataset

    assert not ds.is_equidistant

    ds2 = ds.interp_time(dt=3600)

    assert ds2.is_equidistant

    outfilename = os.path.join(tmpdir.dirname, "interpolated_time.dfs2")

    dfs.write(outfilename, ds2)

    dfs2 = Dfs2(outfilename)
    assert dfs2.timestep == 3600.0

    ds3 = dfs2.read()

    assert ds3.is_equidistant


def test_write_some_time_step(tmpdir):

    filename = r"tests/testdata/waves.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read(time_steps=[1, 2])

    assert ds.data[0].shape[0] == 2
    assert len(ds.time) == 2

    assert dfs.timestep == 86400.0
    assert dfs.start_time.day == 1

    outfilename = os.path.join(tmpdir.dirname, "waves_subset.dfs2")

    dfs.write(outfilename, ds)

    dfs2 = Dfs2(outfilename)
    assert dfs2.timestep == 86400.0
    assert dfs2.start_time.day == 2


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
