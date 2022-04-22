import os
from shutil import copyfile
from datetime import datetime
import numpy as np
import pytest

import mikeio
from mikeio.eum import EUMType, ItemInfo
from mikeio.spatial.geometry import GeometryUndefined
from mikeio.spatial.grid_geometry import Grid2D, Grid3D


def test_dfs3_repr():
    dfs = mikeio.open("tests/testdata/test_dfs3.dfs3")
    assert "<mikeio.Dfs3>" in repr(dfs)
    assert "geometry: Grid3D" in repr(dfs)


def test_dfs3_projection():
    dfs = mikeio.open("tests/testdata/test_dfs3.dfs3")
    assert dfs.projection_string == "LONG/LAT"
    assert dfs.dx == 0.25
    assert dfs.dy == 0.25
    assert dfs.dz == 1.0


def test_dfs3_geometry():
    dfs = mikeio.open("tests/testdata/test_dfs3.dfs3")
    assert isinstance(dfs.geometry, Grid3D)
    assert dfs.geometry.nx == 21
    assert dfs.geometry.ny == 17
    assert dfs.geometry.nz == 34


def test_dfs3_read():
    ds = mikeio.read("tests/testdata/Grid1.dfs3")
    assert ds.n_items == 2
    assert ds.n_timesteps == 30
    da = ds[0]
    assert da.shape == (30, 10, 10, 10)  # t  # z  # y  # x
    assert da.dims == ("time", "z", "y", "x")
    assert da.name == "Item 1"


def test_dfs3_read_time():
    fn = "tests/testdata/test_dfs3.dfs3"
    ds = mikeio.read(fn, time="2020-12-30 00:00")
    assert ds.n_timesteps == 1
    assert isinstance(ds.geometry, Grid3D)

    ds = mikeio.read(fn, time=-1)
    assert ds.n_timesteps == 1
    assert isinstance(ds.geometry, Grid3D)


def test_dfs3_read_1_layer():
    fn = "tests/testdata/test_dfs3.dfs3"
    ds = mikeio.read(fn, layers=-1)
    assert ds.shape == (2, 17, 21)
    assert isinstance(ds.geometry, Grid2D)

    ds = mikeio.read(fn, layers="top")
    assert ds.shape == (2, 17, 21)
    assert isinstance(ds.geometry, Grid2D)

    ds = mikeio.read(fn, layers=[0])
    assert ds.shape == (2, 17, 21)
    assert isinstance(ds.geometry, Grid2D)


def test_dfs3_read_multiple_layers():
    fn = "tests/testdata/test_dfs3.dfs3"
    ds = mikeio.read(fn, layers=(0, 1, 2, 3))
    assert ds.geometry.nz == 4
    assert isinstance(ds.geometry, Grid3D)

    ds = mikeio.read(fn, layers=[1, 5, -3])
    assert isinstance(ds.geometry, GeometryUndefined)
    assert ds.shape == (2, 3, 17, 21)


def test_dfs3_write_single_item(tmpdir):
    outfilename = os.path.join(tmpdir.dirname, "simple.dfs3")
    start_time = datetime(2012, 1, 1)
    items = [ItemInfo(EUMType.Relative_moisture_content)]
    data = []
    #                     t  , z, y, x
    d = np.random.random([20, 2, 5, 10])
    d[:, 0, 0, 0] = 0.0
    data.append(d)
    title = "test dfs3"
    dfs = mikeio.Dfs3()
    dfs.write(
        filename=outfilename,
        data=data,
        start_time=start_time,
        dt=3600.0,
        items=items,
        coordinate=["UTM-33", 450000, 560000, 0],
        dx=0.1,
        dy=0.1,
        dz=10.0,
        title=title,
    )


def test_dfs3_read_write(tmpdir):
    ds = mikeio.read("tests/testdata/Grid1.dfs3")
    outfilename = os.path.join(tmpdir.dirname, "rw.dfs3")
    items = ds.items
    data = ds.to_numpy()
    title = "test dfs3"
    dfs = mikeio.Dfs3()
    dfs.write(
        filename=outfilename,
        data=data,
        start_time=ds.time[0],
        dt=(ds.time[1] - ds.time[0]).total_seconds(),
        items=items,
        coordinate=["LONG/LAT", 5, 10, 0],
        dx=0.1,
        dy=0.1,
        title=title,
    )


def test_read_rotated_grid():
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    assert (
        pytest.approx(ds.geometry._orientation) == 18.1246891021729
    )  # North to Y rotation != Grid rotation


def test_dfs3_to_dfs(tmpdir):
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    # ds = mikeio.read("tests/testdata/Grid1.dfs3")
    outfilename = os.path.join(tmpdir.dirname, "rw.dfs3")
    ds.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert ds.n_items == dsnew.n_items


def test_read_top_layer():
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3", layers="top")
    assert "z" not in ds.dims
    assert isinstance(ds.geometry, Grid2D)


def test_read_bottom_layer():
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3", layers="bottom")
    assert "z" not in ds.dims
    assert isinstance(ds.geometry, Grid2D)
    assert pytest.approx(ds[0].to_numpy()[0, 58, 52]) == 0.05738005042076111


def test_sel_bottom_layer():
    dsall = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    with pytest.raises(NotImplementedError) as excinfo:
        dsall.sel(layer="bottom")  # TODO layers vs layer
    assert "mikeio.read" in str(excinfo.value)
    # assert "z" not in ds.dims
    # assert isinstance(ds.geometry, Grid2D)
    # assert pytest.approx(ds[0].to_numpy()[0, 58, 52]) == 0.05738005042076111
