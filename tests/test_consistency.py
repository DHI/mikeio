from distutils import extension
import pytest

import numpy as np
import mikeio
from mikeio.dataarray import DataArray
from mikeio.spatial.geometry import GeometryUndefined


def test_read_dfs0():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs0",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfs1():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs1",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_dfs1_isel_t():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.isel([0, 1], axis="t")
    assert ds1.dims == ("time", "x")
    assert isinstance(ds1.geometry, type(ds.geometry))
    assert ds1[0].values[0, 8] == pytest.approx(0.203246)


def test_dfs1_isel_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.isel(8, axis="x")
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(0, axis="time").values == pytest.approx(0.203246)


def test_dfs1_sel_t():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.sel(time=slice("2018", "2018-03-10"))
    assert ds1.dims == ("time", "x")
    assert isinstance(ds1.geometry, type(ds.geometry))
    assert ds1[0].values[0, 8] == pytest.approx(0.203246)


def test_dfs1_sel_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.sel(x=7.8)
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(0, axis="time").values == pytest.approx(0.203246)

    da: DataArray = ds[0]
    da1 = da.sel(x=7.8)
    assert da1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert da1.isel(0, axis="time").values == pytest.approx(0.203246)


def test_dfs1_interp_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.interp(x=7.75)
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(0, axis="time").values == pytest.approx(0.20202248)


# Nice to have...
# def test_dfs1_interp_like():
#    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

# ds1 = ds.interp_like(ds.geometry)
#    assert ds1.dims == ("time", "x")
#    assert np.all(ds1[0].values == ds[0].values)


# def test_create_dfs2()

#    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")
#    g = ds.geometry.get_overset_grid(dx=5000)
#    dsi = ds.interp_like(g)
#    dsi.to_dfs("tests/testdata/consistency/oresundHD.dfs2")


def test_read_dfs2():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_sel_line_dfs2():
    x = 350000
    y = 6145000
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs2")
    dsselx = ds.sel(x=x)
    assert isinstance(dsselx.geometry, mikeio.Grid1D)
    dssely = ds.sel(y=y)
    assert isinstance(dssely.geometry, mikeio.Grid1D)
    assert dsselx.geometry != dssely.geometry


def test_sel_mult_line_not_possible():
    xs = [350000, 360000]
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs2")
    with pytest.raises(
        Exception, match="scalar"
    ):  # NotImplemented or ValueError not sure yet
        ds.sel(x=xs)


def test_read_dfs2_single_time():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        time=-1,
    )

    assert ds.n_timesteps == 1
    assert "time" not in ds.dims

    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        time=[-1],  # time as array, forces time dimension to be kept
    )

    assert ds.n_timesteps == 1
    assert "time" in ds.dims


def test_interp_x_y_dfs2():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs2")

    x = 350000
    y = 6145000
    das = ds[0].interp(x=x, y=y)

    assert das.geometry.x == x
    assert das.geometry.y == y

    dss = ds.interp(x=x, y=y)

    assert dss.geometry.x == x
    assert dss.geometry.y == y


def test_sel_x_y_dfsu2d():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")

    x = 350000
    y = 6145000
    dss = ds.sel(x=x, y=y)

    assert dss.geometry.x == pytest.approx(349034.439173)
    assert dss.geometry.y == pytest.approx(6144868.611412)
    assert dss[0].values[0] == pytest.approx(0.179145)


def test_interp_x_y_dfsu2d():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")

    x = 350000
    y = 6145000
    dss = ds.interp(x=x, y=y)

    assert dss.geometry.x == x
    assert dss.geometry.y == y


def test_read_dfsu2d():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfsu2d_single_time():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        time=-1,
    )

    assert ds.n_timesteps == 1
    assert "time" not in ds.dims

    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        time=[-1],
    )

    assert ds.n_timesteps == 1
    assert "time" in ds.dims


def test_read_dfs_time_selection_str():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = "2018-03"
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)

        dsr = mikeio.read(filename=filename, time=time)

        assert all(dsr.time == dssel.time)


def test_read_dfs_time_selection_str_specific():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = "2018-03-08 00:00:00"
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)

        dsr = mikeio.read(filename=filename, time=time)

        assert all(dsr.time == dssel.time)


def test_read_dfs_time_slice_str():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = slice("2018-03-08", "2018-03-09")
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)

        dsr = mikeio.read(filename=filename, time=time)

        assert all(dsr.time == dssel.time)


def test_read_dfs_time_int():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = -1
        ds = mikeio.read(filename=filename)
        dssel = ds.isel(time=time)

        dsr = mikeio.read(filename=filename, time=time)

        assert all(dsr.time == dssel.time)


def test_read_dfs_time_list_int():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = [0, 1]
        ds = mikeio.read(filename=filename)
        dssel = ds.isel(time=time)

        dsr = mikeio.read(filename=filename, time=time)

        assert all(dsr.time == dssel.time)
