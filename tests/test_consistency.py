import numpy as np
import mikeio
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
    assert ds1[0][0, 8].values[0] == 0.203246


def test_dfs1_isel_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.isel(8, axis="x")
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(time=0).values[0] == 0.203246


def test_dfs1_sel_t():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.sel(time=slice("2018", "2018-03-10"))
    assert ds1.dims == ("time", "x")
    assert isinstance(ds1.geometry, type(ds.geometry))
    assert ds1[0][0, 8].values[0] == 0.203246


def test_dfs1_sel_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.sel(x=7.8)
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(time=0).values[0] == 0.203246


def test_dfs1_interp_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.interp(x=7.75)
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(time=0).values[0] == 0.202022475


def test_dfs1_interp_like():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.interp_like(ds.geometry)
    assert ds1.dims == ("time", "x")
    assert np.all(ds1[0].values == ds[0].values)


def test_read_dfs2():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfs2_single_time():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        time=-1,
    )

    assert ds.n_timesteps == 1
    assert "time" not in ds.dims


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

    assert dss.geometry.x == 349034.439173
    assert dss.geometry.y == 6144868.611412
    assert dss[0].values[0] == 0.179145


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
