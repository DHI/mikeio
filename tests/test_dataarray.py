import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import mikeio
from mikeio.eum import EUMType, ItemInfo
from mikeio.spatial.geometry import GeometryPoint3D, GeometryUndefined


@pytest.fixture
def da0():
    time = "2000-01-01 00:00:00"
    da = mikeio.DataArray(
        data=np.array([7.0]),
        time=time,
        item=ItemInfo(name="Foo"),
    )
    return da


@pytest.fixture
def da1():
    nt = 10
    start = 10.0
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    da = mikeio.DataArray(
        data=np.arange(start, start + nt, dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    return da


@pytest.fixture
def da2():
    nt = 10
    nx = 7

    da = mikeio.DataArray(
        data=np.zeros([nt, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, n=nx),
    )

    return da


@pytest.fixture
def da_grid2d():
    nt = 10
    nx = 7
    ny = 14

    da = mikeio.DataArray(
        data=np.zeros([nt, ny, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid2D(x0=10.0, dx=0.1, nx=nx, ny=ny, dy=1.0, y0=-10.0),
    )

    return da


@pytest.fixture
def da_grid2d_proj():
    nt = 10
    nx = 7
    ny = 14

    da = mikeio.DataArray(
        data=np.zeros([nt, ny, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid2D(
            x0=1000, dx=100, nx=nx, ny=ny, dy=10, y0=2000, projection="UTM-32"
        ),
    )

    return da


@pytest.fixture
def da_time_space():
    nt = 10
    start = 10.0
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    da = mikeio.DataArray(
        data=np.zeros(shape=(nt, 2), dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
        geometry=mikeio.Grid1D(n=2, dx=1.0),
    )

    return da


def test_verify_custom_dims():
    nt = 10
    nx = 7

    with pytest.raises(ValueError) as excinfo:
        da = mikeio.DataArray(
            data=np.zeros([nt, nx]) + 0.1,
            time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
            item=ItemInfo("Foo"),
            dims=("space", "ensemble"),  # no time!
            geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, n=nx),
        )
    assert "time" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        da = mikeio.DataArray(
            data=np.zeros([nt, nx]) + 0.1,
            time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
            item=ItemInfo("Foo"),
            dims=("time", "x", "ensemble"),  # inconsistent with data
            geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, n=nx),
        )
    assert "number" in str(excinfo.value).lower()


def test_write_1d(da2, tmp_path):

    outfilename = tmp_path / "grid1d.dfs1"

    da2.to_dfs(outfilename)

    ds = mikeio.read(outfilename)
    assert ds.n_items == 1
    assert isinstance(ds.geometry, mikeio.Grid1D)


def test_data_0d(da0):
    assert da0.ndim == 1
    assert da0.dims == ("time",)
    assert "values" in repr(da0)
    assert "geometry" not in repr(da0)
    assert "values" in repr(da0[:4])

    da0 = da0.squeeze()
    assert da0.ndim == 0
    assert "values" in repr(da0)
    assert "geometry" not in repr(da0)


def test_data_2d_no_geometry_not_allowed():

    nt = 10
    nx = 7
    ny = 14

    with pytest.warns(Warning) as w:
        mikeio.DataArray(
            data=np.zeros([nt, ny, nx]) + 0.1,
            time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
            item=ItemInfo("Foo"),
        )

    assert "geometry" in str(w[0].message).lower()


def test_dataarray_init():
    nt = 10
    start = 10.0
    data = np.arange(start, start + nt, dtype=float)
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    item = ItemInfo(name="Foo")

    da = mikeio.DataArray(data=data, time=time)
    assert isinstance(da, mikeio.DataArray)
    assert da.name == "NoName"  # default name
    assert da.item.type == EUMType.Undefined

    da = mikeio.DataArray(data=data, time=time, item=item)
    assert isinstance(da, mikeio.DataArray)
    assert da.name == "Foo"
    assert da.ndim == 1
    assert da.dims == ("time",)

    da = mikeio.DataArray(data=data, time="2018")
    assert isinstance(da, mikeio.DataArray)
    assert da.n_timesteps == 1
    assert da.ndim == 1
    assert da.dims == ("x",)

    da = mikeio.DataArray(data=data)
    assert da.n_timesteps == 1
    assert da.ndim == 1
    assert da.dims == ("x",)
    assert da.time[0] == pd.Timestamp(2018, 1, 1)


def test_dataarray_init_item_none():

    nt = 10
    data = data = np.zeros([nt, 4]) + 0.1
    time = time = pd.date_range(start="2000-01-01", freq="S", periods=nt)

    da = mikeio.DataArray(data=data, time=time)
    assert da.type == EUMType.Undefined

    with pytest.raises(ValueError, match="Item must be"):
        mikeio.DataArray(data=data, time=time, item=3)


def test_dataarray_init_2d():
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)

    # 2d with time
    ny, nx = 5, 6
    data2d = np.zeros([nt, ny, nx]) + 0.1
    da = mikeio.DataArray(data=data2d, time=time)
    assert da.ndim == 3
    assert da.dims == ("time", "y", "x")

    # singleton time, requires spec of dims
    dims = ("time", "y", "x")
    data2d = np.zeros([1, ny, nx]) + 0.1
    da = mikeio.DataArray(data=data2d, time="2018", dims=dims)
    assert isinstance(da, mikeio.DataArray)
    assert da.n_timesteps == 1
    assert da.ndim == 3
    assert da.dims == dims

    # no time
    data2d = np.zeros([ny, nx]) + 0.1
    da = mikeio.DataArray(data=data2d, time="2018")
    assert isinstance(da, mikeio.DataArray)
    assert da.n_timesteps == 1
    assert da.ndim == 2
    assert da.dims == ("y", "x")

    # x, y swapped
    dims = ("x", "y")
    data2d = np.zeros([nx, ny]) + 0.1
    da = mikeio.DataArray(data=data2d, time="2018", dims=dims)
    assert da.n_timesteps == 1
    assert da.ndim == 2
    assert da.dims == dims


def test_dataarray_init_5d():
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)

    # 5d with named dimensions
    dims = ("x", "y", "layer", "member", "season")
    data5d = np.zeros([2, 4, 5, 3, 3]) + 0.1
    da = mikeio.DataArray(data=data5d, time="2018", dims=dims)
    assert da.n_timesteps == 1
    assert da.ndim == 5
    assert da.dims == dims

    # 5d with named dimensions and time
    dims = ("time", "dummy", "layer", "member", "season")
    data5d = np.zeros([nt, 4, 5, 3, 3]) + 0.1
    da = mikeio.DataArray(data=data5d, time=time, dims=dims)
    assert da.n_timesteps == nt
    assert da.ndim == 5
    assert da.dims == dims


def test_dataarray_init_wrong_dim():
    nt = 10
    start = 10.0
    data = np.arange(start, start + nt, dtype=float)
    time_long = pd.date_range(start="2000-01-01", freq="S", periods=(nt + 1))
    item = ItemInfo(name="Foo")

    with pytest.raises(ValueError):
        mikeio.DataArray(data=data, time=time_long, item=item)

    nt, ny, nx = 10, 5, 6
    data2d = np.zeros([nt, ny, nx]) + 0.1
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time_long)

    # time must be first dim
    dims = ("x", "y", "time")
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time, dims=dims)

    # time must be first dim
    data2d = np.zeros([ny, nt, nx]) + 0.1
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time)


def test_dataarray_init_grid1d():
    nt = 10
    nx = 5
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    data = np.zeros([nt, nx]) + 0.1
    g = mikeio.Grid1D(n=nx, dx=1.0)
    da = mikeio.DataArray(data=data, time=time, geometry=g)
    assert da.ndim == 2
    assert da.dims == ("time", "x")

    # singleton time
    data = np.zeros([1, nx]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 2
    assert da.dims == ("time", "x")

    # no time
    data = np.zeros([nx]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 1
    assert da.dims == ("x",)


def test_dataarray_init_grid2d():
    nt = 10
    ny, nx = 7, 5
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    data = np.zeros([nt, ny, nx]) + 0.1
    g = mikeio.Grid2D(dx=0.5, nx=nx, ny=ny)
    da = mikeio.DataArray(data=data, time=time, geometry=g)
    assert da.ndim == 3
    assert da.dims == ("time", "y", "x")

    # singleton time
    data = np.zeros([1, ny, nx]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 3
    assert da.dims == ("time", "y", "x")  # TODO: fails

    # no time
    data = np.zeros([ny, nx]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 2
    assert da.dims == ("y", "x")


def test_dataarray_init_dfsu2d():
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    filename = "tests/testdata/north_sea_2.mesh"
    dfs = mikeio.open(filename)
    g = dfs.geometry
    ne = g.n_elements

    # time-varying
    data = np.zeros([nt, ne]) + 0.1
    da = mikeio.DataArray(data=data, time=time, geometry=g)
    assert da.ndim == 2
    assert da.dims == ("time", "element")
    assert da.geometry == g

    # singleton time
    data = np.zeros([1, ne]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 2
    assert da.dims == ("time", "element")  # TODO: fails
    assert da.n_timesteps == 1

    # no time
    data = np.zeros([ne]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 1
    assert da.dims == ("element",)


def test_dataarray_init_dfsu3d():
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="S", periods=nt)
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.open(filename)
    g = dfs.geometry
    ne = g.n_elements

    # time-varying
    data = np.zeros([nt, ne]) + 0.1
    da = mikeio.DataArray(data=data, time=time, geometry=g)
    assert da.ndim == 2
    assert da.dims == ("time", "element")
    assert da.geometry == g

    # singleton time
    data = np.zeros([1, ne]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 2
    assert da.dims == ("time", "element")  # TODO: fails

    # no time
    data = np.zeros([ne]) + 0.1
    da = mikeio.DataArray(data=data, time="2018", geometry=g)
    assert da.ndim == 1
    assert da.dims == ("element",)


def test_dataarray_indexing(da1: mikeio.DataArray):

    assert da1.shape == (10,)
    subset = da1[3]
    assert isinstance(subset, mikeio.DataArray)
    assert da1.shape == (10,)
    assert subset.to_numpy() == np.array([13.0])


def test_dataarray_dfsu3d_indexing():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    assert isinstance(
        ds.Salinity.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered
    )

    # indexing in time selecting a single record
    da = ds.Salinity[0, :]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered)

    # indexing in space selecting a single element
    da = ds.Salinity[:, 0]
    assert isinstance(da.geometry, GeometryPoint3D)

    # indexing in space selecting a multiple elements with slice
    da = ds.Salinity[:, 0:45]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered)

    # indexing in space selecting a multiple elements with tuple
    da = ds.Salinity[:, (3, 6, 12)]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered)

    # indexing in both time and space
    da = ds.Salinity[0, 0]
    assert isinstance(da.geometry, GeometryPoint3D)
    assert da.shape == ()


def test_dataarray_grid1d_repr(da2):
    assert "Grid1D" in repr(da2)
    assert "values" not in repr(da2)


def test_dataarray_grid1d_indexing(da2):
    da = da2
    nt, nx = da.shape
    assert da[0].shape == (nx,)
    assert da[0, :].shape == (nx,)
    assert da[:, -1].shape == (nt,)
    assert da[:, :].shape == (nt, nx)
    assert da[0, 0].shape == ()

    assert isinstance(da[:, :].geometry, mikeio.Grid1D)
    assert isinstance(da[:, -1].geometry, GeometryUndefined)


def test_dataarray_grid2d_repr(da_grid2d):
    assert "Grid2D" in repr(da_grid2d)
    assert "values" not in repr(da_grid2d)

    da = da_grid2d[:, -1]
    assert "Grid1D" in repr(da)
    assert "values" not in repr(da)

    da = da_grid2d[:, -1, 0]
    assert "geometry" not in repr(da)
    assert "values" in repr(da)

    da = da_grid2d[0, 0, 0]
    assert "geometry" not in repr(da)
    assert "values" in repr(da)


def test_dataarray_grid2d_indexing(da_grid2d):
    da = da_grid2d
    nt, ny, nx = da.shape  # 10, 14, 7
    assert da[0].shape == (ny, nx)
    assert da[0, :, :].shape == (ny, nx)
    assert da[:, 0, 1:4].shape == (nt, 3)
    assert da[5:, :, 0].shape == (5, ny)
    assert da[0:5, -1, 0].shape == (5,)
    assert da[0, :, 4].shape == (ny,)
    assert da[0, -1, :].shape == (nx,)
    assert da[0, 0, 0].shape == ()

    assert isinstance(da[0, :, :].geometry, mikeio.Grid2D)
    assert isinstance(da[0, 0, :].geometry, mikeio.Grid1D)
    assert isinstance(da[:, :, 0].geometry, mikeio.Grid1D)
    assert isinstance(da[:, -1, 0].geometry, GeometryUndefined)

    # TODO: slices in other than the time direction will give GeometryUndefined
    assert isinstance(da[:, 2:5, 0].geometry, GeometryUndefined)
    assert isinstance(da[:, 2:5, 0:4].geometry, GeometryUndefined)


def test_dataarray_grid2d_indexing_error(da_grid2d):
    with pytest.raises(IndexError, match="Key has more dimensions"):
        da_grid2d[0, :, :, 4]
    with pytest.raises(IndexError):
        da_grid2d[12]
    with pytest.raises(IndexError):
        da_grid2d[14:18]
    with pytest.raises(IndexError):
        da_grid2d[3, :, 100]


def test_da_isel_space(da_grid2d):
    assert da_grid2d.geometry.nx == 7
    assert da_grid2d.geometry.ny == 14
    da_sel = da_grid2d.isel(0, axis="y")
    assert da_sel.dims[0][0] == "t"
    assert da_sel.dims[1] == "x"
    assert isinstance(da_sel.geometry, mikeio.Grid1D)

    da_sel = da_grid2d.isel(0, axis="x")
    assert da_sel.dims[0][0] == "t"
    assert da_sel.dims[1] == "y"
    assert isinstance(da_sel.geometry, mikeio.Grid1D)

    da_sel = da_grid2d.isel(0, axis="t")
    assert da_sel.dims[0] == "y"
    assert da_sel.dims[1] == "x"


def test_da_isel_empty(da_grid2d):
    da_sel = da_grid2d.isel(slice(100, 200), axis="y")
    assert da_sel is None


def test_da_isel_space_multiple_elements(da_grid2d):
    assert da_grid2d.geometry.nx == 7
    assert da_grid2d.geometry.ny == 14
    da_sel = da_grid2d.isel((0, 1, 2, 10), axis="y")
    assert da_sel.dims == ("time", "y", "x")
    assert da_sel.shape == (10, 4, 7)
    assert isinstance(da_sel.geometry, GeometryUndefined)

    da_sel = da_grid2d.isel(slice(None, 3), axis="x")
    assert da_sel.dims == ("time", "y", "x")
    assert da_sel.shape == (10, 14, 3)
    assert isinstance(da_sel.geometry, GeometryUndefined)


def test_da_isel_space_named_axis(da_grid2d: mikeio.DataArray):
    da_sel = da_grid2d.isel(y=0)
    assert da_sel.dims[0] == "time"

    da_sel = da_grid2d.isel(x=0)
    assert da_sel.dims[0] == "time"
    assert da_sel.dims[1] == "y"

    da_sel = da_grid2d.isel(time=0)
    assert da_sel.dims[0] == "y"
    assert da_sel.dims[1] == "x"


def test_da_isel_space_named_missing_axis(da_grid2d: mikeio.DataArray):

    with pytest.raises(ValueError) as excinfo:
        da_grid2d.isel(layer=0)
    assert "layer" in str(excinfo.value)


def test_da_sel_layer():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    da = mikeio.read(filename, items=0)[0]
    assert da.geometry.n_elements == 17118
    assert da.geometry.is_layered

    da1 = da.sel(layer=-1)
    assert da1.geometry.n_elements == 3700
    assert not da1.geometry.is_layered

    da2 = da.sel(layer="top")
    assert da2.geometry.n_elements == 3700
    # assert

    da3 = da.sel(layer="bottom")
    assert da3.geometry.n_elements == 3700


def test_da_sel_area_2d():
    filename = "tests/testdata/FakeLake.dfsu"
    da = mikeio.read(filename, items=0)[0]

    area = [-0.1, 0.15, 0.0, 0.2]
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 14

    area = (-0.1, 0.15, 0.0, 0.2)
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 14


def test_da_sel_area_and_xy_not_ok():
    filename = "tests/testdata/FakeLake.dfsu"
    da = mikeio.read(filename, items=0)[0]

    area = [-0.1, 0.15, 0.0, 0.2]
    with pytest.raises(ValueError) as excinfo:
        da.sel(area=area, x=0.0, y=0.1)
    assert "area" in str(excinfo.value)


def test_da_sel_area_3d():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    da = mikeio.read(filename, items=0)[0]
    assert da.geometry.n_elements == 17118
    assert da.geometry.n_layers == 9

    area = [340000, 6140000, 360000, 6170000]
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 4567
    assert da1.geometry.n_layers == 6


def test_da_sel_area_2dv():
    filename = "tests/testdata/basin_2dv.dfsu"
    da = mikeio.read(filename, items=0)[0]
    assert da.geometry.is_layered

    # TODO
    # area = [100, 10, 300, 30]
    # da1 = da.sel(area=area)
    # assert da1.geometry.n_elements == 128
    # assert da1.geometry.is_layered


def test_describe(da_grid2d):
    df = da_grid2d.describe()
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert "max" in df.index


def test_plot_grid1d(da2):
    # Not very functional tests, but at least it runs without errors
    da2.plot(title="The TITLE")
    da2.plot.line()
    da2.plot.timeseries(figsize=(12, 4))

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    da2.plot.imshow(ax=ax1)
    da2.plot.pcolormesh(ax=ax2)


def test_plot_grid2d_proj(da_grid2d_proj):
    da_grid2d_proj.plot()


def test_timestep(da1):

    assert da1.timestep == 1.0


def test_interp_time(da1):
    da = mikeio.read("tests/testdata/HD2D.dfsu")[0]
    dai = da.interp_time(dt=1800)
    assert dai.timestep == 1800


def test_dims_time(da1):

    assert da1.dims[0][0] == "t"


def test_dims_time_space1d(da_time_space):

    assert da_time_space.dims[1] == "x"


def test_repr(da_time_space):

    text = repr(da_time_space)
    assert "DataArray" in text
    assert "dims: (time:10, x:2)" in text


def test_plot(da1):

    da1.plot()
    assert True


def test_modify_values(da1):

    assert all(~np.isnan(da1.values))
    da1[0] = np.nan
    assert any(np.isnan(da1.values))

    with pytest.raises(ValueError):
        da1.values = np.array([1.0])  # you can not set data to another shape

    # This is allowed
    da1.values = np.zeros_like(da1.values) + 2.0


def test_add_scalar(da1):
    da2 = da1 + 10.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == 10.0)

    da3 = 10.0 + da1  # __radd__
    assert isinstance(da3, mikeio.DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_subtract_scalar(da1):
    da2 = da1 - 10.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == -10.0)

    da3 = 10.0 - da1  # __rsub__
    assert isinstance(da3, mikeio.DataArray)
    assert da3.to_numpy()[-1] == -9.0


def test_multiply_scalar(da1):
    da2 = da1 * 2.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() / da1.to_numpy() == 2.0)

    da3 = 2.0 * da1  # __rmul__
    assert isinstance(da3, mikeio.DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_multiply_two_dataarrays(da1):

    da3 = da1 * da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape

    da3 = da1 * da1.values
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape


def test_multiply_two_dataarrays_broadcasting(da_grid2d):
    da1 = da_grid2d
    da2 = da1 * da1.values[0, 0, :]
    assert isinstance(da2, mikeio.DataArray)
    assert da1.shape == da2.shape

    # nt,ny,nx * ny,nx
    da3 = da1 * da1.max()
    assert isinstance(da3, mikeio.DataArray)
    assert da_grid2d.shape == da3.shape


def test_math_two_dataarrays(da1):

    da3 = da1 + da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape

    da3 = da1 - da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape

    da3 = da1 / da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape

    da3 = da1 * da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape

    da3 = da1 // 23
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape


def test_unary_math_operations(da2):
    assert np.all(da2.values > 0)

    da3 = -da2
    assert np.all(da3.values < 0)
    assert isinstance(da3, mikeio.DataArray)

    da4 = +da2
    assert np.all(da4.values > 0)
    assert np.all(da4.values == da2.values)
    assert isinstance(da4, mikeio.DataArray)

    da5 = abs(da3)
    assert np.all(da5.values == da2.values)
    assert isinstance(da5, mikeio.DataArray)


def test_binary_math_operations(da1):
    da2 = da1**2
    assert np.all(da2.values == da1.values**2)
    assert isinstance(da2, mikeio.DataArray)

    da2 = da1 % 2
    assert isinstance(da2, mikeio.DataArray)


def test_dataarray_masking():
    filename = "tests/testdata/basin_3d.dfsu"
    da = mikeio.read(filename, items="U velocity")[0]
    assert da.shape == (3, 1740)

    mask = da < 0
    assert mask.shape == da.shape
    assert mask.dtype == "bool"
    assert mask.shape == (3, 1740)

    # get values using mask (other values will be np.nan)
    da_mask = da[mask]
    assert isinstance(da_mask, np.ndarray)
    assert da_mask.shape == (2486,)

    # set values smaller than 0 to 0 using mask
    assert da.min(axis=None).values < 0
    da[mask] = 0.0
    assert da.min(axis=None).values == 0

    mask = da > 0
    assert mask.dtype == "bool"

    mask = da == 0
    assert mask.dtype == "bool"

    mask = da != 0
    assert mask.dtype == "bool"

    mask = da >= 0
    assert mask.dtype == "bool"

    mask = da <= 0
    assert mask.dtype == "bool"


def test_daarray_aggregation_dfs2():

    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)
    da = ds.Elevation

    assert da.shape == (1, 264, 216)

    dam = da.nanmean(axis=None)
    assert np.isscalar(dam.values)  # TODO is this what we want

    dasm = da.nanmean(axis="space")
    assert dasm.shape == (1,)


def test_daarray_aggregation():

    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=[3])

    da = ds["Current speed"]
    assert da.ndim == 2
    assert da.dims[0][0] == "t"  # time
    da_max = da.max("time")
    assert da_max.dims[0][0] == "e"  # element
    assert isinstance(da_max, mikeio.DataArray)
    assert da_max.geometry == da.geometry
    assert da_max.start_time == da.start_time  # TODO is this consistent
    assert len(da_max.time) == 1
    # TODO verify values

    da_min = da.min()
    assert isinstance(da_max, mikeio.DataArray)
    assert da_min.geometry == da.geometry
    assert da_min.start_time == da.start_time  # TODO is this consistent
    assert len(da_min.time) == 1
    # TODO verify values

    da_mean = da.mean()
    assert isinstance(da_mean, mikeio.DataArray)
    assert da_mean.geometry == da.geometry
    assert da_mean.start_time == da.start_time  # TODO is this consistent
    assert len(da_mean.time) == 1
    # TODO verify values


def test_daarray_aggregation_nan_versions():

    # TODO find better file, e.g. with flood/dry
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=[3])

    da = ds["Current speed"]
    da_max = da.nanmax()
    assert isinstance(da_max, mikeio.DataArray)
    assert da_max.geometry == da.geometry
    assert da_max.start_time == da.start_time  # TODO is this consistent
    assert len(da_max.time) == 1

    da_min = da.nanmin()
    assert isinstance(da_max, mikeio.DataArray)
    assert da_min.geometry == da.geometry
    assert da_min.start_time == da.start_time  # TODO is this consistent
    assert len(da_min.time) == 1

    da_mean = da.nanmean()
    assert isinstance(da_mean, mikeio.DataArray)
    assert da_mean.geometry == da.geometry
    assert da_mean.start_time == da.start_time  # TODO is this consistent
    assert len(da_mean.time) == 1


def test_da_quantile_axis0(da2):
    assert da2.geometry.n == 7
    assert len(da2.time) == 10
    daq = da2.quantile(q=0.345, axis="time")
    assert daq.geometry.n == 7
    assert len(da2.time) == 10  # this should not change
    assert len(daq.time) == 1  # aggregated

    assert daq.to_numpy()[0] == 0.1
    assert daq.ndim == 1
    assert daq.dims[0] == "x"
    assert daq.n_timesteps == 1

    daqs = da2.quantile(q=0.345, axis="space")
    assert isinstance(
        daqs.geometry, GeometryUndefined
    )  # Aggregating over space doesn't create a well defined geometry
    assert isinstance(da2.geometry, mikeio.Grid1D)  # But this one is intact
    assert len(daqs.time) == 10
    assert daqs.ndim == 1
    assert daqs.dims[0][0] == "t"  # Because it's a mikeio.Grid1D, remember!

    # q as list
    daq = da2.quantile(q=[0.25, 0.75], axis=0)
    assert isinstance(daq, mikeio.Dataset)
    assert daq.n_items == 2
    assert daq[0].to_numpy()[0] == 0.1
    assert daq[1].to_numpy()[0] == 0.1

    assert "Quantile 0.75, " in daq.items[1].name


def test_write_dfs2(tmp_path):

    nt = 10
    g = mikeio.Grid2D(
        x=np.linspace(10, 20, 30),
        y=np.linspace(10, 20, 20),
        projection="LONG/LAT",
    )
    da = mikeio.DataArray(
        np.random.random(size=(nt, g.ny, g.nx)),
        time=pd.date_range(start="2000", freq="H", periods=nt),
        item=ItemInfo("Random"),
        geometry=g,
    )

    fn = str(tmp_path / "test.dfs2")

    da.to_dfs(fn)


def test_write_dfs2_single_time_no_time_dim(tmp_path):

    g = mikeio.Grid2D(
        x=np.linspace(10, 20, 30),
        y=np.linspace(10, 20, 20),
        projection="LONG/LAT",
    )
    da = mikeio.DataArray(
        np.random.random(size=(g.ny, g.nx)),  # No singleton time
        time=pd.date_range(start="2000", periods=1),
        item=ItemInfo("Random"),
        geometry=g,
        dims=("y", "x"),
    )

    fn = str(tmp_path / "test_2.dfs2")

    da.to_dfs(fn)


def test_xzy_selection():
    # select in space via x,y,z coordinates test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    das_xzy = ds.Temperature.sel(x=340000, y=15.75, z=0)

    # check for point geometry after selection
    assert type(das_xzy.geometry) == mikeio.spatial.geometry.GeometryPoint3D


def test_layer_selection():
    # select layer test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    das_layer = ds.Temperature.sel(layer=0)
    # should not be layered after selection
    assert type(das_layer.geometry) == mikeio.spatial.FM_geometry.GeometryFM


def test_time_selection():
    # select time test
    nt = 100
    data = []
    d = np.random.rand(nt)
    data.append(d)
    time = pd.date_range("2000-1-2", freq="H", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset(data, time, items)

    das_t = ds.Foo.sel(time="2000-01-05")

    assert das_t.shape == (24,)
