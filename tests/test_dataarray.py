import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import mikeio
from mikeio.eum import EUMType, ItemInfo
from mikeio.exceptions import OutsideModelDomainError
from mikeio.spatial.geometry import GeometryPoint2D, GeometryPoint3D, GeometryUndefined


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
        geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, nx=nx),
    )

    return da


@pytest.fixture
def da_grid2d():
    nt = 10
    nx = 7
    ny = 14

    da = mikeio.DataArray(
        data=np.zeros([nt, ny, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="H", periods=nt),
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
        geometry=mikeio.Grid1D(nx=2, dx=1.0),
    )

    return da


def test_concat_dataarray_by_time():
    da1 = mikeio.read("tests/testdata/tide1.dfs1")[0]
    da2 = mikeio.read("tests/testdata/tide2.dfs1")[0]
    da3 = mikeio.DataArray.concat([da1, da2])

    assert da3.start_time == da1.start_time
    assert da3.start_time < da2.start_time
    assert da3.end_time == da2.end_time
    assert da3.end_time > da1.end_time
    assert da3.n_timesteps == 145
    assert da3.is_equidistant


def test_verify_custom_dims():
    nt = 10
    nx = 7

    with pytest.raises(ValueError) as excinfo:
        da = mikeio.DataArray(
            data=np.zeros([nt, nx]) + 0.1,
            time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
            item=ItemInfo("Foo"),
            dims=("space", "ensemble"),  # no time!
            geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, nx=nx),
        )
    assert "time" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        da = mikeio.DataArray(
            data=np.zeros([nt, nx]) + 0.1,
            time=pd.date_range(start="2000-01-01", freq="S", periods=nt),
            item=ItemInfo("Foo"),
            dims=("time", "x", "ensemble"),  # inconsistent with data
            geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, nx=nx),
        )
    assert "number" in str(excinfo.value).lower()


def test_write_1d(da2, tmp_path):

    outfilename = tmp_path / "grid1d.dfs1"

    da2.to_dfs(outfilename)

    ds = mikeio.read(outfilename)
    assert ds.n_items == 1
    assert isinstance(ds.geometry, mikeio.Grid1D)


def test_dataset_with_asterisk(da2):

    da2.name = "Foo * Bar"

    ds1 = mikeio.Dataset([da2], validate=False)

    assert ds1[0].name == "Foo * Bar"

    ds2 = mikeio.Dataset({"Foo * Bar": da2})

    assert ds2[0].name == "Foo * Bar"


def test_data_0d(da0):
    assert da0.ndim == 1
    assert da0.dims == ("time",)
    assert "values" in repr(da0)
    assert "values" in repr(da0[:4])

    da0 = da0.squeeze()
    assert da0.ndim == 0
    assert "values" in repr(da0)


def test_create_data_1d_default_grid():

    da = mikeio.DataArray(
        data=np.zeros((10, 5)),
        time=pd.date_range(start="2000-01-01", freq="H", periods=10),
        item=ItemInfo("Foo"),
    )
    assert isinstance(da.geometry, mikeio.Grid1D)


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
    g = mikeio.Grid1D(nx=nx, dx=1.0)
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

    assert isinstance(ds.Salinity.geometry, mikeio.spatial.FM_geometry.GeometryFM3D)

    # indexing in time selecting a single record
    da = ds.Salinity[0, :]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFM3D)

    # indexing in space selecting a single element
    da = ds.Salinity[:, 0]
    assert isinstance(da.geometry, GeometryPoint3D)

    # indexing in space selecting a multiple elements with slice
    da = ds.Salinity[:, 0:45]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFM3D)

    # indexing in space selecting a multiple elements with tuple
    da = ds.Salinity[:, (3, 6, 12)]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFM3D)

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
    assert "geometry: Grid1D" in repr(da)
    assert "values" not in repr(da)

    da = da_grid2d[:, -1, 0]
    assert "geometry: GeometryPoint2D" in repr(da)
    assert "values" in repr(da)

    da = da_grid2d[0, 0, 0]
    assert "geometry: GeometryPoint2D" in repr(da)
    assert "values" in repr(da)


def test_dataarray_grid2d_indexing(da_grid2d):
    da = da_grid2d
    nt, ny, nx = da.shape  # 10, 14, 7
    assert da[0].shape == (ny, nx)
    assert da[0, :, :].shape == (ny, nx)
    assert da[0, [0, 1, 2, 3], [2, 4, 6]].shape == (4, 3)
    assert da[:, 0, 1:4].shape == (nt, 3)
    assert da[5:, :, 0].shape == (5, ny)
    assert da[0:5, -1, 0].shape == (5,)
    assert da[0, :, 4].shape == (ny,)
    assert da[0, -1, :].shape == (nx,)
    assert da[0, 0, 0].shape == ()

    assert isinstance(da[0, :, :].geometry, mikeio.Grid2D)
    assert isinstance(da[0, 0, :].geometry, mikeio.Grid1D)
    assert isinstance(da[:, :, 0].geometry, mikeio.Grid1D)
    assert isinstance(da[:, -1, 0].geometry, GeometryPoint2D)

    # TODO: slices in other than the time direction will give GeometryUndefined
    assert isinstance(da[:, 2:5, 0].geometry, mikeio.Grid1D)
    assert isinstance(da[:, 2:5, 0:4].geometry, mikeio.Grid2D)


def test_dataarray_grid3d_indexing():
    da = mikeio.read("tests/testdata/test_dfs3.dfs3")[0]
    nt, nz, ny, nx = da.shape  # 2, 34, 17, 21
    assert da[0].shape == (nz, ny, nx)
    assert da[0, :, :].shape == (nz, ny, nx)
    assert da[0, [0, 1, 2, 3], [2, 4, 6]].shape == (4, 3, nx)
    assert da[:, 0, 1:4].shape == (nt, 3, nx)
    assert da[:, -1, 0].shape == (nt, nx)
    assert da[:, :, -1, 0].shape == (nt, nz)
    assert da[0, :, 4].shape == (nz, nx)
    assert da[0, -1, :].shape == (ny, nx)
    assert da[0, 0, 0, 0].shape == ()

    assert isinstance(da[0, ::5, ::5, ::5].geometry, mikeio.Grid3D)
    assert isinstance(da[0, :, :].geometry, mikeio.Grid3D)
    assert isinstance(da[0, 0, :].geometry, mikeio.Grid2D)
    assert isinstance(da[:, :, 0].geometry, mikeio.Grid2D)
    assert isinstance(da[:, :, :, -1].geometry, mikeio.Grid2D)
    assert isinstance(da[:, -1, 0].geometry, mikeio.Grid1D)

    # with multi-index along one dimension
    assert isinstance(da[:, 2:5, 0, :].geometry, mikeio.Grid2D)

    # TODO: wait for merge of https://github.com/DHI/mikeio/pull/311
    # assert isinstance(da[:, 1, ::3, :].geometry, mikeio.Grid2D)
    # assert isinstance(da[:, 1, -3, 4:].geometry, mikeio.Grid2D)


def test_dataarray_getitem_time(da_grid2d):
    da = da_grid2d
    # time=pd.date_range("2000-01-01", freq="H", periods=10)
    da_sel = da["2000-1-1"]
    assert da_sel.n_timesteps == da.n_timesteps
    assert da_sel.is_equidistant

    da_sel = da["2000-1-1 02:00":"2000-1-1 05:00"]
    assert da_sel.n_timesteps == 4
    assert da_sel.is_equidistant

    time = ["2000-1-1 02:00", "2000-1-1 04:00", "2000-1-1 06:00"]
    da_sel = da[time]
    assert da_sel.n_timesteps == 3
    assert da_sel.is_equidistant

    time = [da.time[0], da.time[1], da.time[3], da.time[7]]
    da_sel = da[time]
    assert da_sel.n_timesteps == 4
    assert not da_sel.is_equidistant

    da_sel = da[da.time[:5]]
    assert da_sel.n_timesteps == 5
    assert da_sel.is_equidistant


def test_dataarray_grid2d_indexing_error(da_grid2d):
    with pytest.raises(IndexError, match="Key has more dimensions"):
        da_grid2d[0, :, :, 4]
    with pytest.raises(IndexError):
        da_grid2d[12]
    with pytest.raises(IndexError):
        da_grid2d[14:18]
    with pytest.raises(IndexError):
        da_grid2d[3, :, 100]


def test_dropna(da2):
    da2[8:] = np.nan

    da3 = da2.dropna()

    assert da2.n_timesteps == 10
    assert da3.n_timesteps == 8


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
    with pytest.raises(ValueError):
        da_grid2d.isel(slice(100, 200), axis="y")


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
    assert isinstance(da_sel.geometry, mikeio.Grid2D)


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

    da1 = da.sel(layers=-1)
    assert da1.geometry.n_elements == 3700
    assert not da1.geometry.is_layered

    da2 = da.sel(layers="top")
    assert da2.geometry.n_elements == 3700
    # assert

    da3 = da.sel(layers="bottom")
    assert da3.geometry.n_elements == 3700


def test_da_sel_xy_grid2d(da_grid2d):
    # Grid2D(x0=10.0, dx=0.1, nx=7, ny=14, dy=1.0, y0=-10.0),
    da = da_grid2d
    da1 = da.sel(x=10.4, y=0.0)
    assert isinstance(da1.geometry, GeometryPoint2D)
    assert da1.geometry.x == 10.4
    assert da1.geometry.y == 0.0
    assert np.all(da1.to_numpy() == da.to_numpy()[:, 10, 4])

    # da2 = da.sel(x=100.4, y=0.0) # TODO outside grid


def test_da_sel_multi_xy_grid2d(da_grid2d):
    # Grid2D(x0=10.0, dx=0.1, nx=7, ny=14, dy=1.0, y0=-10.0),
    da = da_grid2d
    xx = [10.3, 10.5, 10.4]
    yy = [-1.0, 1.0, -9.0]
    # TODO: not implemented:
    # da1 = da.sel(x=xx, y=yy)
    # assert da1.shape == (10, 3)


def test_da_sel_area_dfsu2d():
    filename = "tests/testdata/FakeLake.dfsu"
    da = mikeio.read(filename, items=0)[0]

    area = [-0.1, 0.15, 0.0, 0.2]
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 14

    area = (-0.1, 0.15, 0.0, 0.2)
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 14


def test_da_sel_area_grid2d():
    filename = "tests/testdata/gebco_sound.dfs2"
    da = mikeio.read(filename, items=0)[0]
    assert da.dims == ("time", "y", "x")

    bbox = [12.4, 55.2, 22.0, 55.6]

    da1 = da.sel(area=bbox)
    assert da1.geometry.nx == 168
    assert da1.geometry.ny == 96

    das = da.squeeze()
    assert das.dims == ("y", "x")

    da = das.sel(area=bbox)
    assert da1.geometry.nx == 168
    assert da1.geometry.ny == 96


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

    plt.close("all")


def test_plot_grid2d_proj(da_grid2d_proj):
    da_grid2d_proj.plot()


def test_timestep(da1):

    assert da1.timestep == 1.0


def test_interp_time(da1):
    da = mikeio.read("tests/testdata/HD2D.dfsu")[0]
    dai = da.interp_time(dt=1800)
    assert dai.timestep == 1800


def test_interp_like_index(da1):
    da = mikeio.read("tests/testdata/HD2D.dfsu")[0]
    dai = da.interp_like(da.time)
    assert any(dai.time == da.time)


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


def test_modify_values_1d(da1):
    assert da1.values[4] == 14.0

    # selecting a slice will return a view. The original is changed.
    da1.isel(slice(4, 6)).values[0] = 13.0
    assert da1.values[4] == 13.0

    # __getitem__ uses isel()
    da1[4:6].values[0] = 12.0
    assert da1.values[4] == 12.0

    # values is scalar, therefore copy by definition. Original is not changed.
    da1.isel(4).values = 11.0
    assert da1.values[4] != 11.0

    # fancy indexing will return copy! Original is *not* changed.
    da1.isel([0, 4, 7]).values[1] = 10.0
    assert da1.values[4] != 10.0


def test_modify_values_2d_all(da2):
    assert da2.shape == (10, 7)
    assert da2.values[2, 5] == 0.1

    da2 += 0.1
    assert da2.values[2, 5] == 0.2

    vals = 0.3 * np.ones(da2.shape)
    da2.values = vals
    assert da2.values[2, 5] == 0.3


def test_modify_values_2d_idx(da2):
    assert da2.shape == (10, 7)
    assert da2.values[2, 5] == 0.1

    # selecting a single index will return a view. The original is changed.
    da2.isel(time=2).values[5] = 0.2
    assert da2.values[2, 5] == 0.2

    da2.isel(x=5).values[2] = 0.3
    assert da2.values[2, 5] == 0.3

    da2.values[2, 5] = 0.4
    assert da2.values[2, 5] == 0.4

    # __getitem__ uses isel()
    da2[2].values[5] = 0.5
    assert da2.values[2, 5] == 0.5

    da2[:, 5].values[2] = 0.6
    assert da2.values[2, 5] == 0.6


def test_modify_values_2d_slice(da2):
    assert da2.shape == (10, 7)
    assert da2.values[2, 5] == 0.1

    # selecting a slice will return a view. The original is changed.
    da2.isel(time=slice(2, 6)).values[0, 5] = 0.4
    assert da2.values[2, 5] == 0.4

    da2.isel(x=slice(5, 7)).values[2, 0] = 0.5
    assert da2.values[2, 5] == 0.5

    # __getitem__ uses isel()
    da2[2:5].values[0, 5] = 0.6
    assert da2.values[2, 5] == 0.6

    da2[:, 5:7].values[2, 0] = 0.7
    assert da2.values[2, 5] == 0.7


def test_modify_values_2d_fancy(da2):
    assert da2.shape == (10, 7)
    assert da2.values[2, 5] == 0.1

    # fancy indexing will return a *copy*. The original is NOT changed.
    da2.isel(time=[2, 3, 4, 5]).values[0, 5] = 0.4
    assert da2.values[2, 5] != 0.4

    da2.isel(x=[5, 6]).values[2, 0] = 0.5
    assert da2.values[2, 5] != 0.5

    # __getitem__ uses isel()
    da2[[2, 3, 4, 5]].values[0, 5] = 0.6
    assert da2.values[2, 5] != 0.6

    da2[:, [5, 6]].values[2, 0] = 0.7
    assert da2.values[2, 5] != 0.7


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


def test_daarray_aggregation_dfs2():

    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)
    da = ds.Elevation

    assert da.shape == (1, 264, 216)

    dam = da.nanmean(axis=None)
    assert np.isscalar(dam.values)  # TODO is this what we want

    dasm = da.nanmean(axis="space")
    assert dasm.shape == (1,)


def test_dataarray_weigthed_average():
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=["Surface elevation"])

    da = ds["Surface elevation"]

    area = da.geometry.get_element_area()

    da2 = da.average(weights=area, axis=1)

    assert isinstance(da2.geometry, GeometryUndefined)
    assert da2.dims == ("time",)


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
    assert da_max.start_time == da.start_time
    assert len(da_max.time) == 1
    assert pytest.approx(da_max.values[0]) == 0.06279723
    assert pytest.approx(da_max.values[778]) == 0.4833801

    da_min = da.min()
    assert isinstance(da_max, mikeio.DataArray)
    assert da_min.geometry == da.geometry
    assert da_min.start_time == da.start_time
    assert len(da_min.time) == 1
    assert pytest.approx(da_min.values[0]) == 0.009865114
    assert pytest.approx(da_min.values[778]) == 0.4032839

    da_mean = da.mean()
    assert isinstance(da_mean, mikeio.DataArray)
    assert da_mean.geometry == da.geometry
    assert da_mean.start_time == da.start_time
    assert len(da_mean.time) == 1
    assert pytest.approx(da_mean.values[0]) == 0.04334851
    assert pytest.approx(da_mean.values[778]) == 0.452692

    da_std = da.std(name="standard deviation")
    assert isinstance(da_std, mikeio.DataArray)
    assert da_std.name == "standard deviation"
    assert da_std.geometry == da.geometry
    assert da_std.start_time == da.start_time
    assert len(da_std.time) == 1
    assert pytest.approx(da_std.values[0]) == 0.015291579

    da_ptp = da.ptp(name="peak to peak (max - min)")
    assert isinstance(da_std, mikeio.DataArray)
    assert da_ptp.geometry == da.geometry
    assert da_ptp.start_time == da.start_time
    assert len(da_ptp.time) == 1
    assert pytest.approx(da_ptp.values[0]) == 0.0529321208596229


def test_daarray_aggregation_no_time():
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=[3], time=-1)
    da = ds["Current speed"]
    assert da.dims == ("element",)

    assert da.max().values == pytest.approx(1.6463733)


def test_daarray_aggregation_nan_versions():

    # TODO find better file, e.g. with flood/dry
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=[3])

    da = ds["Current speed"]
    da_max = da.nanmax()
    assert isinstance(da_max, mikeio.DataArray)
    assert da_max.geometry == da.geometry
    assert da_max.start_time == da.start_time
    assert len(da_max.time) == 1
    assert pytest.approx(da_max.values[0]) == 0.06279723
    assert pytest.approx(da_max.values[778]) == 0.4833801

    da_min = da.nanmin()
    assert isinstance(da_max, mikeio.DataArray)
    assert da_min.geometry == da.geometry
    assert da_min.start_time == da.start_time
    assert len(da_min.time) == 1
    assert pytest.approx(da_min.values[0]) == 0.009865114
    assert pytest.approx(da_min.values[778]) == 0.4032839

    da_mean = da.nanmean()
    assert isinstance(da_mean, mikeio.DataArray)
    assert da_mean.geometry == da.geometry
    assert da_mean.start_time == da.start_time
    assert len(da_mean.time) == 1
    assert pytest.approx(da_mean.values[0]) == 0.04334851
    assert pytest.approx(da_mean.values[778]) == 0.452692

    da_std = da.nanstd()
    assert isinstance(da_std, mikeio.DataArray)
    assert da_std.geometry == da.geometry
    assert da_std.start_time == da.start_time
    assert len(da_std.time) == 1
    assert pytest.approx(da_std.values[0]) == 0.015291579


def test_da_quantile_axis0(da2):
    assert da2.geometry.nx == 7
    assert len(da2.time) == 10
    daq = da2.quantile(q=0.345, axis="time")
    assert daq.geometry.nx == 7
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
        x=np.linspace(10, 20, 11),
        y=np.linspace(15, 25, 21),
        projection="LONG/LAT",
    )
    assert g.origin == (0, 0)
    da = mikeio.DataArray(
        np.random.random(size=(nt, g.ny, g.nx)),
        time=pd.date_range(start="2000", freq="H", periods=nt),
        item=ItemInfo("Random"),
        geometry=g,
    )

    fn = str(tmp_path / "test.dfs2")
    da.to_dfs(fn)

    ds = mikeio.read(fn)
    g2 = ds.geometry
    assert g != g2
    assert np.allclose(g.x, g2.x)
    assert np.allclose(g.y, g2.y)
    assert g2.origin == (10.0, 15.0)
    assert g.projection == g2.projection


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

    das_xzy = ds.Temperature.sel(x=348946, y=6173673, z=0)

    # check for point geometry after selection
    assert type(das_xzy.geometry) == mikeio.spatial.geometry.GeometryPoint3D
    assert das_xzy.values[0] == pytest.approx(17.381)

    # do the same but go one level deeper, but finding the index first
    idx = ds.geometry.find_index(x=348946, y=6173673, z=0)
    das_idx = ds.Temperature.isel(element=idx)
    assert das_idx.values[0] == pytest.approx(17.381)

    # let's try find the same point multiple times
    das_idxs = ds.geometry.find_index(
        x=[348946, 348946], y=[6173673, 6173673], z=[0, 0]
    )
    assert len(das_idxs) == 1  # only one point


def test_xzy_selection_outside_domain():
    # select in space via x,y,z coordinates test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    with pytest.raises(OutsideModelDomainError):
        ds.Temperature.sel(x=340000, y=15.75, z=0)  # this is way outside the domain


def test_layer_selection():
    # select layer test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    das_layer = ds.Temperature.sel(layers=0)
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

    with pytest.raises(IndexError):
        # not in time
        ds.sel(time="1997-09-15 00:00")


def test_interp_na():
    time = pd.date_range("2000", periods=5, freq="D")
    da = mikeio.DataArray(
        data=np.array([np.nan, 1.0, np.nan, np.nan, 4.0]),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    dai = da.interp_na()
    assert np.isnan(dai.to_numpy()[0])
    assert dai.to_numpy()[2] == pytest.approx(2.0)

    dai = da.interp_na(fill_value="extrapolate")
    assert dai.to_numpy()[0] == pytest.approx(0.0)
    assert dai.to_numpy()[2] == pytest.approx(2.0)
