from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import mikeio
from mikeio import EUMType, EUMUnit, ItemInfo, Mesh, DataArray
from mikeio.exceptions import OutsideModelDomainError


@pytest.fixture
def da0() -> mikeio.DataArray:
    time = "2000-01-01 00:00:00"
    da = mikeio.DataArray(
        data=np.array([7.0]),
        time=time,
        item=ItemInfo(name="Foo"),
    )
    return da


@pytest.fixture
def da1() -> mikeio.DataArray:
    nt = 10
    start = 10.0
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
    da = mikeio.DataArray(
        data=np.arange(start, start + nt, dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    return da


@pytest.fixture
def da2() -> mikeio.DataArray:
    nt = 10
    nx = 7

    da = mikeio.DataArray(
        data=np.zeros([nt, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="s", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid1D(x0=1000.0, dx=10.0, nx=nx),
    )

    return da


@pytest.fixture
def da_grid2d() -> mikeio.DataArray:
    nt = 10
    nx = 7
    ny = 14

    da = mikeio.DataArray(
        data=np.zeros([nt, ny, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="h", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid2D(x0=10.0, dx=0.1, nx=nx, ny=ny, dy=1.0, y0=-10.0),
    )

    return da


@pytest.fixture
def da_grid2d_proj() -> mikeio.DataArray:
    nt = 10
    nx = 7
    ny = 14

    da = mikeio.DataArray(
        data=np.zeros([nt, ny, nx]) + 0.1,
        time=pd.date_range(start="2000-01-01", freq="s", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid2D(
            x0=1000, dx=100, nx=nx, ny=ny, dy=10, y0=2000, projection="UTM-32"
        ),
    )

    return da


@pytest.fixture
def da_time_space() -> DataArray:
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
    da = mikeio.DataArray(
        data=np.zeros(shape=(nt, 2), dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
        geometry=mikeio.Grid1D(nx=2, dx=1.0),
    )

    return da


def test_concat_dataarray_by_time() -> None:
    da1 = mikeio.read("tests/testdata/tide1.dfs1")[0]
    da2 = mikeio.read("tests/testdata/tide2.dfs1")[0]
    da3 = mikeio.DataArray.concat([da1, da2])

    assert da3.start_time == da1.start_time
    assert da3.start_time < da2.start_time
    assert da3.end_time == da2.end_time
    assert da3.end_time > da1.end_time
    assert da3.n_timesteps == 145
    assert da3.is_equidistant


def test_write_1d(da2: DataArray, tmp_path: Path) -> None:
    outfilename = tmp_path / "grid1d.dfs1"

    da2.to_dfs(outfilename)

    ds = mikeio.read(outfilename)
    assert ds.n_items == 1
    assert isinstance(ds.geometry, mikeio.Grid1D)


def test_dataset_with_asterisk(da2: DataArray) -> None:
    da2.name = "Foo * Bar"

    ds1 = mikeio.Dataset([da2], validate=False)

    assert ds1[0].name == "Foo * Bar"

    ds2 = mikeio.Dataset({"Foo * Bar": da2})

    assert ds2[0].name == "Foo * Bar"


def test_data_0d(da0: DataArray) -> None:
    assert da0.ndim == 1
    assert da0.dims == ("time",)
    assert "values" in repr(da0)
    assert "values" in repr(da0[:4])

    da0 = da0.squeeze()
    assert da0.ndim == 0
    assert "values" in repr(da0)


def test_create_data_1d_default_grid() -> None:
    da = mikeio.DataArray(
        data=np.zeros((10, 5)),
        time=pd.date_range(start="2000-01-01", freq="h", periods=10),
        item=ItemInfo("Foo"),
    )
    assert isinstance(da.geometry, mikeio.Grid1D)


def test_dataarray_init() -> None:
    nt = 10
    start = 10.0
    data = np.arange(start, start + nt, dtype=float)
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
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


def test_dataarray_init_no_item() -> None:
    nt = 10
    data = data = np.zeros([nt, 4]) + 0.1
    time = time = pd.date_range(start="2000-01-01", freq="s", periods=nt)

    da = mikeio.DataArray(data=data, time=time)
    assert da.type == EUMType.Undefined
    assert da.unit == EUMUnit.undefined


def test_dataarray_init_2d() -> None:
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)

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


def test_dataarray_init_wrong_dim() -> None:
    nt = 10
    start = 10.0
    data = np.arange(start, start + nt, dtype=float)
    time_long = pd.date_range(start="2000-01-01", freq="s", periods=(nt + 1))
    item = ItemInfo(name="Foo")

    with pytest.raises(ValueError):
        mikeio.DataArray(data=data, time=time_long, item=item)

    nt, ny, nx = 10, 5, 6
    data2d = np.zeros([nt, ny, nx]) + 0.1
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time_long)

    # time must be first dim
    dims = ("x", "y", "time")
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time, dims=dims)

    # time must be first dim
    data2d = np.zeros([ny, nt, nx]) + 0.1
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time)


def test_dataarray_init_grid1d() -> None:
    nt = 10
    nx = 5
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
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


def test_dataarray_init_grid2d() -> None:
    nt = 10
    ny, nx = 7, 5
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
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


def test_dataarray_init_dfsu2d() -> None:
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
    filename = "tests/testdata/north_sea_2.mesh"
    msh = Mesh(filename)
    g = msh.geometry
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


def test_dataarray_init_dfsu3d() -> None:
    nt = 10
    time = pd.date_range(start="2000-01-01", freq="s", periods=nt)
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
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


def test_dataarray_indexing(da1: mikeio.DataArray) -> None:
    assert da1.shape == (10,)
    subset = da1[3]
    assert isinstance(subset, mikeio.DataArray)
    assert da1.shape == (10,)
    assert subset.to_numpy() == np.array([13.0])


def test_dataarray_dfsu3d_indexing() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)
    sal = ds["Salinity"]

    assert isinstance(ds["Salinity"].geometry, mikeio.spatial.GeometryFM3D)

    # indexing in time selecting a single record
    da = sal[0, :]  # type: ignore
    assert isinstance(da.geometry, mikeio.spatial.GeometryFM3D)

    # indexing in space selecting a single element
    da = sal[:, 0]  # type: ignore
    assert isinstance(da.geometry, mikeio.spatial.GeometryPoint3D)

    # indexing in space selecting a multiple elements with slice
    da = sal[:, 0:45]  # type: ignore
    assert isinstance(da.geometry, mikeio.spatial.GeometryFM3D)

    # indexing in space selecting a multiple elements with tuple
    da = sal[:, (3, 6, 12)]  # type: ignore
    assert isinstance(da.geometry, mikeio.spatial.GeometryFM3D)

    # indexing in both time and space
    da = sal[0, 0]
    assert isinstance(da.geometry, mikeio.spatial.GeometryPoint3D)
    assert da.shape == ()


def test_dataarray_grid1d_repr(da2: DataArray) -> None:
    assert "Grid1D" in repr(da2)
    assert "values" not in repr(da2)


def test_dataarray_grid1d_indexing(da2: DataArray) -> None:
    da = da2
    nt, nx = da.shape
    assert da[0].shape == (nx,)
    assert da[0, :].shape == (nx,)
    assert da[:, -1].shape == (nt,)
    assert da[:, :].shape == (nt, nx)
    assert da[0, 0].shape == ()

    assert isinstance(da[:, :].geometry, mikeio.Grid1D)
    assert isinstance(da[:, -1].geometry, mikeio.spatial.GeometryUndefined)


def test_dataarray_grid2d_repr(da_grid2d: DataArray) -> None:
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


def test_dataarray_grid2d_indexing(da_grid2d: DataArray) -> None:
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
    assert isinstance(da[:, -1, 0].geometry, mikeio.spatial.GeometryPoint2D)

    # TODO: slices in other than the time direction will give GeometryUndefined
    assert isinstance(da[:, 2:5, 0].geometry, mikeio.Grid1D)
    assert isinstance(da[:, 2:5, 0:4].geometry, mikeio.Grid2D)


def test_dataarray_grid3d_indexing() -> None:
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


def test_dataarray_getitem_time(da_grid2d: DataArray) -> None:
    da = da_grid2d
    # time=pd.date_range("2000-01-01", freq="h", periods=10)
    # deprecated use .sel(time=...) or .isel(time=...) instead
    with pytest.warns(FutureWarning, match="string"):
        da_sel = da["2000-1-1"]
    assert da_sel.n_timesteps == da.n_timesteps
    assert da_sel.is_equidistant

    with pytest.warns(FutureWarning, match="string"):
        da_sel = da["2000-1-1 02:00":"2000-1-1 05:00"]  # type: ignore
    assert da_sel.n_timesteps == 4
    assert da_sel.is_equidistant

    time = ["2000-1-1 02:00", "2000-1-1 04:00", "2000-1-1 06:00"]
    with pytest.warns(FutureWarning, match="string"):
        da_sel = da[time]
    assert da_sel.n_timesteps == 3
    assert da_sel.is_equidistant

    time = [da.time[0], da.time[1], da.time[3], da.time[7]]
    # TODO should this type of indexing be allowed?
    da_sel = da[time]
    assert da_sel.n_timesteps == 4
    assert not da_sel.is_equidistant

    da_sel = da[da.time[:5]]
    assert da_sel.n_timesteps == 5
    assert da_sel.is_equidistant


def test_dataarray_grid2d_indexing_error(da_grid2d: DataArray) -> None:
    with pytest.raises(IndexError, match="Key has more dimensions"):
        da_grid2d[0, :, :, 4]
    with pytest.raises(IndexError):
        da_grid2d[12]
    with pytest.raises(IndexError):
        da_grid2d[14:18]
    with pytest.raises(IndexError):
        da_grid2d[3, :, 100]


def test_dropna(da2: DataArray) -> None:
    da2[8:] = np.nan  # type: ignore

    da3 = da2.dropna()

    assert da2.n_timesteps == 10
    assert da3.n_timesteps == 8


def test_da_isel_space(da_grid2d: DataArray) -> None:
    assert da_grid2d.geometry.nx == 7
    assert da_grid2d.geometry.ny == 14
    da_sel = da_grid2d.isel(y=0)
    assert da_sel.dims == ("time", "x")
    assert isinstance(da_sel.geometry, mikeio.Grid1D)

    da_sel = da_grid2d.isel(x=0)
    assert da_sel.dims == ("time", "y")
    assert isinstance(da_sel.geometry, mikeio.Grid1D)

    da_sel = da_grid2d.isel(time=0)
    assert da_sel.dims == ("y", "x")


def test_da_isel_empty(da_grid2d: DataArray) -> None:
    with pytest.raises(ValueError):
        da_grid2d.isel(y=slice(100, 200))


def test_da_isel_space_multiple_elements(da_grid2d: DataArray) -> None:
    assert da_grid2d.geometry.nx == 7
    assert da_grid2d.geometry.ny == 14
    da_sel = da_grid2d.isel(y=(0, 1, 2, 10))
    assert da_sel.dims == ("time", "y", "x")
    assert da_sel.shape == (10, 4, 7)
    assert isinstance(da_sel.geometry, mikeio.spatial.GeometryUndefined)

    da_sel = da_grid2d.isel(x=slice(None, 3))
    assert da_sel.dims == ("time", "y", "x")
    assert da_sel.shape == (10, 14, 3)
    assert isinstance(da_sel.geometry, mikeio.Grid2D)


def test_da_isel_space_named_axis(da_grid2d: mikeio.DataArray) -> None:
    da_sel = da_grid2d.isel(y=0)
    assert da_sel.dims[0] == "time"

    da_sel = da_grid2d.isel(x=0)
    assert da_sel.dims == ("time", "y")

    da_sel = da_grid2d.isel(time=0)
    assert da_sel.dims == ("y", "x")


def test_da_isel_space_named_missing_axis(da_grid2d: mikeio.DataArray) -> None:
    with pytest.raises(ValueError) as excinfo:
        da_grid2d.isel(layer=0)
    assert "layer" in str(excinfo.value)


def test_da_sel_layer() -> None:
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


def test_da_sel_xy_grid2d(da_grid2d: DataArray) -> None:
    # Grid2D(x0=10.0, dx=0.1, nx=7, ny=14, dy=1.0, y0=-10.0),
    da = da_grid2d
    da1 = da.sel(x=10.4, y=0.0)
    assert isinstance(da1.geometry, mikeio.spatial.GeometryPoint2D)
    assert da1.geometry.x == 10.4
    assert da1.geometry.y == 0.0
    assert np.all(da1.to_numpy() == da.to_numpy()[:, 10, 4])

    # da2 = da.sel(x=100.4, y=0.0) # TODO outside grid


def test_da_sel_multi_xy_grid2d(da_grid2d: DataArray) -> None:
    # Grid2D(x0=10.0, dx=0.1, nx=7, ny=14, dy=1.0, y0=-10.0),
    pass
    # TODO: not implemented:
    # da1 = da.sel(x=xx, y=yy)
    # assert da1.shape == (10, 3)


def test_da_sel_area_dfsu2d() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    da = mikeio.read(filename, items=0)[0]

    area = (-0.1, 0.15, 0.0, 0.2)
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 14

    area = (-0.1, 0.15, 0.0, 0.2)
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 14


def test_da_isel_order_is_important_dfsu2d() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    da = mikeio.read(filename, items=0, time=0)[0]

    # select elements sorted
    da1 = da.isel(element=[0, 1])
    assert da1.values[0] == pytest.approx(-3.2252840995788574)
    assert da1.geometry.element_coordinates[0, 0] == pytest.approx(-0.61049269425)

    # select elements in arbitrary order
    da2 = da.isel(element=[1, 0])
    assert da2.values[1] == pytest.approx(-3.2252840995788574)
    assert da2.geometry.element_coordinates[1, 0] == pytest.approx(-0.61049269425)

    # select same elements multiple times, not sure why, but consistent with NumPy, xarray
    da3 = da.isel(element=[1, 0, 1])
    assert da3.values[1] == pytest.approx(-3.2252840995788574)
    assert da3.geometry.element_coordinates[1, 0] == pytest.approx(-0.61049269425)
    assert len(da3.geometry.element_coordinates) == 3


def test_da_sel_area_grid2d() -> None:
    filename = "tests/testdata/gebco_sound.dfs2"
    da = mikeio.read(filename, items=0)[0]
    assert da.dims == ("time", "y", "x")

    bbox = (12.4, 55.2, 22.0, 55.6)

    da1 = da.sel(area=bbox)
    assert da1.geometry.nx == 168
    assert da1.geometry.ny == 96

    das = da.squeeze()
    assert das.dims == ("y", "x")

    da = das.sel(area=bbox)
    assert da1.geometry.nx == 168
    assert da1.geometry.ny == 96


def test_da_sel_area_and_xy_not_ok() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    da = mikeio.read(filename, items=0)[0]

    area = (-0.1, 0.15, 0.0, 0.2)
    with pytest.raises(ValueError) as excinfo:
        da.sel(area=area, x=0.0, y=0.1)
    assert "area" in str(excinfo.value)


def test_da_sel_area_3d() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    da = mikeio.read(filename, items=0)[0]
    assert da.geometry.n_elements == 17118
    assert da.geometry.n_layers == 9

    area = (340000, 6140000, 360000, 6170000)
    da1 = da.sel(area=area)
    assert da1.geometry.n_elements == 4567
    assert da1.geometry.n_layers == 6


def test_da_sel_area_2dv() -> None:
    filename = "tests/testdata/basin_2dv.dfsu"
    da = mikeio.read(filename, items=0)[0]
    assert da.geometry.is_layered

    # TODO
    # area = [100, 10, 300, 30]
    # da1 = da.sel(area=area)
    # assert da1.geometry.n_elements == 128
    # assert da1.geometry.is_layered


def test_describe(da_grid2d: DataArray) -> None:
    df = da_grid2d.describe()
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert "max" in df.index


def test_plot_grid1d(da2: DataArray) -> None:
    # Not very functional tests, but at least it runs without errors
    da2.plot(title="The TITLE")
    da2.plot.line()
    da2.plot.timeseries(figsize=(12, 4))

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    da2.plot.imshow(ax=ax1)
    da2.plot.pcolormesh(ax=ax2)

    plt.close("all")


def test_plot_grid2d_proj(da_grid2d_proj: DataArray) -> None:
    da_grid2d_proj.plot()


def test_timestep(da1: DataArray) -> None:
    assert da1.timestep == 1.0


def test_interp_time(da1: DataArray) -> None:
    da = mikeio.read("tests/testdata/HD2D.dfsu")[0]
    dai = da.interp_time(dt=1800)
    assert dai.timestep == 1800


def test_interp_like_index(da1: DataArray) -> None:
    da = mikeio.read("tests/testdata/HD2D.dfsu")[0]
    dai = da.interp_like(da.time)
    assert any(dai.time == da.time)


def test_dims_time(da1: DataArray) -> None:
    assert da1.dims[0][0] == "t"


def test_dims_time_space1d(da_time_space: DataArray) -> None:
    assert da_time_space.dims[1] == "x"


def test_repr(da_time_space: DataArray) -> None:
    text = repr(da_time_space)
    assert "DataArray" in text
    assert "dims: (time:10, x:2)" in text


def test_plot(da1: DataArray) -> None:
    da1.plot()
    assert True


def test_modify_values(da1: DataArray) -> None:
    assert all(~np.isnan(da1.values))
    da1[0] = np.nan  # type: ignore
    assert any(np.isnan(da1.values))

    with pytest.raises(ValueError):
        da1.values = np.array([1.0])  # you can not set data to another shape

    # This is allowed
    da1.values = np.zeros_like(da1.values) + 2.0


def test_modify_values_1d(da1: DataArray) -> None:
    assert da1.values[4] == 14.0

    # selecting a slice will return a view. The original is changed.
    da1.isel(slice(4, 6)).values[0] = 13.0
    assert da1.values[4] == 13.0

    # __getitem__ uses isel()
    da1[4:6].values[0] = 12.0
    assert da1.values[4] == 12.0

    # values is scalar, therefore copy by definition. Original is not changed.
    # TODO is the treatment of scalar sensible, i.e. consistent with xarray?
    da1.isel(4).values = 11.0  # type: ignore
    assert da1.values[4] != 11.0

    # fancy indexing will return copy! Original is *not* changed.
    da1.isel([0, 4, 7]).values[1] = 10.0
    assert da1.values[4] != 10.0


def test_get_2d_slice_with_sel(da_grid2d: DataArray) -> None:
    assert da_grid2d.shape == (10, 14, 7)
    da3 = da_grid2d.sel(x=slice(10.0, 10.3))
    assert da3.shape == (10, 14, 3)
    da4 = da_grid2d.sel(y=slice(-5.0, 0.0))
    assert da4.shape == (10, 5, 7)

    da5 = da_grid2d.sel(x=slice(10.0, 10.3), y=slice(-5.0, 0.0))
    assert da5.shape == (10, 5, 3)

    da6 = da_grid2d.sel(x=slice(None, 10.3), y=slice(-4.0, None))
    assert da6.shape == (10, 8, 3)


def test_get_2d_outside_domain_raises_error(da_grid2d: DataArray) -> None:
    with pytest.raises(OutsideModelDomainError):
        da_grid2d.sel(x=0.0)

    with pytest.raises(OutsideModelDomainError):
        da_grid2d.sel(x=slice(0.0, 1.0))


def test_modify_values_2d_all(da2: DataArray) -> None:
    assert da2.shape == (10, 7)
    assert da2.values[2, 5] == 0.1

    da2 += 0.1
    assert da2.values[2, 5] == 0.2

    vals = 0.3 * np.ones(da2.shape)
    da2.values = vals
    assert da2.values[2, 5] == 0.3


def test_modify_values_2d_idx(da2: DataArray) -> None:
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


def test_modify_values_2d_slice(da2: DataArray) -> None:
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


def test_modify_values_2d_fancy(da2: DataArray) -> None:
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


def test_add_scalar(da1: DataArray) -> None:
    da2 = da1 + 10.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == 10.0)

    da3 = 10.0 + da1  # __radd__
    assert isinstance(da3, mikeio.DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_subtract_scalar(da1: DataArray) -> None:
    da2 = da1 - 10.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == -10.0)

    da3 = 10.0 - da1  # __rsub__
    assert isinstance(da3, mikeio.DataArray)
    assert da3.to_numpy()[-1] == -9.0


def test_multiply_scalar(da1: DataArray) -> None:
    da2 = da1 * 2.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() / da1.to_numpy() == 2.0)

    da3 = 2.0 * da1  # __rmul__
    assert isinstance(da3, mikeio.DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_multiply_string_is_not_valid(da1: DataArray) -> None:
    with pytest.raises(TypeError):
        da1 * "2.0"  # type: ignore


def test_multiply_two_dataarrays(da1: DataArray) -> None:
    da3 = da1 * da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape

    da3 = da1 * da1.values
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape


def test_multiply_two_dataarrays_broadcasting(da_grid2d: DataArray) -> None:
    da1 = da_grid2d
    da2 = da1 * da1.values[0, 0, :]
    assert isinstance(da2, mikeio.DataArray)
    assert da1.shape == da2.shape

    # nt,ny,nx * ny,nx
    da3 = da1 * da1.max()
    assert isinstance(da3, mikeio.DataArray)
    assert da_grid2d.shape == da3.shape


def test_math_two_dataarrays(da1: DataArray) -> None:
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


def test_unary_math_operations(da2: DataArray) -> None:
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


def test_binary_math_operations(da1: DataArray) -> None:
    da2 = da1**2
    assert np.all(da2.values == da1.values**2)
    assert isinstance(da2, mikeio.DataArray)

    da2 = da1 % 2
    assert isinstance(da2, mikeio.DataArray)


def test_daarray_aggregation_dfs2() -> None:
    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)
    da = ds["Elevation"]

    assert da.shape == (1, 264, 216)

    dam = da.nanmean(axis=None)
    assert np.isscalar(dam.values)  # TODO is this what we want

    dasm = da.nanmean(axis="space")
    assert dasm.shape == (1,)


def test_dataarray_weigthed_average() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=["Surface elevation"])

    da = ds["Surface elevation"]

    area = da.geometry.get_element_area()

    da2 = da.average(weights=area, axis=1)

    assert isinstance(da2.geometry, mikeio.spatial.GeometryUndefined)
    assert da2.dims == ("time",)


def test_daarray_aggregation() -> None:
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


def test_daarray_aggregation_no_time() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=[3], time=-1)
    da = ds["Current speed"]
    assert da.dims == ("element",)

    assert da.max().values == pytest.approx(1.6463733)


def test_daarray_aggregation_nan_versions() -> None:
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


def test_da_quantile_axis0(da2: DataArray) -> None:
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
        daqs.geometry, mikeio.spatial.GeometryUndefined
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


def test_write_dfs2(tmp_path: Path) -> None:
    nt = 10
    g = mikeio.Grid2D(
        x=np.linspace(10, 20, 11),
        y=np.linspace(15, 25, 21),
        projection="LONG/LAT",
    )
    assert g.origin == (0, 0)
    da = mikeio.DataArray(
        np.random.random(size=(nt, g.ny, g.nx)),
        time=pd.date_range(start="2000", freq="h", periods=nt),
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


def test_write_dfs2_single_time_no_time_dim(tmp_path: Path) -> None:
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


def test_xzy_selection() -> None:
    # select in space via x,y,z coordinates test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    das_xzy = ds["Temperature"].sel(x=348946, y=6173673, z=0)

    # check for point geometry after selection
    assert type(das_xzy.geometry) is mikeio.spatial.GeometryPoint3D
    assert das_xzy.values[0] == pytest.approx(17.381)

    # do the same but go one level deeper, but finding the index first
    idx = ds.geometry.find_index(x=348946, y=6173673, z=0)
    das_idx = ds.Temperature.isel(element=idx)  # type: ignore
    assert das_idx.values[0] == pytest.approx(17.381)

    # let's try find the same point multiple times
    das_idxs = ds.geometry.find_index(
        x=[348946, 348946], y=[6173673, 6173673], z=[0, 0]
    )
    assert len(das_idxs) == 1  # only one point


def test_xzy_selection_outside_domain() -> None:
    # select in space via x,y,z coordinates test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    with pytest.raises(OutsideModelDomainError):
        ds["Temperature"].sel(x=340000, y=15.75, z=0)  # this is way outside the domain


def test_layer_selection() -> None:
    # select layer test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    das_layer = ds["Temperature"].sel(layers=0)
    # should not be layered after selection
    assert type(das_layer.geometry) is mikeio.spatial.GeometryFM2D


def test_time_selection() -> None:
    # select time test
    nt = 100
    data = []
    d = np.random.rand(nt)
    data.append(d)
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    das_t = ds["Foo"].sel(time="2000-01-05")

    assert das_t.shape == (24,)

    with pytest.raises(KeyError):
        # not in time
        ds.sel(time="1997-09-15 00:00")


def test_interp_na() -> None:
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


def test_to_dataframe() -> None:
    time = pd.date_range("2000", periods=5, freq="D")
    da = mikeio.DataArray(
        data=np.ones(5),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    df = da.to_dataframe()
    assert df.shape == (5, 1)
    assert df["Foo"].values[0] == 1.0
    assert df.index[-1].day == 5


def test_to_pandas() -> None:
    time = pd.date_range("2000", periods=5, freq="D")
    da = mikeio.DataArray(
        data=np.ones(5),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    series = da.to_pandas()
    assert series.shape == (5,)
    assert series.index[-1].day == 5
    assert series.values[0] == 1.0
    assert series.name == "Foo"


def test_set_by_mask() -> None:
    fn = "tests/testdata/oresundHD_run1.dfsu"
    da = mikeio.read(fn, items="Surface elevation", time=[0, 2, 4])[0]
    threshold = 0.2
    mask = da < threshold
    wl_capped = da.copy()
    wl_capped[mask] = np.nan  # type: ignore


def test_set_unit() -> None:
    da = mikeio.DataArray(
        data=np.array([0.0, 1.0]),
        item=mikeio.ItemInfo("Water", mikeio.EUMType.Water_Level, mikeio.EUMUnit.meter),
    )

    da.unit = mikeio.EUMUnit.feet

    assert da.unit == mikeio.EUMUnit.feet


def test_set_bad_unit_fails() -> None:
    da = mikeio.DataArray(
        data=np.array([0.0, 1.0]),
        item=mikeio.ItemInfo("Water", mikeio.EUMType.Water_Level, mikeio.EUMUnit.meter),
    )

    with pytest.raises(ValueError, match="unit"):
        da.unit = mikeio.EUMUnit.decibar


def test_create_dataarray_with_name() -> None:
    da = mikeio.DataArray(
        data=[0, 1], time=pd.date_range("2000", periods=2), name="Level"
    )
    assert da.name == "Level"


def test_create_dataarray_with_name_and_type() -> None:
    da = mikeio.DataArray(
        data=[0, 1],
        time=pd.date_range("2000", periods=2),
        name="Level",
        type=mikeio.EUMType.Water_Level,
    )
    assert da.name == "Level"
    assert da.type == mikeio.EUMType.Water_Level


def test_create_dataarray_with_name_type_and_unit() -> None:
    da = mikeio.DataArray(
        data=[0, 1],
        time=pd.date_range("2000", periods=2),
        name="Level",
        type=mikeio.EUMType.Water_Level,
        unit=mikeio.EUMUnit.feet,
    )
    assert da.name == "Level"
    assert da.type == mikeio.EUMType.Water_Level
    assert da.unit == mikeio.EUMUnit.feet


def test_create_dataarray_with_type_can_not_be_passed_along_with_item() -> None:
    with pytest.raises(ValueError, match="item"):
        mikeio.DataArray(
            data=[0, 1],
            time=pd.date_range("2000", periods=2),
            name="Level",
            type=mikeio.EUMType.Water_Level,
            item=mikeio.ItemInfo(mikeio.EUMType.Discharge),
        )


def test_dataarray_to_dataset() -> None:
    ds = mikeio.DataArray(
        data=[0, 1],
        name="Level",
    ).to_dataset()
    da = ds["Level"]
    assert da.name == "Level"


# ===============================================================================================================
# TODO the tests private methods, Options: 1. declare methods as public, 2. Test at a higher level of abstraction


def test_parse_time_None() -> None:
    time = mikeio.DataArray._parse_time(None)
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_constant_str() -> None:
    time = mikeio.DataArray._parse_time("2018")
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_constant_datetime() -> None:
    time = mikeio.DataArray._parse_time(datetime(2018, 1, 1))
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_constant_Timestamp() -> None:
    time = mikeio.DataArray._parse_time(pd.Timestamp(2018, 1, 1))
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_list_str() -> None:
    time = mikeio.DataArray._parse_time(["2018", "2018-1-2", "2018-1-3"])
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 3
    assert time[-1] == pd.Timestamp(2018, 1, 3)


def test_parse_time_list_datetime() -> None:
    time = mikeio.DataArray._parse_time(
        [datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3)]
    )
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 3
    assert time[-1] == pd.Timestamp(2018, 1, 3)


def test_parse_time_list_Timestamp() -> None:
    time = mikeio.DataArray._parse_time(
        [pd.Timestamp(2018, 1, 1), pd.Timestamp(2018, 1, 2), pd.Timestamp(2018, 1, 3)]
    )
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 3
    assert time[-1] == pd.Timestamp(2018, 1, 3)


def test_parse_time_decreasing() -> None:
    times = [
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 1, 15),
    ]

    with pytest.raises(ValueError, match="must be monotonic increasing"):
        mikeio.DataArray._parse_time(times)


# ===============================================================================================================
