from datetime import datetime
import numpy as np
import pandas as pd
import pytest

import mikeio
from mikeio.eum import EUMType, ItemInfo
from mikeio.spatial.geometry import GeometryPoint3D, GeometryUndefined


@pytest.fixture
def da1():
    nt = 10
    start = 10.0
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
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
        time=pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt),
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
        time=pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt),
        item=ItemInfo("Foo"),
        geometry=mikeio.Grid2D(x0=1000.0, dx=10.0, shape=(nx, ny), dy=1.0, y0=-10.0),
    )

    return da


@pytest.fixture
def da_time_space():
    nt = 10
    start = 10.0
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
    da = mikeio.DataArray(
        data=np.zeros(shape=(nt, 2), dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
        geometry=mikeio.Grid1D(n=2, dx=1.0),
    )

    return da


def test_data_2d_no_geometry_not_allowed():

    nt = 10
    nx = 7
    ny = 14

    with pytest.warns(Warning) as w:
        mikeio.DataArray(
            data=np.zeros([nt, ny, nx]) + 0.1,
            time=pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt),
            item=ItemInfo("Foo"),
        )

    assert "geometry" in str(w[0].message).lower()


def test_dataarray_init():
    nt = 10
    start = 10.0
    data = np.arange(start, start + nt, dtype=float)
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
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


def test_dataarray_init_2d():
    nt = 10
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)

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
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)

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
    time_long = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=(nt + 1))
    item = ItemInfo(name="Foo")

    with pytest.raises(ValueError):
        mikeio.DataArray(data=data, time=time_long, item=item)

    nt, ny, nx = 10, 5, 6
    data2d = np.zeros([nt, ny, nx]) + 0.1
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time_long)

    # time must be first dim
    dims = ("x", "y", "time")
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time, dims=dims)

    # time must be first dim
    data2d = np.zeros([ny, nt, nx]) + 0.1
    with pytest.raises(ValueError):
        mikeio.DataArray(data=data2d, time=time)


def test_dataarray_init_grid1d():
    nt = 10
    nx = 5
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
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
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
    data = np.zeros([nt, ny, nx]) + 0.1
    g = mikeio.Grid2D(dx=0.5, shape=(nx, ny))
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
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
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
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
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

    # indexing in space selecting a single item
    da = ds.Salinity[0, :]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered)

    # indexing in space selecting a single item
    da = ds.Salinity[:, 0]
    assert isinstance(da.geometry, GeometryPoint3D)


def test_timestep(da1):

    assert da1.timestep == 1.0


def test_dims_time(da1):

    assert da1.dims[0][0] == "t"


def test_dims_time_space1d(da_time_space):

    assert da_time_space.dims[1] == "x"


def test_repr(da_time_space):

    text = repr(da_time_space)
    assert "DataArray" in text
    assert "Dimensions: (time:10, x:2)" in text


def test_plot(da1):

    da1.plot()
    assert True


def test_modify_values(da1):

    # TODO: Now you are...
    # with pytest.raises(TypeError):
    #     da1[0] = 1.0  # you are not allowed to set individual values

    with pytest.raises(ValueError):
        da1.values = np.array([1.0])  # you can not set data to another shape

    # This is allowed
    da1.values = np.zeros_like(da1.values) + 2.0


def test_add_scalar(da1):
    da2 = da1 + 10.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == 10.0)

    da3 = 10.0 + da1
    assert isinstance(da3, mikeio.DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_subtract_scalar(da1):
    da2 = da1 - 10.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == -10.0)

    da3 = 10.0 - da1
    assert isinstance(da3, mikeio.DataArray)
    assert da3.to_numpy()[-1] == -9.0


def test_multiply_scalar(da1):
    da2 = da1 * 2.0
    assert isinstance(da2, mikeio.DataArray)
    assert np.all(da2.to_numpy() / da1.to_numpy() == 2.0)

    da3 = 2.0 * da1
    assert isinstance(da3, mikeio.DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_add_two_dataarrays(da1):

    da3 = da1 + da1
    assert isinstance(da3, mikeio.DataArray)
    assert da1.shape == da3.shape


def test_daarray_squeeze():

    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)
    da: mikeio.DataArray = ds.Elevation
    assert da.shape == (1, 264, 216)

    das = da.squeeze()
    assert das.shape == (264, 216)
    assert das.dims[0] == "y"


def test_daarray_aggregation_dfs2():

    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)
    da = ds.Elevation

    assert da.shape == (1, 264, 216)
    assert da.dims[0] == "time"

    dam = da.mean(axis="time")

    assert dam.shape == (264, 216)
    assert dam.dims[0] != "time"

    dasm = da.mean(axis="space")

    assert dasm.shape == (1,)
    assert dasm.dims[0] == "time"


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


def test_da_isel_space(da_grid2d):
    assert da_grid2d.geometry.nx == 7
    assert da_grid2d.geometry.ny == 14
    da_sel = da_grid2d.isel(0, axis="y")
    assert da_sel.dims[0][0] == "t"
    assert da_sel.dims[1] == "x"

    da_sel = da_grid2d.isel(0, axis="x")
    assert da_sel.dims[0][0] == "t"
    assert da_sel.dims[1] == "y"

    da_sel = da_grid2d.isel(0, axis="t")
    assert da_sel.dims[0] == "y"
    assert da_sel.dims[1] == "x"


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
    assert not isinstance(
        daqs.geometry, mikeio.Grid1D
    )  # it could be None, or maybe NullGeometry, but not mikeio.Grid1D
    assert isinstance(da2.geometry, mikeio.Grid1D)  # But this one is intact
    assert len(daqs.time) == 10
    assert daqs.ndim == 1
    assert daqs.dims[0][0] == "t"  # Because it's a mikeio.Grid1D, remember!

    # q as list
    # daq = da2.quantile(q=[0.25, 0.75], axis=0)
    # assert daq[0].to_numpy()[0, 0] == 0.1
    # assert daq[1].to_numpy()[0, 0] == 0.1
    # assert daq[2].to_numpy()[0, 0] == 0.2
    # assert daq[3].to_numpy()[0, 0] == 0.2

    # assert daq.n_items == 2 * da2.n_items
    # assert "Quantile 0.75, " in daq.items[1].name
    # assert "Quantile 0.25, " in daq.items[2].name
    # assert "Quantile 0.75, " in daq.items[3].name


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
