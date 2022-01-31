import os
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

import mikeio
from mikeio import DataArray, Dataset, Dfsu, Dfs2, Dfs0
from mikeio.eum import EUMType, ItemInfo, EUMUnit
from mikeio.spatial.grid_geometry import Grid1D


@pytest.fixture
def da1():
    nt = 10
    start = 10.0
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
    da = DataArray(
        data=np.arange(start, start + nt, dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    return da


@pytest.fixture
def da2():
    nt = 10
    nx = 7

    da = DataArray(
        data=np.zeros([nt, nx]) + 0.1,
        time=pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt),
        item=ItemInfo("Foo"),
        geometry=Grid1D(x0=1000.0, dx=10.0, n=nx),
    )

    return da


@pytest.fixture
def da_time_space():
    nt = 10
    start = 10.0
    time = pd.date_range(start=datetime(2000, 1, 1), freq="S", periods=nt)
    da = DataArray(
        data=np.zeros(shape=(nt, 2), dtype=float),
        time=time,
        item=ItemInfo(name="Foo"),
    )

    return da


def test_dataarray_indexing(da1: DataArray):

    assert da1.shape == (10,)
    subset = da1[3]
    assert isinstance(subset, DataArray)
    assert da1.shape == (10,)
    assert subset.to_numpy() == np.array([13.0])


def test_dataarray_dfsu3d_indexing():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfsu = Dfsu(filename)
    ds = dfsu.read()
    assert isinstance(
        ds.Salinity.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered
    )

    # indexing in space selecting a single item
    da = ds.Salinity[0, :]
    assert isinstance(da.geometry, mikeio.spatial.FM_geometry.GeometryFMLayered)

    # indexing in space selecting a single item
    da = ds.Salinity[:, 0]
    assert da.geometry is None


def test_timestep(da1):

    assert da1.timestep == 1.0


def test_dims_time(da1):

    assert da1.dims == ("t",)


def test_dims_time_space1d(da_time_space):

    assert da_time_space.dims == ("t", "x")


def test_repr(da_time_space):

    text = repr(da_time_space)
    assert "DataArray" in text
    assert "Dimensions: (t:10, x:2)" in text


def test_plot(da1):

    da1.plot()
    assert True


def test_modify_values(da1):

    with pytest.raises(TypeError):
        da1[0] = 1.0  # you are not allowed to set individual values

    with pytest.raises(ValueError):
        da1.values = np.array([1.0])  # you can not set data to another shape

    # This is allowed
    da1.values = np.zeros_like(da1.values) + 2.0


def test_add_scalar(da1):
    da2 = da1 + 10.0
    assert isinstance(da2, DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == 10.0)

    da3 = 10.0 + da1
    assert isinstance(da3, DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_subtract_scalar(da1):
    da2 = da1 - 10.0
    assert isinstance(da2, DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == -10.0)

    da3 = 10.0 - da1
    assert isinstance(da3, DataArray)
    assert da3.to_numpy()[-1] == -9.0


def test_multiply_scalar(da1):
    da2 = da1 * 2.0
    assert isinstance(da2, DataArray)
    assert np.all(da2.to_numpy() / da1.to_numpy() == 2.0)

    da3 = 2.0 * da1
    assert isinstance(da3, DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


def test_add_two_dataarrays(da1):

    da3 = da1 + da1
    assert isinstance(da3, DataArray)
    assert da1.shape == da3.shape


def test_daarray_aggregation():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[3])
    da = ds["Current speed"]
    da_max = da.max()
    assert isinstance(da_max, DataArray)
    assert da_max.geometry == da.geometry
    assert da_max.start_time == da.start_time  # TODO is this consistent
    assert len(da_max.time) == 1
    # TODO verify values

    da_min = da.min()
    assert isinstance(da_max, DataArray)
    assert da_min.geometry == da.geometry
    assert da_min.start_time == da.start_time  # TODO is this consistent
    assert len(da_min.time) == 1
    # TODO verify values

    da_mean = da.mean()
    assert isinstance(da_mean, DataArray)
    assert da_mean.geometry == da.geometry
    assert da_mean.start_time == da.start_time  # TODO is this consistent
    assert len(da_mean.time) == 1
    # TODO verify values


def test_daarray_aggregation_nan_versions():

    # TODO find better file, e.g. with flood/dry
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[3])
    da = ds["Current speed"]
    da_max = da.nanmax()
    assert isinstance(da_max, DataArray)
    assert da_max.geometry == da.geometry
    assert da_max.start_time == da.start_time  # TODO is this consistent
    assert len(da_max.time) == 1

    da_min = da.nanmin()
    assert isinstance(da_max, DataArray)
    assert da_min.geometry == da.geometry
    assert da_min.start_time == da.start_time  # TODO is this consistent
    assert len(da_min.time) == 1

    da_mean = da.nanmean()
    assert isinstance(da_mean, DataArray)
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
    assert not isinstance(
        daqs.geometry, Grid1D
    )  # it could be None, or maybe NullGeometry, but not Grid1D
    assert isinstance(da2.geometry, Grid1D)  # But this one is intact
    assert len(daqs.time) == 10
    assert daqs.ndim == 1
    assert daqs.dims[0] == "t"  # Because it's a Grid1D, remember!

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
