import os
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

import mikeio
from mikeio import DataArray, Dataset, Dfsu, Dfs2, Dfs0
from mikeio.eum import EUMType, ItemInfo, EUMUnit


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


def test_dataarray_indexing(da1: DataArray):

    assert da1.shape == (10,)
    subset = da1[3]
    assert isinstance(subset, DataArray)
    assert da1.shape == (10,)
    assert subset.to_numpy() == np.array([13.0])


def test_timestep(da1):

    assert da1.timestep == 1.0


def test_repr(da1):

    text = repr(da1)
    assert "DataArray" in text


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
