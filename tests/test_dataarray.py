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
        data=np.arange(start, start + nt, dtype=float), time=time, item="Foo"
    )

    return da


def test_dataarray_indexing(da1: DataArray):

    subset = da1[3]
    assert isinstance(subset, DataArray)

    assert subset.to_numpy() == np.array([13.0])


def test_add_scalar(da1):
    da2 = da1 + 10.0
    assert isinstance(da2, DataArray)
    assert np.all(da2.to_numpy() - da1.to_numpy() == 10.0)

    da3 = 10.0 + da1
    assert isinstance(da3, DataArray)
    assert np.all(da3.to_numpy() == da2.to_numpy())


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
