"""Tests for NaN handling in aggregation operations.

Documents and locks down behavior when data contains NaN values,
especially all-NaN slices.
"""

import numpy as np
import pandas as pd
import pytest

import mikeio


def _make_da(data: np.ndarray, nt: int | None = None) -> mikeio.DataArray:
    """Create a simple DataArray with time axis."""
    if nt is None:
        nt = data.shape[0]
    return mikeio.DataArray(
        data=data,
        time=pd.date_range("2000", periods=nt, freq="h"),
    )


def test_nanmean_all_nan_returns_nan() -> None:
    """nanmean of an all-NaN column should be NaN, not raise."""
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    da = _make_da(data)

    result = da.nanmean(axis=0)

    assert result.to_numpy()[0] == pytest.approx(2.0)
    assert np.isnan(result.to_numpy()[1])


def test_nanmax_all_nan_returns_nan() -> None:
    """nanmax of an all-NaN slice should return NaN (numpy returns -inf with warning)."""
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    da = _make_da(data)

    result = da.nanmax(axis=0)

    assert result.to_numpy()[0] == pytest.approx(3.0)
    # numpy.nanmax on all-NaN returns nan (with RuntimeWarning suppressed)
    assert np.isnan(result.to_numpy()[1])


def test_nanmin_all_nan_returns_nan() -> None:
    """nanmin of an all-NaN slice should return NaN (numpy returns +inf with warning)."""
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    da = _make_da(data)

    result = da.nanmin(axis=0)

    assert result.to_numpy()[0] == pytest.approx(1.0)
    # numpy.nanmin on all-NaN returns nan (with RuntimeWarning suppressed)
    assert np.isnan(result.to_numpy()[1])


def test_mean_propagates_nan() -> None:
    """mean (not nanmean) propagates NaN: one NaN makes column result NaN."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0]])
    da = _make_da(data)

    result = da.mean(axis=0)

    assert np.isnan(result.to_numpy()[0]), "NaN should propagate through mean"
    assert result.to_numpy()[1] == pytest.approx(4.0)


def test_std_with_single_value_per_column() -> None:
    """std of a single-row DataArray should be 0, not NaN or error."""
    data = np.array([[5.0, 10.0]])
    da = _make_da(data, nt=1)

    result = da.std(axis=0)

    np.testing.assert_allclose(result.to_numpy(), 0.0, atol=1e-15)


def test_aggregation_entirely_nan_array() -> None:
    """All aggregation methods on 100% NaN data should return valid shapes."""
    data = np.full((5, 3), np.nan)
    da = _make_da(data)

    for method_name in ("nanmean", "nanmax", "nanmin", "nanstd"):
        method = getattr(da, method_name)
        result = method(axis=0)
        assert result.shape == (3,), f"{method_name} shape wrong"
        assert all(np.isnan(result.to_numpy())), f"{method_name} should be all NaN"


def test_aggregation_with_nan_dfsu() -> None:
    """nanmean(axis='space') ignores injected NaN elements correctly."""
    ds = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])
    da = ds[0]

    # Inject NaN at specific elements
    data = da.to_numpy().copy()
    data[:, 0] = np.nan
    data[:, 100] = np.nan

    da_nan = mikeio.DataArray(
        data=data,
        time=da.time,
        geometry=da.geometry,
        item=da.item,
    )

    result = da_nan.nanmean(axis="space")
    assert result.shape == (da.n_timesteps,)
    assert not np.any(
        np.isnan(result.to_numpy())
    ), "nanmean over space should ignore NaN elements"
