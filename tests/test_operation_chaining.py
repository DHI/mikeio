"""Tests that chaining operations keeps geometry, data, and metadata aligned."""

from pathlib import Path

import numpy as np

import mikeio


def test_isel_time_then_mean_space() -> None:
    """isel(time=0).mean(axis='space') → scalar."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]

    result = da.isel(time=0).mean(axis="space")

    assert result.shape == ()
    assert np.isscalar(result.to_numpy()) or result.to_numpy().ndim == 0


def test_sel_area_then_nanmax_time() -> None:
    """sel(area=bbox).nanmax(axis=0) → shape matches reduced geometry."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]
    bbox = (606000.0, 6903000.0, 607000.0, 6905000.0)

    subset = da.sel(area=bbox)
    result = subset.nanmax(axis=0)

    assert result.shape == (subset.geometry.n_elements,)


def test_arithmetic_then_roundtrip(tmp_path: Path) -> None:
    """Multiply by 2, write, read back → values doubled (within float32 tol)."""
    ds = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])
    da_orig = ds[0]
    da_doubled = da_orig * 2.0

    ds_out = mikeio.Dataset([da_doubled])
    fp = tmp_path / "doubled.dfsu"
    ds_out.to_dfs(fp)
    ds_back = mikeio.read(fp)

    np.testing.assert_allclose(
        ds_back[0].to_numpy(), da_orig.to_numpy() * 2.0, atol=1e-5
    )


def test_aggregate_preserves_item_info() -> None:
    """mean(axis=0) must preserve item name and EUM type."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]
    orig_name = da.name
    orig_type = da.type

    result = da.mean(axis=0)

    assert result.name == orig_name
    assert result.type == orig_type


def test_double_aggregation_space_then_time() -> None:
    """mean(axis='space').mean(axis=0) → scalar."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]

    result = da.mean(axis="space").mean(axis=0)

    assert result.shape == ()


def test_sel_time_then_sel_area() -> None:
    """Two sequential sel calls must match a single combined selection."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]
    t0 = da.time[0]
    bbox = (606000.0, 6903000.0, 607000.0, 6905000.0)

    result_chained = da.sel(time=t0).sel(area=bbox)
    result_combined = da.sel(time=t0, area=bbox)

    np.testing.assert_array_equal(result_chained.to_numpy(), result_combined.to_numpy())


def test_isel_elements_then_std() -> None:
    """isel(element=[0,1,2]).std(axis=0) → shape (3,)."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]

    result = da.isel(element=[0, 1, 2]).std(axis=0)

    assert result.shape == (3,)
