"""Roundtrip tests: read → write → read must preserve data and metadata."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mikeio


def test_roundtrip_dfs0_data(tmp_path: Path) -> None:
    """Data, item names, EUM types, and time axis survive dfs0 roundtrip."""
    ds_orig = mikeio.read("tests/testdata/random.dfs0")

    fp = tmp_path / "roundtrip.dfs0"
    ds_orig.to_dfs(fp)
    ds_back = mikeio.read(fp)

    assert ds_back.n_items == ds_orig.n_items
    assert ds_back.n_timesteps == ds_orig.n_timesteps
    for i in range(ds_orig.n_items):
        np.testing.assert_allclose(
            ds_back[i].to_numpy(), ds_orig[i].to_numpy(), atol=1e-10
        )
        assert ds_back[i].name == ds_orig[i].name
        assert ds_back[i].type == ds_orig[i].type
    np.testing.assert_array_equal(ds_back.time, ds_orig.time)


def test_roundtrip_dfs1_data(tmp_path: Path) -> None:
    """Data and shape survive dfs1 roundtrip (float32 tolerance)."""
    ds_orig = mikeio.read("tests/testdata/random.dfs1")

    fp = tmp_path / "roundtrip.dfs1"
    ds_orig.to_dfs(fp)
    ds_back = mikeio.read(fp)

    assert ds_back.n_items == ds_orig.n_items
    assert ds_back.shape == ds_orig.shape
    for i in range(ds_orig.n_items):
        np.testing.assert_allclose(
            ds_back[i].to_numpy(), ds_orig[i].to_numpy(), atol=1e-5
        )


def test_roundtrip_dfs2_data(tmp_path: Path) -> None:
    """Data, shape, and geometry survive dfs2 roundtrip."""
    ds_orig = mikeio.read("tests/testdata/random.dfs2")

    fp = tmp_path / "roundtrip.dfs2"
    ds_orig.to_dfs(fp)
    ds_back = mikeio.read(fp)

    assert ds_back.n_items == ds_orig.n_items
    assert ds_back.shape == ds_orig.shape
    g_orig = ds_orig.geometry
    g_back = ds_back.geometry
    assert g_back.nx == g_orig.nx
    assert g_back.ny == g_orig.ny
    assert g_back.dx == pytest.approx(g_orig.dx)
    assert g_back.dy == pytest.approx(g_orig.dy)
    assert g_back.origin == pytest.approx(g_orig.origin)
    assert g_back.projection_string == g_orig.projection_string
    for i in range(ds_orig.n_items):
        np.testing.assert_allclose(
            ds_back[i].to_numpy(), ds_orig[i].to_numpy(), atol=1e-5
        )


def test_roundtrip_dfsu_data(tmp_path: Path) -> None:
    """Data, element/node counts survive dfsu roundtrip."""
    ds_orig = mikeio.read("tests/testdata/HD2D.dfsu")

    fp = tmp_path / "roundtrip.dfsu"
    ds_orig.to_dfs(fp)
    ds_back = mikeio.read(fp)

    assert ds_back.n_items == ds_orig.n_items
    assert ds_back.n_timesteps == ds_orig.n_timesteps
    assert ds_back.geometry.n_elements == ds_orig.geometry.n_elements
    assert ds_back.geometry.n_nodes == ds_orig.geometry.n_nodes
    for i in range(ds_orig.n_items):
        np.testing.assert_allclose(
            ds_back[i].to_numpy(), ds_orig[i].to_numpy(), atol=1e-5
        )


def test_roundtrip_dfs0_nan_preserved(tmp_path: Path) -> None:
    """NaN positions must be identical after NaN → deletevalue → NaN cycle."""
    nt = 50
    data = np.random.default_rng(42).random(nt)
    nan_positions = [3, 10, 22, 49]
    data[nan_positions] = np.nan

    da = mikeio.DataArray(
        data=data,
        time=pd.date_range("2000", periods=nt, freq="h"),
    )
    ds = mikeio.Dataset([da])

    fp = tmp_path / "nan_test.dfs0"
    ds.to_dfs(fp)
    ds_back = mikeio.read(fp)

    orig_nans = np.isnan(ds[0].to_numpy())
    back_nans = np.isnan(ds_back[0].to_numpy())
    np.testing.assert_array_equal(orig_nans, back_nans)


def test_roundtrip_dfs2_nan_preserved(tmp_path: Path) -> None:
    """NaN mask is identical after dfs2 roundtrip (float32 path)."""
    ds_orig = mikeio.read("tests/testdata/random.dfs2")

    # Inject NaNs at known positions
    data = ds_orig[0].to_numpy().copy()
    data[0, 0, 0] = np.nan
    data[0, 5, 1] = np.nan

    da = mikeio.DataArray(
        data=data,
        time=ds_orig.time,
        geometry=ds_orig.geometry,
        item=ds_orig[0].item,
    )
    ds = mikeio.Dataset([da])

    fp = tmp_path / "nan_test.dfs2"
    ds.to_dfs(fp)
    ds_back = mikeio.read(fp)

    orig_mask = np.isnan(ds[0].to_numpy())
    back_mask = np.isnan(ds_back[0].to_numpy())
    np.testing.assert_array_equal(orig_mask, back_mask)


def test_roundtrip_float64_to_float32_documents_precision(tmp_path: Path) -> None:
    """Writing float64 data to dfs2 truncates to float32 precision."""
    # Value with more precision than float32 can represent
    value_f64 = 1.23456789012345
    value_f32 = np.float32(value_f64)
    assert value_f64 != float(value_f32), "Test setup: values must differ"

    ds_orig = mikeio.read("tests/testdata/random.dfs2")
    data = np.full_like(ds_orig[0].to_numpy(), value_f64, dtype=np.float64)

    da = mikeio.DataArray(
        data=data,
        time=ds_orig.time,
        geometry=ds_orig.geometry,
        item=ds_orig[0].item,
    )
    ds = mikeio.Dataset([da])

    fp = tmp_path / "precision.dfs2"
    ds.to_dfs(fp)
    ds_back = mikeio.read(fp)

    # Read-back should match float32-rounded value, not original float64
    np.testing.assert_allclose(ds_back[0].to_numpy(), float(value_f32), atol=1e-10)


def test_roundtrip_dfsu_multi_item(tmp_path: Path) -> None:
    """All items survive dfsu roundtrip with correct ordering."""
    ds_orig = mikeio.read("tests/testdata/HD2D.dfsu")
    assert ds_orig.n_items > 1, "Test setup: need multiple items"

    fp = tmp_path / "multi_item.dfsu"
    ds_orig.to_dfs(fp)
    ds_back = mikeio.read(fp)

    assert ds_back.n_items == ds_orig.n_items
    for i in range(ds_orig.n_items):
        assert ds_back[i].name == ds_orig[i].name
        np.testing.assert_allclose(
            ds_back[i].to_numpy(), ds_orig[i].to_numpy(), atol=1e-5
        )
