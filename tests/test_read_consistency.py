"""Tests that different code paths to the same data produce identical results.

read(area=...) filters at C level; read().sel(area=...) filters in Python.
read(items=[0]) vs read()[0] uses different code paths. These must agree.
"""

import numpy as np

import mikeio


def test_dfsu_read_area_vs_sel_area() -> None:
    """read(area=bbox) must match read().sel(area=bbox) for dfsu."""
    bbox = [-0.3, 0.0, 0.3, 0.3]

    ds_direct = mikeio.read("tests/testdata/FakeLake.dfsu", area=bbox)
    ds_sel = mikeio.read("tests/testdata/FakeLake.dfsu").sel(area=bbox)

    assert ds_direct.shape == ds_sel.shape
    assert ds_direct.geometry.n_elements == ds_sel.geometry.n_elements
    for i in range(ds_direct.n_items):
        np.testing.assert_array_equal(ds_direct[i].to_numpy(), ds_sel[i].to_numpy())


def test_dfs2_read_area_vs_sel_area() -> None:
    """read(area=bbox) must match read().sel(area=bbox) for dfs2."""
    bbox = [5.0, 3.0, 15.0, 8.0]

    ds_direct = mikeio.read("tests/testdata/eq.dfs2", area=bbox)
    ds_sel = mikeio.read("tests/testdata/eq.dfs2").sel(area=bbox)

    assert ds_direct.shape == ds_sel.shape
    for i in range(ds_direct.n_items):
        np.testing.assert_array_equal(ds_direct[i].to_numpy(), ds_sel[i].to_numpy())


def test_read_items_by_index_vs_getitem() -> None:
    """read(items=[0]) must match read()[0] for dfsu and dfs2."""
    for path in ("tests/testdata/HD2D.dfsu", "tests/testdata/eq.dfs2"):
        ds_item = mikeio.read(path, items=[0])
        ds_full = mikeio.read(path)

        np.testing.assert_array_equal(ds_item[0].to_numpy(), ds_full[0].to_numpy())
        assert ds_item[0].name == ds_full[0].name


def test_read_items_by_name_vs_by_index() -> None:
    """read(items=['name']) must match read(items=[0]) when item 0 has that name."""
    ds_by_idx = mikeio.read("tests/testdata/HD2D.dfsu", items=[0])
    item_name = ds_by_idx[0].name
    ds_by_name = mikeio.read("tests/testdata/HD2D.dfsu", items=[item_name])

    np.testing.assert_array_equal(ds_by_idx[0].to_numpy(), ds_by_name[0].to_numpy())


def test_read_time_slice_vs_isel() -> None:
    """read(time=[0,1]) must match read().isel(time=[0,1])."""
    ds_direct = mikeio.read("tests/testdata/eq.dfs2", time=[0, 1])
    ds_isel = mikeio.read("tests/testdata/eq.dfs2").isel(time=[0, 1])

    assert ds_direct.shape == ds_isel.shape
    for i in range(ds_direct.n_items):
        np.testing.assert_array_equal(ds_direct[i].to_numpy(), ds_isel[i].to_numpy())
    np.testing.assert_array_equal(ds_direct.time, ds_isel.time)


def test_dfs2_partial_read_geometry_matches() -> None:
    """read(area=...) geometry must have correct origin, dx, dy, nx, ny."""
    bbox = [5.0, 3.0, 15.0, 8.0]
    ds = mikeio.read("tests/testdata/eq.dfs2", area=bbox)
    g = ds.geometry

    # Geometry must reflect the subset, not the full grid
    assert g.nx == 11
    assert g.ny == 6
    assert g.dx == 1.0
    assert g.dy == 1.0
    # Origin should be at the bbox lower-left (snapped to grid)
    assert g.origin[0] >= bbox[0] - g.dx
    assert g.origin[1] >= bbox[1] - g.dy
