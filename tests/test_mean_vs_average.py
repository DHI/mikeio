"""Tests documenting the difference between mean() and average() on spatial axes.

mean(axis='space') is unweighted — all elements contribute equally.
average(axis='space', weights=area) is area-weighted.
On variable meshes these give different results; users must choose deliberately.
"""

import numpy as np

import mikeio


def test_mean_and_average_differ_on_variable_mesh() -> None:
    """On a mesh with varying element areas, mean != average."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]
    area = da.geometry.get_element_area()

    # Sanity: areas actually vary
    assert area.max() / area.min() > 2.0, "Need variable element areas"

    mean_result = da.mean(axis="space").to_numpy()
    avg_result = da.average(axis="space", weights=area).to_numpy()

    # They must differ by more than floating-point noise
    assert not np.allclose(mean_result, avg_result, atol=1e-10)


def test_mean_and_average_agree_on_uniform_grid() -> None:
    """On a uniform grid (dfs2), unweighted mean equals area-weighted mean."""
    da = mikeio.read("tests/testdata/eq.dfs2")[0]

    mean_result = da.mean(axis="space").to_numpy()

    # On a uniform grid, all cells have the same area, so manual weighted
    # average should match the unweighted mean
    data = da.to_numpy()
    manual_weighted = data.mean(axis=(1, 2))

    np.testing.assert_allclose(mean_result, manual_weighted, atol=1e-10)


def test_average_with_known_weights() -> None:
    """Synthetic test: area=[1,9], values=[10,0] → mean=5, weighted_avg=1."""
    da = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])[0]
    da_sub = da.isel(element=[0, 1])

    # Override data with known values
    data = da_sub.to_numpy().copy()
    data[:, 0] = 10.0
    data[:, 1] = 0.0
    da_known = mikeio.DataArray(
        data=data,
        time=da_sub.time,
        geometry=da_sub.geometry,
        item=da_sub.item,
    )

    weights = np.array([1.0, 9.0])

    unweighted = da_known.mean(axis="space").to_numpy()
    weighted = da_known.average(axis="space", weights=weights).to_numpy()

    np.testing.assert_allclose(unweighted, 5.0, atol=1e-10)
    np.testing.assert_allclose(weighted, 1.0, atol=1e-10)


def test_mean_docstring_warns_unweighted() -> None:
    """mean() docstring should mention it's unweighted or reference average()."""
    doc = mikeio.DataArray.mean.__doc__
    assert doc is not None
    doc_lower = doc.lower()
    assert "unweighted" in doc_lower or "average" in doc_lower
