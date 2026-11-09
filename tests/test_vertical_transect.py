"""Tests for DataArray/Dataset.extract_vertical (vertical transect extraction)."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from mikecore.DfsuFile import DfsuFileType

import mikeio
from mikeio import DataArray, Dataset
from mikeio.spatial import GeometryFM3D, GeometryFMVerticalProfile, Grid2D

SIGMA_Z = "tests/testdata/oresund_sigma_z.dfsu"
# MIKE Zero's own vertical transect of SIGMA_Z (straight line, same items/time)
MIKE_SLICE = "tests/testdata/oresund_vertical_slice.dfsu"
SIGMA = "tests/testdata/basin_3d.dfsu"
DFSU_2D = "tests/testdata/oresundHD_run1.dfsu"


def _mid_transect(geometry: GeometryFM3D) -> tuple[list[float], list[float]]:
    ec = geometry.element_coordinates
    xmin, xmax = ec[:, 0].min(), ec[:, 0].max()
    ymid = 0.5 * (ec[:, 1].min() + ec[:, 1].max())
    dx = 0.05 * (xmax - xmin)
    return [float(xmin + dx), float(xmax - dx)], [float(ymid), float(ymid)]


@pytest.fixture
def ds3d() -> Dataset:
    return mikeio.read(SIGMA_Z)


@pytest.fixture
def da3d(ds3d: Dataset) -> DataArray:
    return ds3d[0]


@pytest.fixture
def mike_slice() -> Dataset:
    return mikeio.read(MIKE_SLICE)


@pytest.fixture
def mike_line(mike_slice: Dataset) -> tuple[list[float], list[float]]:
    """End points of the MIKE transect (first and last node)."""
    nc = mike_slice.geometry.node_coordinates
    return [float(nc[0, 0]), float(nc[-1, 0])], [float(nc[0, 1]), float(nc[-1, 1])]


# ---------------------------------------------------------------------------
# mode="interpolate"
# ---------------------------------------------------------------------------
def test_extract_vertical_interpolate_returns_grid(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    section = da3d.extract_vertical(xs=xs, ys=ys, n_horizontal=20, n_vertical=15)
    assert isinstance(section.geometry, Grid2D)
    assert section.geometry.is_vertical
    assert section.shape[-2:] == (15, 20)
    assert section.n_timesteps == da3d.n_timesteps


def test_extract_vertical_interpolate_no_warnings(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        da3d.extract_vertical(xs=xs, ys=ys)


def test_extract_vertical_single_timestep(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    da0 = da3d.isel(time=0)
    section = da0.extract_vertical(xs=xs, ys=ys, n_horizontal=10, n_vertical=8)
    assert section.shape[-2:] == (8, 10)


def test_extract_vertical_dataset(ds3d: Dataset) -> None:
    xs, ys = _mid_transect(ds3d.geometry)
    section = ds3d.extract_vertical(xs=xs, ys=ys)
    assert isinstance(section, mikeio.Dataset)
    assert section.n_items == ds3d.n_items


# ---------------------------------------------------------------------------
# mode="discrete" against MIKE's own transect
# ---------------------------------------------------------------------------
def test_extract_vertical_discrete_matches_mike(
    ds3d: Dataset, mike_slice: Dataset, mike_line: tuple[list[float], list[float]]
) -> None:
    xs, ys = mike_line
    section = ds3d.extract_vertical(xs=xs, ys=ys, mode="discrete")
    g, gref = section.geometry, mike_slice.geometry
    assert isinstance(g, GeometryFMVerticalProfile)

    assert g.n_elements == gref.n_elements == 441
    assert g.n_nodes == gref.n_nodes == 550
    assert g.n_layers == gref.n_layers == 9
    assert g.n_sigma_layers == gref.n_sigma_layers == 4
    assert len(g.element_table) == len(gref.element_table)
    for e, eref in zip(g.element_table, gref.element_table):
        np.testing.assert_array_equal(e, eref)

    np.testing.assert_allclose(
        g.node_coordinates[:, :2], gref.node_coordinates[:, :2], atol=1e-6
    )
    # static z is the undisturbed level, not the first time step
    np.testing.assert_allclose(
        g.node_coordinates[:, 2], gref.node_coordinates[:, 2], atol=1e-4
    )
    for item in mike_slice.items:
        np.testing.assert_array_equal(
            section[item.name].to_numpy(), mike_slice[item.name].to_numpy()
        )
        assert section[item.name]._zn is not None
        np.testing.assert_allclose(
            section[item.name]._zn, mike_slice[item.name]._zn, atol=1e-4
        )


def test_extract_vertical_discrete_dataarray_matches_dataset(
    ds3d: Dataset, mike_line: tuple[list[float], list[float]]
) -> None:
    xs, ys = mike_line
    da = ds3d[0].extract_vertical(xs=xs, ys=ys, mode="discrete")
    ds = ds3d.extract_vertical(xs=xs, ys=ys, mode="discrete")
    np.testing.assert_array_equal(da.to_numpy(), ds[0].to_numpy())


# ---------------------------------------------------------------------------
# mode="discrete": layer selection (MIKE IO convention, 0 = bottom, -1 = top)
# ---------------------------------------------------------------------------
def test_extract_vertical_discrete_top_layers(
    da3d: DataArray, mike_line: tuple[list[float], list[float]]
) -> None:
    xs, ys = mike_line
    neg = da3d.extract_vertical(
        xs=xs, ys=ys, mode="discrete", layer_min=-3, layer_max=-1
    )
    pos = da3d.extract_vertical(xs=xs, ys=ys, mode="discrete", layer_min=6, layer_max=8)
    assert neg.geometry.n_elements == pos.geometry.n_elements == 297
    assert neg.geometry.n_layers == 3
    assert neg.geometry.n_sigma_layers == 3
    assert neg.geometry._type == DfsuFileType.DfsuVerticalProfileSigma
    np.testing.assert_array_equal(neg.to_numpy(), pos.to_numpy())


def test_extract_vertical_discrete_bottom_layers_are_global(
    da3d: DataArray, mike_line: tuple[list[float], list[float]]
) -> None:
    # layers are surface-aligned: layers 0-2 only exist in the deepest column
    xs, ys = mike_line
    section = da3d.extract_vertical(
        xs=xs, ys=ys, mode="discrete", layer_min=0, layer_max=2
    )
    assert section.geometry.n_elements == 1
    assert section.geometry.n_sigma_layers == 0


def test_extract_vertical_discrete_single_layer(
    da3d: DataArray, mike_line: tuple[list[float], list[float]]
) -> None:
    xs, ys = mike_line
    section = da3d.extract_vertical(
        xs=xs, ys=ys, mode="discrete", layer_min=4, layer_max=4
    )
    assert section.geometry.n_elements == 30
    assert section.geometry.n_layers == 1


def test_extract_vertical_discrete_defaults_to_all_layers(
    da3d: DataArray, mike_line: tuple[list[float], list[float]]
) -> None:
    xs, ys = mike_line
    default = da3d.extract_vertical(xs=xs, ys=ys, mode="discrete")
    explicit = da3d.extract_vertical(
        xs=xs, ys=ys, mode="discrete", layer_min=0, layer_max=-1
    )
    assert default.geometry.n_elements == explicit.geometry.n_elements == 441


@pytest.mark.parametrize(
    "layer_min, layer_max",
    [(-10, -1), (0, 9), (-5, 99), (5, 2), (-1, -2)],
)
def test_extract_vertical_discrete_bad_layers_raise(
    da3d: DataArray, layer_min: int, layer_max: int
) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    with pytest.raises(ValueError):
        da3d.extract_vertical(
            xs=xs, ys=ys, mode="discrete", layer_min=layer_min, layer_max=layer_max
        )


# ---------------------------------------------------------------------------
# mode="discrete": other meshes and polylines
# ---------------------------------------------------------------------------
def test_extract_vertical_discrete_pure_sigma() -> None:
    ds = mikeio.read(SIGMA)
    section = ds.extract_vertical(xs=[10.0, 990.0], ys=[50.0, 50.0], mode="discrete")
    g = section.geometry
    assert isinstance(g, GeometryFMVerticalProfile)
    assert g._type == DfsuFileType.DfsuVerticalProfileSigma
    assert g.n_layers == g.n_sigma_layers == 10
    assert g.n_elements == 430
    # every column has all 10 layers
    assert g.n_elements % 10 == 0
    src = ds[0].to_numpy()
    assert np.all(np.isin(section[0].to_numpy(), src))


def test_extract_vertical_discrete_multi_vertex_polyline(ds3d: Dataset) -> None:
    xs = [356000.0, 362000.0, 366000.0]
    ys = [6145000.0, 6160000.0, 6185000.0]
    da = ds3d[0]
    section = da.extract_vertical(xs=xs, ys=ys, mode="discrete")
    g = section.geometry
    assert g.n_elements == 368
    assert g.n_nodes == 460
    assert np.all(np.diff(g.relative_element_distance) >= 0)

    # every output element takes its values from a 3D element in the 2D column
    # that contains the output element's horizontal centre
    g3 = ds3d.geometry
    src = da.to_numpy()
    out = section.to_numpy()
    for k, el in enumerate(g.element_table):
        cx, cy = g.node_coordinates[el, :2].mean(axis=0)
        col = int(np.atleast_1d(g3.geometry2d.find_index(x=cx, y=cy))[0])
        column = g3.e2_e3_table[col]
        assert any(np.array_equal(src[:, e], out[:, k]) for e in column)


def test_extract_vertical_discrete_roundtrip(
    ds3d: Dataset, mike_line: tuple[list[float], list[float]], tmp_path: Path
) -> None:
    xs, ys = mike_line
    section = ds3d.extract_vertical(xs=xs, ys=ys, mode="discrete")
    fp = tmp_path / "vertical_transect.dfsu"
    section.to_dfs(fp)

    ds2 = mikeio.read(fp)
    assert type(ds2.geometry) is GeometryFMVerticalProfile
    assert ds2.geometry.n_elements == section.geometry.n_elements
    assert ds2.geometry.n_nodes == section.geometry.n_nodes
    # n_layers is not compared: writing a vertical profile with to_dfs does not
    # preserve it (MIKE's own oresund_vertical_slice.dfsu also re-reads with 7
    # instead of 9 layers), which is a writer issue unrelated to the extraction
    assert ds2.geometry.n_sigma_layers == section.geometry.n_sigma_layers
    assert ds2.n_timesteps == section.n_timesteps
    assert ds2.n_items == section.n_items
    np.testing.assert_allclose(
        ds2.geometry.node_coordinates, section.geometry.node_coordinates, atol=1e-4
    )
    for i in range(section.n_items):
        np.testing.assert_allclose(ds2[i].to_numpy(), section[i].to_numpy(), atol=1e-5)
        np.testing.assert_allclose(ds2[i]._zn, section[i]._zn, atol=1e-4)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def test_extract_vertical_bad_mode(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    with pytest.raises(ValueError):
        da3d.extract_vertical(xs=xs, ys=ys, mode="not-a-mode")  # type: ignore[arg-type]


def test_extract_vertical_rejects_2d() -> None:
    ds2d = mikeio.read(DFSU_2D)
    with pytest.raises(NotImplementedError):
        ds2d[0].extract_vertical(xs=[0.0, 1.0], ys=[0.0, 1.0])


def test_extract_vertical_rejects_vertical_column(da3d: DataArray) -> None:
    ec = da3d.geometry.element_coordinates
    column = da3d.sel(x=ec[0, 0], y=ec[0, 1])
    xs, ys = _mid_transect(da3d.geometry)
    with pytest.raises(NotImplementedError):
        column.extract_vertical(xs=xs, ys=ys)
