"""Tests for DataArray/Dataset.extract_vertical (vertical transect extraction)."""
import numpy as np
import pytest

import mikeio
from mikeio import DataArray
from mikeio.spatial import Grid2D, GeometryFM3D, GeometryFMVerticalProfile

SIGMA_Z = "tests/testdata/oresund_sigma_z.dfsu"
DFSU_2D = "tests/testdata/oresundHD_run1.dfsu"


def _mid_transect(geometry: GeometryFM3D) -> tuple[list[float], list[float]]:
    ec = geometry.element_coordinates
    xmin, xmax = ec[:, 0].min(), ec[:, 0].max()
    ymid = 0.5 * (ec[:, 1].min() + ec[:, 1].max())
    dx = 0.05 * (xmax - xmin)
    return [float(xmin + dx), float(xmax - dx)], [float(ymid), float(ymid)]


@pytest.fixture
def da3d() -> DataArray:
    return mikeio.read(SIGMA_Z)[0]


def test_extract_vertical_interpolate_returns_grid(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    section = da3d.extract_vertical(xs=xs, ys=ys, n_horizontal=20, n_vertical=15)
    assert isinstance(section.geometry, Grid2D)
    assert section.geometry.is_vertical
    assert section.shape[-2:] == (15, 20)
    assert section.n_timesteps == da3d.n_timesteps


def test_extract_vertical_discrete_returns_profile(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    n_layers = da3d.geometry.n_layers
    section = da3d.extract_vertical(
        xs=xs, ys=ys, mode="discrete", layer_min=1, layer_max=n_layers
    )
    assert isinstance(section.geometry, GeometryFMVerticalProfile)
    assert section.geometry.n_elements > 0
    assert section._zn is not None


def test_extract_vertical_discrete_values_are_source_elements(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    n_layers = da3d.geometry.n_layers
    section = da3d.extract_vertical(
        xs=xs, ys=ys, mode="discrete", layer_min=1, layer_max=n_layers
    )
    src_vals = np.unique(da3d.to_numpy())
    out_vals = np.unique(section.to_numpy())
    assert np.all(np.isin(out_vals, src_vals))


def test_extract_vertical_single_timestep(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    da0 = da3d.isel(time=0)
    section = da0.extract_vertical(xs=xs, ys=ys, n_horizontal=10, n_vertical=8)
    assert section.shape[-2:] == (8, 10)


def test_extract_vertical_discrete_requires_layers(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    with pytest.raises(ValueError):
        da3d.extract_vertical(xs=xs, ys=ys, mode="discrete")


def test_extract_vertical_bad_mode(da3d: DataArray) -> None:
    xs, ys = _mid_transect(da3d.geometry)
    with pytest.raises(ValueError):
        da3d.extract_vertical(xs=xs, ys=ys, mode="not-a-mode")  # type: ignore[arg-type]


def test_extract_vertical_dataset(da3d: DataArray) -> None:
    ds = mikeio.read(SIGMA_Z)
    xs, ys = _mid_transect(ds.geometry)
    section = ds.extract_vertical(xs=xs, ys=ys)
    assert isinstance(section, mikeio.Dataset)
    assert section.n_items == ds.n_items


def test_extract_vertical_rejects_2d() -> None:
    ds2d = mikeio.read(DFSU_2D)
    with pytest.raises(NotImplementedError):
        ds2d[0].extract_vertical(xs=[0.0, 1.0], ys=[0.0, 1.0])