import numpy as np
import pytest
import matplotlib.pyplot as plt

import mikeio
from mikeio.spatial.FM_geometry import GeometryFMVerticalProfile
from mikeio.spatial.geometry import GeometryPoint3D
from mikecore.DfsuFile import DfsuFileType


@pytest.fixture
def vslice():
    # sigma, z vertical profile (=transect)
    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def vslice_geo():
    # sigma, z vertical profile (=transect)
    # in LONG/LAT, non-straight
    filename = "tests/testdata/kalundborg_transect.dfsu"
    return mikeio.open(filename)


def test_open_transect(vslice):
    dfs = vslice
    assert dfs._type == DfsuFileType.DfsuVerticalProfileSigmaZ
    assert dfs.n_items == 2
    assert dfs.n_elements == 441
    assert dfs.n_sigma_layers == 4
    assert dfs.n_z_layers == 5
    assert dfs.n_timesteps == 3


def test_transect_geometry_properties(vslice):
    g = vslice.geometry
    assert isinstance(g, GeometryFMVerticalProfile)
    assert len(g.top_elements) == 99
    assert len(g.bottom_elements) == 99
    assert len(g.e2_e3_table) == 99
    assert len(g.relative_element_distance) == 441
    d1 = g.get_nearest_relative_distance([3.55e05, 6.145e06])
    assert d1 == pytest.approx(5462.3273)

    with pytest.raises(AttributeError, match="no boundary_polylines property"):
        g.boundary_polylines


def test_transect_geometry_properties_geo(vslice_geo):
    g = vslice_geo.geometry
    ec = g.element_coordinates
    x, y = ec[:, 0], ec[:, 1]
    d0 = np.hypot(x - x[0], y - y[0])  # relative, in degrees
    assert d0.max() < 2

    d = g.relative_element_distance  # in meters and cummulative
    assert d.max() > 38000

    d1 = g.get_nearest_relative_distance([10.77, 55.62])
    assert d1 == pytest.approx(25673.318)


def test_read_transect(vslice):
    ds = vslice.read()
    assert ds.geometry._type == DfsuFileType.DfsuVerticalProfileSigmaZ
    assert type(ds.geometry) == GeometryFMVerticalProfile
    assert ds.n_items == 2
    assert ds.Salinity.name == "Salinity"
    assert ds.geometry.n_elements == 441


def test_getitem_time_transect(vslice):
    da = vslice.read()[1]
    assert da[-1].time[0] == da.time[2]


def test_getitem_element_transect(vslice):
    da = vslice.read().Salinity
    idx = 5
    da2 = da[:, idx]
    assert type(da2.geometry) == GeometryPoint3D
    assert da2.geometry.x == da.geometry.element_coordinates[idx, 0]
    assert da2.geometry.y == da.geometry.element_coordinates[idx, 1]
    assert da2.geometry.z == da.geometry.element_coordinates[idx, 2]
    assert da2.shape == (3,)
    assert np.all(da2.to_numpy() == da.to_numpy()[:, idx])


def test_isel_transect(vslice):
    da = vslice.read().Salinity
    idx = 5
    da2 = da.isel(element=idx)
    assert type(da2.geometry) == GeometryPoint3D
    assert da2.geometry.x == da.geometry.element_coordinates[idx, 0]
    assert da2.geometry.y == da.geometry.element_coordinates[idx, 1]
    assert da2.geometry.z == da.geometry.element_coordinates[idx, 2]
    assert da2.shape == (3,)
    assert np.all(da2.to_numpy() == da.to_numpy()[:, idx])


def test_sel_time_transect(vslice):
    da = vslice.read()[0]
    idx = 1
    da2 = da.sel(time="1997-09-16 00:00")
    assert type(da2.geometry) == GeometryFMVerticalProfile
    assert np.all(da2.to_numpy() == da.to_numpy()[idx, :])
    assert da2.time[0] == da.time[1]
    assert da2.dims == ("element",)
    assert da2.shape == (441,)


# TODO
# def test_sel_xy_transect(vslice):
#     da = vslice.read()[0]
#     da2 = da.sel(x = 0, y = 10)


def test_plot_transect(vslice):
    da = vslice.read().Salinity
    da.plot()
    da.plot(cmin=0, cmax=1)
    da.plot(label="l", title="t", add_colorbar=False)
    da.plot(cmap="Blues", edge_color="0.9")

    # the old way
    vals = da.isel(time=0).to_numpy()
    vslice.plot_vertical_profile(vals, label="l", figsize=(3, 3))
    vslice.plot_vertical_profile(vals, title="t", add_colorbar=False)
    vslice.plot_vertical_profile(vals, cmax=12, cmap="Reds")

    plt.close("all")


def test_plot_transect_geometry(vslice):
    g = vslice.geometry
    g.plot()
    g.plot.mesh()

    plt.close("all")
