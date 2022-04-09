import numpy as np
import pytest
import mikeio
from mikeio.spatial.FM_geometry import GeometryFMVerticalProfile
from mikeio.spatial.geometry import GeometryPoint3D
from mikecore.DfsuFile import DfsuFileType


@pytest.fixture
def vslice():
    # sigma, z vertical profile (=transect)
    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    return mikeio.open(filename)


def test_open_transect(vslice):
    dfs = vslice
    assert dfs._type == DfsuFileType.DfsuVerticalProfileSigmaZ
    assert dfs.n_items == 2
    assert dfs.n_elements == 441
    assert dfs.n_sigma_layers == 4
    assert dfs.n_z_layers == 5
    assert dfs.n_timesteps == 3


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
    da.plot(cmin=0)
    da.plot(cmax=0)
