from mikeio.dataset import Dataset
from mikeio.interpolation import get_idw_interpolant, interp2d, _interp_itemstep
import mikeio
import numpy as np

from mikeio.spatial.geometry import GeometryUndefined

import pytest


def test_get_idw_interpolant():
    d = np.linspace(1, 2, 2)
    w = get_idw_interpolant(d, p=1)
    assert w[0] == 2 / 3
    assert w[1] == 1 / 3


def test_interp2d():
    dfs = mikeio.open("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])

    npts = 5
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 52]
    xy[1, :] = [3, 53]
    xy[2, :] = [7, 54]
    xy[3, :] = [0, 55]
    xy[4, :] = [5, 54]

    elem_ids, weights = dfs.geometry.get_2d_interpolant(xy, n_nearest=1)

    # with pytest.warns(match="Geometry"):
    dati = interp2d(ds, elem_ids, weights)
    assert isinstance(dati, Dataset)
    assert isinstance(dati.geometry, GeometryUndefined)  # There is no suitable file format for this, thus no suitable geometry :-(
    assert np.all(dati.shape == (ds.n_timesteps, npts))
    assert dati[0].values[0, 0] == 8.262675285339355

    dat = ds[0].to_numpy()  # first item, all time steps
    dati = interp2d(dat, elem_ids, weights)
    assert isinstance(dati, np.ndarray)
    assert dati.size == ds.n_timesteps * npts
    assert dati[0, 0] == 8.262675285339355

    elem_ids, weights = dfs.geometry.get_2d_interpolant(xy, n_nearest=3)

    dat = ds[0].values[0, :]  # a single time step
    dati = interp2d(dat, elem_ids, weights)
    assert isinstance(dati, np.ndarray)
    assert dati.size == npts


def test_interp2d_same_points():
    dfs = mikeio.open("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    npts = 3
    # same points as data (could cause IDW to diverge)
    xy = dfs.element_coordinates[:npts, 0:2]
    elem_ids, weights = dfs.geometry.get_2d_interpolant(xy, n_nearest=4)
    assert np.max(weights) <= 1.0
    dat = ds[0].values[0, :]
    dati = interp2d(dat, elem_ids, weights)
    assert np.all(dati == dat[:npts])


def test_interp2d_outside():
    dfs = mikeio.open("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    # outside domain
    npts = 2
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 50]
    xy[1, :] = [3, 51]
    elem_ids, weights = dfs.geometry.get_2d_interpolant(xy, n_nearest=4)
    dati = interp2d(ds[0].values[0, :], elem_ids, weights)
    assert np.all(np.isnan(dati))
    elem_ids, weights = dfs.geometry.get_2d_interpolant(
        xy, n_nearest=4, extrapolate=True
    )
    dati = interp2d(ds[0].values[0, :], elem_ids, weights)
    assert np.all(~np.isnan(dati))


def test_interp_itemstep():
    dfs = mikeio.open("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])

    npts = 5
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 52]
    xy[1, :] = [3, 53]
    xy[2, :] = [7, 54]
    xy[3, :] = [0, 55]
    xy[4, :] = [5, 54]
    elem_ids, weights = dfs.geometry.get_2d_interpolant(xy, n_nearest=1)

    dat = ds[0].values[0, :]
    dati = _interp_itemstep(dat, elem_ids, weights)
    assert len(dati) == npts
    assert dati[0] == 8.262675285339355
