from mikeio.dataset import Dataset
from mikeio._interpolation import get_idw_interpolant, _interp_item
import mikeio
import numpy as np

from mikeio.spatial import GeometryUndefined


def test_get_idw_interpolant() -> None:
    d = np.linspace(1, 2, 2)
    w = get_idw_interpolant(d, p=1)
    assert w[0] == 2 / 3
    assert w[1] == 1 / 3


def test_interp2d() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])

    npts = 5
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 52]
    xy[1, :] = [3, 53]
    xy[2, :] = [7, 54]
    xy[3, :] = [0, 55]
    xy[4, :] = [5, 54]

    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=1)

    # with pytest.warns(match="Geometry"):
    dati = interpolant.interp2d(ds)
    assert isinstance(dati, Dataset)
    assert isinstance(
        dati.geometry, GeometryUndefined
    )  # There is no suitable file format for this, thus no suitable geometry :-(
    assert np.all(dati.shape == (ds.n_timesteps, npts))
    assert dati[0].values[0, 0] == 8.262675285339355

    dat = ds[0].to_numpy()  # first item, all time steps
    dati = interpolant.interp2d(dat)
    assert isinstance(dati, np.ndarray)
    assert dati.size == ds.n_timesteps * npts
    assert dati[0, 0] == 8.262675285339355

    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=3)

    dat = ds[0].values[0, :]  # a single time step
    dati = interpolant.interp2d(dat)
    assert isinstance(dati, np.ndarray)
    assert dati.size == npts


def test_interp2d_same_points() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    npts = 3
    # same points as data (could cause IDW to diverge)
    xy = dfs.geometry.element_coordinates[:npts, 0:2]
    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=4)
    assert np.max(interpolant.weights) <= 1.0
    dat = ds[0].values[0, :]
    dati = interpolant.interp2d(dat)
    assert np.all(dati == dat[:npts])


def test_interp2d_outside() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    # outside domain
    npts = 2
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 50]
    xy[1, :] = [3, 51]
    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=4)
    dati = interpolant.interp2d(ds[0].values[0, :])
    assert np.all(np.isnan(dati))
    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=4, extrapolate=True)
    dati = interpolant.interp2d(ds[0].values[0, :])
    assert np.all(~np.isnan(dati))


def test_interp_itemstep() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])

    npts = 5
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 52]
    xy[1, :] = [3, 53]
    xy[2, :] = [7, 54]
    xy[3, :] = [0, 55]
    xy[4, :] = [5, 54]
    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=1)

    dat = ds[0].values[0, :]
    dati = _interp_item(dat, interpolant)
    assert len(dati) == npts
    assert dati[0] == 8.262675285339355
