import mikeio
import numpy as np


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

    dati = interpolant.interp2d(ds[0].to_numpy())
    assert dati.shape == (ds.n_timesteps, npts)
    assert dati[0, 0] == 8.262675285339355

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
    dati = interpolant.interp2d(dat)
    assert len(dati) == npts
    assert dati[0] == 8.262675285339355


def test_get_idw_interpolant_preserves_float32_dtype() -> None:
    """Test that _get_idw_interpolant preserves float32 dtype.

    DFS files use float32 to save memory, so dtype preservation is critical.
    """
    from mikeio._interpolation import _get_idw_interpolant

    # Create float32 distances (typical for DFS files)
    distances = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=np.float32)

    weights = _get_idw_interpolant(distances, p=2)

    assert weights.dtype == np.float32, f"Expected float32, got {weights.dtype}"


def test_interp2d_preserves_float32_dtype() -> None:
    """Test that interpolation preserves float32 dtype from DFS files.

    DFS files use float32 to save memory. Converting to float64 during
    interpolation would double memory usage.
    """
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])

    npts = 5
    xy = np.zeros((npts, 2))
    xy[0, :] = [2, 52]
    xy[1, :] = [3, 53]
    xy[2, :] = [7, 54]
    xy[3, :] = [0, 55]
    xy[4, :] = [5, 54]

    interpolant = dfs.geometry.get_2d_interpolant(xy, n_nearest=3)

    # DFS data is typically float32
    dat = ds[0].values[0, :].astype(np.float32)
    assert dat.dtype == np.float32

    dati = interpolant.interp2d(dat)

    # Critical: must preserve float32 to avoid doubling memory usage
    assert dati.dtype == np.float32, f"Expected float32, got {dati.dtype}"
