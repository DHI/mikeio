import numpy as np

from .spatial.geometry import GeometryUndefined


# class Interpolation2D:
def get_idw_interpolant(distances, p=2):
    """IDW interpolant for 2d array of distances

    https://pro.arcgis.com/en/pro-app/help/analysis/geostatistical-analyst/how-inverse-distance-weighted-interpolation-works.htm

    Parameters
    ----------
    distances : array-like
        distances between interpolation point and grid point
    p : float, optional
        power of inverse distance weighting, default=2

    Returns
    -------
    np.array
        weights
    """
    is_1d = distances.ndim == 1
    if is_1d:
        distances = np.atleast_2d(distances)

    MIN_DISTANCE = 1e-8
    weights = np.zeros(distances.shape)

    match = distances[:, 0] < MIN_DISTANCE
    weights[match, 0] = 1

    weights[~match, :] = 1 / distances[~match, :] ** p
    denom = weights[~match, :].sum(axis=1).reshape(-1, 1)  # *np.ones((1,n_nearest))
    weights[~match, :] = weights[~match, :] / denom

    if is_1d:
        weights = weights[0]
    return weights


def interp2d(data, elem_ids, weights=None, shape=None):
    """interp spatially in data (2d only)

    Parameters
    ----------
    data : mikeio.Dataset, DataArray, or ndarray
        dfsu data
    elem_ids : ndarray(int)
        n sized array of 1 or more element ids used for interpolation
    weights : ndarray(float), optional
        weights with same size as elem_ids used for interpolation
    shape: tuple, optional
            reshape output

    Returns
    -------
    ndarray, Dataset, or DataArray
        spatially interped data with same type and shape as input

    Examples
    --------
    >>> elem_ids, weights = dfs.get_spatial_interpolant(coords)
    >>> dsi = interp2d(ds, elem_ids, weights)
    """
    from .dataset import DataArray, Dataset

    if isinstance(data, Dataset):
        ds = data.copy()

        ni = len(elem_ids)

        interp_data_vars = {}

        for da in ds:
            key = da.name
            if "time" not in da.dims:
                idatitem = _interp_itemstep(da.to_numpy(), elem_ids, weights)
                if shape:
                    idatitem = idatitem.reshape(*shape)

            else:
                nt, _ = da.shape
                idatitem = np.empty(shape=(nt, ni))
                for step in range(nt):
                    idatitem[step, :] = _interp_itemstep(
                        da[step].to_numpy(), elem_ids, weights
                    )
                if shape:
                    idatitem = idatitem.reshape((nt, *shape))

            dims = ("time", "element")  # TODO is this the best?
            interp_data_vars[key] = DataArray(
                data=idatitem,
                time=da.time,
                dims=dims,
                item=da.item,
                geometry=GeometryUndefined(),
            )

        new_ds = Dataset(interp_data_vars, validate=False)
        return new_ds

    if isinstance(data, DataArray):
        data = data.to_numpy()

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # data is single item and single time step
            idatitem = _interp_itemstep(data, elem_ids, weights)
            if shape:
                idatitem = idatitem.reshape(*shape)
            return idatitem

    ni = len(elem_ids)
    datitem = data
    nt, _ = datitem.shape
    idatitem = np.empty(shape=(nt, ni))
    for step in range(nt):
        idatitem[step, :] = _interp_itemstep(datitem[step], elem_ids, weights)
    if shape:
        idatitem = idatitem.reshape((nt, *shape))
    return idatitem


def _interp_itemstep(data, elem_ids, weights=None):
    """Interpolate a single item and time step

    Parameters
    ----------
    data : ndarray
        data to interpolate
    elem_ids : ndarray(int)
        n sized array of 1 or more element ids used for interpolation
    weights : ndarray(float), optional
        weights with same size as elem_ids used for interpolation

    Returns
    -------
    ndarray
        spatially interpolated data

    Notes
    -----
    This function is used internally by interp2d
    """
    if weights is None:
        return data[elem_ids]
    else:
        idat = data[elem_ids] * weights
        return np.sum(idat, axis=1) if weights.ndim == 2 else idat
