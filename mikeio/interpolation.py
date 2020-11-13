import numpy as np
from .dataset import Dataset

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
    p = 1  # inverse distance order

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
    data : mikeio.Dateset, list(ndarray), or ndarray 
        dfsu data 
    elem_ids : ndarray(int)
        n sized array of 1 or more element ids used for interpolation
    weights : ndarray(float), optional
        weights with same size as elem_ids used for interpolation
    shape: tuple, optional
            reshape output

    Returns
    -------
    ndarray or list(ndarray)
        spatially interped data with same type and shape as input
    
    Examples
    --------    
    >>> elem_ids, weights = dfs.get_2d_interpolant(xy)
    >>> dsi = interp2d(ds, elem_ids, weights)
    """
    is_dataset = False
    if isinstance(data, Dataset):
        is_dataset = True
        ds = data.copy()
        data = ds.data

    is_single_item = False
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # data is single item and single time step
            return _interp_itemstep(data, elem_ids, weights)
        elif data.ndim == 2:
            is_single_item = True
            data = [data]

    idat = []
    ni = len(elem_ids)
    for datitem in data:
        nt, ne = datitem.shape
        idatitem = np.empty(shape=(nt, ni))
        for step in range(nt):
            idatitem[step, :] = _interp_itemstep(datitem[step, :], elem_ids, weights)

        if shape:
            idatitem = idatitem.reshape((nt, *shape))

        idat.append(idatitem)

    if is_dataset:
        ds.data = idat
        idat = ds
    if is_single_item:
        idat = idat[0]

    return idat


def _interp_itemstep(data, elem_ids, weights=None):
    if weights is None:
        # nearest neighbor
        return data[elem_ids]
    ni = len(elem_ids)
    idat = np.empty(ni)
    for j in range(ni):
        idat[j] = np.dot(data[elem_ids[j]], weights[j])
    return idat

