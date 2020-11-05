import numpy as np
from .dataset import Dataset

# class Interpolation2D:
def get_idw_interpolant(distances, p=1):
    """IDW interpolant for 2d array of distances

    Parameters
    ----------
    distances : array-like 
        distances between interpolation point and grid point
    p : int, optional
        order of inverse distance weighting, default=1

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


def interp2d(data, elem_ids, weights=None):

    is_dataset = False
    if isinstance(data, Dataset):
        is_dataset = True
        ds = data.copy()
        data = ds.data

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # data is single item and single time step
            return _interp_itemstep(data, elem_ids, weights)
        elif data.ndim == 2:
            # data is single item
            data = [data]

    idat = []
    ni = len(elem_ids)
    for datitem in data:
        nt, ne = datitem.shape
        idatitem = np.empty(shape=(nt, ni))
        for step in range(nt):
            idatitem[step, :] = _interp_itemstep(datitem[step, :], elem_ids, weights)
        idat.append(idatitem)

    if is_dataset:
        ds.data = idat
        idat = ds

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

