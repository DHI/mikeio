from __future__ import annotations
from typing import TYPE_CHECKING, overload
import numpy as np

if TYPE_CHECKING:
    from .dataset import Dataset, DataArray

from .spatial import GeometryUndefined


def get_idw_interpolant(distances: np.ndarray, p: float = 2) -> np.ndarray:
    """IDW interpolant for 2d array of distances.

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


@overload
def interp2d(
    data: np.ndarray | DataArray,
    elem_ids: np.ndarray,
    weights: np.ndarray | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray: ...


@overload
def interp2d(
    data: Dataset,
    elem_ids: np.ndarray,
    weights: np.ndarray | None = None,
    shape: tuple[int, ...] | None = None,
) -> Dataset: ...


def interp2d(
    data: Dataset | DataArray | np.ndarray,
    elem_ids: np.ndarray,
    weights: np.ndarray | None = None,
    shape: tuple[int, ...] | None = None,
) -> Dataset | np.ndarray:
    """interp spatially in data (2d only).

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
                # use dtype of da
                idatitem = np.empty(shape=(nt, ni), dtype=da.values.dtype)
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
        # TODO why doesn't this return a DataArray?
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
    idatitem = np.empty(shape=(nt, ni), dtype=datitem.dtype)
    for step in range(nt):
        idatitem[step, :] = _interp_itemstep(datitem[step], elem_ids, weights)
    if shape:
        idatitem = idatitem.reshape((nt, *shape))
    return idatitem


def _interp_itemstep(
    data: np.ndarray,
    elem_ids: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if weights is None:
        return data[elem_ids]
    else:
        idat = data[elem_ids] * weights
        return np.sum(idat, axis=1) if weights.ndim == 2 else idat
