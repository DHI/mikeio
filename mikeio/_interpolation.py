from __future__ import annotations
from dataclasses import dataclass
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


@dataclass
class Interpolant:
    ids: np.ndarray

    # TODO should this allowed to be None?
    weights: np.ndarray | None

    # TODO data is inconsistent with interp2d, but I have the feeling this is never used
    def interp1d(self, data: np.ndarray) -> np.ndarray:
        ids = self.ids
        weights = self.weights
        assert weights is not None
        result = np.dot(data[:, ids], weights)
        assert isinstance(result, np.ndarray)
        return result

    @overload
    def interp2d(
        self,
        data: np.ndarray | DataArray,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray: ...

    @overload
    def interp2d(
        self,
        data: Dataset,
        shape: tuple[int, ...] | None = None,
    ) -> Dataset: ...

    def interp2d(
        self,
        data: Dataset | DataArray | np.ndarray,
        shape: tuple[int, ...] | None = None,
    ) -> Dataset | np.ndarray:
        """interp spatially in data (2d only).

        Parameters
        ----------
        data : mikeio.Dataset, DataArray, or ndarray
            dfsu data
        shape: tuple, optional
                reshape output

        Returns
        -------
        ndarray, Dataset, or DataArray
            spatially interped data with same type and shape as input

        """
        from .dataset import DataArray, Dataset

        ni = len(self.ids)

        if isinstance(data, Dataset):
            ds = data.copy()

            interp_data_vars = {}

            for da in ds:
                key = da.name
                if "time" not in da.dims:
                    idatitem = _interp_item(da.to_numpy(), self)
                    if shape:
                        idatitem = idatitem.reshape(*shape)

                else:
                    nt, _ = da.shape
                    idatitem = _interp_item(da.to_numpy(), self)
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
                idatitem = _interp_item(data, self)
                if shape:
                    idatitem = idatitem.reshape(*shape)
                return idatitem

        datitem = data
        nt, _ = datitem.shape
        idatitem = np.empty(shape=(nt, ni), dtype=datitem.dtype)
        for step in range(nt):
            idatitem[step, :] = _interp_item(datitem[step], self)
        if shape:
            idatitem = idatitem.reshape((nt, *shape))
        return idatitem


def _interp_item(
    data: np.ndarray,
    interpolant: Interpolant,
) -> np.ndarray:
    """Vectorized interpolation for 1D or 2D data.

    data: shape (nelem,) or (nt, nelem)

    Returns: shape (ni,) or (nt, ni)

    """
    weights = interpolant.weights
    elem_ids = interpolant.ids

    if data.ndim == 1:
        if weights is None:
            return data[elem_ids]
        else:
            idat = data[elem_ids] * weights
            return np.sum(idat, axis=1) if weights.ndim == 2 else idat
    elif data.ndim == 2:
        # data shape: (nt, nelem)
        if weights is None:
            return data[:, elem_ids]
        else:
            # data[:, elem_ids]: (nt, ni)
            # weights: (ni,) or (ni, nweights)
            idat = data[:, elem_ids] * weights  # broadcasting
            return np.sum(idat, axis=-1) if weights.ndim == 2 else idat
    else:
        raise ValueError("data must be 1D or 2D array")
