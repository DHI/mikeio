from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload
import numpy as np

from .spatial import GeometryUndefined

if TYPE_CHECKING:
    from .dataset import Dataset


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
        data: np.ndarray,
    ) -> np.ndarray: ...

    @overload
    def interp2d(
        self,
        data: Dataset,
    ) -> Dataset: ...

    def interp2d(
        self,
        data: Dataset | np.ndarray,
    ) -> Dataset | np.ndarray:
        """interp spatially in data (2d only).

        Parameters
        ----------
        data : mikeio.Dataset, or ndarray
            dfsu data

        Returns
        -------
        ndarray, Dataset, or DataArray
            spatially interped data with same type as input

        """
        from .dataset import DataArray, Dataset

        if isinstance(data, Dataset):
            das = [
                DataArray(
                    data=self._interp_item(da.to_numpy()),
                    time=da.time,
                    item=da.item,
                    dims=da.dims,
                    geometry=GeometryUndefined(),
                )
                for da in data
            ]

            return Dataset(das, validate=False)

        return self._interp_item(data)

    def _interp_item(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        weights = self.weights
        elem_ids = self.ids

        if data.ndim == 1:
            if weights is None:
                return data[elem_ids]
            else:
                idat = data[elem_ids] * weights
                idat = idat.astype(data.dtype)
                return np.sum(idat, axis=1) if weights.ndim == 2 else idat
        elif data.ndim == 2:
            # data shape: (nt, nelem)
            if weights is None:
                return data[:, elem_ids]
            else:
                # data[:, elem_ids]: (nt, ni)
                # weights: (ni,) or (ni, nweights)
                idat = data[:, elem_ids] * weights  # broadcasting
                idat = idat.astype(data.dtype)
                return np.sum(idat, axis=-1) if weights.ndim == 2 else idat
        else:
            raise ValueError("data must be 1D or 2D array")
