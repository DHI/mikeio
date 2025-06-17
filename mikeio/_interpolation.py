from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def get_idw_interpolant(distances: np.ndarray, p: float = 2) -> np.ndarray:
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
    weights: np.ndarray

    @staticmethod
    def from_distances(distances: np.ndarray, p: float = 2) -> np.ndarray:
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
        return get_idw_interpolant(distances, p)

    def interp1d(self, data: np.ndarray) -> np.ndarray:
        ids = self.ids
        weights = self.weights
        result = np.dot(data[:, ids], weights)
        assert isinstance(result, np.ndarray)
        return result

    def interp2d(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """interp spatially in data (2d only).

        Parameters
        ----------
        data : ndarray
            dfsu data

        Returns
        -------
        ndarray
            spatially interpolated data

        """
        weights = self.weights
        elem_ids = self.ids

        if data.ndim == 1:
            idat = data[elem_ids] * weights.astype(data.dtype)
            return np.sum(idat, axis=1) if weights.ndim == 2 else idat
        elif data.ndim == 2:
            # data shape: (nt, nelem)

            # data[:, elem_ids]: (nt, ni)
            # weights: (ni,) or (ni, nweights)
            idat = data[:, elem_ids] * weights.astype(data.dtype)  # broadcasting
            return np.sum(idat, axis=-1) if weights.ndim == 2 else idat
        else:
            raise ValueError("data must be 1D or 2D array")
