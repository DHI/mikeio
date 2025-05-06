from __future__ import annotations

import numpy as np

from ._geometry import BoundingBox


def xy_to_bbox(xy: np.ndarray, buffer: float = 0.0) -> BoundingBox:
    """Return bounding box for list of coordinates."""
    left = xy[:, 0].min() - buffer
    bottom = xy[:, 1].min() - buffer
    right = xy[:, 0].max() + buffer
    top = xy[:, 1].max() + buffer
    return BoundingBox(left, bottom, right, top)


def dist_in_meters(
    coords: np.ndarray, pt: tuple[float, float] | np.ndarray, is_geo: bool = False
) -> np.ndarray:
    """Get distance between array of coordinates and point.

    Parameters
    ----------
    coords : n-by-2 array
        x, y coordinates
    pt : [float, float]
        x, y coordinate of point
    is_geo : bool, optional
        are coordinates geographical?, by default False

    Returns
    -------
    array
        distances in meter

    """
    coords = np.atleast_2d(coords)
    xe = coords[:, 0]
    ye = coords[:, 1]
    xp = pt[0]
    yp = pt[1]
    if is_geo:
        d = _get_dist_geo(xe, ye, xp, yp)
    else:
        d = np.sqrt(np.square(xe - xp) + np.square(ye - yp))
    return d  # type: ignore


def _get_dist_geo(
    lon: float | np.ndarray, lat: float | np.ndarray, lon1: float, lat1: float
) -> float:
    # assuming input in degrees!
    R = 6371e3  # Earth radius in metres
    dlon = np.deg2rad(lon1 - lon)
    dlon[dlon > np.pi] = dlon[dlon > np.pi] - 2 * np.pi
    dlon[dlon < -np.pi] = dlon[dlon < -np.pi] + 2 * np.pi
    dlat = np.deg2rad(lat1 - lat)
    x = dlon * np.cos(np.deg2rad((lat + lat1) / 2))
    y = dlat
    d = R * np.sqrt(np.square(x) + np.square(y))
    return d  # type: ignore


def relative_cumulative_distance(
    coords: np.ndarray, reference: np.ndarray | None = None, is_geo: bool = False
) -> np.ndarray:
    """Calculate the cumulative relative distance along a path."""
    coords = np.atleast_2d(coords)
    d = np.zeros_like(coords[:, 0])
    if reference is not None:
        pt = (reference[0], reference[1])
        d[0] = dist_in_meters(coords[0, 0:2], pt=pt, is_geo=is_geo)[0]
    for j in range(1, len(d)):
        d[j] = d[j - 1] + dist_in_meters(coords[j, 0:2], coords[j - 1, 0:2], is_geo)[0]
    return d  # type: ignore
