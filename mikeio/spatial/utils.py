import numpy as np

from .geometry import BoundingBox


def xy_to_bbox(xy, buffer=None):
    """return bounding box for list of coordinates"""
    if buffer is None:
        buffer = 0

    left = xy[:, 0].min() - buffer
    bottom = xy[:, 1].min() - buffer
    right = xy[:, 0].max() + buffer
    top = xy[:, 1].max() + buffer
    return BoundingBox(left, bottom, right, top)


def dist_in_meters(coords, pt, is_geo=False):
    """get distance between array of coordinates and point

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
    return d


def _get_dist_geo(lon, lat, lon1, lat1):
    # assuming input in degrees!
    R = 6371e3  # Earth radius in metres
    dlon = np.deg2rad(lon1 - lon)
    dlon[dlon > np.pi] = dlon[dlon > np.pi] - 2 * np.pi
    dlon[dlon < -np.pi] = dlon[dlon < -np.pi] + 2 * np.pi
    dlat = np.deg2rad(lat1 - lat)
    x = dlon * np.cos(np.deg2rad((lat + lat1) / 2))
    y = dlat
    d = R * np.sqrt(np.square(x) + np.square(y))
    return d


def _relative_cumulative_distance(coords, reference=None, is_geo=False):
    """Calculate the cumulative relative distance along a path"""
    coords = np.atleast_2d(coords)
    d = np.zeros_like(coords[:, 0])
    if reference is not None:
        d[0] = dist_in_meters(coords[0, 0:2], reference[0:2], is_geo)
    for j in range(1, len(d)):
        d[j] = d[j - 1] + dist_in_meters(coords[j, 0:2], coords[j - 1, 0:2], is_geo)
    return d
