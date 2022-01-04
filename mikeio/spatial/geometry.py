from collections import namedtuple
import numpy as np


BoundingBox = namedtuple("BoundingBox", ["left", "bottom", "right", "top"])


class _Geometry:
    def __init__(self) -> None:
        self._projstr = None

    @property
    def projection_string(self):
        """The projection string"""
        return self._projstr

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"

    @property
    def is_local_coordinates(self):
        """Are coordinates relative (NON-UTM)?"""
        return self._projstr == "NON-UTM"

    def contains(self, coords):
        raise NotImplementedError

    def nearest_points(self, coords):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    @property
    def coordinates(self):
        raise NotImplementedError


# do we need this?
class GeometryPoint:
    @property
    def ndim(self):
        return 0
