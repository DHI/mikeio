from collections import namedtuple
import numpy as np


BoundingBox = namedtuple("BoundingBox", ["left", "bottom", "right", "top"])


class _Geometry:
    def __init__(self) -> None:
        self._projstr = None

    @property
    def projection_string(self) -> str:
        """The projection string"""
        return self._projstr

    @property
    def projection(self):
        """The projection"""
        return self._projstr

    @property
    def is_geo(self) -> bool:
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"

    @property
    def is_local_coordinates(self) -> bool:
        """Are coordinates relative (NON-UTM)?"""
        return self._projstr == "NON-UTM"

    def contains(self, coords) -> bool:
        raise NotImplementedError

    def nearest_points(self, coords):
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def coordinates(self):
        raise NotImplementedError


class GeometryUndefined(_Geometry):
    def __repr__(self):
        return "GeometryUndefined()"


class GeometryPoint2D(_Geometry):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"GeometryPoint2D(x={self.x}, y={self.y})"

    @property
    def ndim(self):
        return 0


class GeometryPoint3D(_Geometry):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"GeometryPoint3D(x={self.x}, y={self.y}, z={self.z})"

    @property
    def ndim(self):
        return 0
