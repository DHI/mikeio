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

    @staticmethod
    def _area_is_bbox(area):
        is_bbox = False
        if area is not None:
            if not np.isscalar(area):
                area = np.array(area)
                if (area.ndim == 1) & (len(area) == 4):
                    if np.all(np.isreal(area)):
                        is_bbox = True
        return is_bbox

    @staticmethod
    def _area_is_polygon(area) -> bool:
        if area is None:
            return False
        if np.isscalar(area):
            return False
        if not np.all(np.isreal(area)):
            return False
        polygon = np.array(area)
        if polygon.ndim > 2:
            return False

        if polygon.ndim == 1:
            if len(polygon) <= 5:
                return False
            if len(polygon) % 2 != 0:
                return False

        if polygon.ndim == 2:
            if polygon.shape[0] < 3:
                return False
            if polygon.shape[1] != 2:
                return False

        return True

    @staticmethod
    def _inside_polygon(polygon, xy):
        import matplotlib.path as mp

        if polygon.ndim == 1:
            polygon = np.column_stack((polygon[0::2], polygon[1::2]))
        return mp.Path(polygon).contains_points(xy)


class GeometryUndefined(_Geometry):
    def __repr__(self):
        return "GeometryUndefined()"


class GeometryPoint2D(_Geometry):
    def __init__(self, x: float, y: float, projection=None):
        self._projstr = projection
        self.x = x
        self.y = y

    def __repr__(self):
        return f"GeometryPoint2D(x={self.x}, y={self.y})"

    @property
    def ndim(self):
        return 0


class GeometryPoint3D(_Geometry):
    def __init__(self, x: float, y: float, z: float, projection=None):
        self._projstr = projection
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"GeometryPoint3D(x={self.x}, y={self.y}, z={self.z})"

    @property
    def ndim(self):
        return 0
