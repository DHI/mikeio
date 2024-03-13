from abc import ABC, abstractmethod

from collections import namedtuple
from typing import List

from mikecore.Projections import MapProjection

BoundingBox = namedtuple("BoundingBox", ["left", "bottom", "right", "top"])


class _Geometry(ABC):
    def __init__(self, projection: str = "LONG/LAT") -> None:

        if not MapProjection.IsValid(projection):
            raise ValueError(f"{projection=} is not a valid projection string")

        self._projstr = projection

    def default_dims(self, ndim_no_time: int) -> List[str]:
        return {0: [], 1: ["x"], 2: ["y", "x"], 3: ["z", "y", "x"]}[ndim_no_time]  # type: ignore

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

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass


class GeometryUndefined(_Geometry):
    def __repr__(self):
        return "GeometryUndefined()"

    @property
    def ndim(self) -> int:
        return 0


class GeometryPoint2D(_Geometry):
    def __init__(self, x: float, y: float, projection: str = "LONG/LAT"):
        super().__init__(projection)
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"GeometryPoint2D(x={self.x}, y={self.y})"

    # TODO should we use wkt here
    # def __str__(self) -> str:
    #    return self.wkt

    @property
    def wkt(self) -> str:
        return f"POINT ({self.x} {self.y})"

    @property
    def ndim(self) -> int:
        """Geometry dimension"""
        return 0

    def to_shapely(self):
        from shapely.geometry import Point

        return Point(self.x, self.y)


class GeometryPoint3D(_Geometry):
    def __init__(self, x: float, y: float, z: float, projection: str = "LONG/LAT"):
        super().__init__(projection)

        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"GeometryPoint3D(x={self.x}, y={self.y}, z={self.z})"

    @property
    def wkt(self) -> str:
        return f"POINT Z ({self.x} {self.y} {self.z})"

    @property
    def ndim(self) -> int:
        return 0

    def to_shapely(self):
        from shapely.geometry import Point

        return Point(self.x, self.y, self.z)
