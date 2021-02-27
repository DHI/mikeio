from __future__ import annotations

import typing

import DHI.Projections
import pyproj


class CRS:
    def __init__(self, projstr: str) -> None:
        """Create an instance of the CRS class.

        The CRS class provides an interface between the common
        Coordinate Reference System (CRS) projection definitions such as EPSG and WKT
        and DHI MIKE projection strings.

        Parameters
        ----------
        projstr : str
            DHI MIKE projection string.
            Examples include: "LONG/LAT", "UTM-18N"

        """
        # https://manuals.mikepoweredbydhi.help/2021/General/Class_Library/DHI_Projections/html/T_DHI_Projections_Cartography.htm
        self._cartography = DHI.Projections.Cartography(projstr, True)

        # used to cache results of 'to_pyproj' method
        self.__pyproj: typing.Optional[str] = None

    def __repr__(self) -> str:
        summary = [
            " ".join(
                [
                    "Geographic" if self.is_geographic else "Projected",
                    "Coordinate Reference System",
                ]
            ),
            f"DHI projection string: {self.projstr}",
        ]
        return "\n".join(summary)

    @property
    def _map_projection(self) -> DHI.Projections.MapProjection:
        # https://manuals.mikepoweredbydhi.help/2021/General/Class_Library/DHI_Projections/html/T_DHI_Projections_MapProjection.htm
        return self._cartography.get_Projection()

    @property
    def projstr(self) -> str:
        return self._cartography.get_ProjectionName()

    @property
    def is_geographic(self) -> bool:
        return self.projstr == "LONG/LAT"

    @property
    def is_projected(self) -> bool:
        return not self.is_geographic

    # TODO - this is the key method, establish proper mapping between pyproj and CRS
    def to_pyproj(self) -> pyproj.CRS:
        if self.__pyproj is None:
            if self.is_geographic:
                self.__pyproj = pyproj.CRS.from_epsg(code=4326)
            else:
                self.__pyproj = pyproj.CRS.from_wkt(
                    self._cartography.get_ProjectionString()
                )
        return self.__pyproj

    # TODO - this is the key method, establish proper mapping between pyproj and CRS
    @classmethod
    def from_pyproj(cls, pyproj_crs: pyproj.CRS) -> CRS:
        if pyproj_crs.is_geographic:
            projstr = "LONG/LAT"
        else:
            projstr = pyproj_crs.name
        return cls(projstr=projstr)

    def to_epsg(self, confidence: float = 70.0) -> typing.Optional[int]:
        return self.to_pyproj().to_epsg(confidence=confidence)

    @classmethod
    def from_epsg(cls, epsg: int) -> CRS:
        pyproj_crs = pyproj.CRS.from_epsg(code=epsg)
        return cls.from_pyproj(pyproj_crs=pyproj_crs)
