import warnings

from mikecore.Projections import Cartography, MapProjection

# import pyproj


class CRSConversionWarning(Warning):
    """Used when an ad hoc conversion is performed."""

    pass


class CRSConversionError(Exception):
    """Raised when conversion cannot be performed."""

    pass


class CRS:
    def __init__(self, projection_string: str) -> None:
        """Create an instance of the CRS class.

        The CRS class provides an interface between the common
        Coordinate Reference System (CRS) projection definitions such as EPSG and WKT
        and DHI MIKE projection strings.

        Parameters
        ----------
        projection_string : str
            DHI MIKE projection string.
            Either WKT string or a short name which can be recognized
            by DHI MIKE, such as: "LONG/LAT", "UTM-18"

        """
        # https://manuals.mikepoweredbydhi.help/2021/General/Class_Library/DHI_Projections/html/T_DHI_Projections_Cartography.htm
        self.__cartography = Cartography(
            projection_string, validateProjectionString=True
        )

    def __repr__(self) -> str:
        summary = [
            " ".join(
                [
                    "Geographical" if self.is_geographical else "Projected",
                    "Coordinate Reference System",
                ]
            ),
            f"DHI projection name: {self.name}",
            f"DHI projection string: {self.projection_string}",
        ]
        return "\n".join(summary)

    @property
    def map_projection(self) -> MapProjection:
        # https://manuals.mikepoweredbydhi.help/2021/General/Class_Library/DHI_Projections/html/T_DHI_Projections_MapProjection.htm
        return self.__cartography.Projection

    @property
    def name(self) -> str:
        return self.__cartography.ProjectionName

    @property
    def projection_string(self) -> str:
        return self.__cartography.ProjectionString

    @property
    def is_geographical(self) -> bool:
        return MapProjection.IsGeographical(self.projection_string)

    @property
    def is_projected(self) -> bool:
        return not self.is_geographical

    def to_pyproj(self):
        """
        Convert projection to pyptoj.CRS object.

        Returns
        -------
        pyproj.CRS

        """
        import pyproj

        if self.projection_string == "LONG/LAT":
            warnings.warn(
                message="LONG/LAT projection string was interpreted as EPSG:4326",
                category=CRSConversionWarning,
            )
            return pyproj.CRS.from_epsg(4326)
        else:
            return pyproj.CRS.from_string(self.projection_string)

    @classmethod
    def from_pyproj(cls, pyproj_crs):
        """
        Create CRS object from pyproj.CRS object.

        Parameters
        ----------
        pyproj_crs : pyproj.CRS
            pyproj.CRS object.

        Returns
        -------
        CRS
            CRS instance.

        """
        import pyproj

        return cls(projection_string=pyproj_crs.to_wkt(version="WKT1_ESRI"))

    def to_epsg(self, min_confidence: float = 70.0) -> int:
        """
        Convert projection to pyptoj.CRS object.

        Parameters
        ----------
        min_confidence : float, optional
            A value between 0-100 where 100 is the most confident. Default is 70.
            See 'pyproj.CRS.to_epsg' for more details.

        Returns
        -------
        int
            EPSG code.

        Raises
        ------
        CRSConversionError
            Failed to convert projection to EPSG.
        RuntimeError
            Unexpected 'pyproj.to_epsg' return type.

        """
        import pyproj

        epsg_code = self.to_pyproj().to_epsg(min_confidence=min_confidence)
        if epsg_code is None:
            raise CRSConversionError(
                f"cannot convert '{self.projection_string}' to EPSG"
            )
        elif isinstance(epsg_code, int):
            return epsg_code
        else:
            raise RuntimeError(
                f"pyproj.to_epsg returned '{type(epsg_code).__name__}', "
                f"expected None or int"
            )

    @classmethod
    def from_epsg(cls, epsg: int):
        """
        Create CRS object from EPSG code.

        Parameters
        ----------
        epsg : int
            EPSG code.

        Returns
        -------
        CRS
            CRS instance.

        """
        import pyproj

        pyproj_crs = pyproj.CRS.from_epsg(epsg)
        return cls.from_pyproj(pyproj_crs=pyproj_crs)
