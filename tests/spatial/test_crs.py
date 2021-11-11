import pytest
from mikeio.spatial.crs import CRS, CRSConversionError, CRSConversionWarning


pytest.importorskip("pyproj")

import pyproj


class TestCRS:
    @pytest.mark.parametrize(
        ["projection_string", "name", "is_projected"],
        [
            ("LONG/LAT", "LONG/LAT", False),
            ("UTM-32", "UTM-32", True),
        ],
    )
    def test_init(self, projection_string, name, is_projected):
        crs = CRS(projection_string=projection_string)
        assert crs.name == name
        assert crs.is_geographical is not is_projected
        assert crs.is_projected is is_projected

    @pytest.mark.parametrize(
        ["epsg", "is_projected"],
        [
            (4326, False),
            (32632, True),
        ],
    )
    def test_to_from_pyproj(self, epsg, is_projected):
        # Test from_pyproj
        pyproj_crs = pyproj.CRS.from_epsg(epsg)
        crs = CRS.from_pyproj(pyproj_crs=pyproj_crs)
        assert crs.is_projected is is_projected

        # Test to_pyproj
        exported = crs.to_pyproj()
        assert exported.to_epsg() == epsg

    def test_to_pyproj_errors(self):
        with pytest.warns(
            CRSConversionWarning,
            match=r"LONG/LAT projection.+EPSG:4326",
        ):
            CRS(projection_string="LONG/LAT").to_epsg()

    def test_to_epsg(self):
        with pytest.raises(CRSConversionError, match=r"cannot convert.+to EPSG"):
            CRS("UTM-32").to_epsg()

        # Test normal conversion
        wkt = pyproj.CRS.from_epsg(32632).to_wkt(version="WKT1_ESRI")
        crs = CRS(projection_string=wkt)
        assert crs.to_epsg() == 32632

    def test_from_epsg(self):
        # Test WGS84
        crs = CRS.from_epsg(epsg=4326)
        assert crs.is_geographical
        assert crs.name == "GCS_WGS_1984"

        # Test UTM-32
        assert CRS.from_epsg(epsg=32632).is_projected
        assert CRS.from_epsg(epsg=32632).name == "WGS_1984_UTM_Zone_32N"
