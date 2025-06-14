import pytest

from mikeio.spatial import GeometryFM2D


##################################################
# these tests will not run if shapely is not installed
##################################################
pytest.importorskip("shapely")


def test_to_shapely() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    g = GeometryFM2D(nc, el)
    shp = g.to_shapely()
    assert shp.geom_type == "MultiPolygon"
    assert shp.area == 0.5
