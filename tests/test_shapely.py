import pytest
from mikeio.spatial._FM_geometry import GeometryFM2D


##################################################
# these tests will not run if shapely is not installed
##################################################
pytest.importorskip("shapely")


# TODO doesn't run with Numpy 2.1.0
# def test_to_shapely():
#     #     x     y    z
#     nc = [
#         (0.0, 0.0, 0.0),  # 0
#         (1.0, 0.0, 0.0),  # 1
#         (0.5, 1.0, 0.0),  # 2
#     ]

#     el = [(0, 1, 2)]

#     g = GeometryFM2D(nc, el)
#     shp = g.to_shapely()
#     assert shp.geom_type == "MultiPolygon"
#    assert shp.area == 0.5
