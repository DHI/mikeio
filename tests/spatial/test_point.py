from mikeio.spatial import GeometryPoint2D, GeometryPoint3D

# https://www.ogc.org/standard/sfa/


def test_point2d_wkt() -> None:
    p = GeometryPoint2D(10, 20)
    assert p.wkt == "POINT (10 20)"

    p = GeometryPoint2D(x=-5642.5, y=120.1)
    assert p.wkt == "POINT (-5642.5 120.1)"


def test_point3d_wkt() -> None:
    p = GeometryPoint3D(10, 20, 30)
    assert p.wkt == "POINT Z (10 20 30)"


def test_point2d_to_shapely() -> None:
    p = GeometryPoint2D(10, 20)
    sp = p.to_shapely()
    assert sp.x == 10
    assert sp.y == 20
    assert sp.wkt == p.wkt


def test_point3d_to_shapely() -> None:
    p = GeometryPoint3D(10, 20, -1)
    sp = p.to_shapely()
    assert sp.x == 10
    assert sp.y == 20
    assert sp.z == -1
    assert sp.wkt == p.wkt
