import numpy as np
import pytest
import shapely

import mikeio

HD2D = "tests/testdata/HD2D.dfsu"
SIGMA_Z3D = "tests/testdata/oresund_sigma_z.dfsu"


def _centres_inside(geometry: object, polygon: shapely.Geometry) -> np.ndarray:
    xy = geometry.element_coordinates[:, :2]  # type: ignore[attr-defined]
    return np.flatnonzero(shapely.contains_xy(polygon, xy[:, 0], xy[:, 1]))


def _hd2d_polygon(da: mikeio.DataArray, shrink: float) -> shapely.Polygon:
    xy = da.geometry.node_coordinates[:, :2]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    return shapely.box(x0 + dx, y0 + dy, x1 - dx, y1 - dy)


def test_sel_area_shapely_polygon_equals_coordinate_list() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    poly = shapely.Polygon([(606000, 6903000), (607300, 6903000), (606600, 6906000)])
    by_shapely = da.sel(area=poly)
    by_list = da.sel(area=list(poly.exterior.coords)[:-1])
    assert by_shapely.geometry.n_elements == by_list.geometry.n_elements > 0
    assert np.array_equal(by_shapely.values, by_list.values)


def test_sel_area_polygon_with_hole_excludes_hole() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    outer = _hd2d_polygon(da, 0.1)
    hole = _hd2d_polygon(da, 0.4)
    poly = shapely.Polygon(outer.exterior, [hole.exterior])

    idx = _centres_inside(da.geometry, poly)
    assert 0 < len(idx) < len(_centres_inside(da.geometry, outer))
    assert da.sel(area=poly).geometry.n_elements == len(idx)
    assert np.array_equal(da.sel(area=poly).values, da.values[:, idx])


def test_sel_area_multipolygon_is_union_of_parts() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    xy = da.geometry.node_coordinates[:, :2]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    xm = (x0 + x1) / 2
    left = shapely.box(x0, y0, xm - (x1 - x0) * 0.1, y1)
    right = shapely.box(xm + (x1 - x0) * 0.1, y0, x1, y1)
    multi = shapely.MultiPolygon([left, right])

    idx = _centres_inside(da.geometry, multi)
    n_parts = len(_centres_inside(da.geometry, left)) + len(
        _centres_inside(da.geometry, right)
    )
    assert len(idx) == n_parts > 0
    assert np.array_equal(da.sel(area=multi).values, da.values[:, idx])


def test_sel_area_polygon_with_z_coordinates() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    poly = _hd2d_polygon(da, 0.2)
    poly_z = shapely.force_3d(poly)
    assert np.array_equal(da.sel(area=poly_z).values, da.sel(area=poly).values)


def test_read_area_shapely_polygon() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    poly = _hd2d_polygon(da, 0.2)
    read = mikeio.read(HD2D, items=["Surface elevation"], area=poly)[0]
    assert np.array_equal(read.values, da.sel(area=poly).values)


@pytest.mark.parametrize(
    "geom",
    [
        shapely.Point(607000, 6906000),
        shapely.LineString([(607000, 6906000), (609000, 6908000)]),
    ],
)
def test_sel_area_non_polygon_geometry_raises(geom: shapely.Geometry) -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        da.sel(area=geom)


class _FeatureCollection:
    """Stands in for a GeoSeries/GeoDataFrame, which expose a FeatureCollection."""

    def __init__(self, polygon: shapely.Polygon) -> None:
        self._polygon = polygon

    @property
    def __geo_interface__(self) -> dict:
        feature = {"type": "Feature", "geometry": self._polygon.__geo_interface__}
        return {"type": "FeatureCollection", "features": [feature]}


def test_sel_area_feature_collection_raises_and_asks_for_single_polygon() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    with pytest.raises(ValueError, match="single"):
        da.sel(area=_FeatureCollection(_hd2d_polygon(da, 0.2)))


def test_sel_area_empty_polygon_mentions_coordinate_system() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    far_away = shapely.box(0, 0, 1, 1)
    with pytest.raises(ValueError, match="coordinate system"):
        da.sel(area=far_away)


def test_layered_sel_area_multipolygon_selects_whole_columns() -> None:
    n = mikeio.read(SIGMA_Z3D)["Temperature"]
    g = n.geometry
    domain = g.geometry2d.to_shapely()
    x0, y0, x1, y1 = domain.bounds
    xm = (x0 + x1) / 2
    multi = shapely.MultiPolygon(
        [shapely.box(x0, y0, xm - 2000, y1), shapely.box(xm + 2000, y0, x1, y1)]
    )
    idx2d = _centres_inside(g.geometry2d, multi)
    idx3d = np.hstack(g.e2_e3_table[idx2d]).astype(int)
    sub = n.sel(area=multi)
    assert sub.geometry.n_elements == len(idx3d)
    assert np.array_equal(sub.values, n.values[:, idx3d])


def test_sel_area_invalid_polygon_raises_and_points_to_make_valid() -> None:
    da = mikeio.read(HD2D)["Surface elevation"]
    bowtie = shapely.Polygon(
        [(606000, 6903000), (607300, 6906000), (607300, 6903000), (606000, 6906000)]
    )
    assert not bowtie.is_valid
    with pytest.raises(ValueError, match="make_valid"):
        da.sel(area=bowtie)
