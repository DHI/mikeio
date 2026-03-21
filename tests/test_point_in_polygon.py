"""Tests for _point_in_polygon and element search in FM geometry."""

import numpy as np

import mikeio
from mikeio.spatial._FM_geometry import GeometryFM2D


def test_point_in_triangle_inside() -> None:
    """Point inside a triangle is correctly identified."""
    # Counter-clockwise triangle: (0,0), (1,0), (0,1)
    xn = np.array([0.0, 1.0, 0.0])
    yn = np.array([0.0, 0.0, 1.0])

    assert GeometryFM2D._point_in_polygon(xn, yn, 0.2, 0.2) is True


def test_point_in_triangle_outside() -> None:
    """Point outside a triangle is correctly rejected."""
    xn = np.array([0.0, 1.0, 0.0])
    yn = np.array([0.0, 0.0, 1.0])

    assert GeometryFM2D._point_in_polygon(xn, yn, 0.8, 0.8) is False


def test_point_in_polygon_closing_edge() -> None:
    """Point near the closing edge (last→first vertex) is correctly classified.

    The closing edge connects the last vertex back to the first.
    This edge must be checked to get correct results.
    """
    # Triangle: (0,0), (1,0), (0,1)
    xn = np.array([0.0, 1.0, 0.0])
    yn = np.array([0.0, 0.0, 1.0])

    # Just inside the closing edge (hypotenuse)
    assert GeometryFM2D._point_in_polygon(xn, yn, 0.3, 0.3) is True
    # Just outside the closing edge
    assert GeometryFM2D._point_in_polygon(xn, yn, 0.51, 0.51) is False


def test_point_in_quadrilateral() -> None:
    """Point-in-polygon works for quadrilateral elements."""
    # Counter-clockwise square: (0,0), (1,0), (1,1), (0,1)
    xn = np.array([0.0, 1.0, 1.0, 0.0])
    yn = np.array([0.0, 0.0, 1.0, 1.0])

    assert GeometryFM2D._point_in_polygon(xn, yn, 0.5, 0.5) is True
    assert GeometryFM2D._point_in_polygon(xn, yn, 1.5, 0.5) is False


def test_find_element_at_centroid() -> None:
    """find_index at element centroid should return that element."""
    g = mikeio.open("tests/testdata/HD2D.dfsu").geometry
    nc = g.element_coordinates

    for elem_idx in [0, 100, 500, g.n_elements - 1]:
        xy = nc[elem_idx, :2]
        found = g.find_index(x=xy[0], y=xy[1])
        assert found[0] == elem_idx, f"Element {elem_idx}: found {found[0]} instead"


def test_find_element_all_elements_found() -> None:
    """Every element centroid should be findable (no false negatives)."""
    g = mikeio.open("tests/testdata/FakeLake.dfsu").geometry
    nc = g.element_coordinates

    not_found = []
    for i in range(g.n_elements):
        found = g.find_index(x=nc[i, 0], y=nc[i, 1])
        if found[0] != i:
            not_found.append(i)

    assert (
        len(not_found) == 0
    ), f"{len(not_found)} elements not found at their own centroids"
