"""find_index: the Element that contains each point, whatever the mesh (#1133).

find_index first tests the Elements with the nearest centres. When small or
thin Elements crowd the nearest centres, the Element that contains the point
can be further away: beside a coarse Face with Hanging nodes, or in a mesh of
stretched triangles. Shapely is the oracle: the domain is the union of the
Elements as polygons.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import shapely
from hypothesis import given, note
from hypothesis import strategies as st
from numpy.typing import NDArray

from mikeio.exceptions import OutsideModelDomainError
from mikeio.spatial import GeometryFM2D

from .test_dfsu_locally_refined import RTOL, Mesh, domain, extrude, refine, size_of
from .test_dfsu_locally_refined_workflows import square_grid, unbalanced


def stretched(m: Mesh, sx: float, sy: float) -> Mesh:
    return Mesh(m.xy * [sx, sy], m.elements, m.projection)


def graded(m: Mesh, sy: float) -> Mesh:
    """Cells growing to the top right, sy times taller than wide.

    No two neighbouring cells are mirror images, so the centres of the
    Elements on a Face are at different distances from it.
    """
    x, y = m.xy.T
    return Mesh(
        np.column_stack([x * (1 + 0.1 * x), sy * y * (1 + 0.1 * y)]), m.elements
    )


def element_polygons(g: GeometryFM2D) -> NDArray[Any]:
    return np.array(
        [shapely.Polygon(g.node_coordinates[el, :2]) for el in g.element_table]
    )


def assert_found_in_containing_elements(
    g: GeometryFM2D, pts: NDArray[np.float64], idx: NDArray[np.int64]
) -> None:
    """Each point is in its Element, or on its boundary within rounding."""
    polygons = element_polygons(g)[idx]
    distance = shapely.distance(polygons, shapely.points(pts))
    assert distance.max() <= RTOL * size_of(g)


def random_points(
    g: GeometryFM2D, n: int, margin: float = 0.0, seed: int = 0
) -> NDArray[np.float64]:
    xy = g.node_coordinates[:, :2]
    pad = margin * size_of(g)
    return np.random.default_rng(seed).uniform(
        xy.min(axis=0) - pad, xy.max(axis=0) + pad, size=(n, 2)
    )


def inside_and_outside(
    g: GeometryFM2D, pts: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Points clearly inside and clearly outside the domain.

    Points within rounding of the boundary are neither: contains leaves the
    boundary undefined.
    """
    d = domain(g)
    away = shapely.distance(d.boundary, shapely.points(pts)) > RTOL * size_of(g)
    inside = shapely.contains_xy(d, pts[:, 0], pts[:, 1])
    return inside & away, ~inside & away


# Examples


STRETCHED = {
    "tri_x5": lambda: stretched(square_grid(8, quad=False), 5, 1),
    "tri_y5": lambda: stretched(square_grid(8, quad=False), 1, 5),
    "tri_x10": lambda: stretched(square_grid(8, quad=False), 10, 1),
    "refined_tri_y5": lambda: stretched(unbalanced(quad=False), 1, 5),
    "refined_quad_x5": lambda: stretched(unbalanced(quad=True), 5, 1),
}


@pytest.mark.parametrize("make", STRETCHED.values(), ids=STRETCHED.keys())
def test_find_index_in_stretched_elements(make: Callable[[], Mesh]) -> None:
    """In thin Elements, the nearest centres can all be of other Elements."""
    g = make().geometry()
    pts = random_points(g, 6000)
    inside, _ = inside_and_outside(g, pts)
    pts = pts[inside]

    idx = g.find_index(coords=pts)

    assert_found_in_containing_elements(g, pts, idx)


def test_find_index_raises_for_the_points_outside_only() -> None:
    """The error names the points outside, also among points found the slow way."""
    g = stretched(unbalanced(quad=False), 1, 5).geometry()
    pts = random_points(g, 2000, margin=0.2)
    inside, outside = inside_and_outside(g, pts)
    pts = pts[inside | outside]
    expected = np.flatnonzero(outside[inside | outside])
    assert 0 < len(expected) < len(pts)

    with pytest.raises(OutsideModelDomainError) as e:
        g.find_index(coords=pts)

    np.testing.assert_array_equal(e.value.indices, expected)
    np.testing.assert_array_equal(e.value.x, pts[expected, 0])
    np.testing.assert_array_equal(e.value.y, pts[expected, 1])


def test_find_index_on_a_face_takes_the_element_with_the_nearest_centre() -> None:
    """A point on a Face shared by Elements is found in the one with the
    nearest centre, also when the fast check of the nearest centres misses.

    The points are the Face midpoints of a graded mesh refined on the left,
    with cells 10 times taller than wide: on the short Faces, many thin
    Elements are nearer than the ones the point is on.
    """
    g = graded(unbalanced(quad=True), 10).geometry()
    nc = g.node_coordinates[:, :2]
    pts = np.vstack([0.5 * (nc[el] + nc[np.roll(el, -1)]) for el in g.element_table])
    pts = np.unique(pts, axis=0)
    pts = pts[inside_and_outside(g, pts)[0]]  # not on the domain boundary
    polygons = element_polygons(g)
    ec = g.element_coordinates[:, :2]
    expected = []
    clear = []
    for p in pts:
        on = np.flatnonzero(
            shapely.dwithin(polygons, shapely.Point(p), RTOL * size_of(g))
        )
        distance = np.hypot(*(ec[on] - p).T)
        first, second = np.sort(distance)[:2]
        expected.append(on[distance.argmin()])
        # two Elements at the same distance (a diagonal Face): no rule
        clear.append(second - first > RTOL * size_of(g))
    assert sum(clear) > 10
    pts, expected = pts[clear], np.array(expected)[clear]

    idx = g.find_index(coords=pts)

    np.testing.assert_array_equal(idx, expected)


def test_layered_find_index_in_the_column_of_the_containing_element() -> None:
    """A 3D geometry finds the Element of the 2D column, beside a coarse Face."""
    g2 = stretched(unbalanced(quad=True), 1, 5).geometry()
    g3 = extrude(g2, n_layers=3)  # layer 1 from z = -2/3 to -1/3
    pts = random_points(g2, 3000)
    inside, _ = inside_and_outside(g2, pts)
    pts = pts[inside]
    _, column = shapely.STRtree(element_polygons(g2)).query(
        shapely.points(pts), predicate="within"
    )
    expected = {int(g3.e2_e3_table[c][1]) for c in column}

    idx = g3.find_index(coords=np.column_stack([pts, np.full(len(pts), -0.5)]))

    assert set(idx.tolist()) == expected


# Property


@st.composite
def meshes(draw: st.DrawFn) -> Mesh:
    """Grids of up to 8 by 8 cells, stretched up to 10:1, refined in a region.

    Each refinement splits the Elements whose centre is left of a line, so
    beside the line coarse Faces get 1, 3 or 7 Hanging nodes.
    """
    m = square_grid(draw(st.integers(1, 8)), quad=draw(st.booleans()))
    for _ in range(draw(st.integers(0, 3))):
        ec = m.geometry().element_coordinates
        cut = draw(st.floats(float(ec[:, 0].min()), float(ec[:, 0].max())))
        m = refine(m, set(np.flatnonzero(ec[:, 0] <= cut).tolist()))
    s = draw(st.floats(1, 10))
    return stretched(m, *draw(st.sampled_from([(s, 1.0), (1.0, s)])))


@given(m=meshes(), seed=st.integers(0, 2**32 - 1))
def test_find_index_finds_the_containing_element(m: Mesh, seed: int) -> None:
    """Every point inside the domain is found in an Element that contains it;
    find_index raises for the points outside, and names exactly those."""
    g = m.geometry()
    note(repr(m))
    pts = random_points(g, 500, margin=0.1, seed=seed)
    inside, outside = inside_and_outside(g, pts)
    assert g.contains(pts[inside]).all()

    idx = g.find_index(coords=pts[inside])
    assert_found_in_containing_elements(g, pts[inside], idx)

    if outside.any():
        pts = pts[inside | outside]
        with pytest.raises(OutsideModelDomainError) as e:
            g.find_index(coords=pts)
        np.testing.assert_array_equal(
            e.value.indices, np.flatnonzero(outside[inside | outside])
        )


def test_find_index_in_a_mesh_of_one_element() -> None:
    """One id per point, also when there is only one Element to choose."""
    g = square_grid(1, quad=True).geometry()
    pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.9, 0.1]])

    np.testing.assert_array_equal(g.find_index(coords=pts), [0, 0, 0])


def test_find_index_of_no_points() -> None:
    g = square_grid(2, quad=False).geometry()

    assert len(g.find_index(coords=np.empty((0, 2)))) == 0
