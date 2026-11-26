"""Locally refined meshes: Faces with Hanging nodes (MIKE 2027, Submesh type 1).

Property tests draw small meshes, refine some Elements so that their
neighbours get Hanging nodes, and check the public geometry against shapely:
the domain is the union of the Elements as polygons. Each class of edge case
found has a minimal mesh in CATALOGUE as well, so it fails fast without
random search.

Hypothesis profiles are set in conftest.py; `HYPOTHESIS_PROFILE=deep`
searches far wider, for use after changing the geometry code.
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from mikecore.DfsuFile import DfsuFile
from numpy.typing import NDArray

import mikeio
from mikeio.spatial import GeometryFM2D, GeometryFM3D

mpl.use("Agg")

REFINED = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh1_LocallyRefined.dfsu"
NOT_REFINED = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh0.dfsu"

# How far off a Face a Node may be and still hang on it, relative to the
# Face length; the same value as in mikeio.spatial._FM_geometry
TOL = 1e-6
# Rounding error allowed when a result is compared with an oracle, relative
# to its size: far above float64 precision and far below TOL
RTOL = 1e-9


@dataclass
class Mesh:
    """Nodes and Elements as plain lists; prints as a copy-pasteable example."""

    xy: NDArray[np.float64]
    elements: list[list[int]]
    projection: str = "NON-UTM"

    def geometry(self) -> GeometryFM2D:
        nc = np.column_stack([self.xy, np.zeros(len(self.xy))])
        return GeometryFM2D(nc, self.elements, projection=self.projection)

    def __repr__(self) -> str:
        return f"Mesh({self.xy.tolist()}, {self.elements}, {self.projection!r})"


def as_mesh(g: GeometryFM2D) -> Mesh:
    elements = [[int(n) for n in el] for el in g.element_table]
    return Mesh(g.node_coordinates[:, :2].copy(), elements, g.projection_string)


# Minimal meshes, one per class of edge case; all counter-clockwise. In the
# drawings, a Hanging node sits on a coarse Face, between its end Nodes.
CATALOGUE = {
    # coarse triangle 0-1-2; Hanging node 3 on its Face 1-2
    #
    #       2
    #      /| \
    #     / |   \
    #    /  3-----4
    #   /   |   /
    #  /    | /
    # 0-----1
    "one_hanging_tri": Mesh(
        np.array([[0, 0], [1, 0], [1, 2], [1, 1], [2, 1]], float),
        [[0, 1, 2], [1, 4, 3], [3, 4, 2]],
    ),
    # coarse quad 0-1-2-3; Hanging node 6 on its Face 1-2
    #
    # 3-----2-----7
    # |     |     |
    # |     |     |
    # |     6-----5
    # |     |     |
    # |     |     |
    # 0-----1-----4
    "one_hanging_quad": Mesh(
        np.array(
            [[0, 0], [1, 0], [1, 2], [0, 2], [2, 0], [2, 1], [1, 1], [2, 2]], float
        ),
        [[0, 1, 2, 3], [1, 4, 5, 6], [6, 5, 7, 2]],
    ),
    # coarse quad 0-1-2-3; Hanging nodes 6 and 7 on its Face 1-2
    #
    # 3-----2-----9
    # |     |     |
    # |     |     |
    # |     7-----8
    # |     |     |
    # |     |     |
    # |     6-----5
    # |     |     |
    # |     |     |
    # 0-----1-----4
    "two_on_one_face": Mesh(
        np.array(
            [
                [0, 0],
                [1, 0],
                [1, 3],
                [0, 3],
                [2, 0],
                [2, 1],
                [1, 1],
                [1, 2],
                [2, 2],
                [2, 3],
            ],
            float,
        ),
        [[0, 1, 2, 3], [1, 4, 5, 6], [6, 5, 8, 7], [7, 8, 9, 2]],
    ),
    # coarse triangle 0-1-2 beside two fine quads; Hanging node 3 on Face 1-2
    #
    #       2-----6
    #     / |     |
    #   /   |     |
    # 0     3-----5
    #   \   |     |
    #     \ |     |
    #       1-----4
    "coarse_tri_fine_quads": Mesh(
        np.array([[0, 1], [1, 0], [1, 2], [1, 1], [2, 0], [2, 1], [2, 2]], float),
        [[0, 1, 2], [1, 4, 5, 3], [3, 5, 6, 2]],
    ),
    # no Hanging node: boundary Nodes 1 and 4 are in line with the Faces
    # beside them, but at their ends
    #
    # 5-----4-----3
    # |     |     |
    # |     |     |
    # 0-----1-----2
    "face_ends_not_hanging": Mesh(
        np.array([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]], float),
        [[0, 1, 4, 5], [1, 2, 3, 4]],
    ),
    # one_hanging_tri in Geographic coordinates: 4.5e-6 degrees per unit,
    # with Node 0 at 4.5 degrees east on the equator
    "tiny_longlat": Mesh(
        np.array([[0, 0], [1, 0], [1, 2], [1, 1], [2, 1]]) * 4.5e-6 + [4.5, 0],
        [[0, 1, 2], [1, 4, 3], [3, 4, 2]],
        "LONG/LAT",
    ),
    # T-touch, no Hanging node: corner 2 of triangle 2-3-5 touches the middle
    # of Face 1-4 of triangle 0-1-4 from outside, so two domains
    #
    # 4     5
    # | \   | \
    # |   \ |   \
    # |     2-----3
    # |       \
    # |         \
    # 0-----------1
    "t_touch": Mesh(
        np.array([[0, 0], [2, 0], [1, 1], [2, 1], [0, 2], [1, 2]], float),
        [[0, 1, 4], [2, 3, 5]],
    ),
}

# 0-based indices of the Hanging nodes in REFINED (11 and 13 in the file)
REFINED_HANGING_NODES = [10, 12]
HANGING_NODES = {"one_hanging_quad": [6], "two_on_one_face": [6, 7]}


def read_geometry(filename: str | Path) -> GeometryFM2D:
    g = mikeio.Dfsu2DH(filename).geometry
    assert isinstance(g, GeometryFM2D)
    return g


def linear(x: Any, y: Any) -> Any:
    return 2 * x + 3 * y + 10


# Strategies


@st.composite
def grid(draw: st.DrawFn) -> Mesh:
    r"""A grid of 1-3 by 1-3 unit cells: quads, triangles or both.

    Nodes are numbered row by row from the bottom left; nx = 3, ny = 2:

        8-----9----10----11
        |     |     |     |
        4-----5-----6-----7      cell:  d-----c
        |     |     |     |             |     |
        0-----1-----2-----3             a-----b

    A cell is a quad [a, b, c, d] or two triangles, split along either
    diagonal:

        d-----c          d-----c
        |   / |          | \   |
        | /   |          |   \ |
        a-----b          a-----b
        [a, b, c]        [a, b, d]
        [c, d, a]        [b, c, d]

    Half the grids are jittered: each interior Node (5 and 6 above) moves by
    an independent offset of up to 0.2 of the cell size in x and in y. Nodes
    on the outline stay put, so the domain is still the rectangle.
    """
    nx, ny = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    # every cell a quad, every cell two triangles, or each cell drawn at random
    kind = draw(st.sampled_from(["quad", "tri", "mixed"]))
    x, y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0))
    xy = np.column_stack([x.ravel(), y.ravel()])
    if draw(st.booleans()):
        # A corner is 1/sqrt(2) = 0.71 from the diagonal between its
        # neighbours; corner and diagonal each move at most sqrt(2) * 0.2 =
        # 0.28, so every quad stays convex.
        max_jitter = 0.2
        jitter = st.floats(-max_jitter, max_jitter, allow_nan=False)
        j = np.array(draw(st.lists(jitter, min_size=xy.size, max_size=xy.size)))
        interior = (xy[:, 0] > 0) & (xy[:, 0] < nx) & (xy[:, 1] > 0) & (xy[:, 1] < ny)
        xy = xy + j.reshape(-1, 2) * interior[:, None]

    elements: list[list[int]] = []
    for k in range(ny):
        for i in range(nx):
            a, b = k * (nx + 1) + i, k * (nx + 1) + i + 1
            c, d = b + nx + 1, a + nx + 1
            # this cell: quad or triangles, as the grid says or drawn when mixed
            cell = kind if kind != "mixed" else draw(st.sampled_from(["quad", "tri"]))
            if cell == "quad":
                elements.append([a, b, c, d])
            elif draw(st.booleans()):  # which diagonal splits it
                elements += [[a, b, c], [c, d, a]]
            else:
                elements += [[a, b, d], [b, c, d]]
    return Mesh(xy, elements)


@cache
def from_file(filename: str) -> Mesh:
    return as_mesh(read_geometry(filename))


def refine(m: Mesh, selected: set[int]) -> Mesh:
    r"""Split each selected Element into 4 at its Face midpoints.

    A triangle [a, b, c] and a quad [a, b, c, d], with new Nodes at the
    midpoints (ab, bc, ...) and, for the quad, the centre o:

                c                 d----cd-----c
               / \                |     |     |
              /   \               |     |     |
            ca-----bc             da----o-----bc
            / \   / \             |     |     |
           /   \ /   \            |     |     |
          a-----ab----b           a----ab-----b

        [a, ab, ca]  [ab, b, bc]   [a, ab, o, da]  [b, bc, o, ab]
        [ca, bc, c]  [ab, bc, ca]  [c, cd, o, bc]  [d, da, o, cd]

    An unselected neighbour across a-b keeps its Face a-b, so ab becomes a
    Hanging node on it. A midpoint with a Node already there, within
    rounding, reuses it: the midpoint of a Face shared by two split
    Elements, or a Hanging node from an earlier refinement.
    """
    xy = list(m.xy)
    size = float(np.ptp(m.xy, axis=0).max())

    def node_at(p: NDArray[np.float64]) -> int:
        distance = np.hypot(*(np.array(xy) - p).T)
        if distance.min() <= RTOL * size:
            return int(distance.argmin())
        xy.append(p)
        return len(xy) - 1

    out: list[list[int]] = []
    for e, el in enumerate(m.elements):
        if e not in selected:
            out.append(el)
            continue
        p = m.xy[el]
        # p is [a, b, c], rolled [b, c, a]: midpoints [ab, bc, ca]
        mid = [node_at(q) for q in 0.5 * (p + np.roll(p, -1, axis=0))]
        if len(el) == 3:
            (a, b, c), (ab, bc, ca) = el, mid
            out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        else:
            centre = node_at(p.mean(axis=0))
            out += [[el[i], mid[i], centre, mid[i - 1]] for i in range(4)]
    return Mesh(np.array(xy), out, m.projection)


@st.composite
def refined(draw: st.DrawFn, base: st.SearchStrategy[Mesh] | None = None) -> Mesh:
    """A mesh with 1 or 2 levels of refinement of some of its Elements."""
    m = draw(base if base is not None else grid())
    for _ in range(draw(st.integers(1, 2))):
        n = len(m.elements)
        m = refine(
            m, draw(st.sets(st.integers(0, n - 1), min_size=1, max_size=max(1, n - 1)))
        )
    return m


files = st.sampled_from([REFINED, NOT_REFINED]).map(from_file)
meshes = st.one_of(grid(), refined(), refined(files))


@st.composite
def relabelled(
    draw: st.DrawFn, m: Mesh, rotate_start: bool = True
) -> tuple[Mesh, Any, Any]:
    """Renumber Nodes, reorder Elements, and rotate each Element's start Node.

    Returns the new mesh, and for each new Node and Element the old one.
    For example, with node_map [2, 0, 3, 1] and element_map [1, 0]:

        old                new
        3-----2            2-----0
        | e1 /|            | E0 /|      E0 = e1: [0, 2, 3] -> [1, 0, 2],
        |  /  |            |  /  |           rotated to start at 0: [0, 2, 1]
        |/ e0 |            |/ E1 |      E1 = e0: [0, 1, 2] -> [1, 3, 0]
        0-----1            1-----3

    new = argsort(node_map) inverts the map: old Node 0 is new Node
    new[0] = 1, and old Node 2 is new Node new[2] = 0.
    """
    rnd = draw(st.randoms(use_true_random=False))
    node_map = np.array(rnd.sample(range(len(m.xy)), len(m.xy)))
    element_map = np.array(rnd.sample(range(len(m.elements)), len(m.elements)))
    new = np.argsort(node_map)
    elements = []
    for e in element_map:
        el = [int(new[n]) for n in m.elements[e]]
        k = rnd.randrange(len(el)) if rotate_start else 0
        elements.append(el[k:] + el[:k])
    return Mesh(m.xy[node_map], elements, m.projection), node_map, element_map


@st.composite
def subset(draw: st.DrawFn, g: GeometryFM2D) -> NDArray[np.int64]:
    """Elements on one side of a line, or a random set.

    For the line, each Element centre is projected on a direction, d = centre
    . direction, and the Elements below a quantile of d are kept. With angle
    0 the direction is x, and the line is upright:

          x   x   x | x   x
            x   x   |   x   x      ->  direction
          x   x   x | x   x
            kept    | dropped
    """
    if draw(st.booleans()):
        angle = draw(st.floats(0, 2 * np.pi))
        d = g.element_coordinates[:, :2] @ [np.cos(angle), np.sin(angle)]
        fraction = draw(st.floats(0.2, 0.8))  # never all or none of the Elements
        return np.flatnonzero(d <= np.quantile(d, fraction))
    elements = st.integers(0, g.n_elements - 1)
    return np.array(sorted(draw(st.sets(elements, min_size=1))))


def extrude(g: GeometryFM2D, n_layers: int) -> GeometryFM3D:
    """Sigma layers below the 2D mesh, bottom-up in each column.

    2D Node n becomes the column of Nodes n * k to n * k + n_layers, with
    k = n_layers + 1 Nodes per column. With 2 layers, a 2D Face 0-1:

         z   2D Node 0  2D Node 1
         0        2-------5
                  |       |        layer 1: [1, 4, ..] + [2, 5, ..]
       -0.5       1-------4
                  |       |        layer 0: [0, 3, ..] + [1, 4, ..]
        -1        0-------3                 bottom Nodes + top Nodes

    so a 2D triangle [0, 1, 2] gives the prisms [0, 3, 6, 1, 4, 7] and
    [1, 4, 7, 2, 5, 8], in that order: the layers of an Element are
    consecutive.
    """
    k = n_layers + 1  # Nodes per column
    depth = np.linspace(-1.0, 0.0, k)
    nc = [[x, y, z] for x, y, _ in g.node_coordinates for z in depth]
    elements = [
        [n * k + i for n in el] + [n * k + i + 1 for n in el]
        for el in g.element_table
        for i in range(n_layers)
    ]
    return GeometryFM3D(
        node_coordinates=np.array(nc),
        element_table=elements,
        projection="NON-UTM",
        n_layers=n_layers,
        n_sigma=n_layers,
    )


# Oracles: shapely, on the Elements as polygons


def size_of(g: GeometryFM2D) -> float:
    return float(np.ptp(g.node_coordinates[:, :2], axis=0).max())


def domain(g: GeometryFM2D) -> Any:
    """The union of the Elements.

    Each Element is first snapped to the Nodes within rounding of it, which
    puts the Hanging nodes on its Faces; a plain union leaves slivers along
    them.
    """
    nodes = shapely.MultiPoint(g.node_coordinates[:, :2])
    snapped = [shapely.snap(e, nodes, RTOL * size_of(g)) for e in g.to_shapely().geoms]
    return shapely.union_all(snapped)


def touching(g: GeometryFM2D) -> list[set[int]]:
    """The Elements around each Node: those that reach it, within rounding."""
    elements = np.array(g.to_shapely().geoms)
    return [
        set(np.flatnonzero(shapely.dwithin(elements, p, RTOL * size_of(g))).tolist())
        for p in shapely.points(g.node_coordinates[:, :2])
    ]


def pinched(g: GeometryFM2D) -> bool:
    """The boundary touches itself at a point.

    Boundary tracing cannot handle that yet; it is not specific to Locally
    refined meshes (see test_boundary_pinched_at_a_node).
    """
    return not domain(g).boundary.is_simple


def check_domain(g: GeometryFM2D) -> None:
    """boundary_polygons, contains, find_index and the outline agree with the domain."""
    xy = g.node_coordinates[:, :2]
    expected = domain(g)
    parts = list(getattr(expected, "geoms", [expected]))
    polygons = g.boundary_polygons

    assert len(polygons.exteriors) == len(parts)
    assert len(polygons.lines) == sum(1 + len(p.interiors) for p in parts)
    assert sum(p.area for p in polygons.lines) == pytest.approx(expected.area, rel=RTOL)
    perimeter = sum(shapely.LineString(p.xy).length for p in polygons.lines)
    assert perimeter == pytest.approx(expected.length, rel=RTOL)

    # Points at the Elements' centres, Face midpoints and Nodes, and random
    # points in and around the mesh; contains is undefined on the boundary, so
    # only those further than TOL from it
    face_midpoints = np.vstack(
        # [a, b, c] with [b, c, a]: the midpoints of a-b, b-c and c-a
        [0.5 * (xy[el] + xy[np.roll(el, -1)]) for el in g.element_table]
    )
    margin = 0.1 * size_of(g)
    random_points = np.random.default_rng(0).uniform(
        xy.min(axis=0) - margin, xy.max(axis=0) + margin, size=(200, 2)
    )
    points = np.vstack(
        [g.element_coordinates[:, :2], face_midpoints, xy, random_points]
    )
    away = shapely.distance(expected.boundary, shapely.points(points)) > TOL * size_of(
        g
    )
    points = points[away]
    np.testing.assert_array_equal(
        g.contains(points), shapely.contains_xy(expected, points)
    )

    np.testing.assert_array_equal(
        g.find_index(coords=g.element_coordinates), np.arange(g.n_elements)
    )

    ax = g.plot.outline()
    assert len(ax.lines) == len(polygons.lines)
    plt.close("all")


# Properties


@given(m=meshes, data=st.data())
def test_domain_is_the_elements(m: Mesh, data: st.DataObject) -> None:
    """The geometry's domain is the union of its Elements.

    For any mesh, subset and numbering, also after writing to and reading
    from .dfsu or .mesh: boundary_polygons has the union's parts, rings, area
    and perimeter; contains agrees with it; find_index finds each Element at
    its centre; the outline draws each ring.
    """
    g = m.geometry()
    if data.draw(st.booleans()):
        sub = g.isel(data.draw(subset(g)).tolist(), keepdims=True)
        assert isinstance(sub, GeometryFM2D)
        g = sub
    g = data.draw(relabelled(as_mesh(g)))[0].geometry()
    note(repr(as_mesh(g)))
    assume(not pinched(g))
    via = data.draw(st.sampled_from([None, None, "dfsu", "mesh"]))
    if via is not None:  # the geometry as read back from a file
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / f"mesh.{via}"
            if via == "dfsu":
                mikeio.DataArray(np.zeros(g.n_elements), geometry=g).to_dfs(filename)
            else:
                g.to_mesh(filename)
            g = (
                read_geometry(filename)
                if via == "dfsu"
                else mikeio.Mesh(filename).geometry
            )

    check_domain(g)


@given(m=meshes, data=st.data())
def test_node_values_are_a_local_linear_map(m: Mesh, data: st.DataObject) -> None:
    """Node values are a linear map of the Element values that keeps constants.

    A Node depends only on the Elements around it; around a Hanging node that
    includes the coarse Element. Without extrapolation each Node value lies
    between the smallest and largest Element value around it.
    """
    g = m.geometry()
    n = g.n_elements
    bound = 10.0  # on the Element values and the coefficients
    value = st.floats(-bound, bound)
    field = st.lists(value, min_size=n, max_size=n).map(np.array)
    u, w = data.draw(field), data.draw(field)
    a, b, c = data.draw(st.tuples(value, value, value))
    around = touching(g)
    for extrapolate in (True, False):

        def node_values(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return g.get_node_centered_data(v, extrapolate=extrapolate)

        np.testing.assert_allclose(
            node_values(a * u + b * w + c),
            a * node_values(u) + b * node_values(w) + c,
            atol=RTOL * (1 + bound * (abs(a) + abs(b)) + abs(c)),  # result size
        )
        e = data.draw(st.integers(0, n - 1))
        one_element = np.eye(n)[e]  # 1 at Element e, 0 elsewhere
        changed = np.flatnonzero(node_values(one_element))
        assert all(e in around[node] for node in changed)

    zn = g.get_node_centered_data(u, extrapolate=False)
    for node, elements in enumerate(around):
        if elements:
            v = u[sorted(elements)]
            assert v.min() - RTOL * bound <= zn[node] <= v.max() + RTOL * bound


@settings(max_examples=max(5, settings().max_examples // 3))  # matplotlib is slow
@given(m=meshes)
def test_shaded_plot_triangulates_the_domain(m: Mesh) -> None:
    """The shaded plot's triangles tile the domain.

    All counter-clockwise, together the domain's area, using every Node, each
    edge in at most two triangles, and the edges in one as long as the boundary.
    """
    g = m.geometry()
    da = mikeio.DataArray(np.arange(g.n_elements, dtype=float), geometry=g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ax = da.plot(plot_type="shaded", show_outline=False)
    (mesh,) = [c for c in ax.collections if type(c).__name__ == "TriMesh"]
    plt.close("all")
    tri = mesh._triangulation  # no public accessor in matplotlib

    p = np.column_stack([tri.x, tri.y])[tri.triangles]
    ab, ac = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    assert (area > 0).all()
    assert area.sum() == pytest.approx(domain(g).area, rel=RTOL)
    assert set(np.unique(tri.triangles)) == set(range(g.n_nodes))

    edges = np.sort(tri.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2
    once = unique[counts == 1]
    length = np.hypot(
        tri.x[once[:, 0]] - tri.x[once[:, 1]], tri.y[once[:, 0]] - tri.y[once[:, 1]]
    )
    assert length.sum() == pytest.approx(domain(g).length, rel=RTOL)


@given(m=refined(), data=st.data())
def test_invariant_under_relabelling_and_similarity(
    m: Mesh, data: st.DataObject
) -> None:
    """Renumbering, rotating, scaling and moving a mesh changes nothing.

    The same boundary rings through the same Nodes, and every Element centre
    inside. Node values only follow the renumbering, not the start Node of
    each Element or the move (#1125).
    """
    moved, node_map, _ = data.draw(relabelled(m))
    scale = 10.0 ** data.draw(st.integers(-3, 4))
    # a quarter and a half turn keep the Faces of a grid along the axes;
    # 0.3 rad turns them off the axes
    angle = data.draw(st.sampled_from([0.0, 0.3, np.pi / 2, np.pi]))
    # Far from the origin relative to its size, Polygon.area loses precision
    # (test_small_mesh_far_from_the_origin), so the offset is a multiple of
    # the scale: none, a negative non-integer, or far from the mesh
    shift = st.sampled_from([0.0, -123.5, 1e4])
    offset = scale * np.array(data.draw(st.tuples(shift, shift)))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    moved.xy = scale * moved.xy @ rotation.T + offset
    note(repr(moved))

    def ring_nodes(g: GeometryFM2D, to_m: NDArray[np.int64]) -> list[list[int]]:
        """The Nodes of each boundary ring, numbered as in m."""
        xy = g.node_coordinates[:, :2]
        rings = []
        for line in g.boundary_polygons.lines:
            ring = np.asarray(line.xy)[:-1]  # the last point repeats the first
            # distance from each ring point (row) to each Node (column); the
            # nearest Node is the ring point itself
            #                Node 0  Node 1  Node 2
            #   ring point 0    0      1.0     1.4    ->  Node 0
            #   ring point 1   1.0      0      1.0    ->  Node 1
            node = np.linalg.norm(ring[:, None] - xy, axis=2).argmin(axis=1)
            rings.append(sorted(to_m[node].tolist()))
        return sorted(rings)

    before, after = m.geometry(), moved.geometry()
    assert len(after.boundary_polygons.exteriors) == len(
        before.boundary_polygons.exteriors
    )
    assert ring_nodes(after, node_map) == ring_nodes(before, np.arange(len(m.xy)))
    assert after.contains(after.element_coordinates[:, :2]).all()

    # #1125: node values depend on the start Node and position of quads, so
    # for now only check renumbering
    relabelled_mesh, node_map, element_map = data.draw(
        relabelled(m, rotate_start=False)
    )
    z = np.cos(np.arange(before.n_elements))  # any field, of size 1
    zn = before.get_node_centered_data(z)
    zn_relabelled = relabelled_mesh.geometry().get_node_centered_data(z[element_map])
    np.testing.assert_allclose(zn_relabelled, zn[node_map], atol=RTOL)


@given(m=refined(), n_layers=st.integers(1, 3))
def test_layered_mesh_agrees_with_its_2d_geometry(m: Mesh, n_layers: int) -> None:
    """A layered mesh has the domain of its 2D mesh.

    The same boundary polygons and contains, and its top layer has the same area.
    """
    g2 = m.geometry()
    g3 = extrude(g2, n_layers)

    for a, b in zip(
        g3.boundary_polygons.lines, g2.boundary_polygons.lines, strict=True
    ):
        np.testing.assert_array_equal(a.xy, b.xy)
    points = g2.element_coordinates[:, :2]
    np.testing.assert_array_equal(g3.contains(points), g2.contains(points))
    top = (
        mikeio.DataArray(np.ones(g3.n_elements), geometry=g3).sel(layers="top").geometry
    )
    assert top.get_element_area().sum() == pytest.approx(g2.get_element_area().sum())


# Examples


@pytest.mark.parametrize("name", CATALOGUE)
def test_catalogue_domain(name: str) -> None:
    check_domain(CATALOGUE[name].geometry())


def one_hanging_tri_with_node_3_at(x: float, y: float) -> GeometryFM2D:
    m = CATALOGUE["one_hanging_tri"]
    xy = m.xy.copy()
    xy[3] = [x, y]  # on Face 1-2, of length 2, when x is 1
    return Mesh(xy, m.elements).geometry()


@pytest.mark.parametrize(
    ("offset", "n_rings"),
    [(0.0, 1), (0.9 * TOL, 1), (1.1 * TOL, 2)],
    ids=["on the Face", "just within tolerance", "just beyond tolerance"],
)
def test_hanging_node_distance_from_face(offset: float, n_rings: int) -> None:
    g = one_hanging_tri_with_node_3_at(1 + 2 * offset, 1)

    assert len(g.boundary_polygons.lines) == n_rings


@pytest.mark.parametrize(
    ("t", "n_rings"),
    [(0.9 * TOL, 2), (1.1 * TOL, 1)],
    ids=["within tolerance of the Face end", "just beyond it"],
)
def test_hanging_node_distance_from_face_end(t: float, n_rings: int) -> None:
    g = one_hanging_tri_with_node_3_at(1, 2 * t)

    assert len(g.boundary_polygons.lines) == n_rings


@pytest.mark.parametrize("plot_type", ["contour", "contourf"])
def test_refined_plot_of_a_locally_refined_mesh(plot_type: str) -> None:
    """Refining the plot triangulation crashed at Hanging nodes (#1112)."""
    g = read_geometry(REFINED)

    with pytest.warns(FutureWarning, match="n_refinements"):
        getattr(g.plot, plot_type)(n_refinements=1)
    plt.close("all")


@pytest.mark.parametrize("name", HANGING_NODES)
def test_coarse_element_contributes_to_hanging_nodes(name: str) -> None:
    g = CATALOGUE[name].geometry()

    zn = g.get_node_centered_data(np.eye(g.n_elements)[0])  # Element 0 is coarse

    assert (zn[HANGING_NODES[name]] > 0).all()


def test_node_centered_data_at_hanging_nodes() -> None:
    g = read_geometry(REFINED)
    ec, nc = g.element_coordinates, g.node_coordinates

    zn = g.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    # Using only the fine Elements gave an error of 32. The pseudo-laplacian
    # is not exact for a linear field, so some error remains (#1125).
    exact = linear(nc[REFINED_HANGING_NODES, 0], nc[REFINED_HANGING_NODES, 1])
    assert zn[REFINED_HANGING_NODES] == pytest.approx(exact, abs=12)


@pytest.mark.parametrize("name", HANGING_NODES)
def test_quad_mesh_node_centered_data_at_hanging_nodes(name: str) -> None:
    g = CATALOGUE[name].geometry()
    ec, nc = g.element_coordinates, g.node_coordinates

    zn = g.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    # error 2.5 from the fine Elements only
    nodes = HANGING_NODES[name]
    assert zn[nodes] == pytest.approx(linear(nc[nodes, 0], nc[nodes, 1]), abs=0.5)


def test_element_table_is_as_stored_in_file() -> None:
    g = read_geometry(REFINED)
    _ = g.boundary_polygons  # finds the Hanging nodes

    dfs = DfsuFile.Open(REFINED)
    stored = [np.asarray(e) - 1 for e in dfs.ElementTable]
    dfs.Close()

    for a, b in zip(g.element_table, stored, strict=True):
        np.testing.assert_array_equal(a, b)


def submesh_type(filename: str | Path) -> int | None:
    dfs = DfsuFile.Open(str(filename))
    try:
        return dfs.SubMeshType
    finally:
        dfs.Close()


@pytest.mark.parametrize(
    ("filename", "elements", "expected"),
    [
        (REFINED, None, 1),
        (REFINED, range(10), 1),
        (NOT_REFINED, None, 0),
        ("tests/testdata/HD2D.dfsu", None, None),
    ],
    ids=["locally refined", "subset", "not refined", "no Submesh type"],
)
def test_to_dfs_keeps_submesh_type(
    filename: str, elements: range | None, expected: int | None, tmp_path: Path
) -> None:
    ds = mikeio.read(filename)
    if elements is not None:
        ds = ds.isel(element=elements)

    ds.to_dfs(tmp_path / "copy.dfsu")

    assert submesh_type(tmp_path / "copy.dfsu") == expected


# Known bugs, not specific to Locally refined meshes; strict, so a fix shows


@pytest.mark.xfail(
    strict=True, reason="boundary tracing where the boundary touches itself at a Node"
)
def test_boundary_pinched_at_a_node() -> None:
    xy = np.array(
        [[150, 0], [150, 25], [150, 50], [134, 0], [134, 50], [134, 25]], float
    )
    g = Mesh(xy, [[2, 5, 1], [1, 3, 0], [5, 2, 4]]).geometry()

    assert sum(p.area for p in g.boundary_polygons.lines) == pytest.approx(600.0)


@pytest.mark.xfail(
    strict=True, raises=IndexError, reason="Polygon.area far from the origin"
)
def test_small_mesh_far_from_the_origin() -> None:
    g = Mesh(np.array([[0, 0], [1, 0], [0, 1]]) * 0.01 + 1e6, [[0, 1, 2]]).geometry()

    assert g.contains(g.element_coordinates[:, :2]).all()
