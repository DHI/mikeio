"""Locally refined meshes: Faces with Hanging nodes (MIKE 2027, Submesh type 1).

Property tests draw small meshes, refine some Elements so that their
neighbours get Hanging nodes, and check the public geometry against
independent oracles: a brute-force boundary written from the file
specification (split each Face at the Nodes on it), and shapely for the
Elements themselves. Each class of edge case found has a minimal mesh in
CATALOGUE as well, so it fails fast without random search.

Hypothesis profiles are set in conftest.py; `HYPOTHESIS_PROFILE=deep`
searches far wider, for use after changing the geometry code.
"""

from __future__ import annotations

import tempfile
import warnings
from collections import Counter, defaultdict
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
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

import mikeio
from mikeio.spatial import GeometryFM2D, GeometryFM3D

mpl.use("Agg")

REFINED = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh1_LocallyRefined.dfsu"
NOT_REFINED = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh0.dfsu"

# How far off a Face a Node may be and still hang on it, relative to the
# Face length; the same value as in mikeio.spatial._FM_geometry
TOL = 1e-6


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
        xy = np.round(self.xy, 12).tolist()
        return f"Mesh({xy}, {self.elements}, {self.projection!r})"


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
    """A grid of 1-3 by 1-3 quads, triangles or both; interior Nodes jittered."""
    nx, ny = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    kind = draw(st.sampled_from(["quad", "tri", "mixed"]))
    x, y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0))
    xy = np.column_stack([x.ravel(), y.ravel()])
    if draw(st.booleans()):
        # up to 0.2 of the spacing keeps every quad convex
        jitter = st.floats(-0.2, 0.2, allow_nan=False)
        j = np.array(draw(st.lists(jitter, min_size=xy.size, max_size=xy.size)))
        interior = (xy[:, 0] > 0) & (xy[:, 0] < nx) & (xy[:, 1] > 0) & (xy[:, 1] < ny)
        xy = xy + j.reshape(-1, 2) * interior[:, None]

    elements: list[list[int]] = []
    for k in range(ny):
        for i in range(nx):
            a, b = k * (nx + 1) + i, k * (nx + 1) + i + 1
            c, d = b + nx + 1, a + nx + 1
            cell = kind if kind != "mixed" else draw(st.sampled_from(["quad", "tri"]))
            if cell == "quad":
                elements.append([a, b, c, d])
            elif draw(st.booleans()):
                elements += [[a, b, c], [c, d, a]]
            else:
                elements += [[a, b, d], [b, c, d]]
    return Mesh(xy, elements)


@cache
def from_file(filename: str) -> Mesh:
    return as_mesh(read_geometry(filename))


def refine(m: Mesh, selected: set[int]) -> Mesh:
    """Split each selected Element into 4 at its Face midpoints.

    Unselected neighbours of a split Element get a Hanging node. A Node
    already at a midpoint, from an earlier refinement, is reused.
    """
    xy = [p for p in m.xy]
    lo, size = m.xy.min(axis=0), float(np.ptp(m.xy, axis=0).max())
    where = {tuple(np.round((p - lo) / size, 9)): i for i, p in enumerate(xy)}

    def node_at(p: NDArray[np.float64]) -> int:
        key = tuple(np.round((p - lo) / size, 9))
        if key not in where:
            xy.append(p)
            where[key] = len(xy) - 1
        return where[key]

    out: list[list[int]] = []
    for e, el in enumerate(m.elements):
        if e not in selected:
            out.append(el)
            continue
        p = m.xy[el]
        mid = [node_at(0.5 * (p[i] + p[(i + 1) % len(el)])) for i in range(len(el))]
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


def touching(g: GeometryFM2D) -> list[set[int]]:
    """The Elements around each Node: those with the Node, or with it on a Face."""
    xy = g.node_coordinates[:, :2]
    around: list[set[int]] = [set() for _ in range(g.n_nodes)]
    for e, el in enumerate(g.element_table):
        for n in set(el.tolist()) | set(nodes_on_faces(xy, el)):
            around[n].add(e)
    return around


def nodes_on_faces(xy: NDArray[np.float64], el: NDArray[np.int64]) -> list[int]:
    """Nodes strictly inside a Face of the Element, in order along each Face."""
    out = []
    for a, c in zip(el, np.roll(el, -1)):
        d = xy[c] - xy[a]
        r = xy - xy[a]
        t = r @ d / (d @ d)
        offset = np.abs(d[0] * r[:, 1] - d[1] * r[:, 0]) / np.hypot(*d)
        on = np.flatnonzero((t > TOL) & (t < 1 - TOL) & (offset <= TOL * np.hypot(*d)))
        out += on[np.argsort(t[on])].tolist()
    return out


@st.composite
def subset(draw: st.DrawFn, g: GeometryFM2D) -> NDArray[np.int64]:
    """Elements on one side of a line, a random set, or some around a Node.

    Elements around a Node often touch at a single point only.
    """
    kind = draw(st.sampled_from(["half-plane", "random", "around a node"]))
    if kind == "half-plane":
        angle = draw(st.floats(0, 2 * np.pi))
        d = g.element_coordinates[:, :2] @ [np.cos(angle), np.sin(angle)]
        return np.flatnonzero(d <= np.quantile(d, draw(st.floats(0.2, 0.8))))
    if kind == "random":
        elements = st.integers(0, g.n_elements - 1)
        return np.array(sorted(draw(st.sets(elements, min_size=1))))
    around = touching(g)[draw(st.integers(0, g.n_nodes - 1))]
    return np.array(sorted(draw(st.sets(st.sampled_from(sorted(around)), min_size=1))))


def extrude(g: GeometryFM2D, n_layers: int) -> GeometryFM3D:
    """Sigma layers below the 2D mesh, bottom-up in each column."""
    k = n_layers + 1
    nc = [
        [x, y, -1.0 + i / n_layers] for x, y, _ in g.node_coordinates for i in range(k)
    ]
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


# Oracles


def face_pieces(g: GeometryFM2D) -> dict[tuple[int, int], list[int]]:
    """From the spec: each Face split at the Nodes on it, and the Elements of each piece.

    A piece with one Element is on the boundary; with two, it is shared.
    """
    xy = g.node_coordinates[:, :2]
    pieces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for e, el in enumerate(g.element_table):
        for a, c in zip(el, np.roll(el, -1)):
            d, r = xy[c] - xy[a], xy - xy[a]
            t = r @ d / (d @ d)
            offset = np.abs(d[0] * r[:, 1] - d[1] * r[:, 0]) / np.hypot(*d)
            on = np.flatnonzero(
                (t > TOL) & (t < 1 - TOL) & (offset <= TOL * np.hypot(*d))
            )
            chain = [int(a), *on[np.argsort(t[on])].tolist(), int(c)]
            for p, q in zip(chain, chain[1:]):
                pieces[(min(p, q), max(p, q))].append(e)
    return pieces


def boundary_faces(g: GeometryFM2D) -> list[tuple[int, int]]:
    return [face for face, elements in face_pieces(g).items() if len(elements) == 1]


def boundary_length(g: GeometryFM2D) -> float:
    xy = g.node_coordinates[:, :2]
    return float(sum(np.hypot(*(xy[p] - xy[q])) for p, q in boundary_faces(g)))


def connected(n: int, pairs: list[tuple[int, int]]) -> NDArray[np.int64]:
    """A label for each of n vertices: the connected part it belongs to."""
    i, j = np.array(pairs, dtype=np.int64).reshape(-1, 2).T
    graph = coo_array((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def n_domains_and_rings(g: GeometryFM2D) -> tuple[int, int]:
    """Domains: Elements connected by shared pieces. Rings: per domain, boundary pieces connected.

    Counting rings per domain keeps two domains that touch at a point apart.
    """
    pieces = face_pieces(g)
    domain = connected(
        g.n_elements, [(e[0], e[1]) for e in pieces.values() if len(e) == 2]
    )
    rings = 0
    for d in np.unique(domain):
        faces = [
            face for face, e in pieces.items() if len(e) == 1 and domain[e[0]] == d
        ]
        nodes = np.unique(faces)
        label = connected(len(nodes), np.searchsorted(nodes, faces).tolist())
        rings += len(np.unique(label))
    return len(np.unique(domain)), rings


def pinched(g: GeometryFM2D) -> bool:
    """The boundary touches itself at a Node that is a Node of every Element there.

    Boundary tracing cannot handle that yet; it is not specific to Locally
    refined meshes (see test_boundary_pinched_at_a_node).
    """
    degree = Counter(n for face in boundary_faces(g) for n in face)
    on_a_face = {
        n
        for el in g.element_table
        for n in nodes_on_faces(g.node_coordinates[:, :2], el)
    }
    return any(count > 2 and n not in on_a_face for n, count in degree.items())


def ring_length(xy: Any) -> float:
    return float(np.hypot(*np.diff(np.asarray(xy), axis=0).T).sum())


def planar_area(g: GeometryFM2D) -> float:
    """Element area in coordinate units; get_element_area is in m² for LONG/LAT."""
    area = float(sum(p.area for p in g.to_shapely().geoms))
    if g.projection != "LONG/LAT":
        assert g.get_element_area().sum() == pytest.approx(area, rel=1e-9)
    return area


def check_domain(g: GeometryFM2D) -> None:
    """boundary_polygons, contains, find_index and the outline agree with the Elements."""
    xy = g.node_coordinates[:, :2]
    size = float(np.ptp(xy, axis=0).max())
    polygons = g.boundary_polygons

    assert sum(p.area for p in polygons.lines) == pytest.approx(
        planar_area(g), rel=1e-9
    )
    assert (len(polygons.exteriors), len(polygons.lines)) == n_domains_and_rings(g)
    assert sum(ring_length(p.xy) for p in polygons.lines) == pytest.approx(
        boundary_length(g), rel=1e-9
    )

    # inside an Element, away from the boundary; contains is undefined on it
    boundary = shapely.MultiLineString([xy[list(face)] for face in boundary_faces(g)])
    face_midpoints = np.vstack(
        [0.5 * (xy[el] + xy[np.roll(el, -1)]) for el in g.element_table]
    )
    rng = np.random.default_rng(0)
    random_points = rng.uniform(
        xy.min(axis=0) - 0.1 * size, xy.max(axis=0) + 0.1 * size, (200, 2)
    )
    points = np.vstack(
        [g.element_coordinates[:, :2], face_midpoints, xy, random_points]
    )
    points = points[shapely.distance(boundary, shapely.points(points)) > 1e-6 * size]
    # within rounding of an Element: a Hanging node leaves gaps of about 1e-13
    near = shapely.points(points)
    in_an_element = np.any(
        [shapely.dwithin(e, near, 1e-9 * size) for e in g.to_shapely().geoms], axis=0
    )
    np.testing.assert_array_equal(g.contains(points), in_an_element)

    np.testing.assert_array_equal(
        g.find_index(coords=g.element_coordinates), np.arange(g.n_elements)
    )

    ax = g.plot.outline()
    assert len(ax.lines) == len(polygons.lines)
    plt.close("all")


# Properties


@given(m=meshes, data=st.data())
def test_domain_is_the_elements(m: Mesh, data: st.DataObject) -> None:
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
    """Node values are linear in the Element values and depend on nearby Elements only.

    Near a Hanging node that includes the coarse Element. Without extrapolation
    each Node value lies within the values around it.
    """
    g = m.geometry()
    n = g.n_elements
    field = st.lists(st.floats(-10, 10), min_size=n, max_size=n).map(np.array)
    u, w = data.draw(field), data.draw(field)
    a, b, c = data.draw(st.tuples(*[st.floats(-10, 10)] * 3))
    around = touching(g)
    for extrapolate in (True, False):

        def node_values(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return g.get_node_centered_data(v, extrapolate=extrapolate)

        np.testing.assert_allclose(
            node_values(a * u + b * w + c),
            a * node_values(u) + b * node_values(w) + c,
            atol=1e-9 * (1 + 10 * (abs(a) + abs(b)) + abs(c)),
        )
        e = data.draw(st.integers(0, n - 1))
        changed = np.flatnonzero(node_values(np.eye(n)[e]))
        assert all(e in around[node] for node in changed)

    zn = g.get_node_centered_data(u, extrapolate=False)
    for node, elements in enumerate(around):
        if elements:
            v = u[sorted(elements)]
            assert v.min() - 1e-9 <= zn[node] <= v.max() + 1e-9


@settings(max_examples=max(5, settings().max_examples // 3))  # matplotlib is slow
@given(m=meshes)
def test_shaded_plot_triangulates_the_domain(m: Mesh) -> None:
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
    assert area.sum() == pytest.approx(planar_area(g), rel=1e-9)
    assert set(np.unique(tri.triangles)) == set(range(g.n_nodes))

    edges = np.sort(tri.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2
    once = unique[counts == 1]
    length = np.hypot(
        tri.x[once[:, 0]] - tri.x[once[:, 1]], tri.y[once[:, 0]] - tri.y[once[:, 1]]
    )
    assert length.sum() == pytest.approx(boundary_length(g), rel=1e-9)


@given(m=refined(), data=st.data())
def test_invariant_under_relabelling_and_similarity(
    m: Mesh, data: st.DataObject
) -> None:
    """The same Hanging nodes and boundary after renumbering, rotating, scaling and moving."""
    moved, node_map, _ = data.draw(relabelled(m))
    scale = 10.0 ** data.draw(st.integers(-3, 4))
    angle = data.draw(st.sampled_from([0.0, 0.3, np.pi / 2, np.pi]))
    # far from the origin relative to its size, Polygon.area loses precision
    # (test_small_mesh_far_from_the_origin), so offsets scale with the mesh
    offset = scale * np.array(
        data.draw(st.tuples(*[st.sampled_from([0.0, -123.5, 1e4])] * 2))
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    moved.xy = scale * moved.xy @ rotation.T + offset
    note(repr(moved))

    def ring_nodes(g: GeometryFM2D, to_original: Any) -> list[frozenset[Any]]:
        rings = [to_original(np.asarray(p.xy)[:-1]) for p in g.boundary_polygons.lines]
        return sorted(
            (frozenset(map(tuple, np.round(r, 6))) for r in rings), key=sorted
        )

    before, after = m.geometry(), moved.geometry()
    assert len(after.boundary_polygons.exteriors) == len(
        before.boundary_polygons.exteriors
    )
    assert ring_nodes(
        after, lambda xy: ((xy - offset) / scale) @ rotation
    ) == ring_nodes(before, lambda xy: xy)
    assert after.contains(after.element_coordinates[:, :2]).all()

    # #1125: node values depend on the start Node and position of quads, so
    # for now only check renumbering
    relabelled_mesh, node_map, element_map = data.draw(
        relabelled(m, rotate_start=False)
    )
    z = np.arange(before.n_elements, dtype=float) % 5
    zn = before.get_node_centered_data(z)
    zn_relabelled = relabelled_mesh.geometry().get_node_centered_data(z[element_map])
    np.testing.assert_allclose(zn_relabelled, zn[node_map], atol=1e-9)


@given(m=refined(), n_layers=st.integers(1, 3))
def test_layered_mesh_agrees_with_its_2d_geometry(m: Mesh, n_layers: int) -> None:
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

    assert len(g.element_table) == len(stored) == 40
    for a, b in zip(g.element_table, stored):
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
