"""Submesh results: Hanging nodes and the conforming element table.

Only a Submesh result that is a Locally refined mesh differs from an
ordinary Flexible Mesh.

In a Locally refined mesh a coarse Face (A-C) meets two or more fine Faces
(A-B, B-C) at Hanging nodes (B). The element table stores the coarse Element
with Nodes A and C only, so the coarse Face and the fine Faces each occur
once, like Faces on the domain boundary.

The conforming element table inserts each Hanging node as an extra Node of
the coarse Element, between the ends of the Face it lies on. Then every
interior Face is again shared by exactly two Elements.

Hanging nodes are found from the node coordinates and the element table only,
never from the Submesh type in the dfsu header (ADR-014). The same code
therefore handles a Locally refined mesh that has lost its Submesh type, or
that was built in code.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

# Distance tolerance, relative to the length of the Face
_TOLERANCE = 1e-6


def conforming_element_table(
    node_coordinates: NDArray[np.floating],
    element_table: np.ndarray | Sequence[ArrayLike],
) -> list[NDArray[np.int64]] | None:
    """Element table with each Hanging node as an extra Node of its Element.

    The same format as `element_table`: one array of Nodes per Element, in
    counter-clockwise order. Returns None when there are no Hanging nodes.

    Examples
    --------
    A coarse triangle beside two fine triangles. Node 3 is a Hanging node
    on the coarse Face 1-2.

    >>> nc = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 1, 0], [2, 1, 0]])
    >>> table = conforming_element_table(nc, [[0, 1, 2], [1, 4, 3], [3, 4, 2]])
    >>> table[0]
    array([0, 1, 3, 2])
    >>> conforming_element_table(nc, [[0, 1, 2]]) is None
    True

    """
    connectivity, n_nodes = to_connectivity(element_table)
    if n_nodes.size == 0:
        return None

    # Faces that occur once: on the domain boundary, or beside a Hanging node
    faces, element, position = _faces(connectivity, n_nodes)
    _, inverse, counts = np.unique(
        _face_keys(faces), return_inverse=True, return_counts=True
    )
    once = np.flatnonzero(counts[inverse] == 1)
    if once.size < 3:
        return None

    start, end = faces[once, 0], faces[once, 1]
    xy = np.asarray(node_coordinates, dtype=float)[:, :2]
    a, c = xy[start], xy[end]
    d = c - a
    face_length = np.hypot(d[:, 0], d[:, 1])

    # Nodes of once-only Faces within reach of each once-only Face
    candidates = np.unique(np.concatenate([start, end]))
    tree = cKDTree(xy[candidates])
    reach = 0.5 * face_length * (1.0 + _TOLERANCE)
    near = tree.query_ball_point(0.5 * (a + c), reach, return_sorted=False)
    n_near = np.fromiter((len(n) for n in near), dtype=np.int64, count=len(near))
    face = np.repeat(np.arange(once.size), n_near)
    node = candidates[np.concatenate(near).astype(np.int64)]

    keep = (node != start[face]) & (node != end[face])
    face, node = face[keep], node[keep]

    # On the Face, strictly between its ends
    ap = xy[node] - a[face]
    df = d[face]
    length = face_length[face]
    t = (ap[:, 0] * df[:, 0] + ap[:, 1] * df[:, 1]) / length**2
    offset = np.abs(df[:, 0] * ap[:, 1] - df[:, 1] * ap[:, 0]) / length
    on_face = (
        (t > _TOLERANCE) & (t < 1.0 - _TOLERANCE) & (offset <= _TOLERANCE * length)
    )
    if not on_face.any():
        return None
    face, node, t = face[on_face], node[on_face], t[on_face]

    # Rebuild only the Elements that get Hanging nodes. Each Hanging node
    # goes after the start Node of its Face, in order along the Face: sort
    # by Element, position, then distance along the Face (0 for the
    # Element's own Nodes).
    entry = once[face]
    refined = np.unique(element[entry])
    own = np.isin(element, refined)
    order = np.lexsort(
        (
            np.concatenate([np.zeros(own.sum()), t]),
            np.concatenate([position[own], position[entry]]),
            np.concatenate([element[own], element[entry]]),
        )
    )
    nodes = np.concatenate([connectivity[own], node])[order]
    sizes = n_nodes[refined] + np.bincount(element[entry])[refined]

    table = [np.asarray(el) for el in element_table]
    for e, el in zip(refined.tolist(), np.split(nodes, np.cumsum(sizes)[:-1])):
        table[e] = el.astype(table[e].dtype)
    return table


def boundary_faces(
    element_table: np.ndarray | Sequence[ArrayLike],
) -> NDArray[np.int64]:
    """Faces that occur once, as (start, end) Node pairs.

    Ordered by their sorted Node pair; each in the direction of its Element.
    """
    faces, _, _ = _faces(*to_connectivity(element_table))
    _, first, counts = np.unique(
        _face_keys(faces), return_index=True, return_counts=True
    )
    return faces[first[counts == 1]]


def triangulate(
    node_coordinates: NDArray[np.floating],
    element_table: np.ndarray | Sequence[ArrayLike],
    conforming: np.ndarray | Sequence[ArrayLike],
) -> NDArray[np.int64]:
    """Split each Element of the conforming element table into triangles.

    None of the triangles has zero area. A quad without Hanging nodes is
    split as `0-1-2` and `2-3-0`. An Element with Hanging nodes is fanned
    out from a Hanging node that is alone on its Face; when there is none,
    it is split by ear clipping.
    """
    xy = np.asarray(node_coordinates, dtype=float)[:, :2]
    connectivity, n_nodes = to_connectivity(conforming)
    original, original_n_nodes = to_connectivity(element_table)
    element = np.repeat(np.arange(n_nodes.size), n_nodes)
    refined = n_nodes != original_n_nodes

    # A Hanging node is a Node of a refined conforming Element that is not a
    # Node of the Element in the element table
    in_refined = refined[element]
    original_element = np.repeat(np.arange(n_nodes.size), original_n_nodes)
    original_in_refined = refined[original_element]
    key = int(max(connectivity.max(), original.max())) + 1
    hanging = np.zeros(connectivity.size, dtype=bool)
    hanging[in_refined] = ~np.isin(
        element[in_refined] * key + connectivity[in_refined],
        original_element[original_in_refined] * key + original[original_in_refined],
    )

    first = np.zeros(n_nodes.size, dtype=np.int64)
    np.cumsum(n_nodes[:-1], out=first[1:])
    plain = first[~refined & (n_nodes == 3)]
    quads = connectivity[first[~refined & (n_nodes == 4)][:, None] + np.arange(4)]
    tris = [
        connectivity[plain[:, None] + np.arange(3)],
        quads[:, [0, 1, 2]],
        quads[:, [2, 3, 0]],
    ]

    unsplit: list[int] = []
    for n in np.unique(n_nodes[refined]).tolist():
        elements = np.flatnonzero(refined & (n_nodes == n))
        entries = first[elements][:, None] + np.arange(n)
        on_face = hanging[entries]
        alone = on_face & ~np.roll(on_face, 1, axis=1) & ~np.roll(on_face, -1, axis=1)
        apex = np.argmax(alone, axis=1)
        fan = (apex[:, None, None] + _fan(n)) % n
        fan_nodes = connectivity[np.take_along_axis(entries[:, None, :], fan, axis=2)]
        area = _signed_area(xy[fan_nodes])
        ok = alone.any(axis=1) & np.all(
            area > _TOLERANCE * area.sum(axis=1, keepdims=True), axis=1
        )
        tris.append(fan_nodes[ok].reshape(-1, 3))
        unsplit.extend(elements[~ok].tolist())

    for e in unsplit:
        tris.append(_ear_clip(xy, connectivity[first[e] : first[e] + n_nodes[e]]))
    return np.concatenate(tris).astype(np.int64)


def to_connectivity(
    element_table: np.ndarray | Sequence[ArrayLike],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The element table as a dfsu file stores it: Connectivity and No of nodes.

    Connectivity holds the Nodes of all Elements, one Element after the
    other; No of nodes holds the number of Nodes of each Element.
    """
    n_elements = len(element_table)
    n_nodes = np.fromiter(
        (len(el) for el in element_table),  # type: ignore[arg-type]
        dtype=np.int64,
        count=n_elements,
    )
    if n_elements == 0:
        return np.zeros(0, dtype=np.int64), n_nodes
    connectivity = np.concatenate([np.asarray(el) for el in element_table])
    return connectivity.astype(np.int64), n_nodes


def _faces(
    connectivity: NDArray[np.int64], n_nodes: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Faces as (start, end) Node pairs, with their Element and position.

    Face `j` of an Element runs from its Node `j` to Node `j + 1`.
    """
    element = np.repeat(np.arange(n_nodes.size, dtype=np.int64), n_nodes)
    first = np.cumsum(n_nodes) - n_nodes
    position = np.arange(connectivity.size, dtype=np.int64) - first[element]
    following = position + 1
    following[following == n_nodes[element]] = 0
    end = connectivity[first[element] + following]
    return np.column_stack([connectivity, end]), element, position


def _face_keys(faces: NDArray[np.int64]) -> NDArray[np.int64]:
    """One integer per Face, the same in both directions."""
    lo = faces.min(axis=1)
    hi = faces.max(axis=1)
    return lo * (int(hi.max(initial=0)) + 1) + hi


def _fan(n: int) -> NDArray[np.int64]:
    """Triangles fanned out from position 0 of an n-sided polygon."""
    i = np.arange(1, n - 1)
    return np.column_stack([np.zeros_like(i), i, i + 1])


def _signed_area(tri_xy: NDArray[np.floating]) -> NDArray[np.floating]:
    """Signed area of triangles given as (..., 3, 2) coordinates."""
    ab = tri_xy[..., 1, :] - tri_xy[..., 0, :]
    ac = tri_xy[..., 2, :] - tri_xy[..., 0, :]
    return 0.5 * (ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0])


def _ear_clip(xy: NDArray[np.floating], nodes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Triangles of a convex polygon with Nodes on its sides, none with zero area.

    Only a Node where the polygon turns can be clipped, and only when the
    rest of the polygon keeps a non-zero area.
    """
    remaining = nodes.tolist()
    total = abs(_signed_area(xy[np.asarray(remaining)][_fan(len(remaining))]).sum())
    tris: list[list[int]] = []
    while len(remaining) > 3:
        for k in range(len(remaining)):
            ear = [remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]]
            rest = remaining[:k] + remaining[k + 1 :]
            ear_area = _signed_area(xy[ear])
            rest_area = _signed_area(xy[np.asarray(rest)][_fan(len(rest))]).sum()
            if ear_area > _TOLERANCE * total and rest_area > _TOLERANCE * total:
                tris.append(ear)
                remaining = rest
                break
        else:  # a degenerate Element: keep a fan rather than fail
            break
    tris.extend(np.asarray(remaining)[_fan(len(remaining))].tolist())
    return np.asarray(tris, dtype=np.int64)
