"""Submesh results: Hanging nodes and the conforming element table.

A Submesh result is a Globally refined mesh or a Locally refined mesh. A
Globally refined mesh is an ordinary Flexible Mesh and needs nothing here.

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
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

# Distance tolerance, relative to the length of the Face
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FlatElementTable:
    """Element table as one flat array of Nodes.

    Element `i` has the Nodes `nodes[offsets[i]:offsets[i + 1]]`, in
    counter-clockwise order.
    """

    nodes: NDArray[np.int64]
    offsets: NDArray[np.int64]

    @staticmethod
    def from_element_table(element_table: Sequence[ArrayLike]) -> FlatElementTable:
        """Flatten an element table: one array of Nodes per Element."""
        n_elements = len(element_table)
        lengths = np.fromiter(
            (len(el) for el in element_table),  # type: ignore[arg-type]
            dtype=np.int64,
            count=n_elements,
        )
        offsets = np.zeros(n_elements + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        if n_elements == 0:
            return FlatElementTable(np.zeros(0, dtype=np.int64), offsets)
        nodes = np.concatenate([np.asarray(el) for el in element_table])
        return FlatElementTable(nodes.astype(np.int64), offsets)

    @property
    def n_elements(self) -> int:
        """Number of Elements."""
        return len(self.offsets) - 1

    @property
    def lengths(self) -> NDArray[np.int64]:
        """Number of Nodes of each Element."""
        return np.diff(self.offsets)

    @property
    def element(self) -> NDArray[np.int64]:
        """The Element of each entry in `nodes`."""
        return np.repeat(np.arange(self.n_elements, dtype=np.int64), self.lengths)

    @property
    def position(self) -> NDArray[np.int64]:
        """The position of each entry in `nodes` within its Element."""
        return np.arange(self.nodes.size, dtype=np.int64) - self.offsets[self.element]

    def faces(self) -> NDArray[np.int64]:
        """Faces as (start, end) Node pairs, Element by Element.

        Face `j` of an Element runs from its Node `j` to Node `j + 1`.
        """
        element = self.element
        following = self.position + 1
        following[following == self.lengths[element]] = 0
        end = self.nodes[self.offsets[element] + following]
        return np.column_stack([self.nodes, end])

    def boundary_faces(self) -> NDArray[np.int64]:
        """Faces that occur once, in the order of their sorted Node pair."""
        faces = self.faces()
        _, first, counts = np.unique(
            _face_keys(faces), return_index=True, return_counts=True
        )
        return faces[first[counts == 1]]


@dataclass(frozen=True)
class ConformingElementTable(FlatElementTable):
    """Element table with each Hanging node as an extra Node of its Element.

    `hanging[k]` is True when `nodes[k]` is a Hanging node inserted into its
    Element; these Nodes are not in the element table.
    """

    hanging: NDArray[np.bool_]

    @property
    def refined(self) -> NDArray[np.bool_]:
        """True for each Element with Hanging nodes on its Faces."""
        return np.add.reduceat(self.hanging, self.offsets[:-1]) > 0

    def triangles(self, node_coordinates: NDArray[np.floating]) -> NDArray[np.int64]:
        """Split every Element into triangles, none with zero area.

        A quad without Hanging nodes is split as `0-1-2` and `2-3-0`. An
        Element with Hanging nodes is fanned out from a Hanging node that is
        alone on its Face; when there is none, it is split by ear clipping.
        """
        xy = np.asarray(node_coordinates, dtype=float)[:, :2]
        lengths = self.lengths
        refined = self.refined
        first = self.offsets[:-1]

        plain = first[~refined & (lengths == 3)]
        quads = self.nodes[first[~refined & (lengths == 4)][:, None] + np.arange(4)]
        tris = [
            self.nodes[plain[:, None] + np.arange(3)],
            quads[:, [0, 1, 2]],
            quads[:, [2, 3, 0]],
        ]

        unsplit: list[int] = []
        for n in np.unique(lengths[refined]).tolist():
            elements = np.flatnonzero(refined & (lengths == n))
            entries = first[elements][:, None] + np.arange(n)
            hanging = self.hanging[entries]
            alone = (
                hanging & ~np.roll(hanging, 1, axis=1) & ~np.roll(hanging, -1, axis=1)
            )
            apex = np.argmax(alone, axis=1)
            fan = (apex[:, None, None] + _fan(n)) % n
            fan_nodes = self.nodes[np.take_along_axis(entries[:, None, :], fan, axis=2)]
            area = _signed_area(xy[fan_nodes])
            ok = alone.any(axis=1) & np.all(
                area > _TOLERANCE * area.sum(axis=1, keepdims=True), axis=1
            )
            tris.append(fan_nodes[ok].reshape(-1, 3))
            unsplit.extend(elements[~ok].tolist())

        for element in unsplit:
            nodes = self.nodes[self.offsets[element] : self.offsets[element + 1]]
            tris.append(_ear_clip(xy, nodes))
        return np.concatenate(tris).astype(np.int64)


def conforming_element_table(
    node_coordinates: NDArray[np.floating],
    element_table: Sequence[ArrayLike],
) -> ConformingElementTable | None:
    """Element table with each Hanging node as an extra Node of its Element.

    Returns None when there are no Hanging nodes.

    Examples
    --------
    A coarse triangle beside two fine triangles. Node 3 is a Hanging node
    on the coarse Face 1-2.

    >>> nc = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 1, 0], [2, 1, 0]])
    >>> table = conforming_element_table(nc, [[0, 1, 2], [1, 4, 3], [3, 4, 2]])
    >>> table.nodes[table.offsets[0] : table.offsets[1]]
    array([0, 1, 3, 2])
    >>> conforming_element_table(nc, [[0, 1, 2]]) is None
    True

    """
    flat = FlatElementTable.from_element_table(element_table)
    if flat.n_elements == 0:
        return None

    # Faces that occur once: on the domain boundary, or beside a Hanging node
    faces = flat.faces()
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

    # Insert each Hanging node after the start Node of its Face, in order
    # along the Face: sort all entries by Element, position, then distance
    # along the Face (0 for the Element's own Nodes).
    entry = once[face]
    element = np.concatenate([flat.element, flat.element[entry]])
    position = np.concatenate([flat.position, flat.position[entry]])
    along = np.concatenate([np.zeros(flat.nodes.size), t])
    order = np.lexsort((along, position, element))

    nodes = np.concatenate([flat.nodes, node])[order]
    hanging = np.concatenate(
        [np.zeros(flat.nodes.size, dtype=bool), np.ones(node.size, dtype=bool)]
    )[order]
    lengths = flat.lengths + np.bincount(flat.element[entry], minlength=flat.n_elements)
    offsets = np.zeros(flat.n_elements + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return ConformingElementTable(nodes=nodes, offsets=offsets, hanging=hanging)


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
