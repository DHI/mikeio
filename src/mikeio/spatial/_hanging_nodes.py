"""Hanging nodes in a Locally refined mesh.

In a Locally refined mesh a coarse Face (A-C) meets two or more fine Faces
(A-B, B-C) at a Hanging node B. The element table stores the coarse Element
with Nodes A and C only, so the coarse Face and the fine Faces each occur
once, like Faces on the domain boundary.

The conforming element table inserts each Hanging node as an extra Node of
the coarse Element, between the ends of the Face it lies on. Then every
interior Face is again shared by exactly two Elements.

Hanging nodes are found from the node coordinates and the element table only,
never from the Submesh type in the dfsu header.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# Distance tolerance, relative to the length of the Face
_TOLERANCE = 1e-6


def conforming_element_table(node_coordinates: np.ndarray, element_table: Any) -> Any:
    """Element table with each Hanging node as an extra Node of its Element.

    Returns `element_table` itself, the same object, when there are no
    Hanging nodes.
    """
    n_elements = len(element_table)
    if n_elements == 0:
        return element_table

    lengths = np.fromiter(
        (len(el) for el in element_table), dtype=int, count=n_elements
    )
    element_nodes = np.concatenate([np.asarray(el) for el in element_table]).astype(
        np.int64
    )
    first = np.repeat(np.cumsum(lengths) - lengths, lengths)
    local = np.arange(element_nodes.size) - first
    next_nodes = element_nodes[first + (local + 1) % np.repeat(lengths, lengths)]

    # Faces that occur once: on the domain boundary, or beside a Hanging node
    lo = np.minimum(element_nodes, next_nodes)
    hi = np.maximum(element_nodes, next_nodes)
    key = lo * (int(max(element_nodes.max(), next_nodes.max())) + 1) + hi
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    once = np.flatnonzero(counts[inverse] == 1)
    if once.size < 3:
        return element_table

    start, end = element_nodes[once], next_nodes[once]
    xy = np.asarray(node_coordinates)[:, :2]
    a, c = xy[start], xy[end]
    d = c - a
    face_length = np.hypot(d[:, 0], d[:, 1])

    # Nodes of once-only Faces within reach of each once-only Face
    candidates = np.unique(np.concatenate([start, end]))
    tree = cKDTree(xy[candidates])
    reach = 0.5 * face_length * (1.0 + _TOLERANCE)
    near = tree.query_ball_point(0.5 * (a + c), reach, return_sorted=False)
    n_near = np.fromiter((len(n) for n in near), dtype=int, count=len(near))
    face = np.repeat(np.arange(once.size), n_near)
    node = candidates[np.concatenate([np.asarray(n, dtype=int) for n in near])]

    keep = (node != start[face]) & (node != end[face])
    face, node = face[keep], node[keep]

    # On the Face, strictly between its ends
    ap = xy[node] - a[face]
    df = d[face]
    length = face_length[face]
    t = (ap[:, 0] * df[:, 0] + ap[:, 1] * df[:, 1]) / length**2
    offset = np.abs(df[:, 0] * ap[:, 1] - df[:, 1] * ap[:, 0]) / length
    hanging = (
        (t > _TOLERANCE) & (t < 1.0 - _TOLERANCE) & (offset <= _TOLERANCE * length)
    )
    if not hanging.any():
        return element_table

    face, node, t = face[hanging], node[hanging], t[hanging]
    order = np.lexsort((t, face))
    face, node = face[order], node[order]

    owner = np.repeat(np.arange(n_elements), lengths)[once]
    face_local = local[once]
    inserted: dict[int, dict[int, list[int]]] = {}
    for f, n in zip(face.tolist(), node.tolist()):
        on_face = inserted.setdefault(int(owner[f]), {}).setdefault(
            int(face_local[f]), []
        )
        on_face.append(n)

    table = np.empty(n_elements, dtype=object)
    table[:] = list(element_table)
    for element, faces in inserted.items():
        el = np.asarray(element_table[element])
        new: list[int] = []
        for j, node_id in enumerate(el.tolist()):
            new.append(node_id)
            new.extend(faces.get(j, []))
        table[element] = np.asarray(new, dtype=el.dtype)
    return table
