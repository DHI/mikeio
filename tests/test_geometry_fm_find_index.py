"""find_index: the Element that contains each point, whatever the mesh."""

from __future__ import annotations

import numpy as np

from .test_dfsu_locally_refined_workflows import square_grid


def test_find_index_in_a_mesh_of_one_element() -> None:
    """One id per point, also when there is only one Element to choose."""
    g = square_grid(1, quad=True).geometry()
    pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.9, 0.1]])

    np.testing.assert_array_equal(g.find_index(coords=pts), [0, 0, 0])


def test_find_index_of_no_points() -> None:
    g = square_grid(2, quad=False).geometry()

    assert len(g.find_index(coords=np.empty((0, 2)))) == 0
