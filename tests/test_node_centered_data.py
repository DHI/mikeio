"""Node values from Element values: GeometryFM2D.get_node_centered_data (#1125).

Where three or more Elements surround a Node, its value is the value at the
Node of the plane that best fits, by least squares, the Element values at the
Element centres. The properties in test_dfsu_locally_refined.py check that on
generated meshes, with and without Hanging nodes; the tests here cover real
files and the edge cases.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import mikeio
from mikeio.spatial import GeometryFM2D


def jittered_mesh(nx: int = 6, ny: int = 5, jitter: float = 0.3) -> GeometryFM2D:
    """A grid of unit cells, each split into two triangles, with interior Nodes
    moved by up to `jitter` cells."""
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0), indexing="ij")
    interior = (x > 0) & (x < nx) & (y > 0) & (y < ny)
    x[interior] += rng.uniform(-jitter, jitter, interior.sum())
    y[interior] += rng.uniform(-jitter, jitter, interior.sum())
    xyz = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    def node(i: int, j: int) -> int:
        return i * (ny + 1) + j

    elements: list[list[int]] = []
    for i in range(nx):
        for j in range(ny):
            a, b = node(i, j), node(i + 1, j)
            c, d = node(i + 1, j + 1), node(i, j + 1)
            elements += [[a, b, c], [a, c, d]]
    return GeometryFM2D(xyz, elements, projection="NON-UTM")


def elements_at_nodes(g: GeometryFM2D) -> list[list[int]]:
    at: list[list[int]] = [[] for _ in range(g.n_nodes)]
    for e, nodes in enumerate(g.element_table):
        for n in nodes:
            at[n].append(e)
    return at


def nodes_with_a_plane(g: GeometryFM2D) -> list[int]:
    return [n for n, e in enumerate(elements_at_nodes(g)) if len(e) >= 3]


def inverse_distance(
    g: GeometryFM2D, values: NDArray[np.floating], nodes: list[int]
) -> list[float]:
    at = elements_at_nodes(g)
    averages = []
    for n in nodes:
        e = at[n]
        d = g.element_coordinates[e, :2] - g.node_coordinates[n, :2]
        w = 1 / np.hypot(d[:, 0], d[:, 1])
        averages.append(float((w * values[e]).sum() / w.sum()))
    return averages


def random_values(g: GeometryFM2D, seed: int = 1) -> NDArray[np.floating]:
    return np.random.default_rng(seed).uniform(-1, 1, g.n_elements)


@pytest.mark.parametrize("length", [1e3, 2e3, 1e4])
@pytest.mark.parametrize("width", [1e-3, 1e-2, 1e-1, 1.0])
def test_node_whose_weights_sum_to_zero_is_inverse_distance_average(
    length: float, width: float
) -> None:
    """Long slivers fanning out from Node 0: its Element centres lie on a line.

    The weights then sum to zero up to rounding error.
    """
    far = [[length, k * width] for k in range(4)]
    xyz = np.array([[0.0, 0.0, 0.0]] + [[x, y, 0.0] for x, y in far])
    g = GeometryFM2D(xyz, [[0, 1, 2], [0, 2, 3], [0, 3, 4]], projection="NON-UTM")
    values = np.array([1.0, 5.0, 2.0])

    zn = g.get_node_centered_data(values)

    assert np.isfinite(zn).all()
    assert zn[[0]] == pytest.approx(inverse_distance(g, values, [0]), abs=1e-12)


def test_without_extrapolation_some_weights_are_limited() -> None:
    """Limiting the weights changes values on a jittered mesh, so it is not idle."""
    g = jittered_mesh()
    values = random_values(g)

    limited = g.get_node_centered_data(values, extrapolate=False)
    unlimited = g.get_node_centered_data(values, extrapolate=True)

    assert not np.allclose(limited, unlimited)


def linear(x: Any, y: Any) -> Any:
    return 2 * x + 3 * y + 10


@pytest.mark.parametrize(
    "filename",
    [
        "tests/testdata/HD2D.dfsu",
        "tests/testdata/oresundHD_run1.dfsu",
        "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh1_LocallyRefined.dfsu",
    ],
)
def test_linear_field_is_exact_where_three_or_more_elements_meet(
    filename: str,
) -> None:
    g = mikeio.Dfsu2DH(filename).geometry
    assert isinstance(g, GeometryFM2D)
    ec, nc = g.element_coordinates, g.node_coordinates

    zn = g.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    nodes = nodes_with_a_plane(g)
    exact = linear(nc[nodes, 0], nc[nodes, 1])
    assert zn[nodes] == pytest.approx(exact, rel=1e-9)


def test_node_without_elements_is_zero_with_a_warning() -> None:
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 0]])  # Node 3 unused
    g = GeometryFM2D(xyz, [[0, 1, 2]], projection="NON-UTM")

    with pytest.warns(UserWarning, match="1 Node"):
        zn = g.get_node_centered_data(np.array([4.0]))

    assert zn[3] == 0.0
    assert zn[:3] == pytest.approx(4.0)
