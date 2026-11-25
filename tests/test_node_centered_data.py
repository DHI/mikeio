"""Node values from Element values: GeometryFM2D.get_node_centered_data (#1125).

Where three or more Elements surround a Node, its value is the value at the
Node of the plane that best fits, by least squares, the Element values at the
Element centres. That plane fit is the oracle for most tests here.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import mikeio
from mikeio.spatial import GeometryFM2D


def jittered_mesh(
    nx: int = 6,
    ny: int = 5,
    jitter: float = 0.3,
    tri_fraction: float = 1.0,
    seed: int = 0,
    stretch: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> GeometryFM2D:
    """A grid of unit cells with interior Nodes moved by up to `jitter` cells.

    A fraction `tri_fraction` of the cells is split into two triangles; the
    rest stay quads. Then x is multiplied by `stretch` and the mesh is moved
    by `offset`.
    """
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0), indexing="ij")
    interior = (x > 0) & (x < nx) & (y > 0) & (y < ny)
    x[interior] += rng.uniform(-jitter, jitter, interior.sum())
    y[interior] += rng.uniform(-jitter, jitter, interior.sum())
    x, y = x * stretch + offset[0], y + offset[1]
    xyz = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    def node(i: int, j: int) -> int:
        return i * (ny + 1) + j

    elements: list[list[int]] = []
    for i in range(nx):
        for j in range(ny):
            a, b = node(i, j), node(i + 1, j)
            c, d = node(i + 1, j + 1), node(i, j + 1)
            if rng.uniform() < tri_fraction:
                elements += [[a, b, c], [a, c, d]]
            else:
                elements.append([a, b, c, d])
    return GeometryFM2D(xyz, elements, projection="NON-UTM")


def elements_at_nodes(g: GeometryFM2D) -> list[list[int]]:
    at: list[list[int]] = [[] for _ in range(g.n_nodes)]
    for e, nodes in enumerate(g.element_table):
        for n in nodes:
            at[n].append(e)
    return at


def plane_fit(
    g: GeometryFM2D, values: NDArray[np.floating], nodes: list[int]
) -> list[float]:
    """Least-squares plane through the Element values around each Node, at the Node."""
    at = elements_at_nodes(g)
    fits = []
    for n in nodes:
        e = at[n]
        d = g.element_coordinates[e, :2] - g.node_coordinates[n, :2]
        A = np.column_stack([np.ones(len(e)), d])
        coef, *_ = np.linalg.lstsq(A, values[e], rcond=None)
        fits.append(float(coef[0]))
    return fits


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


MESHES = {
    "triangles": dict(tri_fraction=1.0),
    "quads": dict(tri_fraction=0.0),
    "mixed": dict(tri_fraction=0.5),
    "regular-triangles": dict(jitter=0.0),
    "regular-quads": dict(jitter=0.0, tri_fraction=0.0),
    "max-jitter": dict(jitter=0.45, tri_fraction=0.5),
    "stretched": dict(stretch=100.0, tri_fraction=0.5),
    "utm": dict(offset=(500_000.0, 6_200_000.0), tri_fraction=0.5),
}


@pytest.fixture(params=MESHES.values(), ids=list(MESHES))
def mesh_kwargs(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param


@pytest.fixture(params=[0, 1, 2])
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


def random_values(g: GeometryFM2D, seed: int = 1) -> NDArray[np.floating]:
    return np.random.default_rng(seed).uniform(-1, 1, g.n_elements)


def test_node_value_is_the_plane_fit_of_surrounding_elements(
    mesh_kwargs: dict[str, Any], seed: int
) -> None:
    g = jittered_mesh(**mesh_kwargs, seed=seed)
    values = random_values(g, seed)

    zn = g.get_node_centered_data(values)

    nodes = nodes_with_a_plane(g)
    expected = plane_fit(g, values, nodes)
    assert zn[nodes] == pytest.approx(expected, abs=1e-10)


def test_node_with_fewer_than_three_elements_is_inverse_distance_average(
    mesh_kwargs: dict[str, Any], seed: int
) -> None:
    g = jittered_mesh(**mesh_kwargs, seed=seed)
    values = random_values(g, seed)

    zn = g.get_node_centered_data(values)

    nodes = [n for n, e in enumerate(elements_at_nodes(g)) if len(e) < 3]
    assert nodes  # the corners at least
    expected = inverse_distance(g, values, nodes)
    assert zn[nodes] == pytest.approx(expected, abs=1e-12)


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


@pytest.mark.parametrize("extrapolate", [True, False])
def test_constant_field_is_returned_at_every_node(
    mesh_kwargs: dict[str, Any], seed: int, extrapolate: bool
) -> None:
    g = jittered_mesh(**mesh_kwargs, seed=seed)

    zn = g.get_node_centered_data(np.full(g.n_elements, 7.5), extrapolate=extrapolate)

    assert zn == pytest.approx(7.5, abs=1e-12)


def test_without_extrapolation_node_values_stay_within_their_elements(
    mesh_kwargs: dict[str, Any], seed: int
) -> None:
    g = jittered_mesh(**mesh_kwargs, seed=seed)
    values = random_values(g, seed)

    zn = g.get_node_centered_data(values, extrapolate=False)

    for n, e in enumerate(elements_at_nodes(g)):
        assert values[e].min() - 1e-12 <= zn[n] <= values[e].max() + 1e-12


def test_without_extrapolation_some_weights_are_limited() -> None:
    """The mesh used above does limit weights, so the bounds test is not idle."""
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


def test_node_values_on_hd2d_are_the_plane_fit() -> None:
    g = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu").geometry
    assert isinstance(g, GeometryFM2D)
    values = random_values(g)

    zn = g.get_node_centered_data(values)

    nodes = nodes_with_a_plane(g)
    expected = plane_fit(g, values, nodes)
    assert zn[nodes] == pytest.approx(expected, abs=1e-10)
