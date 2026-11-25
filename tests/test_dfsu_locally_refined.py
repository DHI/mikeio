"""Locally refined meshes: Faces with Hanging nodes (MIKE 2027, Submesh type 1)."""

import warnings
from pathlib import Path
from collections.abc import Sequence
from typing import Any, NamedTuple

import matplotlib as mpl
import numpy as np
import pytest
from mikecore.DfsuFile import DfsuFile

import mikeio
from mikeio.spatial import GeometryFM2D, GeometryFM3D

mpl.use("Agg")

REFINED = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh1_LocallyRefined.dfsu"
NOT_REFINED = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh0.dfsu"

# 0-based indices of the Hanging nodes in REFINED (11 and 13 in the file)
HANGING_NODES = [10, 12]


class Mesh(NamedTuple):
    geometry: GeometryFM2D
    hanging_nodes: list[int]
    area: float


def read_geometry(filename: str | Path) -> GeometryFM2D:
    g = mikeio.Dfsu2DH(filename).geometry
    assert isinstance(g, GeometryFM2D)
    return g


def submesh_type(filename: str | Path) -> int | None:
    dfs = DfsuFile.Open(str(filename))
    try:
        return dfs.SubMeshType
    finally:
        dfs.Close()


def linear(x: Any, y: Any) -> Any:
    return 2 * x + 3 * y + 10


def quad_mesh(
    nc: Sequence[Sequence[float]], element_table: list[list[int]]
) -> GeometryFM2D:
    xyz = np.array([[x, y, 0.0] for x, y in nc])
    return GeometryFM2D(xyz, element_table, projection="NON-UTM")


def one_hanging_node() -> Mesh:
    """A coarse quad beside two fine quads: one Hanging node (6).

    y
    2  3 ------- 2 ------- 7
       |         |  fine   |
    1  | coarse  6 ------- 5
       |         |  fine   |
    0  0 ------- 1 ------- 4
       0         1         2  x
    """
    nc = [[0, 0], [1, 0], [1, 2], [0, 2], [2, 0], [2, 1], [1, 1], [2, 2]]
    element_table = [[0, 1, 2, 3], [1, 4, 5, 6], [6, 5, 7, 2]]
    return Mesh(quad_mesh(nc, element_table), hanging_nodes=[6], area=4.0)


def two_hanging_nodes_on_one_face() -> Mesh:
    """A coarse quad beside three fine quads: two Hanging nodes (6, 7).

    y
    3  3 ------- 2 ------- 9
       |         |  fine   |
    2  |         7 ------- 8
       | coarse  |  fine   |
    1  |         6 ------- 5
       |         |  fine   |
    0  0 ------- 1 ------- 4
       0         1         2  x
    """
    nc = [
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
    ]
    element_table = [[0, 1, 2, 3], [1, 4, 5, 6], [6, 5, 8, 7], [7, 8, 9, 2]]
    return Mesh(quad_mesh(nc, element_table), hanging_nodes=[6, 7], area=6.0)


@pytest.fixture
def refined() -> GeometryFM2D:
    return read_geometry(REFINED)


@pytest.fixture
def not_refined() -> GeometryFM2D:
    return read_geometry(NOT_REFINED)


@pytest.fixture(params=[one_hanging_node, two_hanging_nodes_on_one_face])
def mesh(request: pytest.FixtureRequest) -> Mesh:
    return request.param()


@pytest.fixture
def layered() -> GeometryFM3D:
    """The one-Hanging-node mesh, extruded to two sigma layers."""
    g2 = one_hanging_node().geometry
    n_layers = 2
    n_node_layers = n_layers + 1
    nc = np.array(
        [
            [x, y, -1.0 + k / n_layers]
            for x, y, _ in g2.node_coordinates
            for k in range(n_node_layers)
        ]
    )
    element_table = [
        [n * n_node_layers + k for n in el] + [n * n_node_layers + k + 1 for n in el]
        for el in g2.element_table
        for k in range(n_layers)
    ]
    return GeometryFM3D(
        node_coordinates=nc,
        element_table=element_table,
        projection="NON-UTM",
        n_layers=n_layers,
        n_sigma=n_layers,
    )


# Seam 1: Flexible Mesh Geometry


def test_boundary_of_locally_refined_mesh_is_one_domain(
    refined: GeometryFM2D,
) -> None:
    polygons = refined.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert len(polygons.interiors) == 0
    x, y = refined.node_coordinates[:, 0], refined.node_coordinates[:, 1]
    bbox_area = (x.max() - x.min()) * (y.max() - y.min())
    assert polygons.exteriors[0].area == pytest.approx(bbox_area)


def test_boundary_of_mesh_without_hanging_nodes_is_unchanged(
    not_refined: GeometryFM2D,
) -> None:
    assert len(not_refined.boundary_polygons.exteriors) == 1
    assert len(not_refined.boundary_polygons.interiors) == 0


def test_element_table_is_as_stored_in_file(refined: GeometryFM2D) -> None:
    _ = refined.boundary_polygons

    dfs = DfsuFile.Open(REFINED)
    stored = [np.asarray(e) - 1 for e in dfs.ElementTable]
    dfs.Close()

    assert refined.n_elements == 40
    assert len(refined.element_table) == len(stored)
    for a, b in zip(refined.element_table, stored):
        np.testing.assert_array_equal(a, b)


def test_contains_points_on_coarse_fine_transition(refined: GeometryFM2D) -> None:
    hanging = refined.node_coordinates[HANGING_NODES, :2]
    # points just beside the Hanging nodes, on either side of the transition
    pts = np.vstack([hanging, hanging + [0.01, 0.0], hanging - [0.01, 0.0]])

    assert refined.contains(pts).all()


def test_node_centered_data_at_hanging_nodes(refined: GeometryFM2D) -> None:
    ec, nc = refined.element_coordinates, refined.node_coordinates

    zn = refined.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    # Using only the fine Elements gave an error of 32. get_node_centered_data
    # is not exact for a linear field, even at ordinary internal Nodes.
    exact = linear(nc[HANGING_NODES, 0], nc[HANGING_NODES, 1])
    assert zn[HANGING_NODES] == pytest.approx(exact, abs=12)


def test_quad_mesh_with_hanging_nodes_has_one_domain(mesh: Mesh) -> None:
    polygons = mesh.geometry.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert polygons.exteriors[0].area == pytest.approx(mesh.area)
    on_transition = mesh.geometry.node_coordinates[mesh.hanging_nodes, :2]
    assert mesh.geometry.contains(on_transition).all()


def test_quad_mesh_node_centered_data_at_hanging_nodes(mesh: Mesh) -> None:
    g = mesh.geometry
    ec, nc = g.element_coordinates, g.node_coordinates

    zn = g.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    # error 2.5 from the fine Elements only
    exact = linear(nc[mesh.hanging_nodes, 0], nc[mesh.hanging_nodes, 1])
    assert zn[mesh.hanging_nodes] == pytest.approx(exact, abs=0.5)


def test_hanging_node_found_within_rounding() -> None:
    g = one_hanging_node().geometry
    nc = g.node_coordinates.copy()
    nc[6, 0] += 1e-12  # off the coarse Face by rounding only
    g = GeometryFM2D(nc, g.element_table, projection="NON-UTM")

    assert len(g.boundary_polygons.exteriors) == 1


def test_node_off_the_face_is_not_hanging() -> None:
    # move the would-be Hanging node far off the coarse Face: a real gap
    g = one_hanging_node().geometry
    nc = g.node_coordinates.copy()
    nc[6, 0] = 1.2
    g = GeometryFM2D(nc, g.element_table, projection="NON-UTM")

    assert len(g.boundary_polygons.exteriors) == 2


def test_boundary_nodes_touching_face_ends_are_not_hanging() -> None:
    """Each boundary Node lies on the line through its neighbouring Faces,
    but only at their ends.

    y
    1  5 ------- 4 ------- 3
       |         |         |
    0  0 ------- 1 ------- 2
       0         1         2  x
    """
    nc = [[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]]
    g = quad_mesh(nc, [[0, 1, 4, 5], [1, 2, 3, 4]])

    polygons = g.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert len(polygons.exteriors[0].xy) == 7  # 6 Nodes, closed
    assert polygons.exteriors[0].area == pytest.approx(2.0)


def test_layered_mesh_with_hanging_node_has_one_domain(layered: GeometryFM3D) -> None:
    polygons = layered.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert polygons.exteriors[0].area == pytest.approx(4.0)


def test_subset_without_hanging_nodes_is_correct() -> None:
    sub = one_hanging_node().geometry.isel([1, 2])

    assert isinstance(sub, GeometryFM2D)
    assert len(sub.boundary_polygons.exteriors) == 1
    assert sub.boundary_polygons.exteriors[0].area == pytest.approx(2.0)


def test_unchanged_behaviour_on_locally_refined_mesh(refined: GeometryFM2D) -> None:
    ds = mikeio.read(REFINED)

    assert refined.get_element_area().sum() == pytest.approx(150.0 * 100.0)
    assert ds.isel(element=[0, 1, 2]).geometry.n_elements == 3
    x, y = refined.element_coordinates[5, :2]
    assert refined.find_index(x=x, y=y)[0] == 5


# Seam 2: DataArray plotting


def _plot(da: mikeio.DataArray, plot_type: str, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # n_refinements, shaded
        if plot_type == "shaded":
            da.plot(plot_type="shaded", **kwargs)
        else:
            getattr(da.plot, plot_type)(**kwargs)
    plt.close("all")


@pytest.mark.parametrize("plot_type", ["contour", "contourf", "shaded"])
def test_plot_with_refinement_on_locally_refined_mesh(
    plot_type: str, refined: GeometryFM2D
) -> None:
    ec = refined.element_coordinates
    da = mikeio.DataArray(linear(ec[:, 0], ec[:, 1]), geometry=refined)

    _plot(da, plot_type, n_refinements=1)


@pytest.mark.parametrize("plot_type", ["contour", "contourf", "shaded"])
def test_plot_with_refinement_on_quad_mesh_with_hanging_nodes(
    plot_type: str, mesh: Mesh
) -> None:
    ec = mesh.geometry.element_coordinates
    da = mikeio.DataArray(linear(ec[:, 0], ec[:, 1]), geometry=mesh.geometry)

    _plot(da, plot_type, n_refinements=1, show_mesh=True)


def test_outline_of_locally_refined_mesh_has_no_interior_line() -> None:
    import matplotlib.pyplot as plt

    da = mikeio.read(REFINED)[0]

    ax = da.plot.outline()

    assert len(ax.lines) == 1
    plt.close("all")


def test_outline_of_mesh_only_geometry(refined: GeometryFM2D) -> None:
    import matplotlib.pyplot as plt

    mesh = GeometryFM2D(
        refined.node_coordinates,
        refined.element_table,
        codes=refined.codes,
        projection="NON-UTM",
    )

    ax = mesh.plot.outline()

    assert len(ax.lines) == 1
    plt.close("all")


# Seam 3: file round trip


@pytest.mark.parametrize(("filename", "expected"), [(REFINED, 1), (NOT_REFINED, 0)])
def test_to_dfs_keeps_submesh_type(
    filename: str, expected: int, tmp_path: Path
) -> None:
    ds = mikeio.read(filename)
    out = tmp_path / "copy.dfsu"

    ds.to_dfs(out)

    assert submesh_type(out) == expected
    for a, b in zip(
        read_geometry(out).element_table, read_geometry(filename).element_table
    ):
        np.testing.assert_array_equal(a, b)


def test_to_dfs_keeps_submesh_type_after_isel(tmp_path: Path) -> None:
    ds = mikeio.read(REFINED).isel(element=range(10))
    out = tmp_path / "subset.dfsu"

    ds.to_dfs(out)

    assert submesh_type(out) == 1


def test_to_dfs_writes_no_submesh_type_when_there_is_none(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/HD2D.dfsu")
    out = tmp_path / "copy.dfsu"

    ds.to_dfs(out)

    assert submesh_type(out) is None
