"""Locally refined meshes: Faces with Hanging nodes (MIKE 2027, Submesh type 1)."""

import warnings
from pathlib import Path
from typing import Any

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


def geometry(filename: str | Path) -> GeometryFM2D:
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


def quad_mesh_with_hanging_node() -> GeometryFM2D:
    """A coarse quad (left) beside two fine quads (right).

    Node 6 at (1, 1) is a Hanging node on the coarse Face (1, 0)-(1, 2).
    """
    nc = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    element_table = [[0, 1, 2, 3], [1, 4, 5, 6], [6, 5, 7, 2]]
    codes = np.array([1, 1, 1, 1, 1, 1, 0, 1])
    return GeometryFM2D(nc, element_table, codes=codes, projection="NON-UTM")


def layered_quad_mesh_with_hanging_node(n_layers: int = 2) -> GeometryFM3D:
    """The quad mesh above, extruded to n_layers sigma layers."""
    g2 = quad_mesh_with_hanging_node()
    n_node_layers = n_layers + 1
    nc = np.array(
        [
            [x, y, -1.0 + k / n_layers]
            for x, y, _ in g2.node_coordinates
            for k in range(n_node_layers)
        ]
    )
    codes = np.repeat(g2.codes, n_node_layers)
    element_table = [
        [n * n_node_layers + k for n in el] + [n * n_node_layers + k + 1 for n in el]
        for el in g2.element_table
        for k in range(n_layers)
    ]
    return GeometryFM3D(
        node_coordinates=nc,
        element_table=element_table,
        codes=codes,
        projection="NON-UTM",
        n_layers=n_layers,
        n_sigma=n_layers,
    )


# Seam 1: Flexible Mesh Geometry


def test_boundary_of_locally_refined_mesh_is_one_domain() -> None:
    g = geometry(REFINED)

    polygons = g.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert len(polygons.interiors) == 0
    x, y = g.node_coordinates[:, 0], g.node_coordinates[:, 1]
    bbox_area = (x.max() - x.min()) * (y.max() - y.min())
    assert polygons.exteriors[0].area == pytest.approx(bbox_area)


def test_boundary_of_mesh_without_hanging_nodes_is_unchanged() -> None:
    g = geometry(NOT_REFINED)

    assert len(g.boundary_polygons.exteriors) == 1
    assert len(g.boundary_polygons.interiors) == 0


def test_element_table_is_as_stored_in_file() -> None:
    g = geometry(REFINED)
    _ = g.boundary_polygons

    dfs = DfsuFile.Open(REFINED)
    stored = [np.asarray(e) - 1 for e in dfs.ElementTable]
    dfs.Close()

    assert g.n_elements == 40
    assert len(g.element_table) == len(stored)
    for a, b in zip(g.element_table, stored):
        np.testing.assert_array_equal(a, b)


def test_contains_points_on_coarse_fine_transition() -> None:
    g = geometry(REFINED)
    nc = g.node_coordinates
    hanging = nc[HANGING_NODES, :2]
    # points just beside the Hanging nodes, on either side of the transition
    pts = np.vstack([hanging, hanging + [0.01, 0.0], hanging - [0.01, 0.0]])

    assert g.contains(pts).all()


def test_node_centered_data_at_hanging_nodes() -> None:
    g = geometry(REFINED)
    ec, nc = g.element_coordinates, g.node_coordinates

    zn = g.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    # Using only the fine Elements gave an error of 32. The pseudo-laplacian
    # is not exact for a linear field, so some error remains.
    exact = linear(nc[HANGING_NODES, 0], nc[HANGING_NODES, 1])
    assert zn[HANGING_NODES] == pytest.approx(exact, abs=12)


def test_quad_mesh_with_hanging_node_has_one_domain() -> None:
    g = quad_mesh_with_hanging_node()

    polygons = g.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert polygons.exteriors[0].area == pytest.approx(4.0)
    assert g.contains(np.array([[1.0, 1.0], [1.0, 0.5], [1.0, 1.5]])).all()


def test_quad_mesh_node_centered_data_at_hanging_node() -> None:
    g = quad_mesh_with_hanging_node()
    ec = g.element_coordinates

    zn = g.get_node_centered_data(linear(ec[:, 0], ec[:, 1]))

    # 17.5 from the fine Elements only
    assert zn[6] == pytest.approx(linear(1.0, 1.0), abs=0.5)


def test_hanging_node_found_within_rounding() -> None:
    g = quad_mesh_with_hanging_node()
    nc = g.node_coordinates.copy()
    nc[6, 0] += 1e-12  # off the coarse Face by rounding only
    g = GeometryFM2D(nc, g.element_table, codes=g.codes, projection="NON-UTM")

    assert len(g.boundary_polygons.exteriors) == 1


def test_node_off_the_face_is_not_hanging() -> None:
    # move the would-be Hanging node far off the coarse Face: a real gap
    g = quad_mesh_with_hanging_node()
    nc = g.node_coordinates.copy()
    nc[6, 0] = 1.2
    g = GeometryFM2D(nc, g.element_table, codes=g.codes, projection="NON-UTM")

    assert len(g.boundary_polygons.exteriors) == 2


def test_boundary_nodes_touching_face_ends_are_not_hanging() -> None:
    # A strip of two quads with straight sides: each boundary Node lies on
    # the line through its neighbouring Faces, but only at their ends.
    nc = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [0, 1, 0]],
        dtype=float,
    )
    g = GeometryFM2D(nc, [[0, 1, 4, 5], [1, 2, 3, 4]], projection="NON-UTM")

    polygons = g.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert len(polygons.exteriors[0].xy) == 7  # 6 Nodes, closed
    assert polygons.exteriors[0].area == pytest.approx(2.0)


def test_layered_mesh_with_hanging_node_has_one_domain() -> None:
    g = layered_quad_mesh_with_hanging_node()

    polygons = g.boundary_polygons

    assert len(polygons.exteriors) == 1
    assert polygons.exteriors[0].area == pytest.approx(4.0)


def test_subset_without_hanging_nodes_is_correct() -> None:
    g = quad_mesh_with_hanging_node()

    sub = g.isel([1, 2])

    assert isinstance(sub, GeometryFM2D)
    assert len(sub.boundary_polygons.exteriors) == 1
    assert sub.boundary_polygons.exteriors[0].area == pytest.approx(2.0)


def test_unchanged_behaviour_on_locally_refined_mesh() -> None:
    g = geometry(REFINED)
    ds = mikeio.read(REFINED)

    assert g.get_element_area().sum() == pytest.approx(150.0 * 100.0)
    assert ds.isel(element=[0, 1, 2]).geometry.n_elements == 3
    x, y = g.element_coordinates[5, :2]
    assert g.find_index(x=x, y=y)[0] == 5


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
def test_plot_with_refinement_on_locally_refined_mesh(plot_type: str) -> None:
    g = geometry(REFINED)
    ec = g.element_coordinates
    da = mikeio.DataArray(linear(ec[:, 0], ec[:, 1]), geometry=g)

    _plot(da, plot_type, n_refinements=1)


@pytest.mark.parametrize("plot_type", ["contour", "contourf", "shaded"])
def test_plot_with_refinement_on_quad_mesh_with_hanging_node(plot_type: str) -> None:
    g = quad_mesh_with_hanging_node()
    ec = g.element_coordinates
    da = mikeio.DataArray(linear(ec[:, 0], ec[:, 1]), geometry=g)

    _plot(da, plot_type, n_refinements=1, show_mesh=True)


def test_outline_of_locally_refined_mesh_has_no_interior_line() -> None:
    import matplotlib.pyplot as plt

    da = mikeio.read(REFINED)[0]

    ax = da.plot.outline()

    assert len(ax.lines) == 1
    plt.close("all")


def test_outline_of_mesh_only_geometry() -> None:
    import matplotlib.pyplot as plt

    g = geometry(REFINED)
    mesh = GeometryFM2D(
        g.node_coordinates, g.element_table, codes=g.codes, projection="NON-UTM"
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
    for a, b in zip(geometry(out).element_table, geometry(filename).element_table):
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
