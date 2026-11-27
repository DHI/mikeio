"""Core workflows on Locally refined meshes (#1112), beyond the geometry itself.

test_dfsu_locally_refined.py checks the domain, node values and plots of a
Locally refined mesh. This file runs the everyday workflows on such meshes:
point location, interpolation, track extraction, writing dfsu and mesh
files, mikeio.generic, and layered 3D (sigma and sigma-z) files with
Hanging nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shapely
from mikecore.DfsuFile import DfsuFile, DfsuFileType
from numpy.typing import NDArray

import mikeio
import mikeio.generic
from mikeio.spatial import GeometryFM2D, GeometryFM3D

from .test_dfsu_locally_refined import (
    REFINED,
    Mesh,
    as_mesh,
    domain,
    extrude,
    linear,
    read_geometry,
    refine,
)

mpl.use("Agg")


def square_grid(n: int, quad: bool) -> Mesh:
    """n by n unit cells, each a quad or two triangles."""
    x, y = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0))
    elements: list[list[int]] = []
    for k in range(n):
        for i in range(n):
            a, b = k * (n + 1) + i, k * (n + 1) + i + 1
            c, d = b + n + 1, a + n + 1
            elements += [[a, b, c, d]] if quad else [[a, b, c], [c, d, a]]
    return Mesh(np.column_stack([x.ravel(), y.ravel()]), elements)


def refine_where(m: Mesh, inside: Callable[[float, float], bool]) -> Mesh:
    """Split the Elements whose centre is inside, as `refine` does."""
    ec = m.geometry().element_coordinates
    return refine(m, {e for e, (x, y, _) in enumerate(ec) if inside(x, y)})


def nested(quad: bool) -> Mesh:
    """8 by 8 cells, refined twice towards the centre: one Hanging node per Face."""
    m = refine_where(square_grid(8, quad), lambda x, y: 2 < x < 6 and 2 < y < 6)
    return refine_where(m, lambda x, y: 3 < x < 5 and 3 < y < 5)


def unbalanced(quad: bool) -> Mesh:
    """Left half refined twice: three Hanging nodes on each coarse Face beside it."""
    m = square_grid(8, quad)
    for _ in range(2):
        m = refine_where(m, lambda x, y: x < 4)
    return m


MESHES: dict[str, Callable[[], Mesh]] = {
    "file": lambda: as_mesh(read_geometry(REFINED)),
    "nested_tri": lambda: nested(quad=False),
    "nested_quad": lambda: nested(quad=True),
}


def submesh_type(filename: str | Path) -> int | None:
    dfs = DfsuFile.Open(str(filename))
    try:
        return dfs.SubMeshType
    finally:
        dfs.Close()


def hanging_nodes(g: GeometryFM2D) -> set[int]:
    conforming = g._conforming_element_table
    assert conforming is not None
    return {
        int(n)
        for c, e in zip(conforming, g.element_table)
        for n in set(c.tolist()) - set(np.asarray(e).tolist())
    }


def points_beside_coarse_faces(g: GeometryFM2D) -> NDArray[np.float64]:
    """Points on and just inside each coarse Face with Hanging nodes.

    The Hanging nodes, and points along each coarse Face moved a little
    towards the centre of the coarse Element: where the fine Elements beside
    it crowd the nearest Element centres.
    """
    conforming = g._conforming_element_table
    assert conforming is not None
    xy = g.node_coordinates[:, :2]
    points = [xy[sorted(hanging_nodes(g))]]
    for c, e, centre in zip(conforming, g.element_table, g.element_coordinates):
        if len(c) == len(e):
            continue
        for a, b in zip(e, np.roll(e, -1)):
            for t in (0.05, 0.25, 0.5, 0.75, 0.95):
                on_face = (1 - t) * xy[a] + t * xy[b]
                for s in (0.0, 0.05, 0.15, 0.3):
                    points.append((on_face + s * (centre[:2] - on_face))[None, :])
    pts = np.vstack(points)
    # only interior points: a coarse Face can end on the domain boundary
    interior = domain(g).buffer(-1e-9 * float(np.ptp(xy, axis=0).max()))
    return pts[shapely.contains_xy(interior, pts[:, 0], pts[:, 1])]


def linear_dataarray(g: GeometryFM2D, n_timesteps: int = 2) -> mikeio.DataArray:
    ec = g.element_coordinates
    data = np.tile(linear(ec[:, 0], ec[:, 1]), (n_timesteps, 1))
    time = pd.date_range("2000-01-01", periods=n_timesteps, freq="30s")
    return mikeio.DataArray(data, time=time, geometry=g, item=mikeio.ItemInfo("linear"))


# Point location and interpolation


# find_index tests the Elements with the 10 nearest centres only. Beside a
# coarse Face, the fine Elements can crowd out the coarse Element itself.
NEAREST_TEN = (
    "find_index tries only the Elements with the 10 nearest centres; beside a "
    "coarse Face the fine Elements can all be nearer than the coarse Element"
)


@pytest.mark.parametrize(
    "make",
    [
        # in the file, a sliver of about 1.4 m2 of coarse Element 6 is not found:
        # along its Face with Hanging node 11, near (80.5, 48) (file numbering)
        pytest.param(
            MESHES["file"], marks=pytest.mark.xfail(strict=True, reason=NEAREST_TEN)
        ),
        MESHES["nested_tri"],
        MESHES["nested_quad"],
        pytest.param(
            lambda: unbalanced(quad=False),
            marks=pytest.mark.xfail(strict=True, reason=NEAREST_TEN),
        ),
        pytest.param(
            lambda: unbalanced(quad=True),
            marks=pytest.mark.xfail(strict=True, reason=NEAREST_TEN),
        ),
    ],
    ids=["file", "nested_tri", "nested_quad", "unbalanced_tri", "unbalanced_quad"],
)
def test_find_index_beside_coarse_faces(make: Callable[[], Mesh]) -> None:
    """A point on or beside a Face with Hanging nodes is found in its Element."""
    g = make().geometry()
    pts = points_beside_coarse_faces(g)
    assert len(pts) > 0
    assert g.contains(pts).all()

    idx = g.find_index(coords=pts)

    polygons = [shapely.Polygon(g.node_coordinates[el, :2]) for el in g.element_table]
    for p, e in zip(pts, idx):
        assert polygons[e].buffer(1e-9).contains(shapely.Point(p))


@pytest.mark.parametrize("make", MESHES.values(), ids=MESHES.keys())
def test_interp_like_grid_has_values_inside_the_domain(
    make: Callable[[], Mesh],
) -> None:
    g = make().geometry()
    grid = g.get_overset_grid(nx=40, ny=40)
    x, y = np.meshgrid(grid.x, grid.y)
    inside = shapely.contains_xy(domain(g), x, y)

    values = linear_dataarray(g).interp_like(grid).isel(time=0).to_numpy()

    assert not np.isnan(values[inside]).any()


@pytest.mark.parametrize("make", MESHES.values(), ids=MESHES.keys())
def test_extract_track_across_refinement(make: Callable[[], Mesh]) -> None:
    """A track that crosses coarse and fine Elements has values all along."""
    g = make().geometry()
    da = linear_dataarray(g)
    lo, hi = (
        g.node_coordinates[:, :2].min(axis=0),
        g.node_coordinates[:, :2].max(axis=0),
    )
    x = np.linspace(lo[0], hi[0], 200)[1:-1]
    y = lo[1] + (hi[1] - lo[1]) * (0.5 + 0.4 * np.sin(x * 6 / (hi[0] - lo[0])))
    track = pd.DataFrame(
        {"x": x, "y": y}, index=pd.date_range(da.time[0], da.time[-1], periods=len(x))
    )

    values = mikeio.Dataset([da]).extract_track(track, method="inverse_distance")

    assert not np.isnan(values["linear"].to_numpy()).any()


# Files


@pytest.mark.parametrize(
    "operation",
    [
        lambda src, out: mikeio.generic.scale(src, out, factor=2.0),
        lambda src, out: mikeio.generic.extract(src, out, step=1),
        lambda src, out: mikeio.generic.concat([src, src], out),
        lambda src, out: mikeio.generic.add(src, src, out),
        lambda src, out: mikeio.generic.diff(src, src, out),
        lambda src, out: mikeio.generic.avg_time(src, out),
        lambda src, out: mikeio.generic.quantile(src, out, q=0.5),
        lambda src, out: mikeio.generic.fill_corrupt(src, out),
    ],
    ids=[
        "scale",
        "extract",
        "concat",
        "add",
        "diff",
        "avg_time",
        "quantile",
        "fill_corrupt",
    ],
)
def test_generic_keeps_submesh_type(
    operation: Callable[[str, Path], None], tmp_path: Path
) -> None:
    out = tmp_path / "out.dfsu"
    operation(REFINED, out)
    assert submesh_type(out) == 1
    assert len(read_geometry(out).boundary_polygons.exteriors) == 1


def test_mesh_file_round_trip_keeps_hanging_nodes(tmp_path: Path) -> None:
    """A .mesh file has no Submesh type; the Hanging nodes are found again."""
    g = read_geometry(REFINED)
    g.to_mesh(tmp_path / "refined.mesh")

    g2 = mikeio.Mesh(tmp_path / "refined.mesh").geometry

    assert hanging_nodes(g2) == hanging_nodes(g)
    assert len(g2.boundary_polygons.exteriors) == 1


# Layered 3D


def sigma_z(g: GeometryFM2D, bed: Callable[[float, float], float]) -> GeometryFM3D:
    """Sigma-z layers below a 2D mesh: 2 sigma layers above up to 2 z layers.

    z interfaces at -3, -2 and -1, sigma interfaces at -0.5 and 0. An Element
    keeps the z layers above its bed, so columns have 2, 3 or 4 layers. Each
    Node column reaches down to the deepest Element around it.
    """
    levels = np.array([-3.0, -2.0, -1.0, -0.5, 0.0])
    n_z = 2
    n_z_el = [
        int(np.sum(levels[:n_z] >= bed(x, y))) for x, y, _ in g.element_coordinates
    ]
    lowest = np.full(g.n_nodes, n_z)
    for e, el in enumerate(g.element_table):
        lowest[el] = np.minimum(lowest[el], n_z - n_z_el[e])
    nc: list[list[float]] = []
    index: dict[tuple[int, int], int] = {}
    for n, (x, y, _) in enumerate(g.node_coordinates):
        for k in range(lowest[n], len(levels)):
            index[(n, k)] = len(nc)
            nc.append([x, y, levels[k]])
    elements = [
        [index[(n, k)] for n in el] + [index[(n, k + 1)] for n in el]
        for e, el in enumerate(g.element_table)
        for k in range(n_z - n_z_el[e], len(levels) - 1)
    ]
    return GeometryFM3D(
        node_coordinates=np.array(nc),
        element_table=elements,
        projection="NON-UTM",
        n_layers=len(levels) - 1,
        n_sigma=len(levels) - 1 - n_z,
        dfsu_type=DfsuFileType.Dfsu3DSigmaZ,
    )


def layered(make: Callable[[], Mesh], kind: str) -> GeometryFM3D:
    g = make().geometry()
    if kind == "sigma":
        g3 = extrude(g, 3)
        g3._type = DfsuFileType.Dfsu3DSigma
    else:
        x_mid = float(g.node_coordinates[:, 0].mean())
        g3 = sigma_z(g, lambda x, y: -1.5 if x < x_mid else -2.5)
    g3._submesh_type = 1
    return g3


@pytest.mark.parametrize("kind", ["sigma", "sigma-z"])
@pytest.mark.parametrize("make", MESHES.values(), ids=MESHES.keys())
def test_layered_file_with_hanging_nodes(
    make: Callable[[], Mesh], kind: str, tmp_path: Path
) -> None:
    g3 = layered(make, kind)
    g2 = make().geometry()
    zn = np.tile(g3.node_coordinates[:, 2], (2, 1))
    data = np.tile(g3.element_coordinates[:, 0], (2, 1))
    time = pd.date_range("2000-01-01", periods=2, freq="h")
    da = mikeio.DataArray(
        data, time=time, geometry=g3, item=mikeio.ItemInfo("x"), zn=zn
    )
    fn = tmp_path / "layered.dfsu"

    mikeio.Dataset([da]).to_dfs(fn)
    ds = mikeio.read(fn)

    assert submesh_type(fn) == 1
    g = ds.geometry
    assert isinstance(g, GeometryFM3D)
    # one column per 2D Element, and the 2D geometry has the Hanging nodes
    assert len(g.top_elements) == g2.n_elements
    assert hanging_nodes(g.geometry2d) == hanging_nodes(g2)
    assert len(g.boundary_polygons.exteriors) == 1
    assert g.contains(points_beside_coarse_faces(g2)).all()

    top = ds.sel(layers="top")
    top.to_dfs(tmp_path / "top.dfsu")
    assert submesh_type(tmp_path / "top.dfsu") == 1
    top["x"].isel(time=0).plot.contourf()
    plt.close("all")
