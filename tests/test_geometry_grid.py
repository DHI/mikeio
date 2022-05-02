import os
from pathlib import Path

import numpy as np
import pytest
from mikeio import Mesh
from mikeio import Grid2D, Grid1D


def test_create_nx_ny():

    g = Grid2D(x0=180.0, y0=-90.0, dx=0.25, dy=0.25, nx=1440, ny=721)
    assert g.nx == 1440


def test_create_nx_missing_ny():

    with pytest.raises(ValueError) as excinfo:
        Grid2D(x0=180.0, y0=-90.0, dx=0.25, dy=0.25, nx=100)

    assert "ny" in str(excinfo.value)


def test_grid1d_x():
    x0 = 2.0
    x1 = 8.0
    nx = 4
    x = np.linspace(x0, x1, nx)
    g = Grid1D(x=x)
    assert g.x0 == x0
    assert g.x1 == x1


def test_x_y():
    x0 = 2.0
    x1 = 8.0
    nx = 4
    dx = 2.0
    x = np.linspace(x0, x1, nx)
    y0 = 3.0
    y1 = 5
    ny = 3
    dy = 1.0
    y = np.linspace(y0, y1, ny)
    g = Grid2D(x=x, y=y)
    assert g.x0 == x0
    assert g.x1 == x1
    assert g.y0 == y0
    assert g.y1 == y1
    assert np.all(g.x == x)
    assert np.sum(g.y - y) == 0
    assert g.nx == nx
    assert g.ny == ny
    assert g.dx == dx
    assert g.dy == dy

    # BoundingBox(left, bottom, right, top)
    # Is this test good, or just a copy of the implementation?
    assert g.bbox == ((x0 - dx / 2), (y0 - dy / 2), (x1 + dx / 2), (y1 + dy / 2))

    text = repr(g)
    assert "<mikeio.Grid2D>" in text


def test_non_equidistant_axis_grid1d_not_allowed():
    x = [0.0, 1.0, 2.0, 2.9]

    with pytest.raises(Exception) as excinfo:  # this could be allowed in the future
        Grid1D(x=x)

    assert "equidistant" in str(excinfo.value)


def test_non_equidistant_axis_grid2d_not_allowed():
    x = [0, 1]
    y = [0.0, 1.0, 2.0, 2.9]

    with pytest.raises(Exception) as excinfo:  # this could be allowed in the future
        Grid2D(x=x, y=y)

    assert "equidistant" in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:  # this could be allowed in the future
        Grid2D(x=y, y=x)

    assert "equidistant" in str(excinfo.value)


def test_dx_dy_is_positive():

    with pytest.raises(ValueError) as excinfo:
        Grid2D(nx=2, ny=4, dx=-1.0, dy=1.0)

    assert "positive" in str(excinfo.value).lower()

    with pytest.raises(ValueError) as excinfo:
        Grid2D(nx=2, ny=4, dx=1.0, dy=0.0)

    assert "positive" in str(excinfo.value).lower()


def test_x_is_increasing():
    with pytest.raises(ValueError) as excinfo:
        Grid1D(x=[2.0, 1.0])

    assert "increasing" in str(excinfo.value)


def test_x_y_is_increasing():
    with pytest.raises(ValueError) as excinfo:
        Grid2D(x=[2.0, 1.0], y=[0.0, 1.0])

    assert "increasing" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Grid2D(x=[0.0, 1.0], y=[2.0, 1.0])

    assert "increasing" in str(excinfo.value)


def test_xx_yy():
    nx = 4
    ny = 3
    x = np.linspace(1, 7, nx)
    y = np.linspace(3, 5, ny)
    g = Grid2D(x=x, y=y)
    # assert g.n == nx * ny
    assert g._xx[0, 0] == 1.0
    assert g._yy[-1, -1] == 5.0
    assert np.all(g.xy[1] == [3.0, 3.0])
    assert np.all(g.coordinates[1] == [3.0, 3.0])

    g2 = Grid2D(x=x, y=y)

    # Reverse order compared to above makes no difference
    assert g2._yy[-1, -1] == 5.0
    assert g2._xx[0, 0] == 1.0


def test_create_in_bbox():
    bbox = [0, 0, 1, 5]
    g = Grid2D(bbox=bbox, nx=2, ny=5)
    assert g.x0 == 0.25

    g = Grid2D(bbox=bbox, nx=2, ny=None)
    assert g.x0 == 0.25

    g = Grid2D(bbox=bbox)
    assert g.nx == 10
    assert g.ny == 50

    dx = 0.5
    g = Grid2D(bbox=bbox, dx=dx)
    assert g.dx == dx
    assert g.dy == dx
    # assert g.n == 20

    dx = 0.5
    dy = 2.5
    g = Grid2D(bbox=bbox, dx=dx, dy=dy)
    assert g.dx == dx
    assert g.dy == dy
    assert g.nx * g.ny == 4

    # g = Grid2D(bbox=bbox, dx=dx, dy=2.5)
    # assert g.dx == dx
    # assert g.dy == 2.5
    # assert g.n == 4

    # with pytest.raises(ValueError):
    #     Grid2D(bbox, shape=(12, 2, 2))

    # with pytest.raises(ValueError):
    #     Grid2D(bbox=bbox, nx=None, ny=None)


def test_no_parameters():

    with pytest.raises(ValueError):
        Grid2D()


def test_invalid_grid_not_possible():
    bbox = [0, 0, -1, 1]  # x0 > x1
    with pytest.raises(ValueError):
        Grid2D(bbox=bbox, nx=2, ny=2)

    bbox = [0, 0, 1, -1]  # y0 > y1
    with pytest.raises(ValueError):
        Grid2D(bbox=bbox, nx=2, ny=2)


def test_contains():
    bbox = [0, 0, 1, 5]
    g = Grid2D(bbox=bbox)
    xy1 = [0.5, 4.5]
    xy2 = [1.5, 0.5]
    assert g.contains(xy1)
    assert not g.contains(xy2)

    xy = np.vstack([xy1, xy2, xy1])
    inside = g.contains(xy)
    assert inside[0]
    assert not inside[1]

    # inside = g.contains(xy[:, 0], xy[:, 1])
    # assert inside[0]
    # assert not inside[1]


def test_find_index():
    bbox = [0, 0, 1, 5]
    g = Grid2D(bbox=bbox, dx=0.2)
    xy1 = [0.52, 1.52]
    xy2 = [1.5, 0.5]
    i1, j1 = g.find_index(coords=xy1)
    assert i1 == 2
    assert j1 == 7
    i2, j2 = g.find_index(coords=xy2)
    assert i2 == -1
    assert j2 == -1

    xy = np.vstack([xy1, xy2])
    ii, jj = g.find_index(coords=xy)
    assert ii[0] == i1
    assert ii[1] == i2
    assert jj[0] == j1
    assert jj[1] == j2

    xy = np.vstack([xy1, xy2, xy2])
    ii, jj = g.find_index(coords=xy)
    assert ii[0] == i1
    assert jj[0] == j1
    assert ii[2] == i2
    assert jj[2] == j2


def test_to_mesh(tmp_path: Path):
    outfilename = tmp_path / "temp.mesh"

    # 1

    g = Grid2D(bbox=[0, 0, 1, 5])
    g.to_mesh(outfilename)

    assert outfilename.exists()
    mesh = Mesh(outfilename)
    assert mesh.n_elements == g.nx * g.ny
    outfilename.unlink()

    # 2
    nc = g.get_node_coordinates()
    new_z = nc[:, 1] - 10
    g.to_mesh(outfilename, z=new_z)

    assert outfilename.exists()
    mesh = Mesh(outfilename)
    assert mesh.node_coordinates[0, 2] == new_z[2]

    outfilename.unlink()

    # 3

    new_z = -10
    g.to_mesh(outfilename, z=new_z)

    assert outfilename.exists()
    mesh = Mesh(outfilename)
    assert mesh.node_coordinates[0, 2] == -10


# def test_xy_to_bbox():
#     bbox = [0, 0, 1, 5]
#     g = Grid2D(bbox)
#     xy = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 5], [1, 5]], dtype=float)
#     bbox2 = Grid2D.xy_to_bbox(xy)
#     assert bbox[0] == bbox2[0]
#     assert bbox[-1] == bbox2[-1]

#     bbox2 = Grid2D.xy_to_bbox(xy, buffer=0.2)
#     assert bbox2[0] == -0.2
#     assert bbox2[3] == 5.2


def test_isel():
    bbox = [0, 0, 1, 5]
    g = Grid2D(bbox=bbox, nx=10, ny=20)
    assert g.nx == 10

    g1 = g.isel(0, axis=1)

    assert g1.nx == 20
