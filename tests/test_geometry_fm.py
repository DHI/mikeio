import numpy as np
import pytest
from mikeio.spatial.FM_geometry import GeometryFM, GeometryFM3D


def test_basic():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    g = GeometryFM(nc, el)
    assert g.n_elements == 1
    assert g.n_nodes == 3


def test_too_many_elements():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2, 3)]  # There is no node #3

    with pytest.raises(ValueError) as excinfo:
        GeometryFM(nc, el)

    assert "element" in str(excinfo.value).lower()


def test_no_nodes():
    el = [(0, 1, 2)]

    with pytest.raises(ValueError):
        GeometryFM(node_coordinates=None, element_table=el)


def test_no_element_table():
    nc = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, 1.0, 0.0),
    ]

    with pytest.raises(ValueError):
        GeometryFM(node_coordinates=nc, element_table=None)


def test_overset_grid():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    proj = "UTM-33"
    g = GeometryFM(nc, el, projection=proj)
    grid = g.get_overset_grid(dx=0.5)
    assert grid.nx == 2
    assert grid.ny == 2
    assert grid.projection_string == proj


def test_area():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2, 3)]

    g = GeometryFM(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    assert not g.is_tri_only
    area = g.get_element_area()
    assert len(area) == g.n_elements
    assert area > 0.0


def test_plot_mesh():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]

    g = GeometryFM(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    assert g.n_elements == 2
    g.plot.mesh()


def test_layered():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, -1.0),
        (1.0, 1.0, -1.0),
        (0.0, 0.0, -2.0),
        (1.0, 0.0, -2.0),
        (1.0, 1.0, -2.0),
    ]

    el = [(0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)]

    g = GeometryFM3D(
        node_coordinates=nc,
        element_table=el,
        projection="LONG/LAT",
        n_layers=2,
        n_sigma=2,
    )
    assert g.n_elements == 2
    assert g.n_layers == 2
    assert not g.is_2d

    assert len(g.top_elements) == 1
