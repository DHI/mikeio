import os
import pytest
from mikeio.mesh import Mesh


def test_get_number_of_elements():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    m = Mesh()

    m.read(filename)

    assert m.get_number_of_elements() == 654


def test_get_element_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    m = Mesh()

    m.read(filename)

    ec = m.get_element_coords()

    assert ec.shape == (654, 3)
    assert ec[0, 0] > 212000.0
    assert ec[0, 1] > 6153000.0


def test_get_node_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    m = Mesh()

    m.read(filename)

    nc = m.get_node_coords()

    assert nc.shape == (399, 3)


def test_get_land_node_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    m = Mesh()

    m.read(filename)

    nc = m.get_node_coords(code=1)

    assert nc.shape == (134, 3)


def test_get_bad_node_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    m = Mesh()

    m.read(filename)

    with pytest.raises(Exception):
        nc = m.get_node_coords(code="foo")

