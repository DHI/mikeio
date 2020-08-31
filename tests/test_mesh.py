import os
import pytest
from mikeio import Mesh


def test_get_number_of_elements():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    assert msh.n_elements == 654


def test_get_element_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    ec = msh.element_coordinates

    assert ec.shape == (654, 3)
    assert ec[0, 0] > 212000.0
    assert ec[0, 1] > 6153000.0


def test_get_node_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    nc = msh.node_coordinates

    assert nc.shape == (399, 3)


def test_get_land_node_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    nc = msh.node_coordinates[msh.codes==1]

    assert nc.shape == (134, 3)


def test_get_bad_node_coordinates():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    with pytest.raises(Exception):
        nc = msh.get_node_coords(code="foo")


def test_plot_mesh():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    msh.plot()

    assert True


def test_plot_mesh_part():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    msh.plot(elements=list(range(0,100)))

    assert True


def test_plot_mesh_boundary_nodes():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)

    msh.plot_boundary_nodes()

    assert True


def test_set_z():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    zn = msh.node_coordinates[:,2]
    zn[zn<-3] = -3
    msh.set_z(zn)
    zn = msh.node_coordinates[:,2]
    assert zn.min() == -3


def test_set_codes():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    codes = msh.codes
    assert msh.codes[2] == 2
    codes[codes==2] = 7
    msh.set_codes(codes)
    assert msh.codes[2] == 7


def test_write(tmpdir):
    outfilename = os.path.join(tmpdir.dirname, "simple.mesh")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)

    msh.write(outfilename)

    assert os.path.exists(outfilename)


def test_write_part(tmpdir):
    outfilename = os.path.join(tmpdir.dirname, "simple_sub.mesh")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)

    msh.write(outfilename, elements=list(range(0,100)))

    assert os.path.exists(outfilename)
