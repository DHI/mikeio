import os
import numpy as np
import pytest

from mikeio import Dfsu, Mesh

##################################################
# these tests will not run if matplotlib is not installed
##################################################
pytest.importorskip("matplotlib")

def test_plot_bathymetry():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    dfs.plot()
    assert True


def test_plot_2d():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)    
    dfs.plot(cmap='plasma')
    assert True


def test_plot_3d():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    dfs.plot()
    assert True


def test_plot_dfsu_contour():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    dfs.plot(plot_type="contour", levels=5)
    assert True


def test_plot_dfsu_contour_mixedmesh():
    filename = os.path.join("tests", "testdata", "FakeLake.dfsu")
    msh = Mesh(filename)
    msh.plot(plot_type="contour", levels=5)
    assert True


def test_plot_dfsu_n_refinements():
    filename = os.path.join("tests", "testdata", "FakeLake.dfsu")
    msh = Mesh(filename)
    msh.plot(plot_type="contourf", levels=5, n_refinements=1)
    assert True


def test_plot_dfsu_shaded():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    ds = dfs.read(items="Surface elevation", time_steps=0)
    elem40 = np.arange(40)
    wl_40 = ds.data[0][0, elem40]
    dfs.plot(wl_40, elements=elem40, plot_type="shaded", levels=5)
    assert True


def test_plot_dfsu():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    data = dfs.read()
    dfs.plot(z=data[1][0, :])
    assert True


def test_plot_dfsu_arguments():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    data = dfs.read()
    dfs.plot(title='test', label='test', vmin=-23, vmax=23)
    assert True


def test_plot_mesh():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    msh.plot(show_mesh=False)
    assert True


def test_plot_mesh_outline():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    msh.plot(plot_type='outline_only')
    assert True


def test_plot_mesh_part():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    msh.plot(elements=list(range(0, 100)))
    assert True


def test_plot_mesh_ax():
    import matplotlib.pyplot as plt
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    _, ax = plt.subplots()
    msh.plot(ax=ax)
    assert True   


def test_plot_mesh_boundary_nodes():
    filename = os.path.join("tests", "testdata", "odense_rough.mesh")
    msh = Mesh(filename)
    msh.plot_boundary_nodes()
    msh.plot_boundary_nodes(["Land", "Sea"])
    assert True
