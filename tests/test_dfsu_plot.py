import numpy as np
import pytest
import matplotlib as mpl

from mikeio import Mesh
import mikeio

mpl.use("Agg")
mpl.rcParams.update({"figure.max_open_warning": 100})




##################################################
# these tests will not run if matplotlib is not installed
##################################################
pytest.importorskip("matplotlib")


@pytest.fixture
def hd2d_dfs():
    return mikeio.open("tests/testdata/HD2D.dfsu")

def test_plot_bathymetry():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.open(filename)
    with pytest.warns(FutureWarning):
        dfs.plot()
    assert True


def test_plot_bathymetry_no_colorbar():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.open(filename)
    with pytest.warns(FutureWarning):
        dfs.plot(add_colorbar=False)
    assert True


def test_plot_2d(hd2d_dfs):
    dfs = hd2d_dfs
    with pytest.warns(FutureWarning):
        dfs.plot(cmap="plasma")
    assert True


def test_plot_3d():
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.open(filename)
    with pytest.warns(FutureWarning):
        dfs.plot()
    assert True


def test_plot_dfsu_contour(hd2d_dfs):
    dfs = hd2d_dfs
    with pytest.warns(FutureWarning):
        dfs.plot(plot_type="contour", levels=5)
    assert True


def test_plot_dfsu_contourf_levels(hd2d_dfs):
    dfs = hd2d_dfs
    cmap = mpl.colors.ListedColormap(["red", "green", "blue"])
    bounds = [-3, 1, 2, 100]
    with pytest.warns(FutureWarning):
        dfs.plot(levels=bounds, cmap=cmap)
    with pytest.warns(FutureWarning):
        dfs.plot(plot_type="contourf", levels=bounds, cmap=cmap)
    assert True


def test_plot_dfsu_contour_mixedmesh():
    dfs = mikeio.open("tests/testdata/FakeLake.dfsu")
    geometry = dfs.geometry
    geometry.plot(plot_type="contour", levels=5)
    geometry.plot(
        plot_type="contourf",
        title="contourf",
        show_mesh=False,
        levels=[-30, -24, -22, -10, -8],
    )
    assert True


def test_plot_dfsu_n_refinements():
    dfs = mikeio.open("tests/testdata/FakeLake.dfsu")
    geometry = dfs.geometry
    geometry.plot(plot_type="contourf", levels=None, n_refinements=1)
    assert True


def test_plot_dfsu_shaded(hd2d_dfs):
    dfs = hd2d_dfs
    da = dfs.read(items="Surface elevation", time=0)[0]
    elem40 = np.arange(40)

    ax = da.plot(elements=elem40) # this is ok
    assert ax is not None

    wl_40 = da.values[elem40]
    with pytest.warns(FutureWarning):
        dfs.plot(wl_40, elements=elem40, plot_type="shaded", levels=5)

def test_plot_dfsu_contour_subset_not_allowed(hd2d_dfs):
    dfs = hd2d_dfs
    da = dfs.read(items="Surface elevation", time=0)[0]
    elem40 = np.arange(40)
    with pytest.raises(Exception):
        da.plot.contour(elements=elem40)



def test_plot_dfsu(hd2d_dfs):
    dfs = hd2d_dfs
    data = dfs.read()
    with pytest.warns(FutureWarning):
        dfs.plot(z=data[1][0, :], figsize=(3, 3), plot_type="mesh_only")
    assert True


def test_plot_dfsu_squeeze(hd2d_dfs):
    dfs = hd2d_dfs
    data = dfs.read(items=0, time=0)
    with pytest.warns(FutureWarning):
        dfs.plot(z=data)  # 1 item-dataset
    assert True


def test_plot_dfsu_arguments():
    dfs = mikeio.open("tests/testdata/NorthSea_HD_and_windspeed.dfsu")
    dfs.read()
    with pytest.warns(FutureWarning):
        dfs.plot(title="test", label="test", vmin=-23, vmax=23)
    assert True


def test_plot_mesh():
    msh = Mesh("tests/testdata/odense_rough.mesh")
    msh.plot(show_mesh=False)
    assert True


def test_plot_mesh_outline():
    msh = Mesh("tests/testdata/odense_rough.mesh")
    msh.plot(plot_type="outline_only")
    assert True
    msh.plot(plot_type=None)
    assert True


def test_plot_mesh_ax():
    import matplotlib.pyplot as plt

    msh = Mesh("tests/testdata/odense_rough.mesh")
    _, ax = plt.subplots()
    msh.plot(ax=ax)
    assert True


def test_plot_mesh_boundary_nodes():
    msh = Mesh("tests/testdata/odense_rough.mesh")
    msh.plot_boundary_nodes()
    msh.plot_boundary_nodes(["Land", "Sea"])
    assert True


def test_plot_invalid():
    msh = Mesh("tests/testdata/odense_rough.mesh")
    with pytest.raises(Exception):
        msh.plot(plot_type="invalid")
    with pytest.raises(Exception):
        msh.plot(plot_type="invalid")


def test_plot_dfsu_vertical_profile():
    import matplotlib.pyplot as plt

    dfs = mikeio.open("tests/testdata/oresund_vertical_slice.dfsu")
    time_step = 1
    item_number = 1
    data = dfs.read()[item_number].to_numpy()[time_step, :]
    # defaults
    dfs.plot_vertical_profile(data)
    # dfs.plot_vertical_profile(data, time_step, 0, 20)
    dfs.plot_vertical_profile(
        data,
        title="txt",
        label="txt",
        edge_color="0.3",
        cmin=0,
        cmax=20,
        cmap="plasma",
        figsize=(2, 2),
    )
    _, ax = plt.subplots()
    dfs.plot_vertical_profile(data, ax=ax)
    assert True

    plt.close("all")


def test_da_plot():
    import matplotlib.pyplot as plt

    ds = mikeio.read("tests/testdata/FakeLake.dfsu")
    da = ds[0]
    da.plot()
    da.plot(show_mesh=True)
    da.plot.contour()
    da.plot.contourf(show_mesh=True,cmap='terrain', levels=[-30,-20,-10,0])
    da.plot.outline()

    dam = da.max()
    dam.plot.contour()
    dam.plot.contourf()
    dam.plot.outline()

    da.max("space").plot()

    plt.close("all")

def test_plot_non_utm_file():
    
    ds = mikeio.read("tests/testdata/FakeLake_NONUTM.dfsu")
    da = ds[0]
    da.plot()