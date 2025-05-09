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
def hd2d_dfs() -> None:
    return mikeio.open("tests/testdata/HD2D.dfsu")


def test_plot_dfsu_contour_mixedmesh() -> None:
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


def test_plot_dfsu_n_refinements() -> None:
    dfs = mikeio.open("tests/testdata/FakeLake.dfsu")
    geometry = dfs.geometry
    geometry.plot(plot_type="contourf", levels=None, n_refinements=1)
    assert True


def test_plot_dfsu_contour_subset_not_allowed(hd2d_dfs):
    dfs = hd2d_dfs
    da = dfs.read(items="Surface elevation", time=0)[0]
    elem40 = np.arange(40)
    with pytest.raises(Exception):
        da.plot.contour(elements=elem40)


def test_plot_mesh() -> None:
    msh = Mesh("tests/testdata/odense_rough.mesh")
    msh.plot(show_mesh=False)
    assert True


def test_plot_mesh_outline() -> None:
    msh = Mesh("tests/testdata/odense_rough.mesh")
    msh.plot(plot_type="outline_only")
    assert True
    msh.plot(plot_type=None)
    assert True


def test_plot_mesh_ax() -> None:
    import matplotlib.pyplot as plt

    msh = Mesh("tests/testdata/odense_rough.mesh")
    _, ax = plt.subplots()
    msh.plot(ax=ax)
    assert True


def test_plot_mesh_boundary_nodes() -> None:
    msh = Mesh("tests/testdata/odense_rough.mesh")
    msh.plot.boundary_nodes()
    msh.plot.boundary_nodes(["Land", "Sea"])
    assert True


def test_plot_invalid() -> None:
    msh = Mesh("tests/testdata/odense_rough.mesh")
    with pytest.raises(Exception):
        msh.plot(plot_type="invalid")
    with pytest.raises(Exception):
        msh.plot(plot_type="invalid")


def test_plot_dfsu_vertical_profile() -> None:
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


def test_da_plot() -> None:
    import matplotlib.pyplot as plt

    ds = mikeio.read("tests/testdata/FakeLake.dfsu")
    da = ds[0]
    da.plot()
    da.plot(show_mesh=True)
    da.plot.contour()
    da.plot.contourf(show_mesh=True, cmap="terrain", levels=[-30, -20, -10, 0])
    da.plot.outline()

    dam = da.max()
    dam.plot.contour()
    dam.plot.contourf()
    dam.plot.outline()

    da.max("space").plot()

    plt.close("all")


def test_geometry_fm_plot() -> None:
    dfs = mikeio.open("tests/testdata/FakeLake.dfsu")
    g = dfs.geometry
    g.plot()
    g.plot.outline()
    g.plot.mesh()
    g.plot.contour()
    g.plot.contourf()


def test_plot_non_utm_file() -> None:
    ds = mikeio.read("tests/testdata/FakeLake_NONUTM.dfsu")
    da = ds[0]
    da.plot()


def test_plot_timeseries() -> None:
    ds = mikeio.read("tests/testdata/HD2D.dfsu")

    # just make sure it runs
    ds["Surface elevation"].isel(element=0).plot.line()
    ds["Surface elevation"].isel(time=-1).plot.line()


def test_plot_vertical_transect() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")

    dsp = ds.sel(x=333934.1, y=6158101.5)
    da = dsp["Temperature"]
    da.plot.pcolormesh()


def test_plot_point_spectrum() -> None:
    # directional spectra
    da = mikeio.read("tests/testdata/spectra/line_dir_spectra.dfsu")[0]
    da.isel(node=4).plot()

    # frequency spectra
    da2 = mikeio.read("tests/testdata/spectra/line_freq_spectra.dfsu")[0]
    da2_pt = da2.isel(node=4)
    da2_pt.plot()

    # 2d spectra
    da_pt = mikeio.read("tests/testdata/spectra/pt_spectra.dfsu")[0]
    da_pt.plot.patch()
    da_pt.plot.contour()
    da_pt.plot.contourf()
