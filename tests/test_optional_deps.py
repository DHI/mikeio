"""Verify mikeio degrades gracefully when matplotlib/scipy are unavailable.

matplotlib and scipy are always installed in the test environment (see the
test dependency-group), so these tests simulate their absence by forcing
sys.modules entries to None for the exact module names mikeio's
import_optional() requests -- this makes the subsequent `import` raise
ImportError regardless of whether the real package is installed, giving
CI real regression coverage of the "extra not installed" code path.
"""

import sys

import pytest

import mikeio

_MATPLOTLIB_MODULES = [
    "matplotlib",
    "matplotlib.cm",
    "matplotlib.collections",
    "matplotlib.colors",
    "matplotlib.patches",
    "matplotlib.projections.polar",
    "matplotlib.pyplot",
    "matplotlib.tri",
    "mpl_toolkits.axes_grid1",
]
_SCIPY_MODULES = ["scipy.interpolate", "scipy.spatial"]


@pytest.fixture
def no_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _MATPLOTLIB_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def no_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SCIPY_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def no_optional_deps(no_matplotlib: None, no_scipy: None) -> None:
    pass


# -- core functionality must not require matplotlib or scipy -----------------


def test_core_io_works_without_optional_deps(no_optional_deps: None) -> None:
    ds = mikeio.read("tests/testdata/random.dfs0")
    assert ds.n_items == 2

    dfs = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu")
    assert dfs.geometry.n_elements > 0


def test_contains_and_find_index_work_without_matplotlib(
    no_matplotlib: None,
) -> None:
    geom = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu").geometry
    pts = geom.element_coordinates[:5, :2]

    assert geom.contains(pts).all()

    idx = geom.find_index(area=geom.boundary_polygons.exteriors[0].xy)
    assert len(idx) == geom.n_elements


def test_get_node_centered_data_works_without_optional_deps(
    no_optional_deps: None,
) -> None:
    geom = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu").geometry
    data = geom.element_coordinates[:, 0]
    node_data = geom.get_node_centered_data(data)
    assert node_data.shape == (geom.n_nodes,)


# -- plotting/interpolation must fail with a helpful message ------------------


def test_plot_raises_friendly_error_without_matplotlib(no_matplotlib: None) -> None:
    ds = mikeio.read("tests/testdata/random.dfs0")

    with pytest.raises(ModuleNotFoundError, match=r"pip install mikeio\[plot\]"):
        ds[0].plot()

    with pytest.raises(ModuleNotFoundError, match=r"pip install mikeio\[plot\]"):
        ds.plot()


def test_geometry_plot_raises_friendly_error_without_matplotlib(
    no_matplotlib: None,
) -> None:
    geom = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu").geometry

    with pytest.raises(ModuleNotFoundError, match=r"pip install mikeio\[plot\]"):
        geom.plot.mesh()


def test_find_nearest_elements_raises_friendly_error_without_scipy(
    no_scipy: None,
) -> None:
    geom = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu").geometry

    with pytest.raises(ModuleNotFoundError, match=r"pip install mikeio\[interp\]"):
        geom.find_nearest_elements(0, 0)


def test_interp_time_raises_friendly_error_without_scipy(no_scipy: None) -> None:
    ds = mikeio.read("tests/testdata/random.dfs0")

    with pytest.raises(ModuleNotFoundError, match=r"pip install mikeio\[interp\]"):
        ds[0].interp_time(dt=1800)


def test_extract_surface_elevation_raises_friendly_error_without_scipy(
    no_scipy: None,
) -> None:
    dfs = mikeio.Dfsu3D("tests/testdata/basin_3d.dfsu")

    with pytest.raises(ModuleNotFoundError, match=r"pip install mikeio\[interp\]"):
        dfs.extract_surface_elevation_from_3d()
