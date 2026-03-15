from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import mikeio
from mikeio import Dfsu2DV
from mikeio.spatial import (
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
)
from mikeio.spatial import GeometryPoint3D
from mikecore.DfsuFile import DfsuFileType


@pytest.fixture
def vslice() -> Dfsu2DV:
    # sigma, z vertical profile (=transect)
    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    return Dfsu2DV(filename)


@pytest.fixture
def vslice_geo() -> Dfsu2DV:
    # sigma, z vertical profile (=transect)
    # in LONG/LAT, non-straight
    filename = "tests/testdata/kalundborg_transect.dfsu"
    return Dfsu2DV(filename)


def test_transect_open(vslice: Dfsu2DV) -> None:
    dfs = vslice
    assert dfs._type == DfsuFileType.DfsuVerticalProfileSigmaZ
    assert dfs.n_items == 2
    assert dfs.geometry.n_elements == 441
    assert dfs.geometry.n_sigma_layers == 4
    assert dfs.geometry.n_z_layers == 5
    assert dfs.n_timesteps == 3


def test_transect_geometry_properties(vslice: Dfsu2DV) -> None:
    g = vslice.geometry
    assert isinstance(g, GeometryFMVerticalProfile)
    assert len(g.top_elements) == 99
    assert len(g.bottom_elements) == 99
    assert len(g.relative_element_distance) == 441
    d1 = g.get_nearest_relative_distance((3.55e05, 6.145e06))
    assert d1 == pytest.approx(5462.3273)


def test_transect_geometry_properties_geo(vslice_geo: Dfsu2DV) -> None:
    g = vslice_geo.geometry
    ec = g.element_coordinates
    x, y = ec[:, 0], ec[:, 1]
    d0 = np.hypot(x - x[0], y - y[0])  # relative, in degrees
    assert d0.max() < 2

    d = g.relative_element_distance  # in meters and cummulative
    assert d.max() > 38000

    d1 = g.get_nearest_relative_distance((10.77, 55.62))
    assert d1 == pytest.approx(25673.318)


def test_transect_read(vslice: Dfsu2DV) -> None:
    ds = vslice.read()
    assert ds.geometry._type == DfsuFileType.DfsuVerticalProfileSigmaZ
    assert type(ds.geometry) is GeometryFMVerticalProfile
    assert ds.n_items == 2
    assert ds["Salinity"].name == "Salinity"
    assert ds.geometry.n_elements == 441


def test_transect_getitem_time(vslice: Dfsu2DV) -> None:
    da = vslice.read()[1]
    assert da[-1].time[0] == da.time[2]


def test_transect_getitem_element(vslice: Dfsu2DV) -> None:
    da = vslice.read()["Salinity"]
    idx = 5
    da2 = da[:, idx]
    assert type(da2.geometry) is GeometryPoint3D
    assert da2.geometry.x == da.geometry.element_coordinates[idx, 0]
    assert da2.geometry.y == da.geometry.element_coordinates[idx, 1]
    assert da2.geometry.z == da.geometry.element_coordinates[idx, 2]
    assert da2.shape == (3,)
    assert np.all(da2.to_numpy() == da.to_numpy()[:, idx])


def test_transect_isel(vslice: Dfsu2DV) -> None:
    da = vslice.read()["Salinity"]
    idx = 5
    da2 = da.isel(element=idx)
    assert type(da2.geometry) is GeometryPoint3D
    assert da2.geometry.x == da.geometry.element_coordinates[idx, 0]
    assert da2.geometry.y == da.geometry.element_coordinates[idx, 1]
    assert da2.geometry.z == da.geometry.element_coordinates[idx, 2]
    assert da2.shape == (3,)
    assert np.all(da2.to_numpy() == da.to_numpy()[:, idx])


def test_transect_isel_multiple(vslice_geo: Dfsu2DV) -> None:
    ds = vslice_geo.read()
    rd = ds.geometry.relative_element_distance
    idx = np.where(np.logical_and(10000 < rd, rd < 25000))[0]

    ds2 = ds.isel(element=idx)
    assert ds2.geometry.n_elements == 579
    assert type(ds2.geometry) is GeometryFMVerticalProfile
    rd2 = ds2.geometry.relative_element_distance
    assert rd2.max() < 15000


def test_transect_sel_time(vslice: Dfsu2DV) -> None:
    da = vslice.read()[0]
    idx = 1
    da2 = da.sel(time="1997-09-16 00:00")
    assert type(da2.geometry) is GeometryFMVerticalProfile
    assert np.all(da2.to_numpy() == da.to_numpy()[idx, :])
    assert da2.time[0] == da.time[1]
    assert da2.dims == ("element",)
    assert da2.shape == (441,)


def test_transect_sel_xyz(vslice_geo: Dfsu2DV) -> None:
    da = vslice_geo.read()["Temperature"]
    da2 = da.sel(x=10.8, y=55.6, z=-3)
    assert type(da2.geometry) is GeometryPoint3D
    assert da2.geometry.x == pytest.approx(10.802878)
    assert da2.geometry.y == pytest.approx(55.603096)
    assert da2.geometry.z == pytest.approx(-2.3)
    assert da2.n_timesteps == da.n_timesteps


def test_transect_sel_layers(vslice_geo: Dfsu2DV) -> None:
    ds = vslice_geo.read()
    ds2 = ds.sel(layers=range(-6, -1))
    assert type(ds2.geometry) is GeometryFMVerticalProfile


def test_transect_sel_xy(vslice_geo: Dfsu2DV) -> None:
    da = vslice_geo.read()["Temperature"]
    da2 = da.sel(x=10.8, y=55.6)
    assert type(da2.geometry) is GeometryFMVerticalColumn
    gx, gy, _ = da2.geometry.element_coordinates.mean(axis=0)
    assert gx == pytest.approx(10.802878)
    assert gy == pytest.approx(55.603096)
    assert da2.geometry.n_layers == 15
    assert da2.n_timesteps == da.n_timesteps


def test_transect_plot(vslice: Dfsu2DV) -> None:
    da = vslice.read()["Salinity"]
    da.plot()
    da.plot(cmin=0, cmax=1)
    da.plot(label="l", title="t", add_colorbar=False)
    da.plot(cmap="Blues", edge_color="0.9")

    # deprecated remove in 3.1
    vals = da.isel(time=0).to_numpy()

    with pytest.warns(FutureWarning):
        vslice.plot_vertical_profile(vals, label="l", figsize=(3, 3))

    plt.close("all")


def test_transect_plot_geometry(vslice: Dfsu2DV) -> None:
    g = vslice.geometry
    g.plot()
    g.plot.mesh()

    plt.close("all")


def test_write_roundtrip(vslice: Dfsu2DV, tmp_path: Path) -> None:
    ds = vslice.read()
    fp = tmp_path / "vslice_roundtrip.dfsu"
    ds.to_dfs(fp)

    ds2 = mikeio.read(fp)
    assert type(ds2.geometry) is GeometryFMVerticalProfile
    assert ds2.geometry.n_elements == ds.geometry.n_elements
    assert ds2.geometry.n_nodes == ds.geometry.n_nodes
    assert ds2.geometry.n_sigma_layers == ds.geometry.n_sigma_layers
    assert ds2.n_timesteps == ds.n_timesteps
    assert ds2.n_items == ds.n_items
    np.testing.assert_allclose(ds2[0].to_numpy(), ds[0].to_numpy(), atol=1e-5)


def test_create_from_scratch(tmp_path: Path) -> None:
    """Create a vertical profile dfsu from scratch and verify roundtrip."""
    n_points = 5
    depth = np.array([0.0, 10.0, 50.0, 100.0])
    n_depths = len(depth)
    n_layers = n_depths - 1
    time = pd.date_range("2024-01-01", periods=2, freq="D")

    lon = np.linspace(3.5, 5.5, n_points)
    lat = np.linspace(40.5, 41.5, n_points)

    # Build nodes: column-major, bottom-up
    node_x, node_y, node_z = [], [], []
    for pi in range(n_points):
        for di in range(n_depths - 1, -1, -1):
            node_x.append(lon[pi])
            node_y.append(lat[pi])
            node_z.append(-depth[di])
    node_coords = np.column_stack([node_x, node_y, node_z])

    # Build quadrilateral elements
    element_table = []
    nodes_per_col = n_depths
    for pi in range(n_points - 1):
        for li in range(n_layers):
            bl = pi * nodes_per_col + li
            tl = pi * nodes_per_col + li + 1
            br = (pi + 1) * nodes_per_col + li
            tr = (pi + 1) * nodes_per_col + li + 1
            element_table.append(np.array([bl, br, tr, tl]))

    codes = np.zeros(len(node_x), dtype=int)
    codes[:nodes_per_col] = 1
    codes[-nodes_per_col:] = 1

    geometry = GeometryFMVerticalProfile(
        node_coordinates=node_coords,
        element_table=element_table,
        codes=codes,
        projection="LONG/LAT",
        dfsu_type=DfsuFileType.DfsuVerticalProfileSigma,
        n_layers=n_layers,
        n_sigma=n_layers,
    )

    assert geometry.n_elements == (n_points - 1) * n_layers
    assert geometry.n_nodes == n_points * n_depths
    assert geometry.n_layers == n_layers
    assert geometry.n_sigma_layers == n_layers

    # Create data and write
    n_elements = geometry.n_elements
    rng = np.random.default_rng(42)
    data = rng.random((len(time), n_elements)).astype(np.float32)
    zn = np.tile(node_coords[:, 2], (len(time), 1)).astype(np.float32)

    da = mikeio.DataArray(
        data=data,
        time=time,
        geometry=geometry,
        item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature),
        zn=zn,
    )
    ds = mikeio.Dataset([da])

    fp = tmp_path / "created_vprofile.dfsu"
    ds.to_dfs(fp)

    # Read back and verify
    dfs = mikeio.open(fp)
    assert type(dfs) is Dfsu2DV
    assert dfs._type == DfsuFileType.DfsuVerticalProfileSigma

    ds2 = dfs.read()
    assert type(ds2.geometry) is GeometryFMVerticalProfile
    assert ds2.geometry.n_elements == n_elements
    assert ds2.geometry.n_layers == n_layers
    assert ds2.geometry.n_sigma_layers == n_layers
    assert ds2.n_timesteps == len(time)
    np.testing.assert_allclose(ds2[0].to_numpy(), data, atol=1e-5)
