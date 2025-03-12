from pathlib import Path
import pytest
import numpy as np

import mikeio
from mikeio.spatial import GeometryUndefined
from mikeio.spatial import Grid2D, Grid3D


def test_dfs3_repr():
    dfs = mikeio.open("tests/testdata/test_dfs3.dfs3")
    assert "<mikeio.Dfs3>" in repr(dfs)
    assert "geometry: Grid3D" in repr(dfs)


def test_dfs3_projection():
    dfs = mikeio.open("tests/testdata/test_dfs3.dfs3")
    assert dfs.projection_string == "LONG/LAT"
    assert dfs.dx == 0.25
    assert dfs.dy == 0.25
    assert dfs.dz == 1.0


def test_dfs3_geometry():
    dfs = mikeio.open("tests/testdata/test_dfs3.dfs3")
    assert isinstance(dfs.geometry, Grid3D)
    assert dfs.geometry.nx == 21
    assert dfs.geometry.ny == 17
    assert dfs.geometry.nz == 34


def test_dfs_to_xarray():
    ds = mikeio.read("tests/testdata/test_dfs3.dfs3")
    xr_ds = ds.to_xarray()
    assert xr_ds.sizes["time"] == 2

    ds_1d = ds.isel(z=0).isel(y=0)
    xr_ds_1d = ds_1d.to_xarray()
    assert xr_ds_1d.sizes["time"] == 2


def test_dfs3_read():
    ds = mikeio.read("tests/testdata/Grid1.dfs3")
    assert ds.n_items == 2
    assert ds.n_timesteps == 30
    da = ds[0]
    assert da.shape == (30, 10, 10, 10)  # t  # z  # y  # x
    assert da.dims == ("time", "z", "y", "x")
    assert da.name == "Item 1"
    assert da.to_numpy().dtype == np.float32


def test_dfs3_read_double_precision():
    ds = mikeio.read("tests/testdata/Grid1.dfs3", dtype=np.float64)
    assert ds[0].to_numpy().dtype == np.float64


def test_dfs3_read_time():
    fn = "tests/testdata/test_dfs3.dfs3"
    ds = mikeio.read(fn, time="2020-12-30 00:00")
    assert ds.n_timesteps == 1
    assert isinstance(ds.geometry, Grid3D)

    ds = mikeio.read(fn, time=-1)
    assert ds.n_timesteps == 1
    assert isinstance(ds.geometry, Grid3D)


def test_dfs3_read_1_layer():
    fn = "tests/testdata/test_dfs3.dfs3"
    ds = mikeio.read(fn, layers=-1)
    assert ds.shape == (2, 17, 21)
    assert isinstance(ds.geometry, Grid2D)

    ds = mikeio.read(fn, layers="top")
    assert ds.shape == (2, 17, 21)
    assert isinstance(ds.geometry, Grid2D)

    ds = mikeio.read(fn, layers=[0])
    assert ds.shape == (2, 17, 21)
    assert isinstance(ds.geometry, Grid2D)


def test_dfs3_read_multiple_layers():
    fn = "tests/testdata/test_dfs3.dfs3"
    ds = mikeio.read(fn, layers=(0, 1, 2, 3))
    assert ds.geometry.nz == 4
    assert isinstance(ds.geometry, Grid3D)

    with pytest.warns(UserWarning):
        ds = mikeio.read(fn, layers=[1, 5, -3])
    assert isinstance(ds.geometry, GeometryUndefined)
    assert ds.shape == (2, 3, 17, 21)


def test_read_rotated_grid():
    dfs = mikeio.open("tests/testdata/dissolved_oxygen.dfs3")
    # North to Y orientation: 18.124689102173
    # Grid rotation: 17.0003657182497

    # assert dfs._orientation == pytest.approx(18.1246891)
    assert dfs.orientation == pytest.approx(17.0003657)
    assert dfs.geometry.orientation == pytest.approx(17.0003657)  # in own CRS


def test_dfs3_to_dfs(tmp_path):
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    fp = tmp_path / "test.dfs3"
    ds.to_dfs(fp)

    dsnew = mikeio.read(fp)

    assert ds.n_items == dsnew.n_items
    assert ds.geometry == dsnew.geometry


def test_read_top_layer():
    dsall = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3", layers="top")
    assert "z" not in ds.dims
    assert isinstance(ds.geometry, Grid2D)

    # TODO: not yet implemented
    # dssel = dsall.sel(layers="top")
    # assert dssel.geometry == ds.geometry

    dssel = dsall.isel(z=-1)
    assert dssel.geometry == ds.geometry
    dsdiff = dssel - ds
    assert dsdiff.nanmax(axis=None).to_numpy()[0] == 0.0
    assert dsdiff.nanmin(axis=None).to_numpy()[0] == 0.0


def test_read_bottom_layer():
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3", layers="bottom")
    assert "z" not in ds.dims
    assert isinstance(ds.geometry, Grid2D)
    assert pytest.approx(ds[0].to_numpy()[0, 58, 52]) == 0.05738005042076111


def test_sel_bottom_layer():
    dsall = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    with pytest.raises(NotImplementedError) as excinfo:
        dsall.sel(layers="bottom")
    assert "mikeio.read" in str(excinfo.value)
    # assert "z" not in ds.dims
    # assert isinstance(ds.geometry, Grid2D)
    # assert pytest.approx(ds[0].to_numpy()[0, 58, 52]) == 0.05738005042076111


def test_read_single_layer_dfs3():
    fn = "tests/testdata/single_layer.dfs3"
    ds = mikeio.read(fn, keepdims=True)
    assert isinstance(ds.geometry, Grid3D)
    assert ds.dims == ("time", "z", "y", "x")

    ds = mikeio.read(fn, keepdims=False)
    assert isinstance(ds.geometry, Grid2D)
    assert ds.dims == ("time", "y", "x")


def test_read_single_timestep_dfs3():
    fn = "tests/testdata/single_timestep.dfs3"
    ds = mikeio.read(fn, keepdims=True)
    assert ds.dims == ("time", "z", "y", "x")
    assert ds.shape == (1, 5, 17, 21)

    ds = mikeio.read(fn, time=0, keepdims=False)
    assert ds.dims == ("z", "y", "x")
    assert ds.shape == (5, 17, 21)

    ds = mikeio.read(fn, time=0)
    assert ds.dims == ("z", "y", "x")
    assert ds.shape == (5, 17, 21)


def test_read_write_single_layer_as_dfs3(tmp_path):
    fn = "tests/testdata/single_layer.dfs3"
    ds1 = mikeio.read(fn, keepdims=True)
    assert isinstance(ds1.geometry, Grid3D)
    assert ds1.dims == ("time", "z", "y", "x")
    ds2 = mikeio.read(fn, layers=0, keepdims=True)
    assert ds2.dims == ("time", "z", "y", "x")
    assert isinstance(ds2.geometry, Grid3D)

    fp = tmp_path / "single_layer.dfs3"

    ds2.to_dfs(fp)


def test_MIKE_SHE_dfs3_output():
    ds = mikeio.read("tests/testdata/Karup_MIKE_SHE_head_output.dfs3")
    assert ds.n_timesteps == 6
    assert ds.n_items == 1
    g = ds.geometry
    assert g.x[0] == 494329.0
    assert g.y[0] == pytest.approx(6220250.0)
    assert g.origin == pytest.approx((494329.0, 6220250.0))

    ds2 = ds.isel(x=range(30, 45))
    g2 = ds2.geometry
    assert g2.x[0] == g.x[0] + 30 * g.dx
    assert g2.y[0] == g.y[0]  # + 35 * g.dy
    assert g2.origin == pytest.approx((g2.x[0], g2.y[0]))


def test_local_coordinates_read_single_layer_dfs3():
    fn = "tests/testdata/local_coordinates.dfs3"

    ds = mikeio.read(fn)
    assert ds.geometry.x[0] == pytest.approx(0.25)

    ds1 = mikeio.read(fn, layers=1)
    assert ds1.geometry.x[0] == pytest.approx(0.25)


def test_local_coordinates_read_subset_layer_dfs3():
    fn = "tests/testdata/local_coordinates.dfs3"

    ds = mikeio.read(fn, layers=[0, 1])
    assert ds.geometry.x[0] == pytest.approx(0.25)


def test_write_read_local_coordinates(tmp_path: Path) -> None:
    da = mikeio.DataArray(
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        geometry=mikeio.Grid3D(
            nx=3, ny=2, nz=2, dx=0.5, dy=1, dz=1, projection="NON-UTM"
        ),
    )
    fp = tmp_path / "local_coordinates.dfs3"
    da.to_dfs(fp)

    ds = mikeio.read(fp)
    assert da.geometry == ds.geometry


def test_to_xarray() -> None:
    # data is not important here
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    # geographic coordinates
    dag = mikeio.DataArray(
        data=data,
        geometry=mikeio.Grid3D(
            nx=3, ny=2, nz=2, dy=0.5, dz=1, dx=0.5, projection="LONG/LAT"
        ),
    )
    assert dag.geometry.x[0] == pytest.approx(0.0)
    assert dag.geometry.y[0] == pytest.approx(0.0)
    xr_dag = dag.to_xarray()
    assert xr_dag.x[0] == pytest.approx(0.0)
    assert xr_dag.y[0] == pytest.approx(0.0)

    # local coordinates
    da = mikeio.DataArray(
        data=data,
        geometry=mikeio.Grid3D(
            nx=3, ny=2, nz=2, dy=0.5, dz=1, dx=0.5, projection="NON-UTM"
        ),
    )
    # local coordinates (=NON-UTM) have a different convention, geometry.x still refers to element centers
    assert da.geometry.x[0] == pytest.approx(0.25)
    assert da.geometry.y[0] == pytest.approx(0.25)

    xr_da = da.to_xarray()
    assert xr_da.x[0] == pytest.approx(0.25)
    assert xr_da.y[0] == pytest.approx(0.25)


def test_append_dfs3(tmp_path):
    fn = "tests/testdata/Karup_MIKE_SHE_head_output.dfs3"
    ds = mikeio.read(fn, time=[0, 1])
    new_fp = tmp_path / "test_append.dfs3"
    ds.to_dfs(new_fp)

    ds2 = mikeio.read(fn, time=[2, 3])

    dfs = mikeio.open(new_fp)

    dfs.append(ds2)
