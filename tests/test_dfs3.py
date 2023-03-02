import os
import pytest

import mikeio
from mikeio.spatial.geometry import GeometryUndefined
from mikeio.spatial.grid_geometry import Grid2D, Grid3D


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
    assert xr_ds.dims["time"] == 2

    ds_1d = ds.isel(z=0).isel(y=0)
    xr_ds_1d = ds_1d.to_xarray()
    assert xr_ds_1d.dims["time"] == 2


def test_dfs3_read():
    ds = mikeio.read("tests/testdata/Grid1.dfs3")
    assert ds.n_items == 2
    assert ds.n_timesteps == 30
    da = ds[0]
    assert da.shape == (30, 10, 10, 10)  # t  # z  # y  # x
    assert da.dims == ("time", "z", "y", "x")
    assert da.name == "Item 1"


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


def test_dfs3_read_write(tmpdir):
    ds = mikeio.read("tests/testdata/Grid1.dfs3")
    outfilename = os.path.join(tmpdir.dirname, "rw.dfs3")

    ds.to_dfs(outfilename)


def test_read_rotated_grid():
    dfs = mikeio.open("tests/testdata/dissolved_oxygen.dfs3")
    # North to Y orientation: 18.124689102173
    # Grid rotation: 17.0003657182497

    # assert dfs._orientation == pytest.approx(18.1246891)
    assert dfs.orientation == pytest.approx(17.0003657)
    assert dfs.geometry.orientation == pytest.approx(17.0003657)  # in own CRS


def test_dfs3_to_dfs(tmpdir):
    ds = mikeio.read("tests/testdata/dissolved_oxygen.dfs3")
    # ds = mikeio.read("tests/testdata/Grid1.dfs3")
    outfilename = os.path.join(tmpdir.dirname, "rw.dfs3")
    ds.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

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


def test_read_write_single_layer_as_dfs3(tmpdir):
    fn = "tests/testdata/single_layer.dfs3"
    ds1 = mikeio.read(fn, keepdims=True)
    assert isinstance(ds1.geometry, Grid3D)
    assert ds1.dims == ("time", "z", "y", "x")
    ds2 = mikeio.read(fn, layers=0, keepdims=True)
    assert ds2.dims == ("time", "z", "y", "x")
    assert isinstance(ds2.geometry, Grid3D)

    outfile = os.path.join(tmpdir, "single_layer.dfs3")

    ds2.to_dfs(outfile)


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
