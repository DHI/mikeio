import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mikeio
from mikeio import Mesh
from mikeio.spatial import (
    GeometryFM2D,
    GeometryFM3D,
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
    GeometryPoint3D,
)


def test_repr() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    text = repr(dfs)
    assert "number of z layers" in text


def test_read_simple_3d() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    ds = mikeio.read(filename)

    assert ds.to_numpy().shape[0] == 3
    assert len(ds.items) == 3

    assert ds.items[0].name != "Z coordinate"
    assert ds.items[2].name == "W velocity"


def test_read_simple_2dv() -> None:
    filename = "tests/testdata/basin_2dv.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read()

    assert ds.to_numpy().shape[0] == 3
    assert len(ds.items) == 3

    assert ds.items[0].name != "Z coordinate"
    assert ds.items[2].name == "W velocity"


def test_write_read_with_title(tmp_path: Path) -> None:
    tmpfile = tmp_path / "tmp_title.dfsu"
    dfs = mikeio.Dfsu3D("tests/testdata/oresund_sigma_z.dfsu")
    ds = dfs.read()
    ds.to_dfs(tmpfile)
    dfs_tmp = mikeio.Dfsu3D(tmpfile)
    assert dfs.title == dfs_tmp.title


def test_read_returns_correct_items_sigma_z() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read()

    assert len(ds) == 2
    assert ds.items[0].name == "Temperature"
    assert ds.items[1].name == "Salinity"


def test_read_top_layer() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read()  # all data in file
    dstop1 = ds.sel(layers="top")
    assert dstop1.geometry.max_nodes_per_element <= 4

    dstop2 = dfs.read(layers="top")
    assert dstop1.shape == dstop2.shape
    assert dstop1.dims == dstop2.dims
    assert isinstance(dstop1.geometry, GeometryFM2D)
    assert dstop1.geometry._type == dstop2.geometry._type
    assert np.all(dstop1.to_numpy() == dstop2.to_numpy())


def test_read_bottom_layer() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read()  # all data in file
    dsbot1 = ds.sel(layers="bottom")

    dsbot2 = dfs.read(layers="bottom")
    assert dsbot1.shape == dsbot2.shape
    assert dsbot1.dims == dsbot2.dims
    assert isinstance(dsbot1.geometry, GeometryFM2D)
    assert dsbot1.geometry._type == dsbot2.geometry._type
    assert np.all(dsbot1.to_numpy() == dsbot2.to_numpy())
    assert dsbot1.geometry.max_nodes_per_element <= 4


def test_read_single_step_bottom_layer() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read(time=-1)  # Last timestep
    ds.sel(layers="bottom")


def test_read_multiple_layers() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read()  # all data in file
    dstop1 = ds.sel(layers=[-3, -2, -1])

    dstop2 = dfs.read(layers=[-3, -2, -1])
    assert dstop1.shape == dstop2.shape
    assert dstop1.dims == dstop2.dims
    assert isinstance(dstop1.geometry, GeometryFM3D)
    assert dstop1.geometry._type == dstop2.geometry._type
    assert np.all(dstop1.to_numpy() == dstop2.to_numpy())
    assert dstop1.geometry.max_nodes_per_element >= 6


def test_read_dfsu3d_area() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    bbox = (350000, 6192000, 380000, 6198000)

    ds = dfs.read()  # all data in file
    assert ds.geometry.contains((350000, 6192000))

    dsa1 = ds.sel(area=bbox)
    assert not dsa1.geometry.contains((350000, 6192000))
    assert dsa1.geometry.n_layers > 1

    dsa2 = dfs.read(area=bbox)
    assert dsa1.shape == dsa2.shape
    assert dsa1.dims == dsa2.dims
    assert dsa1.geometry._type == dsa2.geometry._type
    assert np.all(dsa1.to_numpy() == dsa2.to_numpy())


def test_read_dfsu3d_area_single_element() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    bbox = (356000, 6144000, 357000, 6144500)
    ds = dfs.read(area=bbox)
    assert ds.geometry.geometry2d.n_elements == 1
    assert ds.geometry.n_elements == 4

    ds = dfs.read(area=bbox, layers="top")
    assert isinstance(ds.geometry, GeometryPoint3D)
    assert ds.dims == ("time",)

    ds = dfs.read(area=bbox, layers="top", time=0)
    assert isinstance(ds.geometry, GeometryPoint3D)
    assert len(ds.dims) == 0


def test_read_dfsu3d_area_empty_fails() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    bbox = (350000, 6192000, 350001, 6192001)
    with pytest.raises(ValueError, match="No elements in selection"):
        dfs.read(area=bbox)


def test_read_dfsu3d_column() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    (x, y) = (333934.1, 6158101.5)

    ds = dfs.read()  # all data in file
    dscol1 = ds.sel(x=x, y=y)
    assert isinstance(dscol1.geometry, GeometryFMVerticalColumn)
    assert dscol1.geometry.n_layers == 4
    assert dscol1.geometry.n_elements == 4
    assert dscol1.geometry.n_nodes == 5 * 3
    assert dscol1.z.nodes.shape == (ds.n_timesteps, 5 * 3)

    dscol2 = dfs.read(x=x, y=y)
    assert isinstance(dscol2.geometry, GeometryFMVerticalColumn)
    assert dscol1.shape == dscol2.shape
    assert dscol1.dims == dscol2.dims
    assert dscol1.geometry._type == dscol2.geometry._type
    assert np.all(dscol1.to_numpy() == dscol2.to_numpy())
    assert dscol2.z.nodes.shape == (ds.n_timesteps, 5 * 3)
    assert np.all(dscol1.z.nodes == dscol2.z.nodes)


def test_flip_column_upside_down() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    (x, y) = (333934.1, 6158101.5)

    ds = dfs.read()  # all data in file
    dscol = ds.sel(x=x, y=y)
    assert dscol.geometry.element_coordinates[0, 2] == pytest.approx(-7.0)
    assert dscol.isel(time=-1)["Temperature"].values[0] == pytest.approx(17.460058)

    idx = list(reversed(range(dscol.geometry.n_elements)))

    dscol_ud = dscol.isel(element=idx)

    assert dscol_ud.geometry.element_coordinates[-1, 2] == pytest.approx(-7.0)
    assert dscol_ud.isel(time=-1)["Temperature"].values[-1] == pytest.approx(17.460058)


def test_read_dfsu3d_column_save(tmp_path: Path) -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.geometry.n_sigma_layers == 4
    assert dfs.geometry.n_z_layers == 5

    (x, y) = (333934.1, 6158101.5)

    ds = dfs.read(x=x, y=y)  # all data in file
    assert isinstance(ds.geometry, GeometryFMVerticalColumn)
    assert ds.geometry.n_sigma_layers == 4
    assert ds.geometry.n_z_layers == 0
    fp = tmp_path / "new_column.dfsu"
    ds.to_dfs(fp)

    (x, y) = (347698.5188405, 6221233.34815)

    ds = dfs.read(x=x, y=y)  # all data in file
    assert isinstance(ds.geometry, GeometryFMVerticalColumn)
    assert ds.geometry.n_layers == 8
    assert ds.geometry.n_sigma_layers == 4
    assert ds.geometry.n_z_layers == 4
    fp = tmp_path / "new_column_2.dfsu"
    ds.to_dfs(fp)


def test_read_dfsu3d_columns_sigma_only() -> None:
    dfs = mikeio.Dfsu3D("tests/testdata/basin_3d.dfsu")
    dscol = dfs.read(x=500, y=50)
    assert isinstance(dscol.geometry, GeometryFMVerticalColumn)
    assert dscol.geometry.n_elements == 10
    assert dscol.n_items == dfs.n_items
    assert dscol["U velocity"].isel(time=-1)[-1].values == pytest.approx(0.363413)

    dscol2 = dfs.read().sel(x=500, y=50)
    assert dscol.shape == dscol2.shape


def test_read_dfsu3d_columns_sigma_only_save(tmp_path: Path) -> None:
    dfs = mikeio.Dfsu3D("tests/testdata/basin_3d.dfsu")
    assert dfs.geometry.n_sigma_layers == 10
    assert dfs.geometry.n_z_layers == 0
    dscol = dfs.read(x=500, y=50)
    assert dscol.geometry.n_sigma_layers == 10
    assert dscol.geometry.n_z_layers == 0
    fp = tmp_path / "new_column.dfsu"
    dscol.to_dfs(fp)


def test_read_dfsu3d_xyz() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    (x, y, z) = (333934.1, 6158101.5, -5)

    ds = dfs.read()  # all data in file
    dspt1 = ds.sel(x=x, y=y, z=z)
    assert isinstance(dspt1.geometry, GeometryPoint3D)
    assert dspt1.geometry.projection == ds.geometry.projection

    dspt2 = dfs.read(x=x, y=y, z=z)
    assert isinstance(dspt2.geometry, GeometryPoint3D)
    assert dspt1.shape == dspt2.shape
    assert dspt1.dims == dspt2.dims
    assert np.all(dspt1.to_numpy() == dspt2.to_numpy())

    dspt3 = dfs.read(time=-1, x=x, y=y, z=z)
    assert dspt3.dims == ()
    assert dspt3[0].values == dspt1[0].values[-1]
    # 20.531237


def test_read_dfsu3d_xyz_to_xarray() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    (x, y, z) = (333934.1, 6158101.5, -5)

    ds = dfs.read()  # all data in file
    dspt1 = ds.sel(x=x, y=y, z=z)

    xr_ds = dspt1.to_xarray()
    assert float(xr_ds.x) == pytest.approx(x)
    assert float(xr_ds.y) == pytest.approx(y)
    assert float(xr_ds.z) == pytest.approx(z)


def test_read_column_select_single_time_plot() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    x, y = 333934.1, 6158101.5

    dsp = dfs.read(x=x, y=y)
    sal_prof = dsp["Salinity"].isel(time=0)
    sal_prof.plot()
    sal_prof.plot.line()


def test_plot_column_selected_from_dataset() -> None:
    import matplotlib.pyplot as plt

    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    dsp = ds.sel(x=333934.1, y=6158101.5)
    assert isinstance(dsp.geometry, GeometryFMVerticalColumn)

    da = dsp["Temperature"]
    da.plot()
    da.plot(extrapolate=False, marker="o")

    plt.close("all")


def test_read_column_interp_time_and_select_time() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    x, y = 333934.1, 6158101.5

    dscol = dfs.read(x=x, y=y)
    dscol_t = dscol.isel(time=0)

    assert "time" not in dscol_t.dims

    dscol_et = dscol.sel(time=dfs.end_time)

    assert "time" not in dscol_et.dims

    ds_15m = dscol.interp_time(dt=900)

    da = ds_15m["Salinity"]

    salinity_it = da.isel(time=0)  # single time-step
    assert salinity_it.n_timesteps == 1

    salinity_st = da.sel(time="1997-09-15 23:00")  # single time-step
    assert salinity_st.n_timesteps == 1

    with pytest.raises(KeyError):
        # not in time
        da.sel(time="1997-09-15 00:00")


def test_number_of_nodes_and_elements_sigma_z() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    assert dfs.geometry.n_elements == 17118
    assert dfs.geometry.n_nodes == 12042


def test_read_and_select_single_element_dfsu_3d() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read()

    selds = ds.isel(idx=1739, axis=1)

    assert selds[0].shape == (3,)


def test_n_layers() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_layers == 10

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_layers == 9

    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_layers == 9


def test_n_sigma_layers() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_sigma_layers == 10

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_sigma_layers == 4

    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_sigma_layers == 4


def test_n_z_layers() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_z_layers == 0

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_z_layers == 5

    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert dfs.n_z_layers == 5


def test_boundary_codes() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.boundary_codes) == 1

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)

    assert len(dfs.geometry.boundary_codes) == 3


def test_boundary_polygons_does_not_warn() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    geometry = mikeio.Dfsu3D(filename).geometry
    assert isinstance(geometry, GeometryFM3D)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        polygons = geometry.boundary_polygons

    assert len(polygons.exteriors) > 0


def test_boundary_polylines_is_deprecated() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    geometry = mikeio.Dfsu3D(filename).geometry
    assert isinstance(geometry, GeometryFM3D)

    with pytest.warns(FutureWarning, match="boundary_polygons"):
        geometry.boundary_polylines


def test_top_elements() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.top_elements) == 174
    assert 39 in dfs.geometry.top_elements
    assert 0 not in dfs.geometry.top_elements
    assert (dfs.geometry.n_elements - 1) in dfs.geometry.top_elements

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.top_elements) == 3700
    assert 16 in dfs.geometry.top_elements
    assert (dfs.geometry.n_elements - 1) in dfs.geometry.top_elements

    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.top_elements) == 99
    assert 19 in dfs.geometry.top_elements
    assert (dfs.geometry.n_elements - 1) in dfs.geometry.top_elements


def test_top_elements_subset() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    g3d: GeometryFM3D = mikeio.Dfsu3D(filename).geometry  # type: ignore
    g2d = g3d.geometry2d

    area = (356000, 6144000, 359000, 6146000)
    idx2d = g2d.find_index(area=area)
    assert len(idx2d) == 6
    assert 3408 in idx2d

    idx3d = g3d.find_index(area=area)
    subg: GeometryFM3D = g3d.isel(idx3d)  # type: ignore

    assert len(subg.top_elements) == 6

    _ = mikeio.Dfsu3D(filename).read(elements=idx3d)


def test_bottom_elements() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.bottom_elements) == 174
    assert dfs.geometry.bottom_elements[3] == 30

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.bottom_elements) == 3700
    assert dfs.geometry.bottom_elements[3] == 13

    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.bottom_elements) == 99
    assert dfs.geometry.bottom_elements[3] == 15


def test_n_layers_per_column() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.n_layers_per_column) == 174
    assert dfs.geometry.n_layers_per_column[3] == 10

    filename = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.n_layers_per_column) == 3700
    assert dfs.geometry.n_layers_per_column[3] == 4
    assert max(dfs.geometry.n_layers_per_column) == dfs.geometry.n_layers

    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = mikeio.Dfsu3D(filename)
    assert len(dfs.geometry.n_layers_per_column) == 99
    assert dfs.geometry.n_layers_per_column[3] == 5


# def test_get_layer_elements() -> None:
#     filename = "tests/testdata/oresund_sigma_z.dfsu"
#     dfs = mikeio.Dfsu3D(filename)

#     elem_ids = dfs.get_layer_elements(-1)
#     assert np.all(elem_ids == dfs.top_elements)

#     elem_ids = dfs.get_layer_elements(-2)
#     assert elem_ids[5] == 23

#     elem_ids = dfs.get_layer_elements(0)
#     assert elem_ids[5] == 8638
#     assert len(elem_ids) == 10

#     elem_ids = dfs.get_layer_elements([0, 2])
#     assert len(elem_ids) == 197

#     with pytest.raises(Exception):
#         elem_ids = dfs.get_layer_elements(11)


def test_write_from_dfsu3D(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/basin_3d.dfsu"
    fp = tmp_path / "basin_3d.dfsu"
    dfs = mikeio.Dfsu3D(sourcefilename)

    ds = dfs.read(items=[0, 1])

    ds.to_dfs(fp)

    assert fp.exists()


def test_extract_top_layer_to_2d(tmp_path: Path) -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"

    dfs = mikeio.Dfsu3D(filename)

    ds = dfs.read(layers="top")

    fp = tmp_path / "toplayer.dfsu"
    ds.to_dfs(fp)

    newdfs = mikeio.Dfsu2DH(fp)
    assert isinstance(newdfs.geometry, GeometryFM2D)


def test_modify_values_in_layer(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    selected_layer = 6  # Zero-based indexing!
    layer_elem_ids = ds.geometry.get_layer_elements(selected_layer)

    # Set values
    ds["Salinity"][:, layer_elem_ids] = 35.0  # type: ignore

    fp = tmp_path / "oresund_modified.dfsu"

    ds.to_dfs(fp)

    ds_sel_layer = mikeio.read(fp, layers=selected_layer)
    assert np.all(np.isclose(ds_sel_layer["Salinity"].to_numpy(), 35.0))


def test_to_mesh_3d(tmp_path: Path) -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"

    dfs = mikeio.Dfsu3D(filename)
    assert isinstance(dfs.geometry, GeometryFM3D)
    fp = tmp_path / "oresund_from_dfs.mesh"
    dfs.geometry.to_mesh(fp)
    assert fp.exists()
    Mesh(fp)

    fp = tmp_path / "oresund_from_geometry.mesh"
    dfs.geometry.to_mesh(fp)
    assert fp.exists()
    Mesh(fp)


def test_extract_surface_elevation_from_3d() -> None:
    dfs = mikeio.Dfsu3D("tests/testdata/oresund_sigma_z.dfsu")
    n_top1 = len(dfs.geometry.top_elements)

    da = dfs.extract_surface_elevation_from_3d()

    assert da.geometry.n_elements == n_top1


def test_dataset_write_dfsu3d(tmp_path: Path) -> None:
    fp = tmp_path / "oresund_sigma_z.dfsu"
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", time=[0, 1])
    ds.to_dfs(fp)

    ds2 = mikeio.read(fp)
    assert ds2.n_timesteps == 2


def test_dataset_write_dfsu3d_max(tmp_path: Path) -> None:
    fp = tmp_path / "oresund_sigma_z.dfsu"
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    assert ds.z.nodes is not None
    ds_max = ds.max("time")
    assert ds_max.z.nodes is not None
    ds_max.to_dfs(fp)

    ds2 = mikeio.read(fp)
    assert ds2.n_timesteps == 1
    assert ds2.geometry.is_layered


def test_read_wildcard_items() -> None:
    dfs = mikeio.Dfsu3D("tests/testdata/oresund_sigma_z.dfsu")
    assert dfs.items[1].name == "Salinity"

    ds = dfs.read(items="Sal*")
    assert ds.items[0].name == "Salinity"
    assert ds.n_items == 1


def test_append_dfsu_3d(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", time=[0])
    assert ds.timestep == pytest.approx(10800)
    ds2 = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", time=[1])
    new_filename = tmp_path / "appended.dfsu"
    ds.to_dfs(new_filename)
    dfs = mikeio.Dfsu3D(new_filename)
    assert dfs.timestep == pytest.approx(10800)
    assert dfs.n_timesteps == 1
    dfs.append(ds2)
    assert dfs.n_timesteps == 2
    assert dfs.time[-1] == ds2.time[-1]

    # verify that the new file can be read
    ds3 = mikeio.read(new_filename)
    assert ds3.n_timesteps == 2
    assert ds3.time[-1] == ds2.time[-1]


def test_read_elements_3d() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", elements=[0, 10])
    assert ds.geometry.element_coordinates[0][0] == pytest.approx(354020.46382194717)
    assert ds["Salinity"].to_numpy()[0, 0] == pytest.approx(23.18906021118164)

    ds2 = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", elements=[10, 0])
    assert ds2.geometry.element_coordinates[1][0] == pytest.approx(354020.46382194717)
    assert ds2["Salinity"].to_numpy()[0, 1] == pytest.approx(23.18906021118164)


def test_write_3d_non_equidistant(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/basin_3d.dfsu"
    fp = tmp_path / "simple.dfsu"
    ds = mikeio.read(sourcefilename)

    # manipulate time
    ds.time = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-10"])

    assert not ds.is_equidistant

    ds.to_dfs(fp)

    ds2 = mikeio.read(fp)

    assert all(ds.time == ds2.time)
    assert not ds2.is_equidistant

    dfs = mikeio.Dfsu3D(fp)

    # it is not possible to get all time without reading the entire file
    with pytest.raises(NotImplementedError):
        dfs.time

    # but getting the end time is not that expensive
    assert dfs.end_time == pd.Timestamp("2000-01-10")


def test_isel_3d_single_time() -> None:
    ds = mikeio.Dfsu3D("tests/testdata/basin_3d.dfsu").read()
    ds1 = ds.isel(element=[0, 1])
    assert ds1.geometry.n_elements == 2

    ds2 = ds.isel(time=-1)
    assert "time" not in ds2.dims
    ds3 = ds2.isel(element=[0, 1])
    assert ds3.geometry.n_elements == 2


def test_z_accessor_nodes_matches_legacy_zn() -> None:
    """S01: da.z.nodes equals the legacy _zn on 3D layered DataArrays."""
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    da = ds[0]
    assert da.z.nodes.shape == (da.n_timesteps, da.geometry.n_nodes)
    assert da.z.nodes.dtype == np.float32 or da.z.nodes.dtype == np.float64
    assert np.isfinite(da.z.nodes).all()


def test_z_accessor_elements_is_node_mean_and_cached() -> None:
    """S02: da.z.elements equals the per-element node-mean and is cached."""
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    da = ds[0]
    ze = da.z.elements
    assert ze.shape == (da.n_timesteps, da.geometry.n_elements)

    zn = da.z.nodes
    expected = np.empty_like(ze)
    for j, nodes in enumerate(da.geometry.element_table):
        expected[:, j] = zn[:, np.asarray(nodes, dtype=int)].mean(axis=1)
    assert np.allclose(ze, expected)

    # Second access returns the same cached array (no recomputation)
    assert da.z.elements is ze


def test_z_accessor_on_vertical_column_slice() -> None:
    """S03: vertical-column slice DataArray retains the z accessor."""
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    dscol = ds.sel(x=333934.1, y=6158101.5)
    da = dscol[0]
    assert isinstance(da.geometry, GeometryFMVerticalColumn)
    assert da.z.nodes.shape[-1] == da.geometry.n_nodes
    assert da.z.nodes.dtype == np.float32 or da.z.nodes.dtype == np.float64
    assert np.isfinite(da.z.nodes).all()


def test_z_accessor_non_layered_raises_attribute_error() -> None:
    """S04: non-layered DataArray raises AttributeError naming the geometry."""
    ds = mikeio.read("tests/testdata/random.dfs2")
    da = ds[0]
    with pytest.raises(AttributeError, match="Grid2D"):
        _ = da.z.nodes
    with pytest.raises(AttributeError, match="has no z-coordinates"):
        _ = da.z.elements


def test_dataset_z_mirrors_first_dataarray() -> None:
    """S05: ds.z.nodes equals ds[0].z.nodes (shared array)."""
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    assert ds.z.nodes is ds[0].z.nodes
    assert np.array_equal(ds.z.elements, ds[0].z.elements)


def test_z_accessor_thickness_is_top_minus_bottom_node_mean() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    dz = da.z.thickness
    assert isinstance(dz, mikeio.DataArray)
    assert dz.name == "Layer thickness"
    assert dz.type == mikeio.EUMType.Layer_Thickness
    assert dz.unit == mikeio.EUMUnit.meter
    assert dz.time.equals(da.time)
    assert dz.geometry == da.geometry

    zn = da.z.nodes
    expected = np.empty(da.shape)
    for j, nodes in enumerate(da.geometry.element_table):
        half = len(nodes) // 2
        expected[:, j] = zn[:, nodes[half:]].mean(axis=1) - zn[:, nodes[:half]].mean(
            axis=1
        )
    assert np.allclose(dz.values, expected)
    assert (dz.values >= 0).all()
    assert da.z.thickness is dz


def test_z_accessor_thickness_sums_to_column_depth() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    g = da.geometry
    dz = da.z.thickness.values
    zn = da.z.nodes
    for column in g.e2_e3_table[:50]:
        bottom, top = g.element_table[column[0]], g.element_table[column[-1]]
        half = len(bottom) // 2
        depth = zn[:, top[half:]].mean(axis=1) - zn[:, bottom[:half]].mean(axis=1)
        assert np.allclose(dz[:, column].sum(axis=1), depth)


def test_z_accessor_volume_is_area_times_thickness() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    vol = da.z.volume
    assert isinstance(vol, mikeio.DataArray)
    assert vol.name == "Element volume"
    assert vol.type == mikeio.EUMType.Element_Volume
    assert vol.unit == mikeio.EUMUnit.meter_pow_3
    assert vol.time.equals(da.time)
    assert vol.geometry == da.geometry
    assert np.allclose(vol.values, da.geometry.element_areas * da.z.thickness.values)
    assert da.z.volume is vol


def test_z_accessor_volume_of_single_timestep() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    vol0 = da.isel(time=0).z.volume
    assert vol0.shape == (da.geometry.n_elements,)
    assert np.allclose(vol0.values, da.z.volume.isel(time=0).values)


def test_z_accessor_volume_follows_area_selection() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    bbox = (340000.0, 6150000.0, 360000.0, 6180000.0)
    assert np.allclose(
        da.sel(area=bbox).z.volume.values, da.z.volume.sel(area=bbox).values
    )


def test_z_accessor_volume_follows_vertical_column_selection() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    x, y = 333934.1, 6158101.5
    assert np.allclose(
        da.sel(x=x, y=y).z.volume.values, da.z.volume.sel(x=x, y=y).values
    )


def test_z_accessor_volume_layer_selection_aligns_with_data() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")["Salinity"]
    g = da.geometry
    top = da.sel(layers="top")
    vol_top = da.z.volume.sel(layers="top")
    assert vol_top.shape == top.shape
    assert np.allclose(vol_top.values, da.z.volume.values[:, g.top_elements])


def test_z_accessor_volume_after_layer_selection_points_to_3d() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    with pytest.raises(AttributeError, match="before selecting layers"):
        _ = da.sel(layers="top").z.volume


def test_z_accessor_volume_non_layered_raises() -> None:
    da = mikeio.read("tests/testdata/random.dfs2")[0]
    with pytest.raises(AttributeError, match="has no z-coordinates"):
        _ = da.z.volume
    with pytest.raises(AttributeError, match="has no z-coordinates"):
        _ = da.z.thickness


def test_dataset_z_volume_mirrors_first_dataarray() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    assert ds.z.volume is ds[0].z.volume


def test_volume_weighted_mean_and_total_mass() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")["Salinity"]
    vol = da.z.volume
    mass = (da.values * vol.values).sum(axis=-1)
    mean = da.average(axis="space", weights=vol)
    assert np.allclose(mean.values, mass / vol.values.sum(axis=-1))


def test_average_rejects_dataarray_weights_on_other_timesteps() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")["Salinity"]
    with pytest.raises(ValueError, match="same time and geometry"):
        da.average(axis="space", weights=da.z.volume.isel(time=0))


def test_average_rejects_dataarray_weights_on_other_elements() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")["Salinity"]
    with pytest.raises(ValueError, match="same time and geometry"):
        da.isel(time=0).average(
            axis="space", weights=da.z.volume.isel(time=0).sel(layers="top")
        )


def test_z_accessor_volume_roundtrips_to_dfsu(tmp_path: Path) -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")[0]
    fn = tmp_path / "volume.dfsu"
    da.z.volume.to_dfs(fn)
    back = mikeio.read(fn)["Element volume"]
    assert back.type == mikeio.EUMType.Element_Volume
    assert np.allclose(back.values, da.z.volume.values, rtol=1e-6)


def test_vertical_profile_has_no_element_areas() -> None:
    g = mikeio.Dfsu2DV("tests/testdata/oresund_vertical_slice.dfsu").geometry
    assert isinstance(g, GeometryFMVerticalProfile)
    assert not hasattr(g, "element_areas")


def test_dataset_volume_weighted_mean() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    mean = ds.average(axis="space", weights=ds.z.volume)
    for name in ["Temperature", "Salinity"]:
        expected = ds[name].average(axis="space", weights=ds.z.volume)
        assert np.allclose(mean[name].values, expected.values)


def test_dataset_average_rejects_dataarray_weights_on_other_timesteps() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    with pytest.raises(ValueError, match="same time and geometry"):
        ds.average(axis="space", weights=ds.z.volume.isel(time=0))


def test_sel_area_on_sigma_only_mesh() -> None:
    da = mikeio.read("tests/testdata/basin_3d.dfsu")[0]
    g = da.geometry
    bbox = (0.0, 0.0, 1000.0, 50.0)
    inside = da.sel(area=bbox)
    ec = g.element_coordinates
    expected = (ec[:, 0] <= 1000.0) & (ec[:, 1] <= 50.0)
    assert inside.geometry.n_elements == expected.sum()
    assert np.array_equal(inside.values, da.values[:, expected])


WQ_FILE = "tests/testdata/odense_rough_3d_wq.dfsu"
NITROGEN = "IN, Inorganic nitrogen, g N/m3"


def test_volume_integral_of_whole_domain() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    total = n.volume_integral()
    assert total.dims == ("time",)
    assert total.time.equals(n.time)
    assert total.name == f"{NITROGEN} (volume integral)"
    assert total.type == mikeio.EUMType.Mass
    assert total.unit == mikeio.EUMUnit.gram
    expected = (n.values * n.z.volume.values).sum(axis=-1)
    assert np.allclose(total.values, expected)
    assert total.values / 1e6 == pytest.approx([122.138, 151.088], abs=1e-3)


def test_volume_integral_of_area_selection() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    bbox = (212000.0, 6155000.0, 217000.0, 6160000.0)
    idx = n.geometry.find_index(area=bbox)
    expected = (n.values[:, idx] * n.z.volume.values[:, idx]).sum(axis=-1)
    assert np.allclose(n.sel(area=bbox).volume_integral().values, expected)


def test_volume_integral_of_single_layer() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    top = n.geometry.top_elements
    expected = (n.values[:, top] * n.z.volume.values[:, top]).sum(axis=-1)
    assert np.allclose(n.volume_integral(layers="top").values, expected)


def test_volume_integral_of_several_layers() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    by_keyword = n.volume_integral(layers=[-3, -2, -1])
    by_sel = n.sel(layers=[-3, -2, -1]).volume_integral()
    assert np.allclose(by_keyword.values, by_sel.values)
    per_layer = sum(n.volume_integral(layers=k).values for k in [-3, -2, -1])
    assert np.allclose(by_keyword.values, per_layer)


def test_volume_integral_layers_add_up_to_whole_domain() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    per_layer = sum(
        n.volume_integral(layers=k).values for k in range(n.geometry.n_layers)
    )
    assert np.allclose(per_layer, n.volume_integral().values)


def test_volume_integral_of_vertical_column() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    x, y = 221040.0, 6163422.0
    idx = n.geometry.find_index(x=x, y=y)
    expected = (n.values[:, idx] * n.z.volume.values[:, idx]).sum(axis=-1)
    assert np.allclose(n.sel(x=x, y=y).volume_integral().values, expected)


def test_volume_integral_of_single_timestep() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    total0 = n.isel(time=1).volume_integral()
    assert total0.values == pytest.approx(n.volume_integral().values[1])


def test_volume_integral_of_sigma_z_file() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")["Salinity"]
    expected = (da.values * da.z.volume.values).sum(axis=-1)
    assert np.allclose(da.volume_integral().values, expected)


def test_volume_integral_unit_without_mass_equivalent_is_undefined() -> None:
    da = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")["Salinity"]
    total = da.volume_integral()
    assert total.type == mikeio.EUMType.Undefined
    assert total.unit == mikeio.EUMUnit.undefined


@pytest.mark.parametrize(
    ("unit", "mass_unit"),
    [
        (mikeio.EUMUnit.mg_per_liter, mikeio.EUMUnit.gram),
        (mikeio.EUMUnit.gram_per_meter_pow_3, mikeio.EUMUnit.gram),
        (mikeio.EUMUnit.mu_g_per_liter, mikeio.EUMUnit.milligram),
        (mikeio.EUMUnit.mg_per_meter_pow_3, mikeio.EUMUnit.milligram),
        (mikeio.EUMUnit.mu_g_per_meter_pow_3, mikeio.EUMUnit.microgram),
        (mikeio.EUMUnit.gram_per_liter, mikeio.EUMUnit.kilogram),
        (mikeio.EUMUnit.kg_per_meter_pow_3, mikeio.EUMUnit.kilogram),
    ],
)
def test_volume_integral_mass_unit(
    unit: mikeio.EUMUnit, mass_unit: mikeio.EUMUnit
) -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    c = mikeio.DataArray(
        n.values,
        time=n.time,
        item=mikeio.ItemInfo("c", mikeio.EUMType.Concentration, unit),
        geometry=n.geometry,
        zn=n.z.nodes,
    )
    total = c.volume_integral()
    assert total.type == mikeio.EUMType.Mass
    assert total.unit == mass_unit
    assert np.allclose(total.values, n.volume_integral().values)


def test_volume_integral_on_selected_single_layer_points_to_layers_keyword() -> None:
    n = mikeio.read(WQ_FILE)[NITROGEN]
    with pytest.raises(ValueError, match=r"volume_integral\(layers="):
        n.sel(layers="top").volume_integral()


def test_volume_integral_on_2d_data_raises() -> None:
    da = mikeio.read("tests/testdata/HD2D.dfsu")["Surface elevation"]
    with pytest.raises(ValueError, match="layered"):
        da.volume_integral()


def test_dataset_volume_integral() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")
    totals = ds.volume_integral(layers="top")
    assert isinstance(totals, mikeio.Dataset)
    assert totals.names == [f"{name} (volume integral)" for name in ds.names]
    for name in ds.names:
        assert np.allclose(
            totals[f"{name} (volume integral)"].values,
            ds[name].volume_integral(layers="top").values,
        )
