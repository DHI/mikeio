import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from mikeio import Dataset, Dfs0, Dfsu, Mesh
from mikeio.custom_exceptions import InvalidGeometry
from mikeio.eum import ItemInfo
from pytest import approx


def test_repr():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    text = repr(dfs)
    assert "Dfsu2D" in text

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    text = repr(dfs)
    assert "number of z layers" in text


def test_read_all_items_returns_all_items_and_names():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read()

    assert dfs.n_items == 4

    ds_text = repr(ds)
    dfs_text = repr(dfs)

    assert len(ds) == 4

    # A filename can be a string or a Path object
    filepath = Path(filename)

    dfs = Dfsu(filepath)

    assert isinstance(filepath, Path)
    assert dfs.n_items == 4


def test_read_item_0():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    assert dfs.n_items == 4

    ds = dfs.read(1)

    assert len(ds) == 1


def test_read_single_precision():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename, dtype=np.float32)

    ds = dfs.read(1)

    assert len(ds) == 1
    assert ds[0].dtype == np.float32


def test_read_int_not_accepted():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    with pytest.raises(Exception):
        dfs = Dfsu(filename, dtype=np.int32)


def test_read_timestep_1():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(time_steps=1)

    assert len(ds.time) == 1


def test_read_simple_3d():
    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read()

    assert len(ds.data) == 3
    assert len(ds.iteminfos) == 3

    assert ds.iteminfos[0].name != "Z coordinate"
    assert ds.iteminfos[2].name == "W velocity"


def test_read_simple_2dv():
    filename = os.path.join("tests", "testdata", "basin_2dv.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read()

    assert len(ds.data) == 3
    assert len(ds.iteminfos) == 3

    assert ds.iteminfos[0].name != "Z coordinate"
    assert ds.iteminfos[2].name == "W velocity"


def test_read_single_item_returns_single_item():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(items=[3])

    assert len(ds.iteminfos) == 1


def test_read_single_item_scalar_index():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read([3])

    assert len(ds) == 1


def test_read_returns_array_time_dimension_first():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read([3])

    assert ds.data[0].shape == (9, 884)


def test_read_selected_item_returns_correct_items():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read([0, 3])

    assert len(ds) == 2
    assert ds.iteminfos[0].name == "Surface elevation"
    assert ds.iteminfos[1].name == "Current speed"


def test_read_selected_item_names_returns_correct_items():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(["Surface elevation", "Current speed"])

    assert len(ds) == 2
    assert ds.iteminfos[0].name == "Surface elevation"
    assert ds.iteminfos[1].name == "Current speed"


def test_read_returns_correct_items_sigma_z():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read()

    assert len(ds) == 2
    # assert ds.iteminfos[0].name == "Z coordinate"
    assert ds.iteminfos[0].name == "Temperature"
    assert ds.iteminfos[1].name == "Salinity"


def test_read_all_time_steps():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds.time) == 9
    assert ds.data[0].shape[0] == 9


def test_read_item_range():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(items=range(1, 3))  # [1,2]

    assert ds.n_items == 2
    assert ds.iteminfos[0].name == "U velocity"


def test_read_all_time_steps_without_progressbar():

    Dfsu.show_progress = True

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")

    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds.time) == 9
    assert ds.data[0].shape[0] == 9


def test_read_single_time_step():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3], time_steps=[1])

    assert len(ds.time) == 1
    assert ds.data[0].shape[0] == 1


def test_read_single_time_step_scalar():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3], time_steps=1)

    assert len(ds.time) == 1
    assert ds.data[0].shape[0] == 1


def test_read_single_time_step_outside_bounds_fails():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    with pytest.raises(Exception):

        dfs.read(items=[0, 3], time_steps=[100])


def test_number_of_time_steps():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    assert dfs.n_timesteps == 9


def test_number_of_nodes_and_elements_sigma_z():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)

    assert dfs.n_elements == 17118
    assert dfs.n_nodes == 12042


def test_get_node_coords():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    nc = dfs.node_coordinates
    assert nc[0, 0] == 607031.4886285994

    nc = dfs.get_node_coords(code=1)
    assert len(nc) > 0


def test_element_coordinates():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ec = dfs.element_coordinates
    assert ec[1, 1] == pytest.approx(6906790.5928664245)


def test_calc_element_coordinates_3d():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)

    # extract dynamic z values for profile
    elem_ids = dfs.find_nearest_profile_elements(333934.1, 6158101.5)
    ds = dfs.read(items=0, elements=elem_ids, time_steps=0)
    zn_dyn = ds[0]._zn  # TODO
    ec = dfs.calc_element_coordinates(elements=elem_ids, zn=zn_dyn)

    assert ec[0, 2] == pytest.approx(-6.981768845)


def test_element_coords_is_inside_nodes():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    nc = dfs.node_coordinates
    ec = dfs.element_coordinates
    nc_min = nc.min(axis=0)
    nc_max = nc.max(axis=0)
    ec_max = ec.max(axis=0)
    ec_min = ec.min(axis=0)

    assert ec_max[0] < nc_max[0]
    assert ec_max[1] < nc_max[1]
    assert ec_min[0] > nc_min[0]
    assert ec_min[1] > nc_min[0]


def test_contains():
    filename = os.path.join("tests", "testdata", "wind_north_sea.dfsu")
    dfs = Dfsu(filename)

    pts = [[4, 54], [0, 50]]
    inside = dfs.contains(pts)
    assert inside[0] == True
    assert inside[1] == False


def test_get_overset_grid():
    filename = os.path.join("tests", "testdata", "FakeLake.dfsu")
    dfs = Dfsu(filename)

    g = dfs.get_overset_grid()
    assert g.nx == 21
    assert g.ny == 10

    g = dfs.get_overset_grid(dx=0.2)
    assert g.dx == 0.2
    assert g.dy == 0.2

    g = dfs.get_overset_grid(dx=(0.4, 0.2))
    assert g.dx == 0.4
    assert g.dy == 0.2

    g = dfs.get_overset_grid(shape=(5, 4))
    assert g.nx == 5
    assert g.ny == 4

    g = dfs.get_overset_grid(shape=(None, 5))
    assert g.nx == 11
    assert g.ny == 5


def test_find_nearest_element_2d():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    elem_id = dfs.find_nearest_elements(606200, 6905480)
    assert elem_id == 317


def test_find_nearest_element_2d_and_distance():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    (elem_id, dist) = dfs.find_nearest_elements(606200, 6905480, return_distances=True)
    assert elem_id == 317

    assert dist > 0.0


def test_dfsu_to_dfs0_via_dataframe(tmpdir):
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert dfs.start_time.year == 1985

    elem_id = dfs.find_nearest_elements(606200, 6905480)

    ds = dfs.read(elements=[elem_id])
    df = ds.to_dataframe()

    outfilename = os.path.join(tmpdir, "out.dfs0")
    df.to_dfs0(outfilename)

    dfs0 = Dfs0(outfilename)
    newds = dfs0.read()

    assert newds[0].name == ds[0].name
    assert ds.time[0] == newds.time[0]
    assert ds.time[-1] == newds.time[-1]


def test_dfsu_to_dfs0(tmpdir):
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert dfs.start_time.year == 1985

    elem_id = dfs.find_nearest_elements(606200, 6905480)

    ds = dfs.read(elements=[elem_id])
    dss = ds.squeeze()

    outfilename = os.path.join(tmpdir, "out.dfs0")

    dfs0 = Dfs0()
    dfs0.write(outfilename, dss)

    dfs0 = Dfs0(outfilename)
    newds = dfs0.read()

    assert newds.iteminfos[0].name == ds.iteminfos[0].name
    assert ds.time[0] == newds.time[0]
    assert ds.time[-1] == newds.time[-1]


def test_find_nearest_elements_2d_array():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    elem_ids = dfs.find_nearest_elements(x=[606200, 606200], y=[6905480, 6905480])
    assert len(elem_ids) == 2
    assert elem_ids[0] == 317
    assert elem_ids[1] == 317


def test_find_nearest_elements_3d():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)

    elem_id = dfs.find_nearest_elements(333934, 6158101)
    assert elem_id == 5323
    assert elem_id in dfs.top_elements

    elem_id = dfs.find_nearest_elements(333934, 6158101, layer=7)
    assert elem_id == 5322

    elem_id = dfs.find_nearest_elements(333934, 6158101, -7)
    assert elem_id == 5320


def find_nearest_profile_elements():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    elem_ids = dfs.find_nearest_profile_elements(333934, 6158101)

    assert elem_ids[0] == 5320
    assert elem_ids[-1] == 5323


def test_read_and_select_single_element():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read()

    assert ds.data[0].shape == (9, 884)

    idx = dfs.find_nearest_elements(606200, 6905480)

    selds = ds.isel(idx=idx, axis=1)

    assert selds.data[0].shape == (9,)


def test_read_and_select_single_element_dfsu_3d():

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)

    ds = dfs.read()

    selds = ds.isel(idx=1739, axis=1)

    assert selds.data[0].shape == (3,)


def test_is_2d():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    assert dfs.is_2d

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)

    assert not dfs.is_2d


def test_n_layers():

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_layers == 10

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_layers == 9

    filename = os.path.join("tests", "testdata", "oresund_vertical_slice.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_layers == 9

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "n_layers")


def test_n_sigma_layers():

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_sigma_layers == 10

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_sigma_layers == 4

    filename = os.path.join("tests", "testdata", "oresund_vertical_slice.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_sigma_layers == 4

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "n_sigma_layers")


def test_n_z_layers():

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_z_layers == 0

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_z_layers == 5

    filename = os.path.join("tests", "testdata", "oresund_vertical_slice.dfsu")
    dfs = Dfsu(filename)
    assert dfs.n_z_layers == 5

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "n_z_layers")


def test_boundary_codes():

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.boundary_codes) == 1

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)

    assert len(dfs.boundary_codes) == 3


def test_top_elements():
    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.top_elements) == 174
    assert dfs.top_elements[3] == 39

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.top_elements) == 3700
    assert dfs.top_elements[3] == 16

    filename = os.path.join("tests", "testdata", "oresund_vertical_slice.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.top_elements) == 99
    assert dfs.top_elements[3] == 19

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "top_elements")


def test_bottom_elements():
    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.bottom_elements) == 174
    assert dfs.bottom_elements[3] == 30

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.bottom_elements) == 3700
    assert dfs.bottom_elements[3] == 13

    filename = os.path.join("tests", "testdata", "oresund_vertical_slice.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.bottom_elements) == 99
    assert dfs.bottom_elements[3] == 15

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "bottom_elements")


def test_n_layers_per_column():
    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.n_layers_per_column) == 174
    assert dfs.n_layers_per_column[3] == 10

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.n_layers_per_column) == 3700
    assert dfs.n_layers_per_column[3] == 4
    assert max(dfs.n_layers_per_column) == dfs.n_layers

    filename = os.path.join("tests", "testdata", "oresund_vertical_slice.dfsu")
    dfs = Dfsu(filename)
    assert len(dfs.n_layers_per_column) == 99
    assert dfs.n_layers_per_column[3] == 5

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "n_layers_per_column")


def test_get_layer_elements():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)

    elem_ids = dfs.get_layer_elements(-1)
    assert np.all(elem_ids == dfs.top_elements)

    elem_ids = dfs.get_layer_elements(-2)
    assert elem_ids[5] == 23

    elem_ids = dfs.get_layer_elements(0)
    assert elem_ids[5] == 8638
    assert len(elem_ids) == 10

    elem_ids = dfs.get_layer_elements([0, 2])
    assert len(elem_ids) == 197

    with pytest.raises(Exception):
        elem_ids = dfs.get_layer_elements(11)


def test_find_nearest_profile_elements():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    elem_ids = dfs.find_nearest_profile_elements(358337, 6196090)
    assert len(elem_ids) == 8
    assert elem_ids[-1] == 3042


def test_is_geo_UTM():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    assert dfs.is_geo is False


def test_is_geo_LONGLAT():
    filename = os.path.join("tests", "testdata", "wind_north_sea.dfsu")
    dfs = Dfsu(filename)
    assert dfs.is_geo is True


def test_is_local_coordinates():
    filename = os.path.join("tests", "testdata", "wind_north_sea.dfsu")
    dfs = Dfsu(filename)
    assert dfs.is_local_coordinates is False


def test_get_element_area_UTM():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    areas = dfs.get_element_area()
    assert areas[0] == 4949.102548750438


def test_get_element_area_3D():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    areas = dfs.get_element_area()
    assert areas[0] == 350186.43530453625


def test_get_element_area_LONGLAT():
    filename = os.path.join("tests", "testdata", "wind_north_sea.dfsu")
    dfs = Dfsu(filename)

    areas = dfs.get_element_area()
    assert areas[0] == 139524218.81411952


def test_get_element_area_tri_quad():
    filename = os.path.join("tests", "testdata", "FakeLake.dfsu")
    dfs = Dfsu(filename)

    areas = dfs.get_element_area()
    assert areas[0] == 0.0006875642143608321


def test_write(tmpdir):

    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)

    n_elements = msh.n_elements
    d = np.zeros((1, n_elements))
    data = []
    data.append(d)

    ds = Dataset(
        data,
        time=[datetime(2000, 1, 1)],
        items=[ItemInfo("Zeros")],
        geometry=msh.geometry,
    )

    dfs = Dfsu(meshfilename)

    dfs.write(outfilename, ds)

    assert os.path.exists(outfilename)


def test_write_from_dfsu(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read([0, 1])

    dfs.write(outfilename, ds)

    assert dfs.start_time.hour == 7

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_incremental_write_from_dfsu(tmpdir):
    "Useful for writing datasets with many timesteps to avoid problems with out of memory"

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time_steps=[0])

    dfs.write(outfilename, ds, keep_open=True)

    for i in range(1, nt):
        ds = dfs.read(time_steps=[i])
        dfs.append(ds)

    dfs.close()

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_incremental_write_from_dfsu_context_manager(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time_steps=[0])

    with dfs.write(outfilename, ds, keep_open=True) as f:
        for i in range(1, nt):
            ds = dfs.read(time_steps=[i])
            f.append(ds)

        # dfs.close() # should be called automagically by context manager

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_write_big_file(tmpdir):

    outfilename = os.path.join(tmpdir.dirname, "big.dfsu")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)

    n_elements = msh.n_elements

    dfs = Dfsu(meshfilename)

    nt = 1000

    n_items = 10

    items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]

    # with dfs.write(outfilename, [], items=items, keep_open=True) as f:
    with dfs.write_header(
        outfilename, start_time=datetime(2000, 1, 1), dt=3600, items=items
    ) as f:
        for i in range(nt):
            data = []
            for i in range(n_items):
                d = np.random.random((1, n_elements))
                data.append(d)
            f.append(data)

    dfsu = Dfsu(outfilename)

    assert dfsu.n_items == n_items
    assert dfsu.n_timesteps == nt
    assert dfsu.start_time.year == 2000


def test_write_from_dfsu_2_time_steps(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read(time_steps=[0, 1])

    assert ds.is_equidistant  # Data with two time steps is per definition equidistant

    dfs.write(outfilename, ds)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time != newdfs.end_time


def test_write_from_dfsu_default_items_numbered(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read([0, 1])

    dfs.write(outfilename, ds.data)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)
    assert newdfs.iteminfos[0].name == "Item 1"


def test_write_from_dfsu3D(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple3D.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read([0, 1])

    dfs.write(outfilename, ds)

    assert os.path.exists(outfilename)


# def test_write_from_dfsu3D_withou_z_coords_fails(tmpdir):

#     sourcefilename = os.path.join("tests", "testdata", "basin_3d.dfsu")
#     outfilename = os.path.join(tmpdir.dirname, "simple3D_fail.dfsu")
#     dfs = Dfsu(sourcefilename)

#     ds = dfs.read([1, 2])
#     with pytest.raises(Exception) as excinfo:
#         dfs.write(outfilename, ds)

#     assert "coord" in str(excinfo.value)
#     assert not os.path.exists(outfilename)


def test_write_invalid_data_closes_and_deletes_file(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfsu")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)

    n_elements = msh.n_elements
    d = np.zeros((1, n_elements - 1))

    assert d.shape[1] != n_elements
    data = []
    data.append(d)

    items = [ItemInfo("Bad data")]

    dfs = Dfsu(meshfilename)

    dfs.write(filename, data, items=items)

    assert not os.path.exists(filename)


def test_write_non_equidistant_is_not_possible(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read(time_steps=[0, 1, 3])

    with pytest.raises(Exception):
        dfs.write(outfilename, ds)

    assert not os.path.exists(outfilename)


def test_temporal_resample_by_reading_selected_timesteps(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time_steps=list(range(0, nt, 2)))
    dfs.write(outfilename, ds)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)

    assert pytest.approx(dfs.timestep) == newdfs.timestep / 2


def test_read_temporal_subset():

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(sourcefilename)

    assert dfs.n_timesteps == 9

    ds = dfs.read(time_steps=slice("1985-08-06 00:00", "1985-08-06 12:00"))

    assert len(ds.time) == 3

    # Specify start
    ds = dfs.read(time_steps=slice("1985-08-06 12:00", None))

    assert len(ds.time) == 7

    # Specify end
    ds = dfs.read(time_steps=slice(None, "1985-08-06 12:00"))

    assert len(ds.time) == 3


def test_read_temporal_subset_string():

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(sourcefilename)

    assert dfs.n_timesteps == 9

    # start,end
    ds = dfs.read(time_steps="1985-08-06 00:00,1985-08-06 12:00")
    assert len(ds.time) == 3

    # start,
    ds = dfs.read(time_steps="1985-08-06 12:00,")
    assert len(ds.time) == 7

    # ,end
    ds = dfs.read(time_steps=",1985-08-06 11:30")
    assert len(ds.time) == 2

    # start=end
    ds = dfs.read(time_steps="1985-08-06 12:00")
    assert len(ds.time) == 1


def test_write_temporal_subset(tmpdir):

    sourcefilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    assert dfs.n_timesteps == 9

    ds = dfs.read()  # TODO read temporal subset with slice e.g. "1985-08-06 12:00":
    selds = ds["1985-08-06 12:00":]
    dfs.write(outfilename, selds)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)

    assert newdfs.start_time.hour == 12
    assert newdfs.n_timesteps == 7


def test_extract_top_layer_to_2d(tmpdir):
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")

    dfs = Dfsu(filename)
    top_ids = dfs.top_elements

    ds = dfs.read(elements=top_ids)

    outfilename = os.path.join(tmpdir, "toplayer.dfsu")
    dfs.write(outfilename, ds, elements=top_ids)

    newdfs = Dfsu(outfilename)
    assert os.path.exists(outfilename)

    assert newdfs.is_2d


def test_geometry_2d():

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")

    dfs = Dfsu(filename)

    geom = dfs.to_2d_geometry()

    assert geom.is_2d


# def test_geometry_2d_2dfile():

#     dfs = Dfsu("tests/testdata/HD2D.dfsu")

#     assert dfs.is_2d
#     geom = dfs.to_2d_geometry()  # No op

#     assert geom.is_2d


# def test_get_layers_2d_error():

#     dfs = Dfsu("tests/testdata/HD2D.dfsu")
#     assert dfs.is_2d

#     with pytest.raises(InvalidGeometry):
#         dfs.get_layer_elements(-1)

#     with pytest.raises(InvalidGeometry):
#         dfs.layer_ids

#     with pytest.raises(InvalidGeometry):
#         dfs.elem2d_ids

#     with pytest.raises(InvalidGeometry):
#         dfs.find_nearest_profile_elements(x=0, y=0)


def test_to_mesh_3d(tmpdir):

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")

    dfs = Dfsu(filename)

    outfilename = os.path.join(tmpdir, "oresund.mesh")

    dfs.to_mesh(outfilename)

    assert os.path.exists(outfilename)

    mesh = Mesh(outfilename)

    assert True


def test_to_mesh_2d(tmpdir):
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)

    outfilename = os.path.join(tmpdir, "hd2d.mesh")

    dfs.to_mesh(outfilename)

    assert os.path.exists(outfilename)

    mesh = Mesh(outfilename)

    assert True


def test_elements_to_geometry():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    prof_ids = dfs.find_nearest_profile_elements(350000, 6150000)
    geom = dfs.elements_to_geometry(prof_ids)

    text = repr(geom)

    assert geom.n_layers == 5
    assert "nodes" in text

    elements = dfs.get_layer_elements(layer=-1)
    geom = dfs.elements_to_geometry(elements, node_layers="top")
    assert not hasattr(geom, "n_layers")
    assert geom.n_elements == len(elements)

    with pytest.raises(Exception):
        geom = dfs.elements_to_geometry(elements, node_layers="center")


def test_element_table():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    eid = 31
    nid = dfs.element_table[eid]
    assert nid[0] == 32
    assert nid[1] == 28
    assert nid[2] == 23


def test_get_node_centered_data():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu(filename)
    ds = dfs.read(items="Surface elevation")
    time_step = 0
    wl_cc = ds.data[0][time_step, :]
    wl_nodes = dfs.get_node_centered_data(wl_cc)

    eid = 31
    assert wl_cc[eid] == pytest.approx(0.4593418836)
    nid = dfs.element_table[eid]
    assert wl_nodes[nid].mean() == pytest.approx(0.4593501736)


def test_interp2d():
    dfs = Dfsu("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    nt = ds.n_timesteps

    g = dfs.get_overset_grid(shape=(20, 10), buffer=-1e-2)
    interpolant = dfs.get_2d_interpolant(g.xy, n_nearest=1)
    dsi = dfs.interp2d(ds, *interpolant)

    assert dsi.shape == (nt, 20 * 10)

    with pytest.raises(Exception):
        dfs.get_2d_interpolant(g.xy, n_nearest=0)


def test_interp2d_radius():
    dfs = Dfsu("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    nt = ds.n_timesteps

    g = dfs.get_overset_grid(shape=(20, 10), buffer=-1e-2)
    interpolant = dfs.get_2d_interpolant(
        g.xy, extrapolate=True, n_nearest=1, radius=0.1
    )
    dsi = dfs.interp2d(ds, *interpolant)

    assert dsi.shape == (nt, 20 * 10)
    assert np.isnan(dsi["Wind speed"].to_numpy()[0][0])


def test_interp2d_reshaped():
    dfs = Dfsu("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"], time_steps=[0, 1])
    nt = ds.n_timesteps

    g = dfs.get_overset_grid(shape=(20, 10), buffer=-1e-2)
    interpolant = dfs.get_2d_interpolant(g.xy, n_nearest=1)
    dsi = dfs.interp2d(ds, *interpolant, shape=(g.ny, g.nx))

    assert dsi.shape == (nt, g.ny, g.nx)


def test_extract_track():
    dfs = Dfsu("tests/testdata/track_extraction_case02_indata.dfsu")
    csv_file = "tests/testdata/track_extraction_case02_track.csv"
    df = pd.read_csv(
        csv_file,
        index_col=0,
        parse_dates=True,
    )
    track = dfs.extract_track(df)

    assert track.data[2][23] == approx(3.6284972794399653)
    assert sum(np.isnan(track.data[2])) == 26
    assert np.all(track.data[1] == df.latitude.values)

    items = ["Sign. Wave Height", "Wind speed"]
    track2 = dfs.extract_track(csv_file, items=items)
    assert track2.data[2][23] == approx(3.6284972794399653)

    track3 = dfs.extract_track(csv_file, method="inverse_distance")
    assert track3.data[2][23] == approx(3.6469911492412463)


def test_extract_bad_track():
    dfs = Dfsu("tests/testdata/track_extraction_case02_indata.dfsu")
    csv_file = "tests/testdata/track_extraction_case02_track.csv"
    df = pd.read_csv(
        csv_file,
        index_col=0,
        parse_dates=True,
    )
    df = df.sort_values("longitude")
    with pytest.raises(AssertionError):
        dfs.extract_track(df)


def test_extract_surface_elevation_from_3d():
    dfs = Dfsu("tests/testdata/oresund_sigma_z.dfsu")
    outputfile = "tests/testdata/oresund_surface_elev_extracted.dfsu"
    n_top1 = len(dfs.top_elements)

    dfs.extract_surface_elevation_from_3d(outputfile, time_steps=-1)

    dfs2 = Dfsu(outputfile)
    assert dfs2.n_elements == n_top1
    os.remove(outputfile)  # clean up


def test_find_nearest_element_in_Zlayer():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    el2dindx = dfs.elem2d_ids[12]
    assert el2dindx == 2
    ids = dfs.find_nearest_elements(357000, 6200000, layer=0)
    el2dindx = dfs.elem2d_ids[ids]
    table = dfs.e2_e3_table[el2dindx]
    assert ids == 3216
    assert el2dindx == 745
    assert len(table) == 9
    ids = dfs.find_nearest_elements(357000, 6200000, layer=8)
    el2dindx = dfs.elem2d_ids[ids]
    table = dfs.e2_e3_table[el2dindx]
    assert ids == 3224
    assert el2dindx == 745
    assert len(table) == 9

    with pytest.raises(Exception):
        # z and layer cannot both be given
        dfs.find_nearest_elements(357000, 6200000, z=-3, layer=8)


def test_e2_e3_table_2d_file():
    filename = os.path.join("tests", "testdata", "NorthSea_HD_and_windspeed.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "e2_e3_table")


# TODO - this is an interim test until Dfsu.to_dfs2 method is finalized
def test_dfsu_to_dfs2(dfsu_hd2d, tmpdir):
    # Create dfs2 file
    dx = 25
    dy = 25
    nx = 100
    ny = 100
    filename = Path(tmpdir.dirname) / "test.dfs2"
    dfs2 = dfsu_hd2d.to_dfs2(
        x0=605900,
        y0=6902400,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        rotation=0,
        epsg=None,
        interpolation_method="nearest",
        filename=filename,
    )

    # Make sure it was saved to the correct location
    assert dfs2._filename == str(filename)

    # Ensure all items are identical
    for i, dfsu_item in enumerate(dfsu_hd2d.iteminfos):
        for parameter in ["data_value_type", "name", "type", "unit"]:
            assert getattr(dfsu_item, parameter) == getattr(
                dfs2.iteminfos[i], parameter
            )

    # Check timesteps
    assert dfs2.timestep == dfsu_hd2d.timestep
    assert dfs2.start_time == dfsu_hd2d.start_time
    assert dfs2.end_time == dfsu_hd2d.end_time

    # Check grid
    assert np.isclose(dfs2.dx, dx, atol=0.1, rtol=0)
    assert np.isclose(dfs2.dy, dy, atol=0.1, rtol=0)
    assert dfs2.shape == (dfsu_hd2d.n_timesteps, ny, nx)

    # Make sure data was interpolated (not all values are nan's)
    assert not np.all(np.isnan(dfs2.read().data))
