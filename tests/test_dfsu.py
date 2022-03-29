import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import mikeio
from mikeio import Dataset, DataArray, Dfs0, Dfsu, Mesh
from mikeio.eum import ItemInfo
from pytest import approx

from mikeio.spatial.FM_geometry import GeometryFM
from mikeio.spatial.grid_geometry import Grid2D


def test_repr():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    text = repr(dfs)
    assert "Dfsu2D" in text

    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    text = repr(dfs)
    assert "number of z layers" in text


def test_read_all_items_returns_all_items_and_names():
    filename = "tests/testdata/HD2D.dfsu"
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
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    assert dfs.n_items == 4

    ds = dfs.read(items=1)

    assert len(ds) == 1


def test_read_single_precision():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename, dtype=np.float32)

    ds = dfs.read(items=1)

    assert len(ds) == 1
    assert ds[0].dtype == np.float32


def test_read_precision_open():
    filename = "tests/testdata/HD2D.dfsu"

    dfs = mikeio.open(filename)
    ds = dfs.read(items=1)
    assert ds[0].dtype == np.float32

    # Double precision
    dfs = mikeio.open(filename, dtype=np.float64)
    ds = dfs.read(items=1)
    assert ds[0].dtype == np.float64


def test_read_int_not_accepted():
    filename = "tests/testdata/HD2D.dfsu"
    with pytest.raises(Exception):
        dfs = Dfsu(filename, dtype=np.int32)


def test_read_timestep_1():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(time=1)

    assert len(ds.time) == 1


def test_read_single_item_returns_single_item():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[3])

    assert len(ds.items) == 1


def test_read_single_item_scalar_index():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[3])

    assert len(ds) == 1


def test_read_returns_array_time_dimension_first():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[3])

    assert ds.data[0].shape == (9, 884)


def test_read_selected_item_returns_correct_items():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds) == 2
    assert ds.items[0].name == "Surface elevation"
    assert ds.items[1].name == "Current speed"


def test_read_selected_item_names_returns_correct_items():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=["Surface elevation", "Current speed"])

    assert len(ds) == 2
    assert ds.items[0].name == "Surface elevation"
    assert ds.items[1].name == "Current speed"


def test_read_all_time_steps():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds.time) == 9
    assert ds.data[0].shape[0] == 9


def test_read_item_range():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=range(1, 3))  # [1,2]

    assert ds.n_items == 2
    assert ds.items[0].name == "U velocity"


def test_read_all_time_steps_without_progressbar():

    Dfsu.show_progress = True

    filename = "tests/testdata/HD2D.dfsu"

    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds.time) == 9
    assert ds.data[0].shape[0] == 9


def test_read_single_time_step():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3], time=1)
    assert "time" not in ds.dims

    ds = dfs.read(items=[0, 3], time=[1])  # this forces time dimension to be kept
    assert "time" in ds.dims


def test_read_single_time_step_scalar():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(items=[0, 3], time=1)

    assert len(ds.time) == 1
    assert ds.data[0].shape[0] == dfs.n_elements


def test_read_single_time_step_outside_bounds_fails():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    with pytest.raises(Exception):

        dfs.read(items=[0, 3], time=[100])


def test_number_of_time_steps():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    assert dfs.n_timesteps == 9


def test_get_node_coords():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    nc = dfs.node_coordinates
    assert nc[0, 0] == 607031.4886285994

    nc = dfs.get_node_coords(code=1)
    assert len(nc) > 0


def test_element_coordinates():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ec = dfs.element_coordinates
    assert ec[1, 1] == pytest.approx(6906790.5928664245)


def test_element_coords_is_inside_nodes():
    filename = "tests/testdata/HD2D.dfsu"
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
    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = Dfsu(filename)

    pts = [[4, 54], [0, 50]]
    inside = dfs.contains(pts)
    assert inside[0] == True
    assert inside[1] == False


def test_get_overset_grid():
    filename = "tests/testdata/FakeLake.dfsu"
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
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    elem_id = dfs.find_nearest_elements(606200, 6905480)
    assert elem_id == 317


def test_find_nearest_element_2d_and_distance():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    (elem_id, dist) = dfs.find_nearest_elements(606200, 6905480, return_distances=True)
    assert elem_id == 317

    assert dist > 0.0


def test_dfsu_to_dfs0_via_dataframe(tmpdir):
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename).sel(x=606200, y=6905480)

    df = ds.to_dataframe()

    outfilename = os.path.join(tmpdir, "out.dfs0")
    df.to_dfs0(outfilename)

    newds = mikeio.read(outfilename)

    assert newds[0].name == ds[0].name
    assert ds.time[0] == newds.time[0]
    assert ds.time[-1] == newds.time[-1]


def test_dfsu_to_dfs0(tmpdir):
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    elem_id = dfs.find_nearest_elements(606200, 6905480)

    ds = dfs.read(elements=[elem_id])
    dss = ds.squeeze()

    outfilename = os.path.join(tmpdir, "out.dfs0")

    dfs0 = Dfs0()
    dfs0.write(outfilename, dss)

    dfs0 = Dfs0(outfilename)
    newds = dfs0.read()

    assert newds.items[0].name == ds.items[0].name
    assert ds.time[0] == newds.time[0]
    assert ds.time[-1] == newds.time[-1]


def test_find_nearest_elements_2d_array():
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    elem_ids = dfs.find_nearest_elements(x=[606200, 606200], y=[6905480, 6905480])
    assert len(elem_ids) == 2
    assert elem_ids[0] == 317
    assert elem_ids[1] == 317


def find_nearest_profile_elements():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = Dfsu(filename)
    elem_ids = dfs.find_nearest_profile_elements(333934, 6158101)

    assert elem_ids[0] == 5320
    assert elem_ids[-1] == 5323


def test_read_and_select_single_element():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read()

    assert ds.data[0].shape == (9, 884)

    idx = dfs.find_nearest_elements(606200, 6905480)

    selds = ds.isel(idx=idx, axis=1)

    assert selds.data[0].shape == (9,)


def test_is_2d():

    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    assert dfs.is_2d

    filename = os.path.join("tests", "testdata", "basin_3d.dfsu")
    dfs = Dfsu(filename)

    assert not dfs.is_2d


def test_is_geo_UTM():
    filename = "tests/testdata/HD2D.dfsu"
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
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)
    areas = dfs.get_element_area()
    assert areas[0] == 4949.102548750438


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

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read(items=[0, 1])

    dfs.write(outfilename, ds)

    assert dfs.start_time.hour == 7

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_incremental_write_from_dfsu(tmpdir):
    "Useful for writing datasets with many timesteps to avoid problems with out of memory"

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time=[0])

    dfs.write(outfilename, ds, keep_open=True)

    for i in range(1, nt):
        ds = dfs.read(time=[i])
        dfs.append(ds)

    dfs.close()

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_incremental_write_from_dfsu_context_manager(tmpdir):

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time=[0])

    with dfs.write(outfilename, ds, keep_open=True) as f:
        for i in range(1, nt):
            ds = dfs.read(time=[i])
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

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read(time=[0, 1])

    assert ds.is_equidistant  # Data with two time steps is per definition equidistant

    dfs.write(outfilename, ds)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time != newdfs.end_time


def test_write_from_dfsu_default_items_numbered(tmpdir):

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read(items=[0, 1])

    dfs.write(outfilename, ds.data)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)
    assert newdfs.items[0].name == "Item 1"


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

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    ds = dfs.read(time=[0, 1, 3])

    with pytest.raises(Exception):
        dfs.write(outfilename, ds)

    assert not os.path.exists(outfilename)


def test_temporal_resample_by_reading_selected_timesteps(tmpdir):

    sourcefilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "simple.dfsu")
    dfs = Dfsu(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time=list(range(0, nt, 2)))
    dfs.write(outfilename, ds)

    assert os.path.exists(outfilename)

    newdfs = Dfsu(outfilename)

    assert pytest.approx(dfs.timestep) == newdfs.timestep / 2


def test_read_temporal_subset():

    sourcefilename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(sourcefilename)

    assert dfs.n_timesteps == 9

    ds = dfs.read(time=slice("1985-08-06 00:00", "1985-08-06 12:00"))

    assert len(ds.time) == 3

    # Specify start
    ds = dfs.read(time=slice("1985-08-06 12:00", None))

    assert len(ds.time) == 7

    # Specify end
    ds = dfs.read(time=slice(None, "1985-08-06 12:00"))

    assert len(ds.time) == 3


def test_read_temporal_subset_string():

    sourcefilename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(sourcefilename)

    assert dfs.n_timesteps == 9

    # start,end
    ds = dfs.read(time=slice("1985-08-06 00:00", "1985-08-06 12:00"))
    assert len(ds.time) == 3

    # start,
    ds = dfs.read(time=slice("1985-08-06 12:00", None))
    assert len(ds.time) == 7

    # ,end
    ds = dfs.read(time=slice(None, "1985-08-06 11:30"))
    assert len(ds.time) == 2

    # start=end
    ds = dfs.read(time="1985-08-06 12:00")
    assert len(ds.time) == 1


def test_write_temporal_subset(tmpdir):

    sourcefilename = "tests/testdata/HD2D.dfsu"
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


def test_to_mesh_2d(tmpdir):
    filename = "tests/testdata/HD2D.dfsu"
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
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)
    eid = 31
    nid = dfs.element_table[eid]
    assert nid[0] == 32
    assert nid[1] == 28
    assert nid[2] == 23


def test_get_node_centered_data():
    filename = "tests/testdata/HD2D.dfsu"
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
        dfs.get_spatial_interpolant(g.xy, n_nearest=0)


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
    ds = dfs.read(items=["Wind speed"], time=[0, 1])
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


def test_e2_e3_table_2d_file():
    filename = os.path.join("tests", "testdata", "NorthSea_HD_and_windspeed.dfsu")
    dfs = Dfsu(filename)
    assert not hasattr(dfs, "e2_e3_table")


def test_dataset_write_dfsu(tmp_path):

    outfilename = tmp_path / "HD2D_start.dfsu"
    ds = mikeio.read("tests/testdata/HD2D.dfsu", time=[0, 1])
    ds.to_dfs(outfilename)

    ds2 = mikeio.read(outfilename)
    assert ds2.n_timesteps == 2


def test_dataset_interp():
    ds = mikeio.read("tests/testdata/oresundHD_run1.dfsu")
    da: DataArray = ds.Surface_elevation

    x = 360000
    y = 6184000

    dai = da.interp(x=x, y=y)

    assert isinstance(dai, DataArray)
    assert dai.shape == (ds.n_timesteps,)
    assert dai.name == da.name
    assert dai.geometry.x == x
    assert dai.geometry.y == y


def test_interp_like_grid():
    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    ws = ds[0]
    grid = ds.geometry.get_overset_grid(dx=0.1)
    ws_grid = ws.interp_like(grid)
    assert ws_grid.n_timesteps == ds.n_timesteps
    assert isinstance(ws_grid, DataArray)
    assert isinstance(ws_grid.geometry, Grid2D)


def test_interp_like_dataarray(tmpdir):

    outfilename = os.path.join(tmpdir, "interp.dfs2")

    da = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")[0]
    da2 = mikeio.read("tests/testdata/consistency/oresundHD.dfs2", time_steps=[0, 1])[0]

    dai = da.interp_like(da2)
    assert isinstance(dai, DataArray)
    assert isinstance(dai.geometry, Grid2D)
    assert dai.n_timesteps == da2.n_timesteps
    assert dai.end_time == da2.end_time

    dae = da.interp_like(da2, extrapolate=True)
    assert isinstance(dae, DataArray)
    assert isinstance(dae.geometry, Grid2D)
    assert dae.n_timesteps == da2.n_timesteps
    assert dae.end_time == da2.end_time


def test_interp_like_dataset(tmpdir):

    outfilename = os.path.join(tmpdir, "interp.dfs2")

    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")
    ds2 = mikeio.read("tests/testdata/consistency/oresundHD.dfs2", time_steps=[0, 1])

    dsi = ds.interp_like(ds2)
    assert isinstance(dsi, Dataset)
    assert isinstance(dsi.geometry, Grid2D)
    assert dsi.n_timesteps == ds2.n_timesteps
    assert dsi.end_time == ds2.end_time

    outfilename = os.path.join(tmpdir, "interp.dfs2")
    dsi.to_dfs(outfilename)

    dse = ds.interp_like(ds2, extrapolate=True)

    outfilename = os.path.join(tmpdir, "extrap.dfs2")
    dse.to_dfs(outfilename)


def test_interp_like_fm():
    msh = Mesh("tests/testdata/north_sea_2.mesh")
    geometry = msh.geometry
    assert isinstance(geometry, GeometryFM)

    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    ws = ds[0]
    wsi = ws.interp_like(geometry)
    assert isinstance(wsi, DataArray)
    assert isinstance(wsi.geometry, GeometryFM)

    wsi2 = ws.interp_like(geometry, n_nearest=5, extrapolate=True)
    assert isinstance(wsi2, DataArray)
    assert isinstance(wsi2.geometry, GeometryFM)


def test_interp_like_fm_dataset():
    msh = Mesh("tests/testdata/north_sea_2.mesh")
    geometry = msh.geometry
    assert isinstance(geometry, GeometryFM)

    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    dsi = ds.interp_like(geometry)
    assert isinstance(dsi, Dataset)
    assert isinstance(dsi.geometry, GeometryFM)
