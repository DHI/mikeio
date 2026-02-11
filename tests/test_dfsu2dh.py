from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import mikeio
from mikeio import Dataset, DataArray, Mesh
from pytest import approx
from mikeio.exceptions import OutsideModelDomainError

from mikeio.spatial._FM_geometry import GeometryFM2D
from mikeio.spatial import GeometryPoint2D
from mikeio.spatial import Grid2D


def test_repr() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    text = repr(dfs)
    assert "Dfsu2D" in text


def test_read_all_items_returns_all_items_and_names() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read()

    assert dfs.n_items == 4

    repr(ds)
    repr(dfs)

    assert len(ds) == 4

    # A filename can be a string or a Path object
    filepath = Path(filename)

    dfs = mikeio.Dfsu2DH(filepath)

    assert isinstance(filepath, Path)
    assert dfs.n_items == 4


def test_read_item_0() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    assert dfs.n_items == 4

    ds = dfs.read(items=1)

    assert len(ds) == 1


def test_read_single_precision() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename, items=1, dtype=np.float32)

    assert len(ds) == 1
    assert ds[0].dtype == np.float32


def test_read_precision_single_and_double() -> None:
    filename = "tests/testdata/HD2D.dfsu"

    ds = mikeio.read(filename, items=1)
    assert ds[0].dtype == np.float32

    # Double precision
    ds = mikeio.read(filename, items=1, dtype=np.float64)
    assert ds[0].dtype == np.float64


def test_read_timestep_1() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(time=1)

    assert len(ds.time) == 1


def test_read_single_item_returns_single_item() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[3])

    assert len(ds.items) == 1


def test_read_single_item_scalar_index() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[3])

    assert len(ds) == 1


def test_read_returns_array_time_dimension_first() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[3])

    assert ds.shape == (9, 884)


def test_read_selected_item_returns_correct_items() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds) == 2
    assert ds.items[0].name == "Surface elevation"
    assert ds.items[1].name == "Current speed"


def test_read_selected_item_names_returns_correct_items() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=["Surface elevation", "Current speed"])

    assert len(ds) == 2
    assert ds.items[0].name == "Surface elevation"
    assert ds.items[1].name == "Current speed"


def test_read_all_time_steps() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[0, 3])

    assert len(ds.time) == 9
    assert ds[0].to_numpy().shape[0] == 9


def test_read_all_time_steps_without_reading_items() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    assert isinstance(dfs.time, pd.DatetimeIndex)
    assert len(dfs.time) == 9


def test_read_item_range() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=range(1, 3))  # [1,2]

    assert ds.n_items == 2
    assert ds.items[0].name == "U velocity"


def test_read_single_time_step() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[0, 3], time=1)
    assert "time" not in ds.dims

    ds = dfs.read(items=[0, 3], time=[1], keepdims=True)
    assert "time" in ds.dims


def test_read_single_time_step_scalar() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=[0, 3], time=1)

    assert len(ds.time) == 1
    assert ds[0].to_numpy().shape[0] == dfs.geometry.n_elements


def test_read_single_time_step_outside_bounds_fails() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    with pytest.raises(Exception):
        dfs.read(items=[0, 3], time=[100])


def test_read_area() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(area=(0, 0, 0.1, 0.1))
    assert isinstance(ds.geometry, GeometryFM2D)
    assert ds.geometry.n_elements == 18


def test_read_area_polygon() -> None:
    polygon = [
        (7.78, 55.20),
        (7.03, 55.46),
        (6.91, 54.98),
        (7.53, 54.73),
        (7.78, 55.20),
    ]

    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    p1 = (4.0, 54.0)
    assert p1 in dfs.geometry

    ds = dfs.read(area=polygon)

    assert p1 not in ds.geometry

    assert ds.geometry.n_elements < dfs.geometry.n_elements

    domain = dfs.geometry.to_shapely().buffer(0)
    subdomain = ds.geometry.to_shapely().buffer(0)

    assert subdomain.within(domain)


def test_read_elements() -> None:
    ds = mikeio.read(filename="tests/testdata/wind_north_sea.dfsu", elements=[0, 10])
    assert ds.geometry.element_coordinates[0][0] == pytest.approx(1.4931853081272184)
    assert ds["Wind speed"].to_numpy()[0, 0] == pytest.approx(9.530759811401367)

    ds2 = mikeio.read(filename="tests/testdata/wind_north_sea.dfsu", elements=[10, 0])
    assert ds2.geometry.element_coordinates[1][0] == pytest.approx(1.4931853081272184)
    assert ds2["Wind speed"].to_numpy()[0, 1] == pytest.approx(9.530759811401367)


def test_read_x_y() -> None:
    x = [1.49318531, 3.69276145]
    y = [53.97088571, 54.08928194]
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    assert isinstance(dfs, mikeio.Dfsu2DH)
    ds = dfs.read(x=x, y=y, time=0, keepdims=True)
    assert isinstance(ds.geometry, mikeio.GeometryFM2D)
    assert ds.geometry.element_coordinates[0][0] == pytest.approx(1.4931853081272184)
    assert ds["Wind speed"].to_numpy()[0, 0] == pytest.approx(9.530759811401367)

    x = x[::-1]
    y = y[::-1]
    ds2 = mikeio.read(
        filename="tests/testdata/wind_north_sea.dfsu", x=x, y=y, time=0, keepdims=True
    )
    assert isinstance(ds2.geometry, mikeio.GeometryFM2D)
    assert ds2.geometry.element_coordinates[1][0] == pytest.approx(1.4931853081272184)
    assert ds2["Wind speed"].to_numpy()[0, 1] == pytest.approx(9.530759811401367)

    x = 1.49318531
    y = 53.97088571
    ds3 = mikeio.read(
        filename="tests/testdata/wind_north_sea.dfsu", x=x, y=y, time=0, keepdims=True
    )
    assert isinstance(ds3.geometry, mikeio.spatial.GeometryPoint2D)
    assert ds3.geometry.x == pytest.approx(1.4931853081272184)
    assert ds3.geometry.y == pytest.approx(53.97088571)


def test_find_index_on_island() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    idx = dfs.geometry.find_index(x=-0.36, y=0.14)
    assert 230 in idx

    with pytest.raises(OutsideModelDomainError) as ex:
        dfs.geometry.find_index(x=-0.36, y=0.15)

    assert ex.value.x == -0.36
    assert ex.value.y == 0.15


def test_read_area_single_element() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(area=(0, 0, 0.02, 0.02))

    # TODO shouldn't this be GeometryFM2D ?
    assert isinstance(ds.geometry, GeometryPoint2D)
    assert ds.dims == ("time",)

    ds = dfs.read(area=(0, 0, 0.02, 0.02), time=0)
    assert isinstance(ds.geometry, GeometryPoint2D)
    assert len(ds.dims) == 0


def test_read_empty_area_fails() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    with pytest.raises(ValueError, match="No elements in selection"):
        dfs.read(area=(0, 0, 0.001, 0.001))


def test_number_of_time_steps() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    assert dfs.n_timesteps == 9


def test_get_node_coords() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    nc = dfs.geometry.node_coordinates
    assert nc[0, 0] == 607031.4886285994

    codes = dfs.geometry.codes
    nc = codes[codes == 1]
    assert len(nc) > 0


def test_element_coordinates() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ec = dfs.geometry.element_coordinates
    assert ec[1, 1] == pytest.approx(6906790.5928664245)


def test_element_coords_is_inside_nodes() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    nc = dfs.geometry.node_coordinates
    ec = dfs.geometry.element_coordinates
    nc_min = nc.min(axis=0)
    nc_max = nc.max(axis=0)
    ec_max = ec.max(axis=0)
    ec_min = ec.min(axis=0)

    assert ec_max[0] < nc_max[0]
    assert ec_max[1] < nc_max[1]
    assert ec_min[0] > nc_min[0]
    assert ec_min[1] > nc_min[0]


def test_contains() -> None:
    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    pts = [[4, 54], [0, 50]]
    inside = dfs.geometry.contains(pts)
    assert inside[0]
    assert not inside[1]


def test_point_in_domain() -> None:
    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    pt = [4, 54]
    assert pt in dfs.geometry

    pt2 = [0, 50]
    assert pt2 not in dfs.geometry

    pts = [pt, pt2]
    inside = [pt in dfs.geometry for pt in pts]
    assert inside[0] is True
    assert inside[1] is False


def test_get_overset_grid() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    g = dfs.get_overset_grid()
    assert g.nx == 21
    assert g.ny == 10

    g = dfs.get_overset_grid(dx=0.2)
    assert pytest.approx(g.dx) == 0.2
    assert pytest.approx(g.dy) == 0.2

    g = dfs.get_overset_grid(dx=0.4, dy=0.2)
    assert pytest.approx(g.dx) == 0.4
    assert pytest.approx(g.dy) == 0.2

    g = dfs.get_overset_grid(nx=5, ny=4)
    assert g.nx == 5
    assert g.ny == 4

    g = dfs.get_overset_grid(ny=5)
    assert g.nx == 11
    assert g.ny == 5


def test_find_nearest_element_2d() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    elem_id = dfs.geometry.find_nearest_elements(606200, 6905480)
    assert elem_id == 317


def test_find_nearest_element_2d_and_distance() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    (elem_id, dist) = dfs.geometry.find_nearest_elements(
        606200, 6905480, return_distances=True
    )
    assert elem_id == 317

    assert dist > 0.0


def test_dfsu_to_dfs0(tmp_path: Path) -> None:
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename).sel(x=606200, y=6905480)

    fp = tmp_path / "out.dfs0"
    ds.to_dfs(fp)

    newds = mikeio.read(fp)

    assert newds[0].name == ds[0].name
    assert ds.time[0] == newds.time[0]
    assert ds.time[-1] == newds.time[-1]


def test_find_nearest_elements_2d_array() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    elem_ids = dfs.geometry.find_nearest_elements(
        x=[606200, 606200], y=[6905480, 6905480]
    )
    assert len(elem_ids) == 2
    assert elem_ids[0] == 317
    assert elem_ids[1] == 317


def test_read_and_select_single_element() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename)

    assert ds.shape == (9, 884)
    selds = ds.sel(x=606200, y=6905480)

    assert selds.shape == (9,)


def test_is_2d() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    assert dfs.geometry.is_2d

    filename = "tests/testdata/basin_3d.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    assert not dfs.geometry.is_2d


def test_is_geo_UTM() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    assert dfs.geometry.is_geo is False


def test_is_geo_LONGLAT() -> None:
    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    assert dfs.geometry.is_geo is True


def test_is_local_coordinates() -> None:
    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    assert dfs.geometry.is_local_coordinates is False


def test_get_element_area_UTM() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    areas = dfs.geometry.get_element_area()
    assert areas[0] == 4949.102548750438


def test_get_element_area_LONGLAT() -> None:
    filename = "tests/testdata/wind_north_sea.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    areas = dfs.geometry.get_element_area()
    assert areas[0] == 139524218.81411952


def test_get_element_area_tri_quad() -> None:
    filename = "tests/testdata/FakeLake.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    areas = dfs.geometry.get_element_area()
    assert areas[0] == 0.0006875642143608321


def test_write(tmp_path: Path) -> None:
    fp = tmp_path / "simple.dfsu"
    meshfilename = "tests/testdata/odense_rough.mesh"

    msh = Mesh(meshfilename)

    da = mikeio.DataArray(np.zeros((1, msh.n_elements)), geometry=msh.geometry)

    da.to_dfs(fp)
    da.isel(time=0).to_dfs(fp)


def test_write_from_dfsu(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "simple.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

    assert dfs.start_time.hour == 7

    ds = dfs.read(items=[0, 1])

    ds.to_dfs(fp)

    newdfs = mikeio.Dfsu2DH(fp)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_incremental_write_using_mikecore(tmp_path: Path) -> None:
    from mikecore.DfsFileFactory import DfsFileFactory

    sourcefilename = "tests/testdata/HD2D.dfsu"

    # copy to tmp_path
    fp = str(tmp_path / "simple.dfsu")
    shutil.copy(sourcefilename, fp)

    nt = 10

    dfs = DfsFileFactory.DfsGenericOpenEdit(fp)
    n_items = len(dfs.ItemInfo)
    n_elements = dfs.ItemInfo[0].ElementCount

    for _ in range(nt):
        for _ in range(n_items):
            data = np.ones(n_elements).astype(np.float32)  # Replace with actual data
            dfs.WriteItemTimeStepNext(0.0, data)
    dfs.Close()


def test_write_from_dfsu_2_time_steps(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "simple.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

    ds = dfs.read(time=[0, 1])

    assert ds.is_equidistant  # Data with two time steps is per definition equidistant

    ds.to_dfs(fp)

    newdfs = mikeio.Dfsu2DH(fp)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time != newdfs.end_time


def test_write_non_equidistant_is_possible(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "simple.dfsu"
    ds = mikeio.read(sourcefilename, time=[0, 1, 3])
    assert not ds.is_equidistant

    ds.to_dfs(fp)

    ds2 = mikeio.read(fp)

    assert all(ds.time == ds2.time)

    dfs = mikeio.Dfsu2DH(fp)

    dfs = mikeio.Dfsu2DH(fp)

    # it is not possible to get all time without reading the entire file
    with pytest.raises(NotImplementedError):
        dfs.time

    # but getting the end time is not that expensive
    assert dfs.end_time == ds.time[-1]

    ds3 = mikeio.read(fp, time=[0, 1])
    assert ds3.n_timesteps == 2


def test_temporal_resample_by_reading_selected_timesteps(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "simple.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time=list(range(0, nt, 2)))
    ds.to_dfs(fp)

    newdfs = mikeio.Dfsu2DH(fp)

    assert pytest.approx(dfs.timestep) == newdfs.timestep / 2


def test_read_temporal_subset() -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

    assert dfs.n_timesteps == 9

    ds = dfs.read(time=slice("1985-08-06 00:00", "1985-08-06 12:00"))

    assert len(ds.time) == 3

    # Specify start
    ds = dfs.read(time=slice("1985-08-06 12:00", None))

    assert len(ds.time) == 7

    # Specify end
    ds = dfs.read(time=slice(None, "1985-08-06 12:00"))

    assert len(ds.time) == 3


def test_read_temporal_subset_string() -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

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


def test_write_temporal_subset(tmp_path: Path) -> None:
    sourcefilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "simple.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

    assert dfs.n_timesteps == 9

    ds = dfs.read(time=slice("1985-08-06 12:00", None))
    ds.to_dfs(fp)

    newdfs = mikeio.Dfsu2DH(fp)

    assert newdfs.start_time.hour == 12
    assert newdfs.n_timesteps == 7


def test_to_mesh_2d(tmp_path: Path) -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    fp = tmp_path / "hd2d.mesh"

    dfs.geometry.to_mesh(fp)

    mesh = Mesh(fp)

    assert mesh.n_elements == dfs.geometry.n_elements


def test_element_table() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    eid = 31
    nid = dfs.geometry.element_table[eid]
    assert nid[0] == 32
    assert nid[1] == 28
    assert nid[2] == 23


def test_get_node_centered_data() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    ds = dfs.read(items="Surface elevation")
    time_step = 0
    wl_cc = ds[0].values[time_step, :]
    wl_nodes = dfs.geometry.get_node_centered_data(wl_cc)

    eid = 31
    assert wl_cc[eid] == pytest.approx(0.4593418836)
    nid = dfs.geometry.element_table[eid]
    assert wl_nodes[nid].mean() == pytest.approx(0.4593501736)


def test_interp2d_radius() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/wind_north_sea.dfsu")
    ds = dfs.read(items=["Wind speed"])
    nt = ds.n_timesteps
    nx = 20
    ny = 10

    g = dfs.get_overset_grid(nx=20, ny=10, buffer=-1e-2)

    dsi = ds.interp_like(g, extrapolate=True, n_nearest=1, radius=0.1)

    assert dsi.shape == (nt, ny, nx)
    assert np.isnan(dsi["Wind speed"].to_numpy()[0, 0, 0])


def test_extract_track() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/track_extraction_case02_indata.dfsu")
    csv_file = "tests/testdata/track_extraction_case02_track.csv"
    df = pd.read_csv(
        csv_file,
        index_col=0,
        parse_dates=True,
    )

    # the csv datetime has inconsistent formatting, which is not supported by pandas 2.0
    # 2008-07-04 12:20:00
    # 2008-07-04 12:20:00.001

    df.index = pd.DatetimeIndex(df.index)

    track = dfs.extract_track(df)

    assert track[2].values[23] == approx(3.6284972794399653)
    assert sum(np.isnan(track[2].to_numpy())) == 26
    assert np.all(track[1].to_numpy() == df.latitude.values)

    items = ["Sign. Wave Height", "Wind speed"]
    track2 = dfs.extract_track(csv_file, items=items)
    assert track2[2].values[23] == approx(3.6284972794399653)
    assert track["Wind speed"].values[23] == approx(12.4430027008056)

    track3 = dfs.extract_track(csv_file, method="inverse_distance")
    assert track3[2].values[23] == approx(3.6469911492412463)


# TODO consider move to test_dataset.py
def test_extract_track_from_dataset(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/track_extraction_case02_indata.dfsu")
    csv_file = "tests/testdata/track_extraction_case02_track.csv"
    df = pd.read_csv(
        csv_file,
        index_col=0,
        parse_dates=True,
    )
    df.index = pd.DatetimeIndex(df.index)
    assert ds[0].name == "Sign. Wave Height"
    track = ds.extract_track(df)

    # This should not change the original dataset
    track = track.rename({"Sign. Wave Height": "Hm0"})
    assert track["Hm0"].name == "Hm0"

    assert ds[0].name == "Sign. Wave Height"

    assert track["Hm0"].values[23] == approx(3.6284972794399653)
    assert sum(np.isnan(track[2].to_numpy())) == 26
    assert np.all(track[1].to_numpy() == df.latitude.values)

    ds2 = ds[["Sign. Wave Height", "Wind speed"]]
    track2 = ds2.extract_track(csv_file)
    assert track2[2].values[23] == approx(3.6284972794399653)

    track3 = ds2.extract_track(csv_file, method="inverse_distance")
    assert track3[2].values[23] == approx(3.6469911492412463)

    # test with dataset
    track_ds = mikeio.from_pandas(df)
    track4 = ds2.extract_track(track_ds)
    assert track4[2].values[23] == approx(3.6284972794399653)

    # test with dfs0 file
    track_ds.to_dfs(tmp_path / "track.dfs0")
    track5 = ds2.extract_track(tmp_path / "track.dfs0")
    assert track5[2].values[23] == approx(3.6284972794399653)

    # test with non-existent file
    with pytest.raises(FileNotFoundError):
        ds2.extract_track("non_existent_file.csv")

    # test with bad file extension
    fp = tmp_path / "track.txt"
    fp.touch()
    with pytest.raises(ValueError):
        ds2.extract_track(fp)


# TODO consider move to test_datarray.py
def test_extract_track_from_dataarray() -> None:
    da = mikeio.read("tests/testdata/track_extraction_case02_indata.dfsu")[0]
    csv_file = "tests/testdata/track_extraction_case02_track.csv"
    df = pd.read_csv(
        csv_file,
        index_col=0,
        parse_dates=True,
    )
    df.index = pd.DatetimeIndex(df.index)
    track = da.extract_track(df)

    assert track[2].values[23] == approx(3.6284972794399653)
    assert sum(np.isnan(track[2].to_numpy())) == 26
    assert np.all(track[1].to_numpy() == df.latitude.values)


def test_extract_bad_track() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/track_extraction_case02_indata.dfsu")
    csv_file = "tests/testdata/track_extraction_case02_track.csv"
    df = pd.read_csv(
        csv_file,
        index_col=0,
        parse_dates=True,
    )
    df = df.sort_values("longitude")
    with pytest.raises(AssertionError):
        dfs.extract_track(df)


def test_dataset_write_dfsu(tmp_path: Path) -> None:
    fp = tmp_path / "HD2D_start.dfsu"
    ds = mikeio.read("tests/testdata/HD2D.dfsu", time=[0, 1])
    ds.to_dfs(fp)

    dfs = mikeio.Dfsu2DH(fp)

    ds2 = dfs.read()
    assert ds2.n_timesteps == 2


def test_dataset_interp() -> None:
    ds = mikeio.read("tests/testdata/oresundHD_run1.dfsu")
    da = ds["Surface elevation"]

    x = 360000
    y = 6184000

    dai = da.interp(x=x, y=y)
    assert isinstance(dai, DataArray)
    assert dai.shape == (ds.n_timesteps,)
    assert dai.name == da.name
    assert dai.geometry.x == x
    assert dai.geometry.y == y
    assert dai.geometry.projection == ds.geometry.projection


def test_dataset_interp_to_xarray() -> None:
    ds = mikeio.read("tests/testdata/oresundHD_run1.dfsu")

    assert not ds.geometry.is_geo

    x = 360000
    y = 6184000

    dsi = ds.interp(x=x, y=y)

    xr_dsi = dsi.to_xarray()
    assert float(xr_dsi.x) == pytest.approx(x)
    assert float(xr_dsi.y) == pytest.approx(y)


def test_interp_like_grid() -> None:
    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    ws = ds[0]
    assert ws.sel(x=0, y=55).to_numpy()[0] == pytest.approx(9.062851)
    with pytest.raises(OutsideModelDomainError):
        assert ws.sel(x=6, y=51)
    assert ws.values.dtype == np.float32
    grid = ds.geometry.get_overset_grid(dx=0.1)
    ws_grid = ws.interp_like(grid, n_nearest=1)
    # outside the domain, but inside the grid
    assert ws_grid.sel(x=0, y=55).to_numpy()[0] == pytest.approx(9.062851)
    assert np.isnan(ws_grid.sel(x=6, y=51).to_numpy()[0])
    assert ws_grid.values.dtype == np.float32
    assert ws_grid.n_timesteps == ds.n_timesteps
    assert isinstance(ws_grid, DataArray)
    assert isinstance(ws_grid.geometry, Grid2D)


def test_interp_like_grid_time_invariant() -> None:
    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu", time=-1)
    assert "time" not in ds.dims
    grid = ds.geometry.get_overset_grid(dx=0.1)
    ds_grid = ds.interp_like(grid)
    assert ds_grid.n_timesteps == ds.n_timesteps
    assert isinstance(ds_grid, Dataset)
    assert isinstance(ds_grid.geometry, Grid2D)

    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu", time=-1)
    assert "time" not in ds.dims
    ws = ds[0]
    grid = ds.geometry.get_overset_grid(dx=0.1)
    ws_grid = ws.interp_like(grid)
    assert ws_grid.n_timesteps == ds.n_timesteps
    assert isinstance(ws_grid, DataArray)
    assert isinstance(ws_grid.geometry, Grid2D)


def test_interp_like_dataarray(tmp_path: Path) -> None:
    tmp_path / "interp.dfs2"

    da = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")[0]
    da2 = mikeio.read("tests/testdata/consistency/oresundHD.dfs2", time=[0, 1])[0]

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


def test_interp_like_dataset(tmp_path: Path) -> None:
    fp = tmp_path / "interp.dfs2"

    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")
    ds2 = mikeio.read("tests/testdata/consistency/oresundHD.dfs2", time=[0, 1])

    dsi = ds.interp_like(ds2)
    assert isinstance(dsi, Dataset)
    assert isinstance(dsi.geometry, Grid2D)
    assert dsi.n_timesteps == ds2.n_timesteps
    assert dsi.end_time == ds2.end_time

    fp = tmp_path / "interp.dfs2"
    dsi.to_dfs(fp)

    dse = ds.interp_like(ds2, extrapolate=True)

    fp = tmp_path / "extrap.dfs2"
    dse.to_dfs(fp)


def test_interp_like_fm() -> None:
    msh = Mesh("tests/testdata/north_sea_2.mesh")
    geometry = msh.geometry
    assert isinstance(geometry, GeometryFM2D)

    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    ws = ds[0]
    wsi = ws.interp_like(geometry)
    assert isinstance(wsi, DataArray)
    assert isinstance(wsi.geometry, GeometryFM2D)

    wsi2 = ws.interp_like(geometry, n_nearest=5, extrapolate=True)
    assert isinstance(wsi2, DataArray)
    assert isinstance(wsi2.geometry, GeometryFM2D)


def test_interp_like_fm_dataset() -> None:
    msh = Mesh("tests/testdata/north_sea_2.mesh")
    geometry = msh.geometry
    assert isinstance(geometry, GeometryFM2D)

    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    dsi = ds.interp_like(geometry)
    assert isinstance(dsi, Dataset)
    assert isinstance(dsi.geometry, GeometryFM2D)


def test_append_dfsu_2d(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu", time=[0, 1])
    ds2 = mikeio.read("tests/testdata/consistency/oresundHD.dfsu", time=[2, 3])
    new_filename = tmp_path / "appended.dfsu"
    ds.to_dfs(new_filename)
    dfs = mikeio.Dfsu2DH(new_filename)
    assert dfs.time[-1] == ds.time[-1]
    dfs.append(ds2)
    assert dfs.time[-1] == ds2.time[-1]

    ds3 = mikeio.read(new_filename)
    assert ds3.n_timesteps == 4
    assert ds3.time[-1] == ds2.time[-1]
    assert (
        ds3["V velocity"].isel(time=3).values[0]
        == ds2["V velocity"].isel(time=1).values[0]
    )


def test_repr_dfsu_many_items_only_shows_number_of_items() -> None:
    ds = mikeio.read("tests/testdata/random_data_20_items_2d.dfsu")
    txt = repr(ds)
    assert "number of items: 20" in txt

    # repeat for mikeio.Dfsu2DH
    dfs = mikeio.Dfsu2DH("tests/testdata/random_data_20_items_2d.dfsu")
    txt_dfs = repr(dfs)
    assert "number of items: 20" in txt_dfs


def test_dfsu_to_xarray_has_element_coordinates() -> None:
    ds = mikeio.read("tests/testdata/FakeLake.dfsu")
    da = ds[0]

    example_tri_element = 5
    example_tri_coordinates = [
        -0.6155248853333334,
        0.175671869,
        -4.971006711324056,
    ]

    example_quad_element = 150
    example_quad_coordinates = [
        -0.4014337815,
        0.069547867575,
        -26.11973285675049,
    ]

    xr_da = da.to_xarray()

    # coordinate existence and dimension checks
    for coord in ["x", "y", "z"]:
        assert coord in xr_da.coords
        assert xr_da.coords[coord].dims == ("element",)
        assert xr_da.coords[coord].sizes["element"] == da.geometry.n_elements

    # TRI element values match
    assert xr_da.x.values[example_tri_element] == approx(example_tri_coordinates[0])
    assert xr_da.y.values[example_tri_element] == approx(example_tri_coordinates[1])
    assert xr_da.z.values[example_tri_element] == approx(example_tri_coordinates[2])

    # QUAD element values match
    assert xr_da.x.values[example_quad_element] == approx(example_quad_coordinates[0])
    assert xr_da.y.values[example_quad_element] == approx(example_quad_coordinates[1])
    assert xr_da.z.values[example_quad_element] == approx(example_quad_coordinates[2])


def test_write_dfsu_with_title(tmp_path: Path) -> None:
    """Test writing a dfsu file with a custom title and reading it back."""
    sourcefilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "with_title.dfsu"

    # Read source file
    dfs = mikeio.Dfsu2DH(sourcefilename)
    ds = dfs.read(items=[0])

    # Write with custom title
    custom_title = "Test DFSU with Custom Title"
    ds.to_dfs(fp, title=custom_title)

    # Read back and verify title
    newdfs = mikeio.Dfsu2DH(fp)
    assert newdfs.title == custom_title


def test_read_dfsu_title(tmp_path: Path) -> None:
    """Test reading the title from an existing dfsu file."""
    sourcefilename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(sourcefilename)

    # Dfsu files should have a title property
    assert hasattr(dfs, "title")
    assert isinstance(dfs.title, str)
