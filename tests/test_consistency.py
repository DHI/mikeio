import pandas as pd
from datetime import datetime
import pytest

import mikeio
from mikeio.dataarray import DataArray
from mikeio.spatial.geometry import GeometryUndefined
import mikeio.generic


def test_read_dfs0():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs0",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfs1():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs1",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_dfs1_isel_t():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.isel([0, 1], axis="t")
    assert ds1.dims == ("time", "x")
    assert isinstance(ds1.geometry, type(ds.geometry))
    assert ds1[0].values[0, 8] == pytest.approx(0.203246)


def test_dfs1_isel_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.isel(8, axis="x")
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(0, axis="time").values == pytest.approx(0.203246)


def test_dfs1_sel_t():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.sel(time=slice("2018", "2018-03-10"))
    assert ds1.dims == ("time", "x")
    assert isinstance(ds1.geometry, type(ds.geometry))
    assert ds1[0].values[0, 8] == pytest.approx(0.203246)


def test_dfs1_sel_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.sel(x=7.8)
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(0, axis="time").values == pytest.approx(0.203246)

    da: DataArray = ds[0]
    da1 = da.sel(x=7.8)
    assert da1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert da1.isel(0, axis="time").values == pytest.approx(0.203246)


def test_dfs1_interp_x():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

    ds1 = ds.interp(x=7.75)
    assert ds1.dims == ("time",)
    assert isinstance(ds1.geometry, GeometryUndefined)
    assert ds1[0].isel(0, axis="time").values == pytest.approx(0.20202248)


# Nice to have...
# def test_dfs1_interp_like():
#    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs1")

# ds1 = ds.interp_like(ds.geometry)
#    assert ds1.dims == ("time", "x")
#    assert np.all(ds1[0].values == ds[0].values)


# def test_create_dfs2()

#    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")
#    g = ds.geometry.get_overset_grid(dx=5000)
#    dsi = ds.interp_like(g)
#    dsi.to_dfs("tests/testdata/consistency/oresundHD.dfs2")


def test_read_dfs2():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_sel_line_dfs2():
    x = 350000
    y = 6145000
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs2")
    dsselx = ds.sel(x=x)
    assert isinstance(dsselx.geometry, mikeio.Grid1D)
    dssely = ds.sel(y=y)
    assert isinstance(dssely.geometry, mikeio.Grid1D)
    assert dsselx.geometry != dssely.geometry


def test_sel_mult_line_not_possible():
    xs = [350000, 360000]
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs2")
    with pytest.raises(
        Exception, match="scalar"
    ):  # NotImplemented or ValueError not sure yet
        ds.sel(x=xs)


def test_read_dfs2_single_time():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        time=-1,
    )

    assert ds.n_timesteps == 1
    assert "time" not in ds.dims

    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        time="2018-03-10",
    )

    assert ds.n_timesteps == 1
    assert "time" not in ds.dims

    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2", time=[-1], keepdims=True
    )

    assert ds.n_timesteps == 1
    assert "time" in ds.dims


def test_read_single_row_dfs2_single_time_step():
    ds = mikeio.read("tests/testdata/single_row.dfs2", time="2000-01-01")
    assert ds.n_timesteps == 1
    assert "time" not in ds.dims

    ds2 = mikeio.read("tests/testdata/single_row.dfs2").sel(time="2000-01-01")

    assert ds.dims == ds2.dims
    assert all(ds.time == ds2.time)


def test_interp_x_y_dfs2():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfs2")

    x = 350000
    y = 6145000
    das = ds[0].interp(x=x, y=y)

    assert das.geometry.x == x
    assert das.geometry.y == y

    dss = ds.interp(x=x, y=y)

    assert dss.geometry.x == x
    assert dss.geometry.y == y


def test_sel_x_y_dfsu2d():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")

    x = 350000
    y = 6145000
    dss = ds.sel(x=x, y=y)

    assert dss.geometry.x == pytest.approx(349034.439173)
    assert dss.geometry.y == pytest.approx(6144868.611412)
    assert dss[0].values[0] == pytest.approx(0.179145)


def test_interp_x_y_dfsu2d():
    ds = mikeio.read("tests/testdata/consistency/oresundHD.dfsu")

    x = 350000
    y = 6145000
    dss = ds.interp(x=x, y=y)

    assert dss.geometry.x == x
    assert dss.geometry.y == y

def test_interp_x_y_dfsu3d_not_yet_implemented():
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu")

    x = 350000
    y = 6145000
    
    with pytest.raises(NotImplementedError, match="3d"):
        ds.interp(x=x, y=y)

    


def test_read_dfsu2d():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        items=[0, 1],
        time=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfsu2d_single_time():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        time=-1,
    )

    assert ds.n_timesteps == 1
    assert "time" not in ds.dims

    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        time=[-1],
        keepdims=True,
    )

    assert ds.n_timesteps == 1
    assert "time" in ds.dims


def test_read_dfs_time_selection_str():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = "2018-03"
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 5

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_selection_str_specific():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = "2018-03-08 00:00:00"
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 1
        assert "time" not in dssel.dims

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_selection_list_str():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = ["2018-03-08 00:00", "2018-03-10 00:00"]
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 2

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_selection_pdTimestamp():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = pd.Timestamp("2018-03-08 00:00:00")
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 1
        assert "time" not in dssel.dims

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_selection_pdDatetimeIndex():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = pd.date_range("2018-03-08", end="2018-03-10", freq="D")
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 3

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_selection_datetime():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = datetime(2018, 3, 8, 0, 0, 0)
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 1
        assert "time" not in dssel.dims

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape

        dsr2 = mikeio.read(filename=filename, time=pd.Timestamp(time))
        assert all(dsr2.time == dsr.time)
        assert dsr2.shape == dsr.shape


def test_read_dfs_time_list_datetime():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = [datetime(2018, 3, 8), datetime(2018, 3, 10)]
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 2

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_slice_datetime():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = slice(datetime(2018, 3, 8), datetime(2018, 3, 10))
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 3

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_slice_str():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = slice("2018-03-08", "2018-03-10")
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 3

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_selection_str_comma():

    extensions = ["dfs0", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = "2018-03-08,2018-03-10"
        ds = mikeio.read(filename=filename)
        dssel = ds.sel(time=time)
        assert dssel.n_timesteps == 3

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        dsgetitem = ds[time]
        assert all(dsr.time == dsgetitem.time)
        assert dsr.shape == dsgetitem.shape


def test_read_dfs_time_int():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = -1
        ds = mikeio.read(filename=filename)
        dssel = ds.isel(time=time)
        assert dssel.n_timesteps == 1
        assert "time" not in dssel.dims

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        # integer time selection for DataArray (not Dataset)
        dsgetitem = ds[0][time]
        assert all(dsr[0].time == dsgetitem.time)
        assert dsr[0].shape == dsgetitem.shape


def test_read_dfs_time_list_int():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        time = [0, 1]
        ds = mikeio.read(filename=filename)
        dssel = ds.isel(time=time)
        assert dssel.n_timesteps == 2

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape

        # integer time selection for DataArray (not Dataset)
        dsgetitem = ds[0][time]
        assert all(dsr[0].time == dsgetitem.time)
        assert dsr[0].shape == dsgetitem.shape

def test_read_dfs_time_slice_int():

    extensions = ["dfsu", "dfs2", "dfs1", "dfs0"]
    for ext in extensions:
        filename = f"tests/testdata/consistency/oresundHD.{ext}"
        
        time = slice(1,3) # 1,2 (not 3)
        dssel = mikeio.read(filename=filename).isel(time=time)
        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dssel.n_timesteps == 2

        time = slice(None,-1) # Skip last step
        dssel = mikeio.read(filename=filename).isel(time=time)
        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dssel.n_timesteps == 4
        
        time = slice(1,None) # Skip first step
        dssel = mikeio.read(filename=filename).isel(time=time)
        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dssel.n_timesteps == 4

        dsr = mikeio.read(filename=filename, time=time)
        assert all(dsr.time == dssel.time)
        assert dsr.shape == dssel.shape


def test_filter_items_dfs0():

    ds = mikeio.read("tests/testdata/sw_points.dfs0", items="*Point 42*")
    assert ds.n_items == 15

    for da in ds:
        assert "Point 42" in da.name

    ds = mikeio.read("tests/testdata/sw_points.dfs0", items="*Height*")
    assert ds.n_items == 12

    for da in ds:
        assert "Height" in da.name

    with pytest.raises(KeyError):
        mikeio.read(
            "tests/testdata/sw_points.dfs0", items="height*"
        )  # Note missing wildcard in beginning


def test_filter_items_wildcard_getitem():
    dsall = mikeio.read("tests/testdata/sw_points.dfs0")

    ds = dsall["*Height*"]
    assert ds.n_items == 12


def test_filter_items_dfsu():

    ds = mikeio.read("tests/testdata/wind_north_sea.dfsu", items="*direction*")
    assert ds.n_items == 1


def test_filter_items_dfsu_getitem():

    dsall = mikeio.read("tests/testdata/wind_north_sea.dfsu")
    ds = dsall["*direction*"]
    assert ds.n_items == 1
    assert "direction" in ds[0].name


def test_concat_dfsu3d_single_timesteps_generic_vs_dataset(tmp_path):
    filename = "tests/testdata/basin_3d.dfsu"

    fn_1 = tmp_path / "ts_1.dfsu"
    fn_2 = tmp_path / "ts_2.dfsu"

    mikeio.generic.extract(
        filename, start=0, end=1, outfilename=fn_1
    )  # TODO change start, end to time
    mikeio.generic.extract(
        filename, start=1, end=2, outfilename=fn_2
    )  # TODO change start, end to time

    fn_3 = tmp_path / "ts_all.dfsu"

    mikeio.generic.concat([fn_1, fn_2], fn_3)

    ds4 = mikeio.read(fn_3)

    ds1 = mikeio.read(fn_1)
    ds2 = mikeio.read(fn_2)
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert ds3.start_time == ds4.start_time
    assert ds3.end_time == ds4.end_time
    assert ds3.n_items == ds4.n_items
    assert ds3.n_timesteps == ds4.n_timesteps
