import os
from shutil import copyfile
from datetime import datetime
import numpy as np
from mikeio.dfs3 import Dfs3
from mikeio.eum import EUMType, ItemInfo


def test_read_dfs3():
    dfs = Dfs3("tests/testdata/Grid1.dfs3")
    ds = dfs.read()
    assert ds.n_items == 2
    assert ds.n_timesteps == 30
    da = ds[0]
    assert da.shape == (30, 10, 10, 10)  # t  # z  # y  # x
    assert da.name == "Item 1"


def test_write_single_item(tmpdir):
    outfilename = os.path.join(tmpdir.dirname, "simple.dfs3")
    start_time = datetime(2012, 1, 1)
    items = [ItemInfo(EUMType.Relative_moisture_content)]
    data = []
    #                     t  , z, y, x
    d = np.random.random([20, 2, 5, 10])
    d[:, 0, 0, 0] = 0.0
    data.append(d)
    title = "test dfs3"
    dfs = Dfs3()
    dfs.write(
        filename=outfilename,
        data=data,
        start_time=start_time,
        dt=3600.0,
        items=items,
        coordinate=["UTM-33", 450000, 560000, 0],
        dx=0.1,
        dy=0.1,
        dz=10.0,
        title=title,
    )


def test_read_write(tmpdir):
    dfs = Dfs3("tests/testdata/Grid1.dfs3")
    ds = dfs.read()
    outfilename = os.path.join(tmpdir.dirname, "rw.dfs3")
    start_time = datetime(2012, 1, 1)
    items = ds.items
    data = ds.data
    title = "test dfs3"
    dfs = Dfs3()
    dfs.write(
        filename=outfilename,
        data=data,
        start_time=ds.time[0],
        dt=(ds.time[1] - ds.time[0]).total_seconds(),
        items=items,
        coordinate=["LONG/LAT", 5, 10, 0],
        dx=0.1,
        dy=0.1,
        title=title,
    )


def test_dfs3_projection():
    dfs = Dfs3("tests/testdata/test_dfs3.dfs3")
    assert dfs.projection_string == "LONG/LAT"
    assert dfs.dx == 0.25
    assert dfs.dy == 0.25
    assert dfs.dz == 1.0


def test_dfs3_get_bottom_data():
    dfs = Dfs3("tests/testdata/test_dfs3.dfs3")
    data_bottom = dfs.get_bottom_values()
    assert len(data_bottom) > 0


def test_read_dfs3_timesteps_data():
    dfs = Dfs3("tests/testdata/test_dfs3.dfs3")
    dfs.read(time_steps=[1])
