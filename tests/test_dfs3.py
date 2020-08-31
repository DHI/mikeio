import os
from shutil import copyfile
from datetime import datetime
import numpy as np
from mikeio.dfs3 import Dfs3
from mikeio.eum import EUMType, ItemInfo, TimeStep


def test_read_dfs3():
    dfs = Dfs3("tests/testdata/Grid1.dfs3")
    ds = dfs.read()

    assert len(ds.data) == 2
    assert len(ds.time) == 30
    assert ds.data[0].shape == (30, 10, 10, 10)  # t  # z  # y  # x
    assert ds.items[0].name == "Untitled"


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
        timeseries_unit=TimeStep.SECOND,
        dt=3600.0,
        items=items,
        coordinate=["UTM-33", 450000, 560000, 0],
        length_x=0.1,
        length_y=0.1,
        length_z=10.0,
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
        timeseries_unit=TimeStep.SECOND,
        dt=(ds.time[1] - ds.time[0]).total_seconds(),
        items=items,
        coordinate=["LONG/LAT", 5, 10, 0],
        length_x=0.1,
        length_y=0.1,
        title=title,
    )
