import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import pytest

import mikeio

from mikeio.dataset import Dataset
from mikeio.dfs2 import Dfs2
from mikeio.eum import EUMType, ItemInfo, EUMUnit
from mikeio.custom_exceptions import (
    DataDimensionMismatch,
    ItemsError,
)


@pytest.fixture
def dfs2_random():
    filepath = Path("tests/testdata/random.dfs2")
    return Dfs2(filepath)


@pytest.fixture
def dfs2_random_2items():
    filepath = Path("tests/testdata/random_two_item.dfs2")
    return Dfs2(filepath)


@pytest.fixture
def dfs2_pt_spectrum():
    filepath = Path("tests/testdata/pt_spectra.dfs2")
    return Dfs2(filepath)


@pytest.fixture
def dfs2_gebco():
    filepath = Path("tests/testdata/gebco_sound.dfs2")
    return Dfs2(filepath)


@pytest.fixture
def dfs2_gebco_rotate():
    filepath = Path("tests/testdata/gebco_sound_crop_rotate.dfs2")
    return Dfs2(filepath)


def test_simple_write(tmp_path):

    filepath = tmp_path / "simple.dfs2"

    data = []

    nt = 100
    nx = 20
    ny = 5
    d = np.random.random([nt, ny, nx])

    data.append(d)

    dfs = Dfs2()

    dfs.write(filepath, data=data)

    newdfs = Dfs2(filepath)

    ds = newdfs.read()

    assert len(ds) == 1
    assert ds.items[0].type == EUMType.Undefined


def test_write_inconsistent_shape(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

    data = []

    nt = 100
    nx = 20
    ny = 5
    d1 = np.random.random([nt, ny, nx])
    d2 = np.random.random([nt, ny, nx + 1])

    data = Dataset(
        data=[d1, d2], time=pd.date_range(start="2000", periods=nt, freq="H")
    )

    dfs = Dfs2()

    with pytest.raises(DataDimensionMismatch):
        dfs.write(filename=filename, data=data)


def test_write_single_item(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

    data = []
    d = np.random.random([100, 2, 3])
    d[10, :, :] = np.nan
    d[11, :, :] = 0
    d[12, :, :] = 1e-10
    d[13, :, :] = 1e10

    data.append(d)
    # >>> from pyproj import Proj
    # >>> utm = Proj(32633)
    # >>> utm(12.0, 55.0)
    # east = 308124
    # north = 6098907

    ds = Dataset(
        data=data,
        time=pd.date_range("2012-1-1", freq="s", periods=100),
        items=[ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)],
    )

    dfs = Dfs2()

    dfs.write(
        filename=filename,
        data=ds,
        coordinate=["UTM-33", 12.0, 55.0, 0.0],
        dx=100,
        dy=200,
        title="test dfs2",
    )

    newdfs = Dfs2(filename)
    assert newdfs.projection_string == "UTM-33"
    assert pytest.approx(newdfs.longitude) == 12.0
    assert pytest.approx(newdfs.latitude) == 55.0
    assert newdfs.dx == 100.0
    assert newdfs.dy == 200.0


def test_read(dfs2_random):

    dfs = dfs2_random
    ds = dfs.read(["testing water level"])
    data = ds.data[0]
    assert data[0, 88, 0] == 0
    assert np.isnan(data[0, 89, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_bad_item(dfs2_random):
    dfs = dfs2_random
    with pytest.raises(ItemsError):
        dfs.read(items=100)


def test_read_temporal_subset_slice():

    filename = r"tests/testdata/eq.dfs2"
    dfs = Dfs2(filename)
    ds = dfs.read(time_steps=slice("2000-01-01 00:00", "2000-01-01 12:00"))

    assert len(ds.time) == 13


def test_read_numbered_access(dfs2_random_2items):

    dfs = dfs2_random_2items

    res = dfs.read([1])

    assert np.isnan(res.data[0][0, 0, 0])
    assert res.time is not None
    assert res.items[0].name == "Untitled"


def test_properties_pt_spectrum(dfs2_pt_spectrum):
    dfs = dfs2_pt_spectrum
    assert dfs.x0 == pytest.approx(0.055)
    assert dfs.y0 == 0
    assert dfs.dx == pytest.approx(1.1)
    assert dfs.dy == 22.5
    assert dfs.nx == 25
    assert dfs.ny == 16
    assert dfs.longitude == 0
    assert dfs.latitude == 0
    assert dfs.orientation == 0
    assert dfs.n_items == 1
    assert dfs.n_timesteps == 31


def test_properties_rotated(dfs2_gebco_rotate):
    dfs = dfs2_gebco_rotate
    assert dfs.x0 == 0
    assert dfs.y0 == 0
    assert dfs.dx == pytest.approx(0.00416667)
    assert dfs.dy == pytest.approx(0.00416667)
    assert dfs.nx == 140
    assert dfs.ny == 150
    assert dfs.longitude == pytest.approx(12.2854167)
    assert dfs.latitude == pytest.approx(55.3270833)
    assert dfs.orientation == 45
    assert dfs.n_items == 1
    assert dfs.n_timesteps == 1


def test_write_selected_item_to_new_file(dfs2_random_2items, tmpdir):

    dfs = dfs2_random_2items

    outfilename = os.path.join(tmpdir.dirname, "simple.dfs2")

    ds = dfs.read(["Untitled"])

    dfs.write(outfilename, ds)

    dfs2 = Dfs2(outfilename)

    ds2 = dfs2.read()

    assert len(ds2) == 1
    assert ds.items[0].name == "Untitled"
    assert dfs.start_time == dfs2.start_time
    assert dfs.end_time == dfs2.end_time
    assert dfs.projection_string == dfs2.projection_string
    assert dfs.longitude == dfs2.longitude
    assert dfs.latitude == dfs2.latitude
    assert dfs.orientation == dfs2.orientation


def test_repr(dfs2_gebco):

    text = repr(dfs2_gebco)

    assert "Dfs2" in text
    assert "Items" in text
    assert "dx" in text


def test_repr_empty():

    dfs = Dfs2()

    text = repr(dfs)

    assert "Dfs2" in text


def test_repr_time(dfs2_random):

    dfs = dfs2_random
    text = repr(dfs)

    assert "Dfs2" in text
    assert "Items" in text
    assert "dx" in text
    assert "steps" in text


def test_write_modified_data_to_new_file(dfs2_gebco, tmpdir):

    dfs = dfs2_gebco

    outfilename = os.path.join(tmpdir.dirname, "mod.dfs2")

    ds = dfs.read()

    ds.data[0] = ds.data[0] + 10.0

    dfs.write(outfilename, ds)

    dfsmod = Dfs2(outfilename)

    assert dfs._longitude == dfsmod._longitude


def test_read_some_time_step(dfs2_random_2items):

    dfs = dfs2_random_2items
    res = dfs.read(time_steps=[1, 2])

    assert res.data[0].shape[0] == 2
    assert len(res.time) == 2


def test_interpolate_non_equidistant_data(tmpdir):

    filename = r"tests/testdata/eq.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read(time_steps=[0, 2, 3, 6])  # non-equidistant dataset

    assert not ds.is_equidistant

    ds2 = ds.interp_time(dt=3600)

    assert ds2.is_equidistant

    outfilename = os.path.join(tmpdir.dirname, "interpolated_time.dfs2")

    dfs.write(outfilename, ds2)

    dfs2 = Dfs2(outfilename)
    assert dfs2.timestep == 3600.0

    ds3 = dfs2.read()

    assert ds3.is_equidistant


def test_write_some_time_step(tmpdir):

    filename = r"tests/testdata/waves.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read(time_steps=[1, 2])

    assert ds.data[0].shape[0] == 2
    assert len(ds.time) == 2

    assert dfs.timestep == 86400.0
    assert dfs.start_time.day == 1

    outfilename = os.path.join(tmpdir.dirname, "waves_subset.dfs2")

    dfs.write(outfilename, ds)

    dfs2 = Dfs2(outfilename)
    assert dfs2.timestep == 86400.0
    assert dfs2.start_time.day == 2


# TODO when we have the .sel method
# def test_find_by_x_y():

#  ds = mikeio.read("tests/testdata/gebco_sound.dfs2")

#  assert ds.Elevation.sel(x=12.74792, y=55.865) == -43.0


def test_write_accumulated_datatype(tmpdir):
    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

    data = []
    d = np.random.random([100, 2, 3])
    data.append(d)

    ds = Dataset(
        data=data,
        time=pd.date_range("2021-1-1", periods=100, freq="s"),
        items=[
            ItemInfo(
                "testing water level",
                EUMType.Water_Level,
                EUMUnit.meter,
                data_value_type="MeanStepBackward",
            )
        ],
    )

    dfs = Dfs2()

    dfs.write(
        filename=filename,
        data=ds,
        title="test dfs2",
    )

    newdfs = Dfs2(filename)
    assert newdfs.items[0].data_value_type == 3


def test_write_default_datatype(tmpdir):
    filename = os.path.join(tmpdir.dirname, "simple.dfs2")

    data = []
    d = np.random.random([100, 2, 3])
    data.append(d)

    dfs = Dfs2()

    dfs.write(
        filename=filename,
        data=data,
        start_time=datetime.datetime(2012, 1, 1),
        dt=12,
        items=[ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)],
        title="test dfs2",
    )

    newdfs = Dfs2(filename)
    assert newdfs.items[0].data_value_type == 0


def test_write_NonEqCalendarAxis(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs2")
    data = []
    d = np.random.random([6, 5, 10])
    d[1, :, :] = np.nan
    d[2, :, :] = 0
    d[3, 3:, :] = 2
    d[4, :, 4:] = 5
    data.append(d)
    # east = 308124 # Not supported, supply lat/lon of origin also for projected coords
    # north = 6098907
    orientation = 0
    dateTime = [
        datetime.datetime(2012, 1, 1),
        datetime.datetime(2012, 1, 4),
        datetime.datetime(2012, 1, 5),
        datetime.datetime(2012, 1, 10),
        datetime.datetime(2012, 1, 15),
        datetime.datetime(2012, 1, 28),
    ]
    dfs = Dfs2()
    dfs.write(
        filename=filename,
        data=data,
        # start_time=datetime.datetime(2012, 1, 1),
        # dt=12,
        datetimes=dateTime,
        items=[ItemInfo("testing water level", EUMType.Water_Level, EUMUnit.meter)],
        coordinate=["UTM-33", 12.0, 55.0, orientation],
        dx=100,
        dy=200,
        title="test dfs2",
    )

    newdfs = Dfs2(filename)
    assert newdfs.projection_string == "UTM-33"
    assert pytest.approx(newdfs.longitude) == 12.0
    assert pytest.approx(newdfs.latitude) == 55.0
    assert newdfs.dx == 100.0
    assert newdfs.dy == 200.0
    assert newdfs._is_equidistant == False


def test_write_non_equidistant_data(tmpdir):

    filename = r"tests/testdata/eq.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read(time_steps=[0, 2, 3, 6])  # non-equidistant dataset

    assert not ds.is_equidistant

    outfilename = os.path.join(tmpdir.dirname, "neq_from_dataset.dfs2")

    dfs.write(outfilename, ds)

    dfs2 = Dfs2(outfilename)
    ds3 = dfs2.read()

    assert not ds3.is_equidistant


def test_incremental_write_from_dfs2(tmpdir):
    "Useful for writing datasets with many timesteps to avoid problems with out of memory"

    sourcefilename = "tests/testdata/eq.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "appended.dfs2")
    dfs = Dfs2(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time_steps=[0])

    dfs_to_write = Dfs2()
    dfs_to_write.write(outfilename, ds, dt=dfs.timestep, keep_open=True)

    for i in range(1, nt):
        ds = dfs.read(time_steps=[i])
        dfs_to_write.append(ds)

    dfs_to_write.close()

    newdfs = Dfs2(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_incremental_write_from_dfs2_context_manager(tmpdir):
    "Useful for writing datasets with many timesteps to avoid problems with out of memory"

    sourcefilename = "tests/testdata/eq.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "appended.dfs2")
    dfs = Dfs2(sourcefilename)

    nt = dfs.n_timesteps

    ds = dfs.read(time_steps=[0])

    dfs_to_write = Dfs2()
    with dfs_to_write.write(outfilename, ds, dt=dfs.timestep, keep_open=True) as f:

        for i in range(1, nt):
            ds = dfs.read(time_steps=[i])
            f.append(ds)

        # dfs_to_write.close() # called automagically by context manager

    newdfs = Dfs2(outfilename)
    assert dfs.start_time == newdfs.start_time
    assert dfs.timestep == newdfs.timestep
    assert dfs.end_time == newdfs.end_time


def test_dfs2_plot():

    dfs = Dfs2("tests/testdata/random.dfs2")
    ds = dfs.read(items=0)
    # aggregate in time
    dss = ds.aggregate(axis="time", func=np.std)

    assert len(dss) == 1
    assert dss[0].shape[0] == 1

    dfs.plot(dss)

    assert True


def test_spatial_aggregation_dfs2_to_dfs0(tmp_path):

    outfilename = tmp_path / "waves_max.dfs0"

    ds = mikeio.read("tests/testdata/waves.dfs2")
    ds_max = ds.nanmax(axis="space")
    ds_max.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == ds.n_timesteps
    assert dsnew.n_items == ds.n_items
