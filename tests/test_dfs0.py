import os
import numpy as np
import pandas as pd
import datetime
import mikeio
from mikeio.dfs0 import Dfs0
from mikeio.eum import TimeStep, EUMType, EUMUnit, ItemInfo
from datetime import timedelta
from shutil import copyfile
import pytest


def test_simple_create(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs0")

    data = []

    nt = 100
    d = np.random.random([nt])
    data.append(d)

    dfs = Dfs0()

    dfs.create(filename=filename, data=data)

    assert True


def test_read_units_create_new(tmpdir):

    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    copyfile(dfs0file, tmpfile)
    dfs = Dfs0()
    res = dfs.read(tmpfile)
    data = res.data

    # Create new file
    dfs.create(tmpfile, data=data, items=res.items)

    # Verify that new file has same variables/units as original
    ds = dfs.read(tmpfile)

    assert res.items[0].type == ds.items[0].type
    assert res.items[0].unit == ds.items[0].unit


def test_multiple_create():

    dfs0File = r"zeros_ones.dfs0"

    data = []

    nt = 10
    d1 = np.zeros(nt)
    data.append(d1)
    d2 = np.ones(nt)
    data.append(d2)

    items = [ItemInfo("Zeros"), ItemInfo("Ones")]

    dfs = Dfs0()

    dfs.create(filename=dfs0File, data=data, items=items, title="Zeros and ones")

    assert True
    os.remove(dfs0File)


def test_create_timestep_7days():

    dfs0File = r"zeros_ones.dfs0"

    data = []

    nt = 10
    d1 = np.zeros(nt)
    data.append(d1)
    d2 = np.ones(nt)
    data.append(d2)

    items = [ItemInfo("Zeros"), ItemInfo("Ones")]

    dfs = Dfs0()

    dfs.create(
        filename=dfs0File,
        data=data,
        items=items,
        title="Zeros and ones",
        timeseries_unit=TimeStep.DAY,
        dt=7,
    )

    assert True

    res = dfs.read(dfs0File)

    dt = res.time[1] - res.time[0]

    assert dt == timedelta(days=7)

    os.remove(dfs0File)


def test_create_equidistant_calendar():

    dfs0file = r"random.dfs0"
    d1 = np.random.random([1000])
    d2 = np.random.random([1000])
    data = []
    data.append(d1)
    data.append(d2)
    start_time = datetime.datetime(2017, 1, 1)
    timeseries_unit = 1402
    title = "Hello Test"
    items = [ItemInfo("VarFun01", 100000, 1000), ItemInfo("NotFun", 100000, 1000)]

    data_value_type = [0, 1]  # TODO add data_value_type to ItemInfo
    dt = 5
    dfs = Dfs0()
    dfs.create(
        filename=dfs0file,
        data=data,
        start_time=start_time,
        timeseries_unit=timeseries_unit,
        dt=dt,
        items=items,
        title=title,
        data_value_type=data_value_type,
    )

    os.remove(dfs0file)
    assert True


def test_create_non_equidistant_calendar():
    dfs0file = r"neq.dfs0"
    d1 = np.random.random([1000])
    d2 = np.random.random([1000])
    data = []
    data.append(d1)
    data.append(d2)
    start_time = datetime.datetime(2017, 1, 1)
    time_vector = []
    for i in range(1000):
        time_vector.append(start_time + datetime.timedelta(hours=i * 0.1))
    title = "Hello Test"
    items = [ItemInfo("VarFun01", 100000, 1000), ItemInfo("NotFun", 100000, 1000)]
    data_value_type = [0, 1]

    dfs = Dfs0()
    dfs.create(
        filename=dfs0file,
        data=data,
        datetimes=time_vector,
        items=items,
        title=title,
        data_value_type=data_value_type,
    )

    assert True
    os.remove(dfs0file)


def test_read_equidistant_dfs0_to_dataframe_fixed_freq():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    df = dfs.to_dataframe(dfs0file)

    assert df.index.freq is not None


def test_read_nonequidistant_dfs0_to_dataframe_no_freq():

    dfs0file = r"tests/testdata/da_diagnostic.dfs0"

    dfs = Dfs0()
    df = dfs.to_dataframe(dfs0file)

    assert df.index.freq is None


def test_read_dfs0_delete_value_conversion():

    dfs0file = r"tests/testdata/da_diagnostic.dfs0"

    dfs = Dfs0()
    ds = dfs.read(dfs0file)

    assert np.isnan(ds.data[3][1])

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    ds = dfs.read(dfs0file)

    assert np.isnan(ds.data[0][2])


def test_read_dfs0_small_value_not_delete_value(tmpdir):

    filename = os.path.join(tmpdir.dirname, "small.dfs0")

    data = []

    d = np.array([0.0, 0.0000001, -0.0001])

    assert np.isclose(d, -1e-35, atol=1e-33).any()
    data.append(d)

    dfs = Dfs0()

    dfs.create(filename=filename, data=data)

    ds = dfs.read(filename)

    assert not np.isnan(ds.data[0]).any()


def test_create_from_data_frame(tmpdir):

    df = pd.read_csv(
        "tests/testdata/co2-mm-mlo.csv",
        parse_dates=True,
        index_col="Date",
        na_values=-99.99,
    )

    filename = os.path.join(tmpdir.dirname, "dataframe.dfs0")
    Dfs0.from_dataframe(
        df, filename, itemtype=EUMType.Concentration, unit=EUMUnit.gram_per_meter_pow_3
    )  # Could not find better type

    ds = mikeio.read(filename)

    assert len(ds.items) == 5
    assert ds.items[0].type == EUMType.Concentration
    assert ds.items[0].unit == EUMUnit.gram_per_meter_pow_3


def test_create_from_data_frame_monkey_patched(tmpdir):

    df = pd.read_csv(
        "tests/testdata/co2-mm-mlo.csv",
        parse_dates=True,
        index_col="Date",
        na_values=-99.99,
    )

    filename = os.path.join(tmpdir.dirname, "dataframe.dfs0")

    df.to_dfs0(
        filename, itemtype=EUMType.Concentration, unit=EUMUnit.gram_per_meter_pow_3
    )

    ds = mikeio.read(filename)

    assert len(ds.items) == 5
    assert ds.items[0].type == EUMType.Concentration
    assert ds.items[0].unit == EUMUnit.gram_per_meter_pow_3


def test_create_from_data_frame_different_types(tmpdir):

    df = pd.read_csv(
        "tests/testdata/co2-mm-mlo.csv",
        parse_dates=True,
        index_col="Date",
        na_values=-99.99,
    )

    df = df[["Average", "Trend"]]

    filename = os.path.join(tmpdir.dirname, "dataframe.dfs0")

    items = [
        ItemInfo("Average", EUMType.Concentration, EUMUnit.gram_per_meter_pow_3),
        ItemInfo("Trend", EUMType.Undefined),
    ]

    Dfs0.from_dataframe(df, filename, items=items)

    ds = mikeio.read(filename)

    assert len(ds.items) == 2
    assert ds.items[0].type == EUMType.Concentration
    assert ds.items[0].unit == EUMUnit.gram_per_meter_pow_3

    assert ds.items[1].type == EUMType.Undefined
    assert ds.items[1].unit == EUMUnit.undefined


def test_read_dfs0_single_item():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    (data, t, items) = dfs.read(dfs0file, item_numbers=[1])

    assert len(data) == 1


def test_read_dfs0_single_item_named_access():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    res = dfs.read(dfs0file, item_numbers=[1])
    data = res.data

    assert len(data) == 1


def test_read_dfs0_single_item_read_by_name():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    res = dfs.read(
        dfs0file, item_names=["NotFun", "VarFun01"]
    )  # reversed order compare to original file
    data = res.data

    assert len(data) == 2
    assert res.items[0].name == "NotFun"
    assert res.items[0].type == EUMType.Water_Level
    assert res.items[0].unit == EUMUnit.meter
    assert repr(res.items[0].unit) == "meter"


def test_read_dfs0_to_dataframe():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    pd = dfs.to_dataframe(dfs0file)

    assert np.isnan(pd[pd.columns[0]][2])


def test_read_dfs0_to_matrix():
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    (data, t, items) = dfs.read(filename=dfs0file)

    assert len(data) == 2


def test_write(tmpdir):
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    copyfile(dfs0file, tmpfile)
    dfs = Dfs0()
    res = dfs.read(tmpfile)
    data = res.data

    # Do something with the data
    data[0] = np.zeros_like(data[0])
    data[1] = np.ones_like(data[0])

    # Overwrite the file
    dfs.write(tmpfile, data)

def test_write_data_with_missing_values(tmpdir):
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    copyfile(dfs0file, tmpfile)
    dfs = Dfs0()
    res = dfs.read(tmpfile)
    data = res.data

    # Do something with the data
    data[0] = np.zeros_like(data[0])
    data[1] = np.ones_like(data[0])

    # Add some NaNs
    data[1][0:10] = np.nan

    # Overwrite the file
    dfs.write(tmpfile, data)

    # Write operation does not modify the data
    assert(np.isnan(data[1][1]))

    modified = dfs.read(tmpfile)
    assert(np.isnan(modified.data[1][5]))



def test_write_wrong_n_items(tmpdir):
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    copyfile(dfs0file, tmpfile)
    dfs = Dfs0()
    res = dfs.read(tmpfile)
    data = res.data

    # One item too many...
    data[0] = np.zeros_like(data[0])
    data[1] = np.ones_like(data[0])
    data.append(np.ones_like(data[0]))

    # Overwrite the file
    with pytest.raises(Exception):
        dfs.write(tmpfile, data)


def test_write_no_existing_file():
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0()
    res = dfs.read(dfs0file)
    data = res.data

    # Overwrite the file
    with pytest.raises(Exception):
        dfs.write("not_a_file", data)


def test_read_dfs0_main_module():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = mikeio.Dfs0()
    (data, t, items) = dfs.read(dfs0file, item_numbers=[1])

    assert len(data) == 1
