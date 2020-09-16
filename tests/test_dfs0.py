import os
import numpy as np
import pandas as pd
import datetime
import mikeio
from mikeio.dfs0 import Dfs0
from mikeio.eum import TimeStepUnit, EUMType, EUMUnit, ItemInfo
from datetime import timedelta

import pytest


def test_repr():
    filename = os.path.join("tests", "testdata", "da_diagnostic.dfs0")
    dfs = Dfs0(filename)

    text = repr(dfs)

    assert "NonEquidistant" in text


def test_repr_equidistant():
    filename = os.path.join("tests", "testdata", "random.dfs0")
    dfs = Dfs0(filename)

    text = repr(dfs)

    assert "Dfs0" in text
    assert "Equidistant" in text
    assert "NonEquidistant" not in text


def test_simple_write(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfs0")

    data = []

    nt = 100
    d = np.random.random([nt])
    data.append(d)

    dfs = Dfs0()

    dfs.write(filename=filename, data=data)

    assert os.path.exists(filename)


def test_read_units_write_new(tmpdir):

    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    dfs = Dfs0(dfs0file)
    ds = dfs.read()

    # write new file
    dfs.write(tmpfile, ds)

    # Verify that new file has same variables/units as original
    newdfs = Dfs0(tmpfile)
    ds2 = newdfs.read()

    assert ds2.items[0].type == ds.items[0].type
    assert ds2.items[0].unit == ds.items[0].unit


def test_multiple_write(tmpdir):

    filename = os.path.join(tmpdir.dirname, "random.dfs0")

    data = []

    nt = 10
    d1 = np.zeros(nt)
    data.append(d1)
    d2 = np.ones(nt)
    data.append(d2)

    items = [ItemInfo("Zeros"), ItemInfo("Ones")]

    dfs = Dfs0()

    dfs.write(filename=filename, data=data, items=items, title="Zeros and ones")

    assert os.path.exists(filename)


def test_write_timestep_7days(tmpdir):

    filename = os.path.join(tmpdir.dirname, "random.dfs0")

    data = []

    nt = 10
    d1 = np.zeros(nt)
    data.append(d1)
    d2 = np.ones(nt)
    data.append(d2)

    items = [ItemInfo("Zeros"), ItemInfo("Ones")]

    dfs = Dfs0()

    dfs.write(
        filename=filename,
        data=data,
        items=items,
        title="Zeros and ones",
        timeseries_unit=TimeStepUnit.DAY,
        dt=7,
    )

    assert os.path.exists(filename)

    newdfs = Dfs0(filename)
    res = newdfs.read()

    dt = res.time[1] - res.time[0]

    assert dt == timedelta(days=7)
    assert res["Zeros"][-1] == 0.0
    assert res["Ones"][-1] == 1.0


def test_write_equidistant_calendar(tmpdir):

    dfs0file = os.path.join(tmpdir.dirname, "random.dfs0")
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
    dfs.write(
        filename=dfs0file,
        data=data,
        start_time=start_time,
        timeseries_unit=timeseries_unit,
        dt=dt,
        items=items,
        title=title,
        data_value_type=data_value_type,
    )


def test_write_non_equidistant_calendar(tmpdir):
    dfs0file = os.path.join(tmpdir.dirname, "neq.dfs0")
    d1 = np.zeros(1000)
    d2 = np.ones([1000])
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
    dfs.write(
        filename=dfs0file,
        data=data,
        datetimes=time_vector,
        items=items,
        title=title,
        data_value_type=data_value_type,
    )

    assert os.path.exists(dfs0file)


def test_read_equidistant_dfs0_to_dataframe_fixed_freq():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    df = dfs.to_dataframe()

    assert df.index.freq is not None

    df = dfs.to_dataframe(round_time=False)


def test_read_equidistant_dfs0_to_dataframe_unit_in_name():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    df = dfs.to_dataframe(unit_in_name=True)

    assert "meter" in df.columns[0]


def test_read_nonequidistant_dfs0_to_dataframe_no_freq():

    dfs0file = r"tests/testdata/da_diagnostic.dfs0"

    dfs = Dfs0(dfs0file)
    df = dfs.to_dataframe()

    assert df.index.freq is None


def test_read_dfs0_delete_value_conversion():

    dfs0file = r"tests/testdata/da_diagnostic.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read()

    assert np.isnan(ds.data[3][1])

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read()

    assert np.isnan(ds.data[0][2])


def test_read_dfs0_small_value_not_delete_value(tmpdir):

    filename = os.path.join(tmpdir.dirname, "small.dfs0")

    data = []

    d = np.array([0.0, 0.0000001, -0.0001])

    assert np.isclose(d, -1e-35, atol=1e-33).any()
    data.append(d)

    dfs = Dfs0()

    dfs.write(filename=filename, data=data)

    dfs = Dfs0(filename)
    ds = dfs.read()

    assert not np.isnan(ds.data[0]).any()


def test_write_from_data_frame(tmpdir):

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

    assert os.path.exists(filename)

    ds = mikeio.read(filename)

    assert len(ds.items) == 5
    assert ds.items[0].type == EUMType.Concentration
    assert ds.items[0].unit == EUMUnit.gram_per_meter_pow_3


def test_write_from_data_frame_monkey_patched(tmpdir):

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
    assert np.isnan(ds["Average"][3])
    assert ds.time[0].year == 1958


def test_write_from_data_frame_different_types(tmpdir):

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

    dfs = Dfs0(dfs0file)
    ds = dfs.read([1])

    assert len(ds.data) == 1


def test_read_dfs0_single_item_named_access():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    res = dfs.read(items=[1])
    data = res.data

    assert len(data) == 1


def test_read_dfs0_temporal_subset():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read(time_steps=[1, 2])

    assert len(ds.time) == 2
    assert ds.time[0].strftime("%H") == "05"


def test_read_dfs0_single_item_read_by_name():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    res = dfs.read(["NotFun", "VarFun01"])  # reversed order compare to original file
    data = res.data

    assert len(data) == 2
    assert res.items[0].name == "NotFun"
    assert res.items[0].type == EUMType.Water_Level
    assert res.items[0].unit == EUMUnit.meter
    assert repr(res.items[0].unit) == "meter"


def test_read_dfs0_to_dataframe():

    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    pd = dfs.to_dataframe()

    assert np.isnan(pd[pd.columns[0]][2])


def test_read_dfs0_to_matrix():
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read()

    assert len(ds.data) == 2


def test_write_data_with_missing_values(tmpdir):
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = os.path.join(tmpdir.dirname, "random.dfs0")

    dfs = Dfs0(dfs0file)
    ds = dfs.read()

    # Do something with the data
    ds.data[0] = np.zeros_like(ds.data[0])
    ds.data[1] = np.ones_like(ds.data[0])

    # Add some NaNs
    ds.data[1][0:10] = np.nan

    # Overwrite the file
    dfs.write(tmpfile, ds)

    # Write operation does not modify the data
    assert np.isnan(ds.data[1][1])

    moddfs = Dfs0(tmpfile)
    modified = moddfs.read()
    assert np.isnan(modified.data[1][5])

