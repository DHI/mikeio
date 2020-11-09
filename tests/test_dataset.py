import os
from datetime import datetime
from dateutil.rrule import rrule, SECONDLY, HOURLY, DAILY
import numpy as np
import pandas as pd
import pytest
from mikeio import Dataset, Dfsu, Dfs2
from mikeio.eum import EUMType, ItemInfo, EUMUnit


def _get_time(nt):
    return list(rrule(freq=SECONDLY, count=nt, dtstart=datetime(2000, 1, 1)))


def test_get_names():

    data = []
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.items[0].name == "Foo"
    assert ds.items[0].type == EUMType.Undefined
    assert repr(ds.items[0].unit) == "undefined"


def test_select_subset_isel():

    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    selds = ds.isel(10, axis=1)

    assert len(selds.items) == 2
    assert len(selds.data) == 2
    assert selds["Foo"].shape == (100, 30)
    assert selds["Foo"][0, 0] == 2.0
    assert selds["Bar"][0, 0] == 3.0


def test_select_temporal_subset_by_idx():

    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    selds = ds.isel([0, 1, 2], axis=0)

    assert len(selds) == 2
    assert selds["Foo"].shape == (3, 100, 30)


def test_temporal_subset_fancy():

    nt = (24 * 31) + 1
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    data = [d1, d2]

    time = list(rrule(freq=HOURLY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    assert ds.time[0].hour == 0
    assert ds.time[-1].hour == 0

    selds = ds["2000-01-01 00:00":"2000-01-02 00:00"]

    assert len(selds) == 2
    assert selds["Foo"].shape == (25, 100, 30)

    selds = ds[:"2000-01-02 00:00"]
    assert selds["Foo"].shape == (25, 100, 30)

    selds = ds["2000-01-31 00:00":]
    assert selds["Foo"].shape == (25, 100, 30)

    selds = ds["2000-01-30":]
    assert selds["Foo"].shape == (49, 100, 30)


def test_subset_with_datetime_is_not_supported():
    nt = (24 * 31) + 1
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    data = [d1, d2]

    time = list(rrule(freq=HOURLY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    with pytest.raises(ValueError):
        ds[datetime(2000, 1, 1)]


def test_select_item_by_name():
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    foo_data = ds["Foo"]
    assert foo_data[0, 10, 0] == 2.0


def test_select_multiple_items_by_name():
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    d3 = np.zeros([nt, 100, 30]) + 3.0

    data = [d1, d2, d3]

    time = _get_time(nt)
    # items = [ItemInfo("Foo"), ItemInfo("Bar"), ItemInfo("Baz")]
    items = [ItemInfo(x) for x in ["Foo", "Bar", "Baz"]]
    ds = Dataset(data, time, items)

    assert len(ds) == 3  # Length of a dataset is the number of items

    newds = ds[["Baz", "Foo"]]
    assert newds.items[0].name == "Baz"
    assert newds.items[1].name == "Foo"
    assert newds["Foo"][0, 10, 0] == 1.5

    assert len(newds) == 2


def test_select_multiple_items_by_index():
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    d3 = np.zeros([nt, 100, 30]) + 3.0

    data = [d1, d2, d3]

    time = _get_time(nt)
    items = [ItemInfo(x) for x in ["Foo", "Bar", "Baz"]]
    ds = Dataset(data, time, items)

    assert len(ds) == 3  # Length of a dataset is the number of items

    newds = ds[[2, 0]]
    assert newds.items[0].name == "Baz"
    assert newds.items[1].name == "Foo"
    assert newds["Foo"][0, 10, 0] == 1.5

    assert len(newds) == 2


def test_select_item_by_iteminfo():
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    foo_item = items[0]

    foo_data = ds[foo_item]
    assert foo_data[0, 10, 0] == 2.0


def test_select_subset_isel_multiple_idxs():

    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    selds = ds.isel([10, 15], axis=1)

    assert len(selds.items) == 2
    assert len(selds.data) == 2
    assert selds["Foo"].shape == (100, 2, 30)


def test_to_dataframe():

    nt = 100
    d1 = np.zeros([nt])
    d2 = np.zeros([nt])

    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)
    df = ds.to_dataframe()

    assert list(df.columns) == ["Foo", "Bar"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_multidimensional_to_dataframe_no_supported():

    nt = 100
    d1 = np.zeros([nt, 2])

    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset([d1], time, items)

    with pytest.raises(ValueError):
        ds.to_dataframe()


def test_get_data():

    data = []
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.data[0].shape == (100, 100, 30)


def test_interp_time():

    data = []
    nt = 4
    d = np.zeros([nt, 10, 3])
    d[1] = 2.0
    d[3] = 4.0
    data.append(d)
    time = list(rrule(freq=DAILY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.data[0].shape == (nt, 10, 3)

    dsi = ds.interp_time(dt=3600)

    assert ds.time[0] == dsi.time[0]
    assert dsi.data[0].shape == (73, 10, 3)


def test_interp_time_to_other_dataset():

    # Arrange
    ## Dataset 1
    nt = 4
    data = [np.zeros([nt, 10, 3])]
    time = list(rrule(freq=DAILY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo")]
    ds1 = Dataset(data, time, items)
    assert ds1.data[0].shape == (nt, 10, 3)

    ## Dataset 2
    nt = 12
    data = [np.ones([nt, 10, 3])]
    time = list(rrule(freq=HOURLY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo")]
    ds2 = Dataset(data, time, items)

    # Act
    ## interp
    dsi = ds1.interp_time(dt=ds2.time)

    # Assert
    assert dsi.time[0] == ds2.time[0]
    assert dsi.time[-1] == ds2.time[-1]
    assert len(dsi.time) == len(ds2.time)
    assert dsi.data[0].shape[0] == ds2.data[0].shape[0]


def test_extrapolate():
    # Arrange
    ## Dataset 1
    nt = 2
    data = [np.zeros([nt, 10, 3])]
    time = list(rrule(freq=DAILY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo")]
    ds1 = Dataset(data, time, items)
    assert ds1.data[0].shape == (nt, 10, 3)

    ## Dataset 2 partly overlapping with Dataset 1
    nt = 3
    data = [np.ones([nt, 10, 3])]
    time = list(rrule(freq=HOURLY, count=nt, dtstart=datetime(2000, 1, 2)))
    items = [ItemInfo("Foo")]
    ds2 = Dataset(data, time, items)

    # Act
    ## interp
    dsi = ds1.interp_time(dt=ds2.time, fill_value=1.0)

    # Assert
    assert dsi.time[0] == ds2.time[0]
    assert dsi.time[-1] == ds2.time[-1]
    assert len(dsi.time) == len(ds2.time)
    assert dsi.data[0][0] == pytest.approx(0.0)
    assert dsi.data[0][1] == pytest.approx(1.0)  # filled
    assert dsi.data[0][2] == pytest.approx(1.0)  # filled


def test_extrapolate_not_allowed():
    ## Dataset 1
    nt = 2
    data = [np.zeros([nt, 10, 3])]
    time = list(rrule(freq=DAILY, count=nt, dtstart=datetime(2000, 1, 1)))
    items = [ItemInfo("Foo")]
    ds1 = Dataset(data, time, items)
    assert ds1.data[0].shape == (nt, 10, 3)

    ## Dataset 2 partly overlapping with Dataset 1
    nt = 3
    data = [np.ones([nt, 10, 3])]
    time = list(rrule(freq=HOURLY, count=nt, dtstart=datetime(2000, 1, 2)))
    items = [ItemInfo("Foo")]
    ds2 = Dataset(data, time, items)

    with pytest.raises(ValueError):
        dsi = ds1.interp_time(dt=ds2.time, fill_value=1.0, extrapolate=False)


def test_get_data_2():

    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert data[0].shape == (100, 100, 30)


def test_get_data_name():

    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds["Foo"].shape == (100, 100, 30)


def test_get_bad_name():
    nt = 100
    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    with pytest.raises(Exception):
        ds["BAR"]


def test_head():

    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    dshead = ds.head()

    assert len(dshead.time) == 5
    assert ds.time[0] == dshead.time[0]

    dshead10 = ds.head(n=10)

    assert len(dshead10.time) == 10


def test_head_small_dataset():

    nt = 2
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    dshead = ds.head()

    assert len(dshead.time) == nt


def test_tail():

    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    dstail = ds.tail()

    assert len(dstail.time) == 5
    assert ds.time[-1] == dstail.time[-1]

    dstail10 = ds.tail(n=10)

    assert len(dstail10.time) == 10


def test_thin():

    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    dsthin = ds.thin(2)

    assert len(dsthin.time) == 50


def test_tail_small_dataset():
    nt = 2
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    dstail = ds.tail()

    assert len(dstail.time) == nt


def test_flipud():

    nt = 2
    d = np.random.random([nt, 100, 30])
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset([d], time, items)

    dsud = ds.copy()
    dsud.flipud()

    assert dsud.shape == ds.shape
    assert dsud["Foo"][0, 0, 0] == ds["Foo"][0, -1, 0]


def test_aggregation_workflows(tmpdir):
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(["Surface elevation", "Current speed"])
    ds2 = ds.max()

    outfilename = os.path.join(tmpdir.dirname, "max.dfs0")
    ds2.to_dfs0(outfilename)
    assert os.path.isfile(outfilename)

    ds3 = ds.min()

    outfilename = os.path.join(tmpdir.dirname, "min.dfs0")
    ds3.to_dfs0(outfilename)
    assert os.path.isfile(outfilename)


def test_aggregations(tmpdir):
    filename = "tests/testdata/gebco_sound.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read()

    for axis in [0, 1, 2]:
        ds.mean(axis=axis)
        ds.nanmean(axis=axis)
        ds.nanmin(axis=axis)
        ds.nanmax(axis=axis)

    assert True


def test_weighted_average(tmpdir):
    filename = "tests/testdata/HD2D.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read(["Surface elevation", "Current speed"])

    area = dfs.get_element_area()
    ds2 = ds.average(weights=area)

    outfilename = os.path.join(tmpdir.dirname, "average.dfs0")
    ds2.to_dfs0(outfilename)
    assert os.path.isfile(outfilename)


def test_copy():
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    data = [d1, d2]

    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = Dataset(data, time, items)

    assert len(ds.items) == 2
    assert len(ds.data) == 2
    assert ds.items[0].name == "Foo"

    ds2 = ds.copy()

    ds2.items[0].name = "New name"

    assert ds2.items[0].name == "New name"
    assert ds.items[0].name == "Foo"


def test_default_type():

    item = ItemInfo("Foo")
    assert item.type == EUMType.Undefined
    assert repr(item.unit) == "undefined"


def test_int_is_valid_type_info():

    item = ItemInfo("Foo", 100123)
    assert item.type == EUMType.Viscosity

    item = ItemInfo("U", 100002)
    assert item.type == EUMType.Wind_Velocity


def test_int_is_valid_unit_info():

    item = ItemInfo("U", 100002, 2000)
    assert item.type == EUMType.Wind_Velocity
    assert item.unit == EUMUnit.meter_per_sec
    assert repr(item.unit) == "meter per sec"  # TODO replace _per_ with /


def test_default_unit_from_type():

    item = ItemInfo("Foo", EUMType.Water_Level)
    assert item.type == EUMType.Water_Level
    assert item.unit == EUMUnit.meter
    assert repr(item.unit) == "meter"

    item = ItemInfo("Tp", EUMType.Wave_period)
    assert item.type == EUMType.Wave_period
    assert item.unit == EUMUnit.second
    assert repr(item.unit) == "second"

    item = ItemInfo("Temperature", EUMType.Temperature)
    assert item.type == EUMType.Temperature
    assert item.unit == EUMUnit.degree_Celsius
    assert repr(item.unit) == "degree Celsius"


def test_default_name_from_type():

    item = ItemInfo(EUMType.Current_Speed)
    assert item.name == "Current Speed"
    assert item.unit == EUMUnit.meter_per_sec

    item2 = ItemInfo(EUMType.Current_Direction, EUMUnit.degree)
    assert item2.unit == EUMUnit.degree
    item3 = ItemInfo(
        "Current direction (going to)", EUMType.Current_Direction, EUMUnit.degree
    )
    assert item3.type == EUMType.Current_Direction
    assert item3.unit == EUMUnit.degree


def test_iteminfo_string_type_should_fail_with_helpful_message():

    with pytest.raises(ValueError):

        item = ItemInfo("Water level", "Water level")


def test_item_search():

    res = EUMType.search("level")

    assert len(res) > 0
    assert isinstance(res[0], EUMType)


def test_dfsu3d_dataset():

    filename = "tests/testdata/oresund_sigma_z.dfsu"

    dfsu = Dfsu(filename)

    ds = dfsu.read()

    text = repr(ds)

    assert len(ds) == 3  # Z coordinate, Salinity, Temperature

    dsagg = ds.nanmean(axis=0)  # Time averaged

    assert len(dsagg) == 2  # Salinity, Temperature

    assert dsagg[0].shape[0] == 1

    assert dsagg.time[0] == ds.time[0]  # Time-averaged data index by start time

    ds_elm = dfsu.read(elements=[0])

    assert len(ds_elm) == 3  # Z coordinate, Salinity, Temperature

    dss = ds_elm.squeeze()

    assert len(dss) == 2  # Salinity, Temperature


def test_items_data_mismatch():

    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    time = _get_time(nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]  # Two items is not correct!

    with pytest.raises(ValueError):
        Dataset([d], time, items)


def test_time_data_mismatch():

    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    time = _get_time(nt + 1)  # 101 timesteps is not correct!
    items = [ItemInfo("Foo")]

    with pytest.raises(ValueError):
        Dataset([d], time, items)


def test_properties_dfs2():
    filename = "tests/testdata/gebco_sound.dfs2"
    dfs = Dfs2(filename)

    ds = dfs.read()
    assert ds.n_timesteps == 1
    assert ds.n_items == 1
    assert np.all(ds.shape == (1, 264, 216))
    assert ds.n_elements == (264 * 216)
    assert ds._first_non_z_item == 0
    assert ds.is_equidistant


def test_properties_dfsu():
    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    dfs = Dfsu(filename)

    ds = dfs.read()
    assert ds.n_timesteps == 3
    assert ds.n_items == 3
    assert np.all(ds.shape == (3, 441))
    assert ds.n_elements == 441
    assert ds._first_non_z_item == 1
    assert ds.is_equidistant
