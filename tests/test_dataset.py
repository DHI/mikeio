from datetime import datetime
from dateutil.rrule import rrule, SECONDLY
import numpy as np
import pandas as pd
import pytest
from mikeio.dutil import Dataset
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


def test_get_data():

    data = []
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = _get_time(nt)
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.data[0].shape == (100, 100, 30)


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
