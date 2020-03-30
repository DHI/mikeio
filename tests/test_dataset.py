from datetime import datetime
import numpy as np
import pytest
from mikeio.dutil import Dataset
from mikeio.eum import EUMType, ItemInfo, EUMUnit


def test_get_names():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.items[0].name == "Foo"
    assert ds.items[0].type == EUMType.Undefined
    assert repr(ds.items[0].unit) == "undefined"


def test_get_data():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.data[0].shape == (100, 100, 30)


def test_get_data_2():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert data[0].shape == (100, 100, 30)


def test_get_data_name():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds["Foo"].shape == (100, 100, 30)


def test_get_bad_name():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    with pytest.raises(Exception):
        ds["BAR"]


def test_get_data_mulitple_name_fails():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    with pytest.raises(Exception):
        ds[["Foo", "Foo"]]


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


def test_iteminfo_string_type_should_fail_with_helpful_message():

    with pytest.raises(ValueError):

        item = ItemInfo("Water level", "Water level")


def test_item_search():

    res = EUMType.search("level")

    assert len(res) > 0
    assert isinstance(res[0], EUMType)


if __name__ == "__main__":
    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    (data, time, names) = Dataset(data, time, items)
