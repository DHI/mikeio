from datetime import datetime
import numpy as np
import pytest
from mikeio.dutil import Dataset
from mikeio.eum import Item, ItemInfo, Unit


def test_get_names():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    ds = Dataset(data, time, items)

    assert ds.items[0].name == "Foo"
    assert ds.items[0].item == Item.Undefined
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
    assert item.item == Item.Undefined
    assert repr(item.unit) == "undefined"


def test_int_is_valid_type_info():

    item = ItemInfo("Foo", 100123)
    assert item.item == Item.Viscosity

    item = ItemInfo("U", 100002)
    assert item.item == Item.Wind_Velocity


def test_int_is_valid_unit_info():

    item = ItemInfo("U", 100002, 2000)
    assert item.item == Item.Wind_Velocity
    assert item.unit == Unit.meter_per_sec
    assert repr(item.unit) == "meter per sec"  # TODO replace _per_ with /


def test_default_unit_from_type():

    item = ItemInfo("Foo", Item.Water_Level)
    assert item.item == Item.Water_Level
    assert item.unit == Unit.meter
    assert repr(item.unit) == "meter"

    item = ItemInfo("Tp", Item.Wave_period)
    assert item.item == Item.Wave_period
    assert item.unit == Unit.second
    assert repr(item.unit) == "second"

    item = ItemInfo("Temperature", Item.Temperature)
    assert item.item == Item.Temperature
    assert item.unit == Unit.degree_Celsius
    assert repr(item.unit) == "degree Celsius"


def test_iteminfo_string_type_should_fail_with_helpful_message():

    with pytest.raises(ValueError):

        item = ItemInfo("Water level", "Water level")


if __name__ == "__main__":
    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    (data, time, names) = Dataset(data, time, items)
