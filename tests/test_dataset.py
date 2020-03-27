from datetime import datetime
import numpy as np
import pytest
from mikeio.dutil import Dataset
from mikeio.eum import Item, ItemInfo


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


if __name__ == "__main__":
    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    items = [ItemInfo("Foo")]
    (data, time, names) = Dataset(data, time, items)
