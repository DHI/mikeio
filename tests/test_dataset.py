from datetime import datetime
import numpy as np
import pytest
from mikeio.dutil import Dataset


def test_get_names():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    ds = Dataset(data, time, names)

    assert ds.names[0] == "Foo"


def test_get_data():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    ds = Dataset(data, time, names)

    assert ds.data[0].shape == (100, 100, 30)


def test_get_data_2():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    (data, time, names) = Dataset(data, time, names)

    assert data[0].shape == (100, 100, 30)


def test_get_data_name():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    ds = Dataset(data, time, names)

    assert ds["Foo"].shape == (100, 100, 30)


def test_get_bad_name():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    ds = Dataset(data, time, names)

    with pytest.raises(Exception):
        ds["BAR"]


def test_get_data_mulitple_name_fails():

    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    ds = Dataset(data, time, names)

    with pytest.raises(Exception):
        ds[["Foo", "Foo"]]


if __name__ == "__main__":
    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = [datetime.now()]
    names = ["Foo"]
    (data, time, names) = Dataset(data, time, names)
