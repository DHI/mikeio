import pandas as pd
import pytest
from datetime import datetime

from mikeio._time import DateTimeSelector


def test_date_time_selector_valid() -> None:
    idx = pd.date_range("2000-01-01", periods=4, freq="D")
    assert len(idx) == 4

    dts = DateTimeSelector(idx)

    assert dts.isel() == [0, 1, 2, 3]
    assert dts.isel(None) == [0, 1, 2, 3]
    assert dts.isel(0) == [0]
    assert dts.isel(-1) == [3]
    assert dts.isel([0, 1]) == [0, 1]
    assert dts.isel("2000-01-02") == [1]
    assert dts.isel(["2000-01-02", "2000-01-03"]) == [1, 2]
    assert dts.isel(idx) == [0, 1, 2, 3]
    assert dts.isel(slice(1, 4)) == [1, 2, 3]
    assert dts.isel(slice("2000-01-02", "2000-01-04")) == [1, 2, 3]
    assert dts.isel(datetime(2000, 1, 2)) == [1]


def test_out_of_range_int() -> None:
    idx = pd.date_range("2000-01-01", periods=4, freq="D")
    dts = DateTimeSelector(idx)
    with pytest.raises(IndexError):
        dts.isel(4)


def test_out_of_range_str() -> None:
    idx = pd.date_range("2000-01-01", periods=4, freq="D")
    dts = DateTimeSelector(idx)
    with pytest.raises(KeyError):
        dts.isel("2000-01-05")


def test_out_of_range_datetime() -> None:
    idx = pd.date_range("2000-01-01", periods=4, freq="D")
    dts = DateTimeSelector(idx)
    with pytest.raises(KeyError):
        dts.isel(datetime(2000, 1, 5))
