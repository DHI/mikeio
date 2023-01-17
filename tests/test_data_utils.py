from datetime import datetime
import pandas as pd
import pytest

# TODO this file tests private methods, Options: 1. declare methods as public, 2. Test at a higher level of abstraction

from mikeio.data_utils import DataUtilsMixin as du


def test_parse_time_None():
    time = du._parse_time(None)
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_constant_str():
    time = du._parse_time("2018")
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_constant_datetime():
    time = du._parse_time(datetime(2018, 1, 1))
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_constant_Timestamp():
    time = du._parse_time(pd.Timestamp(2018, 1, 1))
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 1
    assert time[0] == pd.Timestamp(2018, 1, 1)


def test_parse_time_list_str():
    time = du._parse_time(["2018", "2018-1-2", "2018-1-3"])
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 3
    assert time[-1] == pd.Timestamp(2018, 1, 3)


def test_parse_time_list_datetime():
    time = du._parse_time(
        [datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3)]
    )
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 3
    assert time[-1] == pd.Timestamp(2018, 1, 3)


def test_parse_time_list_Timestamp():
    time = du._parse_time(
        [pd.Timestamp(2018, 1, 1), pd.Timestamp(2018, 1, 2), pd.Timestamp(2018, 1, 3)]
    )
    assert isinstance(time, pd.DatetimeIndex)
    assert len(time) == 3
    assert time[-1] == pd.Timestamp(2018, 1, 3)


def test_parse_time_decreasing():
    times = [
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 1, 15),
    ]

    with pytest.raises(ValueError, match="must be monotonic increasing"):
        du._parse_time(times)


def test_safe_name_noop():

    good_name = "MSLP"

    assert du._to_safe_name(good_name) == good_name


def test_safe_name_bad():

    # fmt: off
    bad_name   = "MSLP., 1:st level\n 2nd chain"
    safe_name  = "MSLP_1_st_level_2nd_chain"
    assert du._to_safe_name(bad_name) == safe_name
    # fmt : on
