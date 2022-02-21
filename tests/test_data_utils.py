from datetime import datetime
import pandas as pd


import mikeio.data_utils as du


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
