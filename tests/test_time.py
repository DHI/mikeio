import pandas as pd

from mikeio._time import DateTimeSelector

def test_date_time_selector():

    idx = pd.date_range("2000-01-01", periods=4, freq="D")
    assert len(idx) == 4
    
    dts = DateTimeSelector(idx)

    assert dts.isel(None) == [0,1,2,3]
    assert dts.isel(0) == [0]
    assert dts.isel(-1) == [3]
    assert dts.isel([0,1]) == [0,1]
    assert dts.isel("2000-01-02") == [1]
    assert dts.isel(["2000-01-02", "2000-01-03"]) == [1,2]
    assert dts.isel(idx) == [0,1,2,3]
    assert dts.isel(slice(1,4)) == [1,2,3]
    assert dts.isel(slice("2000-01-02", "2000-01-04")) == [1,2,3]