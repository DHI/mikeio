from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
from typing import List, Iterable, Optional

import pandas as pd


@dataclass
class  DateTimeSelector:

    index: pd.DatetimeIndex

    @property
    def __len__(self):
        return len(self.index)

    def isel(self, time: Optional[int | Iterable[int] | str | datetime | pd.DatetimeIndex | slice]) -> List[int]:
        
        indices = list(range(len(self.index)))

        if time is None:
            return indices
        
        if isinstance(time, int):
            return [indices[time]]
        
        if isinstance(time, (datetime, str)):
            loc = self.index.get_loc(time)
            if isinstance(loc, int):
                return [loc]
            elif isinstance(loc, slice):
                return list(range(loc.start, loc.stop))

        if isinstance(time, slice):
            if isinstance(time.start, int) or isinstance(time.stop, int):
                return indices[time]
            else:
                s = self.index.slice_indexer(time.start, time.stop)
                return list(range(s.start, s.stop))
        
        if isinstance(time, Iterable):
            
            # recursive call
            return [self.isel(t)[0] for t in time]
        
        return indices

    


if __name__ == "__main__":
        
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

        