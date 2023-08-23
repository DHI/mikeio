from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
from typing import List, Iterable, Optional

import pandas as pd


@dataclass
class DateTimeSelector:
    """Helper class for selecting time steps from a pandas DatetimeIndex"""

    index: pd.DatetimeIndex

    def isel(
        self,
        x: Optional[
            int | Iterable[int] | str | datetime | pd.DatetimeIndex | slice
        ] = None,
    ) -> List[int]:
        """Select time steps from a pandas DatetimeIndex

        Parameters
        ----------
        x : int, Iterable[int], str, datetime, pd.DatetimeIndex, slice
            Time steps to select, negative indices are supported

        Returns
        -------
        List[int]
            List of indices in the range (0, len(index)
        Examples
        --------
        >>> idx = pd.date_range("2000-01-01", periods=4, freq="D")
        >>> dts = DateTimeSelector(idx)
        >>> dts.isel(None)
        [0, 1, 2, 3]
        >>> dts.isel(0)
        [0]
        >>> dts.isel(-1)
        [3]
        """

        indices = list(range(len(self.index)))

        if x is None:
            return indices

        if isinstance(x, int):
            return [indices[x]]

        if isinstance(x, (datetime, str)):
            loc = self.index.get_loc(x)
            if isinstance(loc, int):
                return [loc]
            elif isinstance(loc, slice):
                return list(range(loc.start, loc.stop))

        if isinstance(x, slice):
            if isinstance(x.start, int) or isinstance(x.stop, int):
                return indices[x]
            else:
                s = self.index.slice_indexer(x.start, x.stop)
                return list(range(s.start, s.stop))

        if isinstance(x, Iterable):
            return [self.isel(t)[0] for t in x]

        return indices