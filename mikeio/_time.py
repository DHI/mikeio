from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
from collections.abc import Iterable

import pandas as pd


@dataclass
class DateTimeSelector:
    """Helper class for selecting time steps from a pandas DatetimeIndex."""

    index: pd.DatetimeIndex

    def isel(
        self,
        x: (
            int | Iterable[int] | str | datetime | pd.DatetimeIndex | slice | None
        ) = None,
    ) -> list[int]:
        """Select time steps from a pandas DatetimeIndex.

        Parameters
        ----------
        x : int, Iterable[int], str, datetime, pd.DatetimeIndex, slice
            Time steps to select, negative indices are supported

        Returns
        -------
        list[int]
            List of indices in the range (0, len(index)
        Examples
        --------
        ```{python}
        import mikeio
        import pandas as pd
        idx = pd.date_range("2000-01-01", periods=4, freq="D")
        dts = DateTimeSelector(idx)
        dts.isel(None)
        ```

        ```{python}
        dts.isel(0)
        ```

        ```{python}
        dts.isel(-1)
        ```

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
